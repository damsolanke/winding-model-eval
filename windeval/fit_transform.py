"""Fit an affine map from one scan's voxel frame to another by CT block matching.

Block centres are mesh vertices (so they sit on papyrus) in the source frame.
Each centre is carried into the target frame by an initial transform; a
cubic block of the target volume around it is compared with the source
volume resampled onto the same target grid, and phase correlation gives the
residual shift. The shifted centres and the source centres form point
correspondences for a robust least-squares affine fit. Residuals are
reported on the fitting blocks and on held-out blocks (two-fold split).

Usage: python -m windeval.fit_transform SPEC.json
SPEC keys: source_volume (array dir), target_volume (array dir), segments
(tifxyz dir in source-array units), init_matrix (3x4, source array xyz to
target array xyz) or init_transform_json (+ init_source_scale /
init_target_scale), z_range (source units), n_blocks, block, seed, out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windeval import env

env.setup()

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

import vc  # noqa: E402


def expand(path: str) -> Path:
    return env.expand_path(path)


def sample_block(volume, center_xyz, size, mapping=None):
    """Block [z, y, x] of `size`^3 voxels on the integer grid around a target-frame centre.

    With `mapping` (3x4, target xyz -> volume xyz) the grid is pulled from another frame.
    """
    half = size // 2
    offsets = np.arange(-half, size - half, dtype=np.float64)
    zz, yy, xx = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    grid = np.stack([xx, yy, zz], -1).reshape(-1, 3) + np.round(center_xyz)
    if mapping is not None:
        grid = grid @ mapping[:, :3].T + mapping[:, 3]
    values, _ = volume._sample(grid.astype(np.float32), "trilinear")
    return values.reshape(size, size, size).astype(np.float32)


def phase_shift(fixed, moving):
    """Sub-voxel shift s (z, y, x) such that moving(x) ~ fixed(x + s), and the NCC there."""
    n = fixed.shape[0]
    win = np.hanning(n)
    w = win[:, None, None] * win[None, :, None] * win[None, None, :]
    fa = np.fft.fftn((fixed - fixed.mean()) * w)
    fb = np.fft.fftn((moving - moving.mean()) * w)
    cross = fa * np.conj(fb)
    corr = np.real(np.fft.ifftn(cross / (np.abs(cross) + 1e-9)))
    peak = np.array(np.unravel_index(np.argmax(corr), corr.shape))
    shift = peak.astype(np.float64)
    for axis in range(3):
        lo, hi = peak.copy(), peak.copy()
        lo[axis] = (peak[axis] - 1) % n
        hi[axis] = (peak[axis] + 1) % n
        c0, cm, cp = corr[tuple(peak)], corr[tuple(lo)], corr[tuple(hi)]
        denom = cm - 2 * c0 + cp
        if denom < 0:
            shift[axis] += 0.5 * (cm - cp) / denom
    shift = np.where(shift > n / 2, shift - n, shift)
    s = np.round(shift).astype(int)
    a = np.roll(fixed, tuple(-s), axis=(0, 1, 2))
    core = slice(abs(s).max() + 1, n - abs(s).max() - 1)
    aa, bb = a[core, core, core].ravel(), moving[core, core, core].ravel()
    ncc = float(np.corrcoef(aa, bb)[0, 1]) if aa.size > 10 and aa.std() > 0 and bb.std() > 0 else 0.0
    return shift, ncc, float(corr.max())


def fit_affine(src, dst, iters=5, k=3.0):
    """Robust least squares dst ~ src @ A.T + b; returns (3x4, inlier mask)."""
    keep = np.ones(len(src), dtype=bool)
    for _ in range(iters):
        X = np.c_[src[keep], np.ones(keep.sum())]
        sol, *_ = np.linalg.lstsq(X, dst[keep], rcond=None)
        resid = np.linalg.norm(np.c_[src, np.ones(len(src))] @ sol - dst, axis=1)
        mad = np.median(np.abs(resid[keep] - np.median(resid[keep]))) + 1e-9
        new_keep = resid <= np.median(resid[keep]) + k * 1.4826 * mad
        if (new_keep == keep).all():
            break
        keep = new_keep
    return sol.T, keep


def apply(m, pts):
    return pts @ m[:, :3].T + m[:, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=Path)
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    src_vol = vc.Volume.open(str(expand(spec["source_volume"])))
    dst_vol = vc.Volume.open(str(expand(spec["target_volume"])))
    if "init_matrix" in spec:
        init = np.asarray(spec["init_matrix"], dtype=np.float64)
    else:
        # The source volume's transform.json maps its base voxels into the target volume's base
        # voxels; rescale to the opened arrays (e.g. level 2 of the source: scale 4).
        values = np.asarray(json.loads(expand(spec["init_transform_json"]).read_text())["transformation_matrix"])
        s_src, s_dst = float(spec.get("init_source_scale", 1.0)), float(spec.get("init_target_scale", 1.0))
        init = np.c_[values[:, :3] * s_src / s_dst, values[:, 3] / s_dst]

    rng = np.random.default_rng(int(spec.get("seed", 0)))
    zlo, zhi = spec["z_range"]
    centers = []
    for seg in sorted(p for p in expand(spec["segments"]).iterdir() if (p / "z.tif").exists()):
        x, y, z = (tifffile.imread(seg / f"{c}.tif").astype(np.float64) for c in "xyz")
        valid = (z > 0) & np.isfinite(z) & (z >= zlo) & (z < zhi)
        pts = np.stack([x[valid], y[valid], z[valid]], -1)
        if len(pts):
            centers.append(pts[rng.choice(len(pts), min(len(pts), 64), replace=False)])
    centers = np.concatenate(centers)
    centers = centers[rng.choice(len(centers), min(len(centers), int(spec.get("n_blocks", 240))), replace=False)]
    size = int(spec.get("block", 48))

    ncc_min = float(spec.get("ncc_min", 0.3))
    n_iter = int(spec.get("iterations", 4))

    def match(current):
        """Measure each block's residual shift under `current` (source -> target)."""
        inverse = np.linalg.inv(np.r_[current, [[0, 0, 0, 1]]])[:3]
        rows = []
        for c_src in centers:
            c_dst = apply(current, c_src[None])[0]
            fixed = sample_block(dst_vol, c_dst, size)
            moving = sample_block(src_vol, c_dst, size, mapping=inverse)
            if fixed.std() < 1 or moving.std() < 1:
                continue
            shift_zyx, ncc, peak = phase_shift(fixed, moving)
            rows.append({"source_xyz": apply(inverse, np.round(c_dst)[None])[0].tolist(),
                         "target_xyz": (np.round(c_dst) + shift_zyx[::-1]).tolist(),
                         "shift_xyz": shift_zyx[::-1].tolist(), "ncc": ncc, "peak": peak})
        good = [r for r in rows if r["ncc"] >= ncc_min and np.abs(r["shift_xyz"]).max() < size / 3]
        return rows, good

    # Phase correlation with a fixed window under-reads sub-voxel shifts (it is
    # pulled toward zero), so the fit is iterated: once the residual shifts are
    # near zero the bias no longer matters.
    current = init
    history = []
    for iteration in range(n_iter):
        rows, good = match(current)
        src = np.array([r["source_xyz"] for r in good])
        dst = np.array([r["target_xyz"] for r in good])
        shifts = np.linalg.norm(np.array([r["shift_xyz"] for r in good]), axis=1)
        history.append({"iteration": iteration, "blocks_used": len(good),
                        "median_shift": float(np.median(shifts)), "p90_shift": float(np.percentile(shifts, 90))})
        print(json.dumps(history[-1]), flush=True)
        matrix, inliers = fit_affine(src, dst)
        current = matrix
    # held-out check: refit on half the blocks, measure the other half's shift under that fit
    fold = rng.permutation(len(src)) % 2
    held = []
    for f in (0, 1):
        m_f, _ = fit_affine(src[fold != f], dst[fold != f])
        held.append(np.linalg.norm(apply(m_f, src[fold == f]) - dst[fold == f], axis=1))
    held = np.concatenate(held)
    final_rows, final_good = match(current)
    final_shift = np.linalg.norm(np.array([r["shift_xyz"] for r in final_good]), axis=1)
    resid = np.linalg.norm(apply(matrix, src) - dst, axis=1)
    init_rows, init_good = (rows, good) if n_iter == 1 else match(init)
    init_shift = np.linalg.norm(np.array([r["shift_xyz"] for r in init_good]), axis=1)
    stats = lambda r: {"median": float(np.median(r)), "p90": float(np.percentile(r, 90)), "mean": float(np.mean(r))}
    out = {
        "source_volume": spec["source_volume"], "target_volume": spec["target_volume"],
        "segments": spec["segments"], "z_range": spec["z_range"], "block": size,
        "blocks_tried": len(centers), "blocks_used_final": len(final_good),
        "inliers_last_fit": int(inliers.sum()), "ncc_min": ncc_min,
        "ncc_median_final": float(np.median([r["ncc"] for r in final_good])),
        "iterations": history,
        "matrix_source_to_target_xyz": matrix.tolist(),
        "init_matrix_source_to_target_xyz": init.tolist(),
        "measured_shift_under_init_target_voxels": stats(init_shift),
        "measured_shift_under_fit_target_voxels": stats(final_shift),
        "residual_fit_target_voxels": stats(resid[inliers]),
        "residual_heldout_target_voxels": stats(held),
        "note": "measured shifts come from windowed phase correlation, which under-reads sub-voxel shifts; "
                "the iterated fit drives them toward zero",
        "pairs_final": final_rows,
    }
    path = expand(spec["out"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("pairs_final",)}, indent=1))


if __name__ == "__main__":
    main()
