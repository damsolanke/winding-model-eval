"""Spacing between consecutive winding meshes, measured along the sheet normal.

For each segment w_k (vertices inside a z band) the nearest vertex of the
next labelled winding w_{k+1} is found and the offset is projected on w_k's
normal. Reports median and 10th / 90th percentiles per consecutive pair, in
the meshes' coordinate units. Several meshes may carry the same winding
number; each pair of meshes is reported.

Usage: winding_spacing.py SEGMENTS_DIR Z_LO Z_HI OUT.json [--max-points 20000]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

WINDING = re.compile(r"(?:^|-)w(?P<index>\d+)(?:_|$)")


def load(seg, zlo, zhi, rng, max_points):
    xyz = np.stack([tifffile.imread(seg / f"{c}.tif") for c in "xyz"], -1).astype(np.float64)
    valid = xyz[..., 2] > 0
    du, dv = np.gradient(xyz, axis=0), np.gradient(xyz, axis=1)
    normal = np.cross(du, dv)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12
    interior = valid.copy()
    interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
    interior[1:-1, 1:-1] &= valid[:-2, 1:-1] & valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
    band = interior & (xyz[..., 2] >= zlo) & (xyz[..., 2] < zhi)
    idx = np.argwhere(band)
    if len(idx) > max_points:
        idx = idx[rng.choice(len(idx), max_points, replace=False)]
    all_band = valid & (xyz[..., 2] >= zlo - 50) & (xyz[..., 2] < zhi + 50)
    return xyz[idx[:, 0], idx[:, 1]], normal[idx[:, 0], idx[:, 1]], xyz[all_band]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segments", type=Path)
    ap.add_argument("z_lo", type=float)
    ap.add_argument("z_hi", type=float)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-points", type=int, default=20000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    meshes = {}
    for seg in sorted(p for p in args.segments.iterdir() if (p / "z.tif").exists()):
        m = WINDING.search(seg.name)
        if m:
            meshes.setdefault(int(m["index"]), []).append((seg.name, *load(seg, args.z_lo, args.z_hi, rng, args.max_points)))
    pairs = []
    for w in sorted(meshes):
        if w + 1 not in meshes:
            continue
        for name_a, pts, normals, _ in meshes[w]:
            for name_b, _, _, cloud in meshes[w + 1]:
                if len(pts) < 50 or len(cloud) < 50:
                    continue
                _, nearest = cKDTree(cloud).query(pts)
                along = np.abs(np.einsum("ij,ij->i", cloud[nearest] - pts, normals))
                pairs.append({"from": w, "to": w + 1, "from_mesh": name_a, "to_mesh": name_b, "n": int(len(pts)),
                              "median": float(np.median(along)), "p10": float(np.percentile(along, 10)),
                              "p90": float(np.percentile(along, 90))})
                print(f"w{w:03d} -> w{w + 1:03d}: median {pairs[-1]['median']:.1f}  p10 {pairs[-1]['p10']:.1f}  p90 {pairs[-1]['p90']:.1f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"segments": args.segments.name, "z_band": [args.z_lo, args.z_hi],
                                    "units": "segment coordinate units", "pairs": pairs}, indent=1))


if __name__ == "__main__":
    main()
