"""Diagnostics for a winding-model evaluation config.

For a handful of slabs drawn exactly as evaluate.py draws them, this
1. checks the vc shim's trilinear slab against scipy.ndimage.map_coordinates
   on the same coordinates (independent implementation),
2. collects signed residuals between each labelled crossing and the nearest
   decoded phase passage (after the per-slab offset registration the metric
   uses), in ray samples, to separate "wrong count" from "right count, shifted",
3. writes villa's own training visualization for the first slabs.

Usage: python -m windeval.diagnose CONFIG.json [--num-slabs N] [--out DIR]
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from windeval import env

env.setup()

import numpy as np  # noqa: E402
import scipy.ndimage  # noqa: E402
import torch  # noqa: E402

from vesuvius.neural_tracing.winding_models import winding_targets  # noqa: E402
from vesuvius.neural_tracing.winding_models.winding_model import WindingModel  # noqa: E402
from vesuvius.neural_tracing.winding_models.winding_model_dataset import WindingModelDataset  # noqa: E402
from vesuvius.neural_tracing.winding_models.winding_visualization import make_winding_visualization  # noqa: E402

from windeval.evaluate import expand  # noqa: E402
from windeval.sampling import restrict_sources  # noqa: E402


def shim_vs_scipy(dataset, volume_idx, direction, origin) -> dict:
    extractor = dataset.slab_extractor
    image, valid, frame = extractor.extract(volume_idx, direction, origin)
    size, length = extractor.transverse_size, extractor.ray_length
    ii, jj, kk = np.meshgrid(np.arange(size), np.arange(size), np.arange(length), indexing="ij")
    seg_xyz = frame.to_world(np.stack([ii, jj, kk], -1).reshape(-1, 3).astype(np.float64))
    transform = extractor.segment_to_volume_xyz[volume_idx]
    vol_xyz = seg_xyz @ transform[:3, :3].T + transform[:3, 3]
    lo = np.floor(vol_xyz.min(0)).astype(int) - 1
    hi = np.ceil(vol_xyz.max(0)).astype(int) + 2
    vol = extractor._volume(volume_idx)
    store = vol.store
    lo = np.maximum(lo, 0)
    block = np.zeros((hi[2] - lo[2], hi[1] - lo[1], hi[0] - lo[0]), np.uint8)
    cz, cy, cx = store.chunks
    for kz in range(lo[2] // cz, (hi[2] - 1) // cz + 1):
        for ky in range(lo[1] // cy, (hi[1] - 1) // cy + 1):
            for kx in range(lo[0] // cx, (hi[0] - 1) // cx + 1):
                data = store.get(kz, ky, kx)
                if data is None:
                    continue
                z0, y0, x0 = kz * cz, ky * cy, kx * cx
                sz = slice(max(z0, lo[2]), min(z0 + cz, hi[2]))
                sy = slice(max(y0, lo[1]), min(y0 + cy, hi[1]))
                sx = slice(max(x0, lo[0]), min(x0 + cx, hi[0]))
                block[sz.start - lo[2]:sz.stop - lo[2], sy.start - lo[1]:sy.stop - lo[1], sx.start - lo[0]:sx.stop - lo[0]] = \
                    data[sz.start - z0:sz.stop - z0, sy.start - y0:sy.stop - y0, sx.start - x0:sx.stop - x0]
    coords = (vol_xyz - lo)[:, ::-1].T
    ref = scipy.ndimage.map_coordinates(block.astype(np.float32), coords, order=1, mode="constant", cval=0.0)
    ref = np.floor(np.clip(ref, 0, 255) + 1e-4).astype(np.int32).reshape(image.shape)
    diff = np.abs(image.astype(np.int32) - ref)
    return {"max_abs_diff": int(diff.max()), "frac_diff_gt1": float((diff > 1).mean()),
            "frac_equal": float((diff == 0).mean()), "mean_intensity": float(image.mean())}


def residuals(output, batch, config, max_abs=8.0):
    """Signed (passage - label) offsets, per labelled crossing, in samples."""
    phase = output["phase"][0].float().cpu().numpy()
    target = batch["phase_target"][0].numpy()
    valid = batch["phase_valid"][0].numpy().astype(bool)
    ct = batch["crossing_t"][0].numpy()
    counts = batch["num_crossings"][0].numpy()
    crossing_valid = batch["crossing_valid"][0].numpy().astype(bool)
    spacing = float(config.get("spacing", 1.0))
    offset = float(phase[valid].mean() - target[valid].mean()) if valid.any() else 0.0
    out = []
    for i, j in np.argwhere(counts >= 2):
        positions, _ = winding_targets.phase_passages(phase[i, j] - offset)
        peaks = np.clip(np.rint(positions).astype(int), 0, phase.shape[-1] - 1)
        positions = positions[crossing_valid[i, j][peaks]]
        if not len(positions):
            continue
        for t in ct[i, j, : counts[i, j]] / spacing:
            d = positions - t
            k = np.argmin(np.abs(d))
            if abs(d[k]) <= max_abs:
                out.append(d[k])
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--num-slabs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--source-windings", type=int, nargs="*")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    spec = json.loads(args.config.read_text())
    ckpt = torch.load(expand(spec["checkpoint"]), map_location="cpu", weights_only=False)
    config = copy.deepcopy(ckpt["config"])
    config.update(spec.get("overrides", {}))
    config["datasets"] = [{**d, "volume_path": expand(d["volume_path"]), "segments_path": expand(d["segments_path"])}
                          for d in spec["datasets"]]
    config["segment_cache_dir"] = expand("${WINDEVAL_DATA}/cache/winding_segments")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = WindingModelDataset(config)
    if spec.get("region_zyx"):
        restrict_sources(dataset, spec["region_zyx"]["lo"], spec["region_zyx"]["hi"],
                         args.source_windings or spec.get("source_windings"))
    device = env.pick_device(spec.get("device", "auto"))
    model = WindingModel(config["model"])
    model.load_state_dict(ckpt[spec.get("weights", "model")])
    model.to(device).eval()
    out_dir = (args.out or env.expand_path("${WINDEVAL_DATA}/runs/diagnose")) / (spec["name"] + args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_res, report = [], []
    for n in range(args.num_slabs):
        sample = dataset[0]
        batch = winding_targets.collate_winding_batch([sample])
        with torch.no_grad():
            output = model(batch["slab_image"].to(device), batch["slab_valid"].to(device))
        output = {k: v.cpu() for k, v in output.items()}
        res = residuals(output, batch, config)
        all_res.append(res)
        entry = {"slab": n, "n_matched_within_8": int(len(res)),
                 "median_residual": float(np.median(res)) if len(res) else None,
                 "mean_abs_residual": float(np.mean(np.abs(res))) if len(res) else None,
                 "frac_within_2": float(np.mean(np.abs(res) <= 2.0)) if len(res) else None}
        if n == 0:
            direction = sample["ray_direction_zyx"].numpy()[::-1].astype(np.float64)
            # extract() wants the ray origin, not the frame corner (slab_origin_zyx).
            ray_origin = sample["ray_origin_zyx"].numpy()[::-1].astype(np.float64)
            entry["shim_vs_scipy"] = shim_vs_scipy(dataset, int(sample["volume_idx"]), direction, ray_origin)
        if n < 3:
            make_winding_visualization(batch, output, str(out_dir / f"slab{n:02d}.png"), spacing=1.0,
                                       peak_threshold=float(config.get("crossing_peak_threshold", 0.3)),
                                       peak_min_distance=2,
                                       density_min_gap_wv=float(config.get("density_min_gap_wv", 4.0)),
                                       crossing_sigma_wv=float(config.get("crossing_sigma_wv", 1.0)))
        report.append(entry)
        print(json.dumps(entry), flush=True)
    res = np.concatenate(all_res) if all_res else np.zeros(0)
    hist, edges = np.histogram(res, bins=np.arange(-8.0, 8.5, 0.5))
    summary = {"n": int(len(res)), "median": float(np.median(res)), "mean": float(res.mean()),
               "frac_within_1": float(np.mean(np.abs(res) <= 1)), "frac_within_2": float(np.mean(np.abs(res) <= 2)),
               "hist_edges": edges.tolist(), "hist": hist.tolist()}
    (out_dir / "diagnose.json").write_text(json.dumps({"slabs": report, "residuals": summary}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("hist")}))
    for lo_edge, count in zip(edges[:-1], hist):
        print(f"{lo_edge:+5.1f} {'#' * int(60 * count / max(1, hist.max()))} {count}")


if __name__ == "__main__":
    main()
