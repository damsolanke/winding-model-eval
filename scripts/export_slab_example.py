"""Export labelled crossings and decoded phase passages for example slabs (no CT pixels).

Draws slabs exactly as evaluate.py does for CONFIG (restricted to rays that start
on SOURCE windings), runs the model, registers each slab's phase offset the way
accumulate_winding_metrics does, and writes, for the centre row of columns, the
labelled crossing positions with their winding numbers and the decoded passage
positions. Also the centre column's target and registered predicted phase.

Usage: python scripts/export_slab_example.py CONFIG.json OUT.json --source-windings 41 43 [--slabs 4] [--seed 7]
"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windeval import env  # noqa: E402

env.setup()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vesuvius.neural_tracing.winding_models import winding_targets  # noqa: E402
from vesuvius.neural_tracing.winding_models.winding_model import WindingModel  # noqa: E402
from windeval.lazy import build_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--source-windings", type=int, nargs="*")
    ap.add_argument("--slabs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    spec = json.loads(args.config.read_text())
    ckpt = torch.load(env.expand_path(spec["checkpoint"]), map_location="cpu", weights_only=False)
    config = copy.deepcopy(ckpt["config"])
    config.update(spec.get("overrides", {}))
    config["datasets"] = [{**d, "volume_path": str(env.expand_path(d["volume_path"])),
                           "segments_path": str(env.expand_path(d["segments_path"]))} for d in spec["datasets"]]
    config["segment_cache_dir"] = str(env.expand_path("${WINDEVAL_DATA}/cache/winding_segments"))
    torch.manual_seed(args.seed)
    dataset = build_dataset(config, spec.get("region_zyx"), args.source_windings)
    device = env.pick_device("auto")
    model = WindingModel(config["model"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    examples = []
    for _ in range(args.slabs):
        sample = dataset[0]
        batch = winding_targets.collate_winding_batch([sample])
        with torch.no_grad():
            out = model(batch["slab_image"].to(device), batch["slab_valid"].to(device))
        phase = out["phase"][0].float().cpu().numpy()
        target = batch["phase_target"][0].numpy()
        valid = batch["phase_valid"][0].numpy().astype(bool)
        cvalid = batch["crossing_valid"][0].numpy().astype(bool)
        offset = float(phase[valid].mean() - target[valid].mean())
        ct, ci, n = (batch["crossing_t"][0].numpy(), batch["crossing_indices"][0].numpy(),
                     batch["num_crossings"][0].numpy())
        row = phase.shape[0] // 2
        columns = []
        for j in range(phase.shape[1]):
            pos, _ = winding_targets.phase_passages(phase[row, j] - offset)
            peaks = np.clip(np.rint(pos).astype(int), 0, phase.shape[-1] - 1)
            columns.append({
                "labels_t": [float(t) for t in ct[row, j, :n[row, j]]],
                "labels_w": [abs(int(w)) for w in ci[row, j, :n[row, j]]],
                "passages_t": [float(p) for p, k in zip(pos, cvalid[row, j][peaks]) if k],
                "supervised_from": float(np.flatnonzero(valid[row, j])[0]) if valid[row, j].any() else None,
                "supervised_to": float(np.flatnonzero(valid[row, j])[-1]) if valid[row, j].any() else None,
            })
        col = phase.shape[1] // 2
        examples.append({
            "centre_row_columns": columns,
            "centre_column_target_phase": [float(v) if m else None for v, m in zip(target[row, col], valid[row, col])],
            "centre_column_pred_phase": [float(v) for v in phase[row, col] - offset],
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": spec["name"], "source_windings": args.source_windings,
                                    "seed": args.seed, "examples": examples}))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
