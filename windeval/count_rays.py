"""Count how many supervised rays a segment set yields under villa's own sampling rules.

Stage 1 asks the dataset's ray builders directly (`_labelled_ray` for segments
with a winding number, `_multi_wrap_ray` for the others) how often a random
origin gives an ordered crossing list with at least two windings. Stage 2
draws full samples with `dataset[i]` (each call makes up to
max_sample_attempts attempts) and reports supervised columns and crossings.

Usage: python -m windeval.count_rays CONFIG.json [--attempts 2000] [--samples 16]
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
from pathlib import Path

from windeval import env

env.setup()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from windeval.lazy import build_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--attempts", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    spec = json.loads(args.config.read_text())
    ckpt = torch.load(env.expand_path(spec["checkpoint"]), map_location="cpu", weights_only=False)
    config = copy.deepcopy(ckpt["config"])
    config.update(spec.get("overrides", {}))
    config["datasets"] = [{**d, "volume_path": str(env.expand_path(d["volume_path"])),
                           "segments_path": str(env.expand_path(d["segments_path"]))} for d in spec["datasets"]]
    config["segment_cache_dir"] = str(env.expand_path("${WINDEVAL_DATA}/cache/winding_segments"))
    if "segment_to_volume_scale" in spec:
        config["_segment_to_volume_scale"] = spec["segment_to_volume_scale"]
    torch.manual_seed(args.seed)
    dataset = build_dataset(config, spec.get("region_zyx"), spec.get("source_windings"))

    stage1 = collections.Counter()
    crossings_per_ray = []
    for _ in range(args.attempts):
        volume_idx, segment = dataset._sample_volume_and_segment()
        volume = dataset.volumes[volume_idx]
        kind = "labelled" if segment.winding_idx is not None else "multi_wrap"
        ray = (dataset._labelled_ray(volume, segment) if kind == "labelled"
               else dataset._multi_wrap_ray(segment))
        stage1[f"{kind}_tried"] += 1
        if ray is not None:
            stage1[f"{kind}_rays"] += 1
            crossings_per_ray.append(len(ray[2]))

    stage2 = {"calls": args.samples, "samples": 0, "failed_calls": 0, "supervised_columns": [], "crossings": []}
    for i in range(args.samples):
        try:
            sample = dataset[i]
        except RuntimeError as exc:
            if "Could not find a slab" not in str(exc):
                raise
            stage2["failed_calls"] += 1
            continue
        counts = sample["num_crossings"].numpy()
        stage2["samples"] += 1
        stage2["supervised_columns"].append(int((counts >= 2).sum()))
        stage2["crossings"].append(int(counts[counts >= 2].sum()))
    out = {
        "config": str(args.config),
        "segments": [(int(s.winding_idx) if s.winding_idx is not None else None, int(len(s.sample_cells)))
                     for v in dataset.volumes for s in v.segments],
        "stage1": dict(stage1),
        "stage1_crossings_per_ray": (np.percentile(crossings_per_ray, [0, 50, 100]).tolist()
                                     if crossings_per_ray else None),
        "stage2": {**stage2,
                   "attempts_per_call": dataset.max_sample_attempts,
                   "median_supervised_columns": float(np.median(stage2["supervised_columns"])) if stage2["supervised_columns"] else 0,
                   "median_crossings": float(np.median(stage2["crossings"])) if stage2["crossings"] else 0},
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
