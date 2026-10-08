"""Summarize evaluate.py runs next to the released training-batch metrics.

Usage: python -m windeval.report RUN_DIR [RUN_DIR ...] [--metrics PATH] [--sectors 8]

For each run: pooled metrics (the trainer's own pooling), a slab bootstrap and
a spatial block bootstrap (blocks = angular sectors of the ray origin around
the centroid of all origins in the run), plus the training-batch rows of
metrics.jsonl for reference (last row, and mean / sd / min / max of the last
20 logged rows).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

KEYS = ("val_crossing_precision", "val_crossing_recall", "val_winding_count_error", "val_phase_mae")
SHORT = {"val_crossing_precision": "precision", "val_crossing_recall": "recall",
         "val_winding_count_error": "count_err", "val_phase_mae": "phase_mae"}


def pooled(sums_list):
    tot = {}
    for sums in sums_list:
        for key, value in sums.items():
            tot[key] = tot.get(key, 0.0) + value
    tp, fp, fn = tot["crossing_tp"], tot["crossing_fp"], tot["crossing_fn"]
    return {
        "val_crossing_precision": tp / max(1.0, tp + fp),
        "val_crossing_recall": tp / max(1.0, tp + fn),
        "val_winding_count_error": tot["count_error"] / max(1.0, tot["count_columns"]),
        "val_phase_mae": tot["phase_abs_error"] / max(1.0, tot["phase_slabs"]),
    }


def block_bootstrap(groups, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(groups)
    draws = {k: [] for k in KEYS}
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        sums = [s for i in pick for s in groups[keys[i]]]
        m = pooled(sums)
        for k in KEYS:
            draws[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in draws.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--metrics", type=Path,
                    default=None)
    ap.add_argument("--sectors", type=int, default=8)
    args = ap.parse_args()

    if args.metrics is None:
        from windeval.env import expand_path

        args.metrics = expand_path("${WINDEVAL_DATA}/hf/winding_model_9um/metrics.jsonl")
    rows = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    last, tail = rows[-1], rows[-20:]
    print(f"training-batch reference (metrics.jsonl, step {last['step']}):")
    for k in KEYS:
        vals = [r[k] for r in tail]
        print(f"  {SHORT[k]:10s} last {last[k]:.3f} | last-20 mean {statistics.mean(vals):.3f} "
              f"sd {statistics.pstdev(vals):.3f} min {min(vals):.3f} max {max(vals):.3f}")

    for run in args.runs:
        result = json.loads((run / "result.json").read_text())
        slabs = [json.loads(line) for line in (run / "slabs.jsonl").read_text().splitlines()]
        origins = np.array([s["origin_vol_zyx"] for s in slabs], dtype=np.float64)
        center = origins[:, 1:].mean(axis=0)
        angle = np.arctan2(origins[:, 1] - center[0], origins[:, 2] - center[1])
        sector = np.floor((angle + np.pi) / (2 * np.pi) * args.sectors).astype(int) % args.sectors
        groups = {}
        for s, b in zip(slabs, sector):
            groups.setdefault(int(b), []).append(s["sums"])
        m = pooled([s["sums"] for s in slabs])
        boot = result.get("bootstrap_95") or {}
        bb = block_bootstrap(groups)
        print(f"\n{result['name']}: {len(slabs)} slabs, {result['segments']} segments, "
              f"{result['supervised_columns']} supervised columns, {result['crossings_labelled']} labelled crossings, "
              f"weights={result['weights']}, precision={result.get('precision')}, device={result['device']}")
        names = {"val_crossing_precision": "precision", "val_crossing_recall": "recall",
                 "val_winding_count_error": "count_error", "val_phase_mae": "phase_mae"}
        for k in KEYS:
            lo, hi = boot.get(names[k], (float("nan"), float("nan")))
            print(f"  {SHORT[k]:10s} {m[k]:.3f}  slab-boot95 [{lo:.3f}, {hi:.3f}]  "
                  f"sector-block-boot95 [{bb[k][0]:.3f}, {bb[k][1]:.3f}] ({len(groups)} blocks)")
        gaps = np.array([s.get("median_winding_gap") or np.nan for s in slabs], dtype=np.float64)
        if np.isfinite(gaps).sum() >= 9:
            cuts = np.nanpercentile(gaps, [100 / 3, 200 / 3])
            labels = np.digitize(gaps, cuts)
            parts = []
            for lab, name in enumerate(("compressed", "middle", "open")):
                sel = [s["sums"] for s, l, g in zip(slabs, labels, gaps) if l == lab and np.isfinite(g)]
                if sel:
                    v = pooled(sel)
                    parts.append(f"{name} (n={len(sel)}) P={v['val_crossing_precision']:.2f} "
                                 f"R={v['val_crossing_recall']:.2f} CE={v['val_winding_count_error']:.2f} "
                                 f"PMAE={v['val_phase_mae']:.3f}")
            print(f"  by winding spacing (tertile cuts {cuts[0]:.1f}, {cuts[1]:.1f} samples/winding): " + "; ".join(parts))
        per_sector = {b: pooled(g) for b, g in sorted(groups.items())}
        print("  by sector: " + "; ".join(
            f"{b}: n={len(groups[b])} P={v['val_crossing_precision']:.2f} R={v['val_crossing_recall']:.2f} "
            f"CE={v['val_winding_count_error']:.2f}" for b, v in per_sector.items()))
        t = result["timing_s"]
        mem = result["memory_bytes"]
        data = result["data"]
        print(f"  timing: wall {t['wall']:.0f} s, {t['per_slab_wall']:.2f} s/slab; forward per batch of "
              f"{result['batch_size']} mean {t['forward_per_batch']['mean']:.2f} s (p95 {t['forward_per_batch']['p95']:.2f}); "
              f"data wait mean {t['data_wait_per_batch']['mean']:.2f} s; metrics {t['metrics_per_batch']['mean']:.2f} s")
        print(f"  memory: MPS driver peak {mem.get('mps_driver', 0) / 2**30:.1f} GiB, tensors {mem.get('mps_current', 0) / 2**30:.2f} GiB, "
              f"main RSS {mem['rss_main'] / 2**30:.1f} GiB, workers RSS {mem['rss_workers'] / 2**30:.1f} GiB")
        print(f"  data: streamed this run {data['bytes_streamed_this_run'] / 1e9:.2f} GB, mirror total "
              f"{data['mirror_bytes_total'] / 1e9:.2f} GB in {data['mirror_chunk_files']} chunks")


if __name__ == "__main__":
    main()
