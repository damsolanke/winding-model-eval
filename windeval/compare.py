"""Paired comparison of evaluate.py runs that share their slabs.

For every run: pooled metrics over all slabs and over two groups split by the
labelled winding bracket (slabs whose bracket touches windings LO..HI versus
slabs that avoid them; for runs on other slabs the split uses their own
brackets), each with a 95 % slab-bootstrap interval. Runs drawn on
exactly the same slabs as the reference (same ray origins) also get a paired
bootstrap interval for the difference to the reference.

Usage: python -m windeval.compare --ref RUN_DIR RUN_DIR [RUN_DIR ...]
       [--band 40 49] [--labels name ...] [--json OUT.json] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ("crossing_tp", "crossing_fp", "crossing_fn", "count_error", "count_columns",
        "phase_abs_error", "phase_slabs")
METRICS = ("precision", "recall", "count_error", "phase_mae")


def table(slabs):
    return np.array([[s["sums"].get(k, 0.0) for k in KEYS] for s in slabs], dtype=np.float64)


def metrics(sums):
    tp, fp, fn, ce, cc, pe, ps = sums
    return {"precision": tp / max(1.0, tp + fp), "recall": tp / max(1.0, tp + fn),
            "count_error": ce / max(1.0, cc), "phase_mae": pe / max(1.0, ps)}


def boot(tab, idx_sets, rng):
    out = {m: [] for m in METRICS}
    for idx in idx_sets:
        m = metrics(tab[idx].sum(axis=0))
        for k in METRICS:
            out[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def in_band(slab, lo, hi):
    span = slab.get("winding_span")
    return bool(span) and span[0] <= hi and span[1] >= lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--band", type=int, nargs=2, default=None)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    def load(run):
        return [json.loads(line) for line in (run / "slabs.jsonl").read_text().splitlines()]

    ref = load(args.ref)
    n = len(ref)
    groups = {"all": np.arange(n)}
    if args.band:
        lo, hi = args.band
        mask = np.array([in_band(s, lo, hi) for s in ref])
        groups[f"outside w{lo:03d}-w{hi:03d}"] = np.flatnonzero(~mask)
        groups[f"touching w{lo:03d}-w{hi:03d}"] = np.flatnonzero(mask)
    resamples = {g: [rng.choice(idx, len(idx)) for _ in range(args.n_boot)] for g, idx in groups.items()}
    ref_tab = table(ref)
    labels = args.labels or [r.name for r in [args.ref] + list(args.runs)]
    results = []
    for label, run in zip(labels, [args.ref] + list(args.runs)):
        slabs = load(run)
        paired = len(slabs) == n and all(a["origin_vol_zyx"] == b["origin_vol_zyx"] for a, b in zip(ref, slabs))
        tab = table(slabs)
        entry = {"run": run.name, "label": label, "paired_with_ref": paired, "groups": {}}
        own_groups = {"all": np.arange(len(slabs))}
        if args.band and not paired:
            own_mask = np.array([in_band(s, *args.band) for s in slabs])
            names = list(groups)
            own_groups[names[1]] = np.flatnonzero(~own_mask)
            own_groups[names[2]] = np.flatnonzero(own_mask)
        for g, idx in groups.items():
            use = idx if paired else own_groups[g]
            if not len(use):
                continue
            res = resamples[g] if paired else [rng.choice(use, len(use)) for _ in range(args.n_boot)]
            point = metrics(tab[use].sum(axis=0))
            info = {"n": int(len(use)), **{k: float(v) for k, v in point.items()},
                    "ci95": boot(tab, res, rng)}
            if paired and run != args.ref:
                ref_point = metrics(ref_tab[idx].sum(axis=0))
                deltas = {k: [] for k in METRICS}
                for r in res:
                    a, b = metrics(tab[r].sum(axis=0)), metrics(ref_tab[r].sum(axis=0))
                    for k in METRICS:
                        deltas[k].append(a[k] - b[k])
                info["delta_vs_ref"] = {k: float(point[k] - ref_point[k]) for k in METRICS}
                info["delta_ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                      for k, v in deltas.items()}
            entry["groups"][g] = info
        results.append(entry)

    for g in groups:
        print(f"\n### {g} (n = {len(groups[g])} slabs of the reference)\n")
        print("| Run | Precision | Recall | Count error | Phase MAE |")
        print("|---|---|---|---|---|")
        for e in results:
            info = e["groups"].get(g)
            if not info:
                continue
            cells = []
            for k in METRICS:
                lo, hi = info["ci95"][k]
                cell = f"{info[k]:.3f} ({lo:.3f} to {hi:.3f})"
                if "delta_vs_ref" in info:
                    dl, dh = info["delta_ci95"][k]
                    cell += f"; change {info['delta_vs_ref'][k]:+.3f} ({dl:+.3f} to {dh:+.3f})"
                cells.append(cell)
            print(f"| {e['label']} | " + " | ".join(cells) + " |")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"ref": args.ref.name, "band": args.band, "runs": results}, indent=1))


if __name__ == "__main__":
    main()
