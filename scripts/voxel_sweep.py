"""Voxel-size sweep (conditions 5 and 6): paired changes against the training-voxel reading.

Every run reads the same rays as its reference, one sample per `sample_spacing`
reference voxels, which is what the model sees on a scan with that voxel size
read at one sample per voxel. Condition 6 is a real 8.64 um scan (PHerc0343,
two adjacent winding meshes), read natively and resampled to the training voxel. Labels are re-rendered for each step by villa's
rasterizer (windeval/replay.py). A slab whose labels cannot be re-rendered at
a step (fewer supervised columns than villa's minimum once the ray is shorter)
is dropped from both sides of that pair. Point estimates pool slab sums as
villa does; intervals are 95 % slab bootstraps, paired for the changes
(2,000 resamples, seed 0, as in windeval/compare.py).

Usage: python scripts/voxel_sweep.py [--runs results/runs] [--json results/compare_voxel_sweep.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from windeval.compare import METRICS, in_band, metrics, table  # noqa: E402

# One 2.399 um level-2 voxel measured on the 9.362 um scan by the fitted affine
# (results/transform_pherc0139_2399l2_to_9362.json: scale 1.02866 x 9.362 um).
UNIT_UM = 9.6303
SWEEPS = {
    "PHerc0139": {
        "scan": "9.362 um, 113 keV, 1.2 m scan, re-sampled along the 2.399 um level-2 rays",
        "unit_um": UNIT_UM,
        "ref": "cond2fair_pherc0139_9362_default",
        "band": (40, 49),
        "runs": ["cond5_pherc0139_9362_eq790", "cond5_pherc0139_9362_eq830", "cond5_pherc0139_9362_eq864",
                 "cond5_pherc0139_9362_eq900", "cond4_pherc0139_9362_default_native",
                 "cond5_pherc0139_9362_eq1020", "cond5_pherc0139_9362_eq1080"],
    },
    "PHerc1667": {
        "scan": "2.399 um, 78 keV, 0.2 m scan at level 2, re-sampled in place",
        "unit_um": UNIT_UM,
        "ref": "gate_pherc1667_2399_l2_z5000",
        "band": None,
        "runs": ["cond5_pherc1667_2399_l2_eq830", "cond5_pherc1667_2399_l2_eq864",
                 "cond5_pherc1667_2399_l2_eq936", "cond5_pherc1667_2399_l2_eq1020"],
    },
    "PHerc0343": {
        "scan": "8.64 um, 116 keV, 1.2 m scan at level 0, labels w047 and w048 (Nieuwlaar)",
        "unit_um": 8.64,
        "ref": "cond6_pherc0343_864_to963",
        "band": None,
        "tilt_split": 0.30,
        "per_winding": True,
        "runs": ["cond6_pherc0343_864_native"],
    },
}


def load(run_dir: Path):
    result = json.loads((run_dir / "result.json").read_text())
    slabs = [json.loads(line) for line in (run_dir / "slabs.jsonl").read_text().splitlines()]
    return result, slabs


def spacing(result) -> float:
    return float((result.get("replay") or {}).get("sample_spacing", 1.0))


def extent_table(slabs):
    """Per slab: count error sum, labelled crossings and supervised columns.

    Crossings minus columns counts labelled winding steps (k - 1 per column with
    k crossings), which approximates the labelled phase change the count error
    is measured against.
    """
    return np.array([[s["sums"]["count_error"], s["crossings"], s["supervised_columns"]] for s in slabs],
                    dtype=np.float64)


def count_error_per_step(ext, idx):
    ce, crossings, columns = ext[idx].sum(axis=0)
    return ce / max(1.0, crossings - columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "runs")
    ap.add_argument("--json", type=Path, default=ROOT / "results" / "compare_voxel_sweep.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    out = {"n_boot": args.n_boot, "scrolls": {}}
    for scroll, spec in SWEEPS.items():
        _, ref = load(args.runs / spec["ref"])
        ref_tab, ref_ext = table(ref), extent_table(ref)
        n = len(ref)
        groups = {"all": np.arange(n)}
        if spec["band"]:
            lo, hi = spec["band"]
            mask = np.array([in_band(s, lo, hi) for s in ref])
            groups[f"outside w{lo:03d}-w{hi:03d}"] = np.flatnonzero(~mask)
            groups[f"touching w{lo:03d}-w{hi:03d}"] = np.flatnonzero(mask)
        if spec.get("tilt_split"):
            # Ray tilt out of the transverse plane (|z| of the unit ray direction).
            steep = np.array([abs(s["direction_zyx"][0]) >= spec["tilt_split"] for s in ref])
            groups[f"ray |dz| < {spec['tilt_split']:.2f}"] = np.flatnonzero(~steep)
            groups[f"ray |dz| >= {spec['tilt_split']:.2f}"] = np.flatnonzero(steep)
        rows = []
        for name in [spec["ref"]] + spec["runs"]:
            result, slabs = load(args.runs / name)
            if len(slabs) != n or any(a["origin_vol_zyx"] != b["origin_vol_zyx"]
                                      or a["direction_zyx"] != b["direction_zyx"] for a, b in zip(ref, slabs)):
                raise SystemExit(f"{name} is not on the reference rays")
            tab, ext = table(slabs), extent_table(slabs)
            full = metrics(tab.sum(axis=0))
            logged = result["metrics"]
            for key, val in (("precision", "val_crossing_precision"), ("recall", "val_crossing_recall"),
                             ("count_error", "val_winding_count_error"), ("phase_mae", "val_phase_mae")):
                if abs(full[key] - logged[val]) > 1e-9:
                    raise SystemExit(f"{name}: pooled {key} {full[key]} differs from result.json {logged[val]}")
            ok = np.array([s.get("replay_ok", True) is not False and r.get("replay_ok", True) is not False
                           for s, r in zip(slabs, ref)])
            row = {"run": name, "sample_spacing": spacing(result),
                   "voxel_um": round(spacing(result) * spec["unit_um"], 3),
                   "slabs": n, "slabs_dropped": int((~ok).sum()), "all_slabs": full, "groups": {}}
            if spec.get("per_winding"):
                row["label_agreement_by_winding"] = result["label_agreement_by_winding"]
            for g, gidx in groups.items():
                idx = gidx[ok[gidx]]
                rng = np.random.default_rng(0)
                res = [rng.choice(idx, len(idx)) for _ in range(args.n_boot)]
                point, ref_point = metrics(tab[idx].sum(axis=0)), metrics(ref_tab[idx].sum(axis=0))
                draws = {k: [] for k in METRICS}
                deltas = {k: [] for k in METRICS}
                step_ratio = []
                for r in res:
                    a, b = metrics(tab[r].sum(axis=0)), metrics(ref_tab[r].sum(axis=0))
                    for k in METRICS:
                        draws[k].append(a[k])
                        deltas[k].append(a[k] - b[k])
                    step_ratio.append(count_error_per_step(ext, r) / max(1e-12, count_error_per_step(ref_ext, r)))
                _, crossings, columns = ext[idx].sum(axis=0)
                _, ref_crossings, ref_columns = ref_ext[idx].sum(axis=0)
                pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
                row["groups"][g] = {
                    "n": int(len(idx)),
                    **{k: float(point[k]) for k in METRICS},
                    "ci95": {k: pct(draws[k]) for k in METRICS},
                    "ref": {k: float(ref_point[k]) for k in METRICS},
                    "delta": {k: float(point[k] - ref_point[k]) for k in METRICS},
                    "delta_ci95": {k: pct(deltas[k]) for k in METRICS},
                    "crossings_per_column": float(crossings / max(1.0, columns)),
                    "ref_crossings_per_column": float(ref_crossings / max(1.0, ref_columns)),
                    "count_error_per_step_ratio": float(count_error_per_step(ext, idx)
                                                        / max(1e-12, count_error_per_step(ref_ext, idx))),
                    "count_error_per_step_ratio_ci95": pct(step_ratio),
                }
            rows.append(row)
        rows.sort(key=lambda r: r["sample_spacing"])
        out["scrolls"][scroll] = {"scan": spec["scan"], "unit_um": spec["unit_um"], "ref": spec["ref"],
                                  "band": spec["band"], "runs": rows}

        for g in groups:
            print(f"\n### {scroll}, {g}: change against the training-voxel reading (9.63 um), paired 95 % intervals\n")
            print("| Voxel size (um) | Slabs | Precision | Recall | Count error | Phase MAE |")
            print("|---|---|---|---|---|---|")
            for row in rows:
                info = row["groups"][g]
                cells = []
                for k in METRICS:
                    cell = f"{info[k]:.3f}"
                    if row["run"] != spec["ref"]:
                        dl, dh = info["delta_ci95"][k]
                        cell += f", {info['delta'][k]:+.3f} ({dl:+.3f} to {dh:+.3f})"
                    cells.append(cell)
                print(f"| {row['voxel_um']:.2f} | {info['n']} | " + " | ".join(cells) + " |")

        print(f"\n### {scroll}, all slabs: ray extent and count error per labelled winding step\n")
        print("| Voxel size (um) | Labelled crossings per supervised column | Count error per step, ratio to 9.63 um |")
        print("|---|---|---|")
        for row in rows:
            info = row["groups"]["all"]
            lo, hi = info["count_error_per_step_ratio_ci95"]
            print(f"| {row['voxel_um']:.2f} | {info['crossings_per_column']:.2f} | "
                  f"{info['count_error_per_step_ratio']:.3f} ({lo:.3f} to {hi:.3f}) |")
        if spec.get("per_winding"):
            print(f"\n### {scroll}: labels with a decoded passage within tolerance, and mean signed offset "
                  "(samples, + toward the higher winding) of passages within 8 samples\n")
            print("| Voxel size (um) | Winding | Labels | Within tolerance | Mean signed offset |")
            print("|---|---|---|---|---|")
            for row in rows:
                for w, e in sorted(row["label_agreement_by_winding"].items(), key=lambda kv: int(kv[0])):
                    print(f"| {row['voxel_um']:.2f} | w{int(w):03d} | {e['labels']} | "
                          f"{e['frac_within_tolerance']:.3f} | {e['mean_oriented_residual_within_8']:+.2f} |")

    args.json.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.json.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
