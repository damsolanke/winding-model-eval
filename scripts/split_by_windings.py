"""Pool a run's per-slab metric sums by whether the labelled winding bracket covers given windings.

Usage: split_by_windings.py RUN_DIR W [W ...]
"""
import collections
import json
import sys
from pathlib import Path


def pooled(sums):
    t = collections.Counter()
    for s in sums:
        t.update(s)
    return "n=%3d P=%.3f R=%.3f CE=%.3f PMAE=%.3f" % (
        len(sums), t["crossing_tp"] / max(1, t["crossing_tp"] + t["crossing_fp"]),
        t["crossing_tp"] / max(1, t["crossing_tp"] + t["crossing_fn"]),
        t["count_error"] / max(1, t["count_columns"]), t["phase_abs_error"] / max(1, t["phase_slabs"]))


def main():
    run = Path(sys.argv[1])
    targets = [int(w) for w in sys.argv[2:]]
    slabs = [json.loads(line) for line in (run / "slabs.jsonl").read_text().splitlines()]
    groups = collections.defaultdict(list)
    for s in slabs:
        lo, hi = s["winding_span"] or (0, 0)
        span = hi - lo
        length = "4-6" if span <= 6 else "7-12" if span <= 12 else "13+"
        hit = any(lo <= w <= hi for w in targets)
        groups[("all", hit)].append(s["sums"])
        groups[(length, hit)].append(s["sums"])
    for length in ("all", "4-6", "7-12", "13+"):
        for hit in (False, True):
            if groups[(length, hit)]:
                label = "covers " + ",".join(f"w{w:03d}" for w in targets) if hit else "covers none"
                print(f"bracket {length:5s} {label:30s} {pooled(groups[(length, hit)])}")


if __name__ == "__main__":
    main()
