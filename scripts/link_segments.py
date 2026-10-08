"""Build a flat segments folder for WindingModelDataset from published per-segment meshes.

The dataset reads a segment's winding index from its tifxyz folder name (regex
`(?:^|-)w(\\d+)(?:_|$)`), while published mesh folders are named
`<id>-on-<volume>.tifxyz`. This links each segment's mesh for one volume frame into
OUT/<segment uuid>/ (e.g. OUT/w029_2025010827/{x,y,z}.tif, meta.json).

Usage: link_segments.py SEGMENTS_ROOT MESH_SUFFIX OUT_DIR [--exclude w046 ...] [--keep-unlabelled]
"""
import argparse
import re
from pathlib import Path

WINDING = re.compile(r"(?:^|-)w(?P<index>\d+)(?:_|$)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segments_root", type=Path)
    ap.add_argument("mesh_suffix")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--keep-unlabelled", action="store_true",
                    help="also link segments without a winding index (the dataset treats them as multi-wrap)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    linked, skipped = [], []
    for seg in sorted(p for p in args.segments_root.iterdir() if p.is_dir()):
        uuid = seg.name.split("-", 1)[1] if "-" in seg.name else seg.name
        meshes = sorted((seg / "mesh").glob(f"*{args.mesh_suffix}"))
        if not WINDING.search(uuid) and not args.keep_unlabelled:
            skipped.append((seg.name, "no winding index in name"))
            continue
        if any(uuid.startswith(prefix) for prefix in args.exclude):
            skipped.append((seg.name, "excluded"))
            continue
        if len(meshes) != 1:
            skipped.append((seg.name, f"{len(meshes)} meshes match"))
            continue
        dest = args.out_dir / uuid
        dest.mkdir(exist_ok=True)
        for name in ("x.tif", "y.tif", "z.tif", "meta.json"):
            link = dest / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to((meshes[0] / name).resolve())
        linked.append(uuid)
    print(f"linked {len(linked)} segments into {args.out_dir}")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")


if __name__ == "__main__":
    main()
