"""Write tifxyz copies with coordinates divided by a factor (segment_downscale semantics).

The released winding model's config reads PHerc0139 segments with segment_downscale=4 against
volume_scale=2, i.e. segment coordinates in units of 2.399 um voxels / 4, and samples slabs at
spacing 1 in those units. The published `-on-20260102150214-2.399um` meshes are in full-resolution
voxels, so this divides valid vertices by FACTOR (invalid vertices keep their -1 marker) and
writes uncompressed float32 x/y/z.tif plus meta.json (bbox / FACTOR, grid scale * FACTOR).

Usage: scale_segments.py IN_DIR OUT_DIR FACTOR   (IN_DIR holds one tifxyz folder per segment)
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile


def main():
    src, dst, factor = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    dst.mkdir(parents=True, exist_ok=True)
    for seg in sorted(p for p in src.iterdir() if (p / "meta.json").exists()):
        x, y, z = (tifffile.imread(seg / f"{c}.tif").astype(np.float32) for c in "xyz")
        valid = (z > 0) & np.isfinite(z)
        out = dst / seg.name
        out.mkdir(exist_ok=True)
        for name, arr in zip("xyz", (x, y, z)):
            scaled = np.where(valid, arr / np.float32(factor), arr).astype(np.float32)
            tifffile.imwrite(out / f"{name}.tif", scaled, compression=None, photometric="minisblack")
        meta = json.loads((seg / "meta.json").read_text())
        if "bbox" in meta:
            meta["bbox"] = [[v / factor for v in corner] for corner in meta["bbox"]]
        if "scale" in meta:
            meta["scale"] = [s * factor for s in meta["scale"]]
        meta.pop("area_vx2", None)
        meta["uuid"] = seg.name
        meta["windeval_note"] = f"coordinates divided by {factor:g} from {(seg / 'x.tif').resolve().parent}"
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(seg.name, x.shape, int(valid.sum()))


if __name__ == "__main__":
    main()
