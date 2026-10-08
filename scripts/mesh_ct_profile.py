"""Mean CT intensity along the mesh normal, to check that a mesh sits on papyrus in a volume frame.

Usage: mesh_ct_profile.py SEGMENTS_DIR VOLUME_ARRAY_DIR [--scale S] [--z LO HI] [--n 4000]
Segment coordinates are multiplied by S to get volume voxels. Prints mean intensity at offsets
-12..12 voxels along the normal; a mesh on papyrus shows a peak or an edge near 0, a mesh in the
wrong frame shows a flat, low profile.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vcshim"))
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
import vc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segments", type=Path)
    ap.add_argument("volume")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--z", type=float, nargs=2)
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()
    vol = vc.Volume.open(args.volume)
    rng = np.random.default_rng(0)
    offsets = np.arange(-12, 13)
    for seg in sorted(p for p in args.segments.iterdir() if (p / "z.tif").exists()):
        xyz = np.stack([tifffile.imread(seg / f"{c}.tif") for c in "xyz"], -1).astype(np.float64) * args.scale
        valid = (xyz[..., 2] > 0)
        du, dv = np.gradient(xyz, axis=0), np.gradient(xyz, axis=1)
        n = np.cross(du, dv)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
        valid[0, :] = valid[-1, :] = valid[:, 0] = valid[:, -1] = False
        if args.z:
            valid &= (xyz[..., 2] >= args.z[0]) & (xyz[..., 2] < args.z[1])
        idx = np.argwhere(valid)
        if not len(idx):
            continue
        idx = idx[rng.choice(len(idx), min(args.n, len(idx)), replace=False)]
        p, nn = xyz[idx[:, 0], idx[:, 1]], n[idx[:, 0], idx[:, 1]]
        pts = (p[:, None, :] + offsets[None, :, None] * nn[:, None, :]).reshape(-1, 3).astype(np.float32)
        values, _ = vol._sample(pts, "trilinear")
        prof = values.reshape(len(p), len(offsets)).astype(np.float64).mean(0)
        print(f"{seg.name[:28]:28s} n={len(p):5d} " + " ".join(f"{v:4.0f}" for v in prof))


if __name__ == "__main__":
    main()
