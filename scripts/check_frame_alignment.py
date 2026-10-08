"""Estimate the integer voxel shift between two volumes' arrays over one block (phase correlation).

Usage: check_frame_alignment.py REF_ARRAY_DIR MOVING_ARRAY_DIR z0 y0 x0 size
Array dirs are local (partial) zarr mirrors made by make_mirror.py; chunks are fetched by the vc shim.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vcshim"))
import numpy as np  # noqa: E402
import vc  # noqa: E402


def block(vol, z0, y0, x0, n):
    zz, yy, xx = np.meshgrid(np.arange(z0, z0 + n), np.arange(y0, y0 + n), np.arange(x0, x0 + n), indexing="ij")
    pts = np.stack([xx, yy, zz], -1).reshape(-1, 3).astype(np.float32) + np.float32(1e-3)
    vals, _ = vol._sample(pts, "nearest")
    return vals.reshape(n, n, n).astype(np.float32)


def main():
    ref, mov = vc.Volume.open(sys.argv[1]), vc.Volume.open(sys.argv[2])
    z0, y0, x0, n = map(int, sys.argv[3:7])
    a, b = block(ref, z0, y0, x0, n), block(mov, z0, y0, x0, n)
    print("mean intensity ref %.1f moving %.1f; corr at zero shift %.4f"
          % (a.mean(), b.mean(), np.corrcoef(a.ravel(), b.ravel())[0, 1]))
    win = np.hanning(n)
    w = win[:, None, None] * win[None, :, None] * win[None, None, :]
    fa, fb = np.fft.rfftn((a - a.mean()) * w), np.fft.rfftn((b - b.mean()) * w)
    cross = fa * np.conj(fb)
    corr = np.fft.irfftn(cross / (np.abs(cross) + 1e-9), s=a.shape)
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    shift = [p if p <= n // 2 else p - n for p in peak]
    print("phase-correlation peak shift (z, y, x) of ref relative to moving:", shift, "peak %.3f" % corr.max())


if __name__ == "__main__":
    main()
