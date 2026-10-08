"""NumPy stand-in for the two `vc.Volume` calls the winding dataset makes.

villa's `vc` package is the nanobind binding of volume-cartographer's C++
core, which does not build on macOS without open PR #1975. The winding
dataset (`winding_model_dataset.py` / `volume_slab_extractor.py`) only needs:

* ``vc.Volume.open(path)``
* ``Volume.sample_planes(origins, x_steps, y_steps, shape, sampling=, tile_size=)``
* ``Volume.sample_coords(coords, valid_mask, sampling=, tile_size=)``
* ``vc.set_chunk_cache_budget`` / ``vc.set_chunk_cache_io_threads``
* ``vc.grid_raycast.GridRaycaster`` (built unmodified from
  volume-cartographer/python/vc/grid_raycast.cpp, see windeval/native).

The sampling below follows ``ChunkedPlaneSampler::
sampleCoordsLevelBlockingRequestedLevel`` with ``samplePinnedTrilinear`` /
``samplePinnedNearest`` (volume-cartographer/core/src/render/
ChunkedPlaneSampler.cpp) and the binding code in python/vc/volume.cpp:

* coordinates are XYZ voxel positions of the opened array (level 0);
* sentinel coordinates (non-finite, any component == -1, or all zero) are
  skipped: value 0, valid 0;
* points outside [0, shape) get the array fill value and stay valid;
* trilinear: floor corner, float32 fractions, corners outside the array
  read the fill value, a corner in a chunk absent from the store makes the
  whole sample 0 (status Missing), the blend is clamped to [0, 255] and
  truncated to uint8;
* nearest: int(p + 0.5) clamped to the array.

Chunks are read from a local zarr v2 directory (uint8, no compressor, '/'
separator). When the directory carries a ``.windeval_remote`` file holding the
array's base URL, chunks that are not on disk yet are fetched once over HTTP
and written into the same directory, so it becomes a partial mirror of the
remote array. Present-chunk listings per z chunk row are cached in
``.windeval_present/`` so absent chunks are never requested.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np

__all__ = ["Volume", "set_chunk_cache_budget", "set_chunk_cache_io_threads", "io_stats"]

_CACHE_BUDGET = [512 << 20]
_IO_THREADS = [16]
_STATS = {
    "chunks_fetched": 0,
    "bytes_fetched": 0,
    "fetch_seconds": 0.0,
    "chunks_read_disk": 0,
    "chunks_absent": 0,
    "listings": 0,
}
_STATS_LOCK = threading.Lock()


def set_chunk_cache_budget(nbytes: int) -> None:
    _CACHE_BUDGET[0] = int(nbytes)


def set_chunk_cache_io_threads(count: int) -> None:
    _IO_THREADS[0] = max(1, int(count))


def io_stats() -> dict:
    with _STATS_LOCK:
        return dict(_STATS)


def _bump(key, value=1):
    with _STATS_LOCK:
        _STATS[key] += value


def _http_get(url: str, attempts: int = 6) -> bytes | None:
    """GET with retries; None on 404 (absent object)."""
    delay = 1.0
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt + 1 == attempts:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt + 1 == attempts:
                raise
        time.sleep(delay)
        delay *= 2
    return None


class _ChunkStore:
    def __init__(self, array_dir: Path):
        self.dir = Path(array_dir)
        meta = json.loads((self.dir / ".zarray").read_text())
        if meta.get("dtype") != "|u1":
            raise ValueError("vc shim supports uint8 arrays only")
        if meta.get("compressor") is not None or meta.get("filters"):
            raise ValueError("vc shim supports uncompressed arrays only")
        if meta.get("order", "C") != "C" or meta.get("dimension_separator", ".") != "/":
            raise ValueError("vc shim expects C order and '/' chunk keys")
        self.shape = tuple(int(v) for v in meta["shape"])
        self.chunks = tuple(int(v) for v in meta["chunks"])
        fill = meta.get("fill_value") or 0
        self.fill = np.uint8(min(max(int(round(float(fill))), 0), 255))
        remote_file = self.dir / ".windeval_remote"
        self.remote = remote_file.read_text().strip().rstrip("/") if remote_file.is_file() else None
        self._present: dict[int, set] = {}
        self._row_locks: dict[int, threading.Lock] = {}
        self._mem: OrderedDict = OrderedDict()
        self._mem_bytes = 0
        self._lock = threading.Lock()

    # presence -----------------------------------------------------------
    def _row_presence(self, cz: int) -> set | None:
        with self._lock:
            if cz in self._present:
                return self._present[cz]
            row_lock = self._row_locks.setdefault(cz, threading.Lock())
        with row_lock:
            if cz in self._present:
                return self._present[cz]
            path = self.dir / ".windeval_present" / f"{cz}.txt"
            if path.is_file():
                present = set(path.read_text().split())
            elif self.remote is not None and self.remote.startswith("https://") and ".s3." in self.remote:
                present = self._list_row(cz)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_text("\n".join(sorted(present)))
                tmp.replace(path)
            else:
                present = None
            self._present[cz] = present
            return present

    def _list_row(self, cz: int) -> set:
        parsed = urllib.parse.urlparse(self.remote)
        bucket_url = f"{parsed.scheme}://{parsed.netloc}"
        prefix = parsed.path.lstrip("/") + f"/{cz}/"
        present, token = set(), None
        while True:
            url = f"{bucket_url}/?list-type=2&prefix={urllib.parse.quote(prefix)}"
            if token:
                url += "&continuation-token=" + urllib.parse.quote(token)
            text = _http_get(url).decode()
            for key in re.findall(r"<Key>([^<]*)</Key>", text):
                present.add(key[len(prefix):])
            match = re.search(r"<NextContinuationToken>([^<]*)<", text)
            if not match:
                break
            token = match.group(1)
        _bump("listings")
        return present

    # chunks -------------------------------------------------------------
    def get(self, cz: int, cy: int, cx: int) -> np.ndarray | None:
        """Chunk array, or None when the chunk is absent from the store."""
        key = (cz, cy, cx)
        with self._lock:
            if key in self._mem:
                self._mem.move_to_end(key)
                return self._mem[key]
        present = self._row_presence(cz)
        if present is not None and f"{cy}/{cx}" not in present:
            _bump("chunks_absent")
            return self._remember(key, None)
        path = self.dir / str(cz) / str(cy) / str(cx)
        size = self.chunks[0] * self.chunks[1] * self.chunks[2]
        if path.is_file():
            data = np.fromfile(path, dtype=np.uint8)
            _bump("chunks_read_disk")
        elif self.remote is not None:
            t0 = time.time()
            payload = _http_get(f"{self.remote}/{cz}/{cy}/{cx}")
            _bump("fetch_seconds", time.time() - t0)
            if payload is None:
                _bump("chunks_absent")
                return self._remember(key, None)
            _bump("chunks_fetched")
            _bump("bytes_fetched", len(payload))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_bytes(payload)
            tmp.replace(path)
            data = np.frombuffer(payload, dtype=np.uint8)
        else:
            _bump("chunks_absent")
            return self._remember(key, None)
        if data.size < size:
            raise RuntimeError("decoded chunk byte size does not match full chunk shape")
        return self._remember(key, data[:size].reshape(self.chunks))

    def _remember(self, key, value):
        with self._lock:
            nbytes = 0 if value is None else value.nbytes
            self._mem[key] = value
            self._mem_bytes += nbytes
            while self._mem_bytes > _CACHE_BUDGET[0] and len(self._mem) > 1:
                _, old = self._mem.popitem(last=False)
                self._mem_bytes -= 0 if old is None else old.nbytes
        return value

    def get_many(self, keys):
        keys = list(keys)
        threads = max(1, _IO_THREADS[0])
        if threads == 1 or len(keys) <= 1:
            return {key: self.get(*key) for key in keys}
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(min(threads, len(keys))) as pool:
            values = list(pool.map(lambda key: self.get(*key), keys))
        return dict(zip(keys, values))


def _is_sentinel(points: np.ndarray) -> np.ndarray:
    finite = np.isfinite(points).all(axis=-1)
    minus_one = (points == -1.0).any(axis=-1)
    zero = (points == 0.0).all(axis=-1)
    return ~finite | minus_one | zero


class Volume:
    """Single-level uint8 zarr array opened from a local (partially mirrored) directory."""

    def __init__(self, path: Path):
        path = Path(path)
        if not (path / ".zarray").is_file() and (path / "0" / ".zarray").is_file():
            path = path / "0"
        self.path = path
        self.store = _ChunkStore(path)

    @classmethod
    def open(cls, path: str) -> "Volume":
        return cls(Path(path))

    @property
    def shape(self):
        return self.store.shape

    # ------------------------------------------------------------------
    def _sample(self, points: np.ndarray, sampling: str) -> tuple[np.ndarray, np.ndarray]:
        """points [N, 3] float32 XYZ -> (uint8 values [N], bool valid [N])."""
        store = self.store
        Z, Y, X = store.shape
        cz_, cy_, cx_ = store.chunks
        n = len(points)
        values = np.zeros(n, dtype=np.uint8)
        sentinel = _is_sentinel(points)
        valid = ~sentinel
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        inb = valid & (x >= 0) & (y >= 0) & (z >= 0) & (x < X) & (y < Y) & (z < Z)
        values[valid & ~inb] = store.fill
        idx = np.flatnonzero(inb)
        if not len(idx):
            return values, valid
        p = points[idx]
        if sampling == "nearest":
            corner = np.floor(p + np.float32(0.5)).astype(np.int64)
            np.clip(corner[:, 0], 0, X - 1, out=corner[:, 0])
            np.clip(corner[:, 1], 0, Y - 1, out=corner[:, 1])
            np.clip(corner[:, 2], 0, Z - 1, out=corner[:, 2])
            offsets = [(0, 0, 0)]
        elif sampling == "trilinear":
            corner = np.floor(p).astype(np.int64)
            offsets = [(dz, dy, dx) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
        else:
            raise ValueError("sampling must be one of: nearest, trilinear")

        # Dense block over the voxels any corner can touch, assembled from chunks.
        lo = corner.min(axis=0)  # x, y, z
        hi = corner.max(axis=0) + (1 if sampling == "trilinear" else 0)
        hi = np.minimum(hi, np.array([X - 1, Y - 1, Z - 1]))
        lo = np.maximum(lo, 0)
        c_lo = lo // np.array([cx_, cy_, cz_])
        c_hi = hi // np.array([cx_, cy_, cz_])
        # Chunks actually touched: chunk of each corner (cheap: unique over floor keys,
        # plus neighbours for corners sitting on a chunk's last voxel).
        needed = set()
        grid = (c_hi - c_lo + 1)
        def add_keys(ix, iy, iz):
            key = ((iz // cz_ - c_lo[2]) * grid[1] + (iy // cy_ - c_lo[1])) * grid[0] + (ix // cx_ - c_lo[0])
            for k in np.unique(key):
                k = int(k)
                kx = k % grid[0]
                ky = (k // grid[0]) % grid[1]
                kz = k // (grid[0] * grid[1])
                needed.add((int(kz + c_lo[2]), int(ky + c_lo[1]), int(kx + c_lo[0])))
        cx0 = np.clip(corner[:, 0], 0, X - 1)
        cy0 = np.clip(corner[:, 1], 0, Y - 1)
        cz0 = np.clip(corner[:, 2], 0, Z - 1)
        add_keys(cx0, cy0, cz0)
        if sampling == "trilinear":
            edge = ((corner[:, 0] % cx_) == cx_ - 1) | ((corner[:, 1] % cy_) == cy_ - 1) | ((corner[:, 2] % cz_) == cz_ - 1)
            if edge.any():
                e = np.flatnonzero(edge)
                for dz, dy, dx in offsets[1:]:
                    add_keys(np.clip(corner[e, 0] + dx, 0, X - 1),
                             np.clip(corner[e, 1] + dy, 0, Y - 1),
                             np.clip(corner[e, 2] + dz, 0, Z - 1))
        chunk_data = store.get_many(sorted(needed))

        shape_zyx = (int(hi[2] - lo[2] + 1), int(hi[1] - lo[1] + 1), int(hi[0] - lo[0] + 1))
        block = np.zeros(shape_zyx, dtype=np.uint8)
        absent = np.zeros(shape_zyx, dtype=bool)
        for (kz, ky, kx), data in chunk_data.items():
            z0, y0, x0 = kz * cz_, ky * cy_, kx * cx_
            sz = slice(max(z0, lo[2]), min(z0 + cz_, hi[2] + 1))
            sy = slice(max(y0, lo[1]), min(y0 + cy_, hi[1] + 1))
            sx = slice(max(x0, lo[0]), min(x0 + cx_, hi[0] + 1))
            dst = (slice(sz.start - lo[2], sz.stop - lo[2]),
                   slice(sy.start - lo[1], sy.stop - lo[1]),
                   slice(sx.start - lo[0], sx.stop - lo[0]))
            if data is None:
                absent[dst] = True
            else:
                block[dst] = data[sz.start - z0:sz.stop - z0, sy.start - y0:sy.stop - y0, sx.start - x0:sx.stop - x0]

        def read(dz, dy, dx):
            ix = corner[:, 0] + dx
            iy = corner[:, 1] + dy
            iz = corner[:, 2] + dz
            outside = (ix >= X) | (iy >= Y) | (iz >= Z)
            jx = np.minimum(ix, X - 1) - lo[0]
            jy = np.minimum(iy, Y - 1) - lo[1]
            jz = np.minimum(iz, Z - 1) - lo[2]
            v = block[jz, jy, jx]
            a = absent[jz, jy, jx] & ~outside
            v = np.where(outside, store.fill, v)
            return v, a

        if sampling == "nearest":
            v, a = read(0, 0, 0)
            values[idx] = np.where(a, 0, v)
            return values, valid

        f = (p - corner.astype(np.float32)).astype(np.float32)
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        corners = {}
        missing = np.zeros(len(idx), dtype=bool)
        for off in offsets:
            v, a = read(*off)
            corners[off] = v.astype(np.float32)
            missing |= a
        c00 = fx * (corners[(0, 0, 1)] - corners[(0, 0, 0)]) + corners[(0, 0, 0)]
        c01 = fx * (corners[(0, 1, 1)] - corners[(0, 1, 0)]) + corners[(0, 1, 0)]
        c10 = fx * (corners[(1, 0, 1)] - corners[(1, 0, 0)]) + corners[(1, 0, 0)]
        c11 = fx * (corners[(1, 1, 1)] - corners[(1, 1, 0)]) + corners[(1, 1, 0)]
        c0 = fy * (c01 - c00) + c00
        c1 = fy * (c11 - c10) + c10
        value = np.clip(fz * (c1 - c0) + c0, np.float32(0.0), np.float32(255.0))
        out = value.astype(np.uint8)  # truncation, like static_cast<uint8_t>
        out[missing] = 0
        values[idx] = out
        return values, valid

    # ------------------------------------------------------------------
    def sample_planes(self, origins_xyz, x_steps_xyz, y_steps_xyz, shape, level=0,
                      sampling="trilinear", tile_size=32):
        if level != 0:
            raise ValueError("vc shim opens one physical array; level must be 0")
        origins = np.asarray(origins_xyz, dtype=np.float32)
        xs = np.asarray(x_steps_xyz, dtype=np.float32)
        ys = np.asarray(y_steps_xyz, dtype=np.float32)
        height, width = int(shape[0]), int(shape[1])
        yy = np.arange(height, dtype=np.float32)
        xx = np.arange(width, dtype=np.float32)
        # float32 arithmetic in the same order as samplePlanes: row origin first.
        row_origin = origins[:, None, :] + yy[None, :, None] * ys[:, None, :]
        coords = row_origin[:, :, None, :] + xx[None, None, :, None] * xs[:, None, None, :]
        values, valid = self._sample(coords.reshape(-1, 3), sampling)
        n = len(origins)
        stats = {"requested_level_only": True}
        return (values.reshape(n, height, width),
                valid.astype(np.uint8).reshape(n, height, width),
                stats)

    def sample_coords(self, coords_xyz, valid_mask, level=0, sampling="trilinear",
                      tile_size=32, blocking=True):
        if level != 0:
            raise ValueError("vc shim opens one physical array; level must be 0")
        coords = np.asarray(coords_xyz, dtype=np.float32)
        mask = np.asarray(valid_mask, dtype=bool)
        h, w = coords.shape[:2]
        flat = coords.reshape(-1, 3)
        take = np.flatnonzero(mask.reshape(-1))
        image = np.zeros(h * w, dtype=np.uint8)
        valid = np.zeros(h * w, dtype=np.uint8)
        if len(take):
            v, ok = self._sample(flat[take], sampling)
            image[take] = v
            valid[take] = ok
        return image.reshape(h, w, 1), valid.reshape(h, w, 1), {"requested_level_only": True}
