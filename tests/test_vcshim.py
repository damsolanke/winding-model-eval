"""Unit checks for the NumPy `vc.Volume` stand-in against its C++ reference semantics."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vcshim"))
import vc  # noqa: E402


@pytest.fixture()
def volume(tmp_path):
    rng = np.random.default_rng(0)
    shape, chunks = (40, 36, 44), (16, 16, 16)
    data = rng.integers(0, 256, size=shape, dtype=np.uint8)
    arr = tmp_path / "vol.zarr" / "0"
    arr.mkdir(parents=True)
    (arr / ".zarray").write_text(json.dumps({
        "shape": list(shape), "chunks": list(chunks), "dtype": "|u1", "fill_value": 0,
        "order": "C", "filters": None, "dimension_separator": "/", "compressor": None, "zarr_format": 2}))
    for cz in range(-(-shape[0] // chunks[0])):
        for cy in range(-(-shape[1] // chunks[1])):
            for cx in range(-(-shape[2] // chunks[2])):
                if (cz, cy, cx) == (1, 1, 1):
                    continue  # absent chunk
                block = np.zeros(chunks, np.uint8)
                part = data[cz * 16:(cz + 1) * 16, cy * 16:(cy + 1) * 16, cx * 16:(cx + 1) * 16]
                block[:part.shape[0], :part.shape[1], :part.shape[2]] = part
                path = arr / str(cz) / str(cy)
                path.mkdir(parents=True, exist_ok=True)
                block.tofile(path / str(cx))
    present = data.copy()
    present[16:32, 16:32, 16:32] = 0
    return vc.Volume.open(str(arr.parent)), present


def test_trilinear_matches_scipy_away_from_absent_chunk(volume):
    vol, data = volume
    rng = np.random.default_rng(1)
    pts = rng.uniform([0, 0, 0], [43.0, 35.0, 39.0], size=(5000, 3)).astype(np.float32)
    corner = np.floor(pts).astype(int)
    near_absent = ((corner[:, ::-1] >= 15) & (corner[:, ::-1] <= 32)).all(axis=1)
    pts = pts[~near_absent]
    values, valid = vol._sample(pts, "trilinear")
    ref = scipy.ndimage.map_coordinates(data.astype(np.float64), pts[:, ::-1].T, order=1, mode="constant", cval=0)
    assert valid.all()
    assert np.abs(values.astype(int) - np.floor(ref + 1e-6).astype(int)).max() <= 1


def test_integer_coordinates_read_voxels(volume):
    vol, data = volume
    idx = np.array([[3, 5, 7], [43, 35, 39], [0, 0, 1]], dtype=np.float32)
    values, _ = vol._sample(idx, "trilinear")
    assert values.tolist() == [data[7, 5, 3], data[39, 35, 43], data[1, 0, 0]]


def test_out_of_bounds_and_sentinels(volume):
    vol, _ = volume
    pts = np.array([[-0.5, 3, 3], [44.0, 3, 3], [3, -1.0, 3], [0, 0, 0], [np.nan, 1, 1]], dtype=np.float32)
    values, valid = vol._sample(pts, "trilinear")
    assert values.tolist() == [0, 0, 0, 0, 0]
    assert valid.tolist() == [True, True, False, False, False]


def test_absent_chunk_zeroes_whole_sample(volume):
    vol, data = volume
    pts = np.array([[15.5, 15.5, 15.5]], dtype=np.float32)  # corners straddle the absent chunk
    values, valid = vol._sample(pts, "trilinear")
    assert values[0] == 0 and valid[0]


def test_sample_planes_layout(volume):
    vol, data = volume
    origins = np.array([[2, 3, 4], [2, 4, 4]], dtype=np.float32)
    xs = np.array([[1, 0, 0]] * 2, dtype=np.float32)
    ys = np.array([[0, 0, 1]] * 2, dtype=np.float32)
    images, valid, _ = vol.sample_planes(origins, xs, ys, (3, 5))
    assert images.shape == (2, 3, 5) and valid.all()
    # plane n, row r, column c samples x = 2 + c, y = 3 + n, z = 4 + r
    for n in range(2):
        for r in range(3):
            for c in range(5):
                assert images[n, r, c] == data[4 + r, 3 + n, 2 + c]
