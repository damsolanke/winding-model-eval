"""Unit checks for the transform fit and the paired comparison helpers."""

import sys
from pathlib import Path

import numpy as np
import scipy.ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windeval.compare import in_band, metrics  # noqa: E402
from windeval.fit_transform import apply, fit_affine, phase_shift  # noqa: E402


def test_fit_affine_recovers_matrix_and_ignores_outliers():
    rng = np.random.default_rng(0)
    true = np.array([[1.03, 0.04, -0.01, 5.0], [-0.04, 1.02, 0.0, -3.0], [0.01, 0.0, 0.99, 12.0]])
    src = rng.uniform(0, 1000, size=(200, 3))
    dst = apply(true, src) + rng.normal(0, 0.1, size=(200, 3))
    dst[:10] += 25.0
    est, inliers = fit_affine(src, dst)
    grid = rng.uniform(0, 1000, size=(500, 3))
    assert np.abs(apply(est, grid) - apply(true, grid)).max() < 0.15
    assert not inliers[:10].any() and inliers[10:].mean() > 0.95


def test_phase_shift_direction_and_bias():
    # moved(x) = block(x - s), so the reported shift (moving(x) ~ fixed(x + shift)) is -s.
    # Windowed phase correlation reads sub-voxel shifts low; fit_transform iterates for that reason.
    rng = np.random.default_rng(1)
    big = scipy.ndimage.gaussian_filter(rng.normal(size=(96, 96, 96)), 2.0).astype(np.float32)
    for true in ((2.4, -1.0, -3.6), (0.6, 0.0, 0.0)):
        moved = scipy.ndimage.shift(big, true, order=3, mode="nearest")
        shift, ncc, _ = phase_shift(big[24:72, 24:72, 24:72], moved[24:72, 24:72, 24:72])
        assert np.all(np.sign(np.round(shift, 1)) == -np.sign(true))
        assert np.all(np.abs(shift + np.array(true)) <= 0.5)
        assert ncc > 0.9


def test_metrics_and_band():
    m = metrics(np.array([6.0, 4.0, 2.0, 3.0, 6.0, 1.0, 2.0]))
    assert m == {"precision": 0.6, "recall": 0.75, "count_error": 0.5, "phase_mae": 0.5}
    assert in_band({"winding_span": [35, 41]}, 40, 49)
    assert not in_band({"winding_span": [50, 55]}, 40, 49)
    assert not in_band({"winding_span": None}, 40, 49)
