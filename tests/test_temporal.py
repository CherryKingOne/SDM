"""时序平滑与闪烁度量测试。"""

import numpy as np
import pytest

from sdm.temporal import smooth_along_time, temporal_gradient


def noisy_seq(n=20, amplitude=1.0, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(1.0, 2.0, n, dtype=np.float32)[:, None, None]
    return base + amplitude * rng.standard_normal((n, 8, 8)).astype(np.float32)


def test_alpha_one_is_identity():
    seq = noisy_seq()
    assert np.array_equal(smooth_along_time(seq, alpha=1.0), seq)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_invalid_alpha_raises(alpha):
    with pytest.raises(ValueError):
        smooth_along_time(noisy_seq(), alpha=alpha)


def test_smoothing_reduces_frame_to_frame_flicker():
    seq = noisy_seq()
    assert temporal_gradient(smooth_along_time(seq, alpha=0.2)).mean() < temporal_gradient(seq).mean()


def test_smoothing_reduces_deviation_from_trend():
    """因果 EMA 会引入少量滞后，但应让序列更靠近真实趋势（降噪为主目的）。"""
    n = 40
    seq = noisy_seq(n=n, amplitude=0.3)
    trend = np.linspace(1.0, 2.0, n, dtype=np.float32)[:, None, None]
    smoothed = smooth_along_time(seq, alpha=0.3)
    assert np.std(smoothed - trend) < np.std(seq - trend)


def test_temporal_gradient_single_frame():
    assert temporal_gradient(np.zeros((1, 4, 4))).size == 1
