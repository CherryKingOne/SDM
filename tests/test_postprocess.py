"""纯数值后处理逻辑测试（不依赖模型与网络）。"""

import numpy as np
import pytest

from sdm.postprocess import colorize, normalize_to_uint8, robust_range, save_depth_png16


def make_depth():
    # 用足够多的像素，让 percentile 类断言不受小样本插值影响
    return np.linspace(0.0, 10.0, 40 * 40, dtype=np.float32).reshape(40, 40)


def test_robust_range_ignores_outliers():
    d = make_depth()
    d[0, 0] = 1e6
    lo, hi = robust_range(d)
    assert lo < hi < 1e5


def test_normalize_endpoints_map_to_0_and_255():
    d = make_depth()
    g = normalize_to_uint8(d, 0.0, 10.0)
    assert g.dtype == np.uint8
    assert g.min() == 0 and g.max() == 255


def test_near_is_bright_inverts():
    d = make_depth()
    a = normalize_to_uint8(d, 0.0, 10.0, near_is_bright=True)
    b = normalize_to_uint8(d, 0.0, 10.0, near_is_bright=False)
    assert np.allclose(a.astype(int) + b.astype(int), 255, atol=1)


def test_normalize_is_clipped_not_wrapped():
    d = make_depth()
    g = normalize_to_uint8(d, 4.0, 5.0)
    assert g.min() == 0 and g.max() == 255


def test_colorize_produces_bgr():
    g = normalize_to_uint8(make_depth(), 0.0, 10.0)
    out = colorize(g)
    assert out.ndim == 3 and out.shape[2] == 3


def test_save_depth_png16_is_lossless_readable(tmp_path):
    import cv2

    d = make_depth()
    p = tmp_path / "d.png"
    save_depth_png16(str(p), d, 0.0, 10.0)
    loaded = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    assert loaded.dtype == np.uint16
    assert loaded[0, 0] == 0 and loaded[-1, -1] == pytest.approx(65535, abs=2)


def test_normalize_degenerate_range_does_not_divide_by_zero():
    d = np.full((4, 4), 3.0, dtype=np.float32)
    g = normalize_to_uint8(d, 3.0, 3.0)
    assert np.array_equal(g, np.full((4, 4), 0, dtype=np.uint8))
