"""深度估计常用指标（需有 GT 深度时使用；无 GT 时可用 temporal_gradient 看闪烁）。"""

from __future__ import annotations

import numpy as np


def _check(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if pred.shape != gt.shape:
        raise ValueError(f"形状不一致：pred={pred.shape} gt={gt.shape}")
    valid = np.ones(pred.shape, dtype=bool) if mask is None else mask.astype(bool)
    valid &= np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    return pred[valid], gt[valid]


def depth_errors(pred: np.ndarray, gt: np.ndarray, *, mask: np.ndarray | None = None) -> dict:
    """标准单目深度指标：AbsRel、SqRel、RMSE、Log10、δ1/δ2/δ3。

    注意：DA V2 输出是相对深度（disparity 量纲），与米制 GT 直接比较需要先做
    尺度+偏移最小二乘对齐（scale-and-shift alignment），否则数值无意义。
    """
    p, g = _check(pred, gt, mask)
    if p.size == 0:
        raise ValueError("没有有效像素可评估")
    eps = 1e-8
    diff = p - g
    abs_rel = float(np.mean(np.abs(diff) / np.maximum(g, eps)))
    sq_rel = float(np.mean(diff**2 / np.maximum(g, eps)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    log_diff = np.log10(np.maximum(p, eps)) - np.log10(np.maximum(g, eps))
    log10 = float(np.mean(np.abs(log_diff)))
    ratio = np.maximum(p, g) / np.minimum(p, g)
    thresh = [1.25, 1.25**2, 1.25**3]
    acc = {f"delta{i+1}": float(np.mean(ratio < t) * 100.0) for i, t in enumerate(thresh)}
    return {"abs_rel": abs_rel, "sq_rel": sq_rel, "rmse": rmse, "rmse_log": float(np.sqrt(np.mean(log_diff**2))), "log10": log10, **acc}


def align_scale_shift(pred: np.ndarray, gt: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
    """对 pred 做 scale-and-shift 对齐到 gt（最小二乘），用于把相对深度转成可比尺度。"""
    p, g = _check(pred, gt, mask)
    a, b = np.polyfit(p, g, 1)
    return pred * a + b
