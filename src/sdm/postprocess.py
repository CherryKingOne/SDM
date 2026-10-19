"""深度图后处理：归一化、伪彩着色、16bit 无损导出。"""

from __future__ import annotations

import cv2
import numpy as np


def robust_range(depth: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> tuple[float, float]:
    """用百分位统计深度范围，抑制离群值导致的整段对比度塌陷。"""
    return float(np.percentile(depth, lo_pct)), float(np.percentile(depth, hi_pct))


def normalize_to_uint8(depth: np.ndarray, vmin: float, vmax: float, *, near_is_bright: bool = True) -> np.ndarray:
    """按给定全局范围把浮点深度线性映射到 uint8。

    Args:
        near_is_bright: DA V2 输出是 disparity（值越大越近）。
            True（默认）→ 近处亮/白、远处暗/黑（业界常见深度可视化约定）；
            False → 反转。
    """
    span = max(vmax - vmin, 1e-6)
    scaled = (depth - vmin) / span
    if not near_is_bright:
        scaled = 1.0 - scaled
    return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def colorize(gray: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """灰度深度图转伪彩 BGR（仅用于预览，不建议作为交付主格式）。"""
    return cv2.applyColorMap(gray, colormap)


def save_depth_png16(path: str, depth: np.ndarray, vmin: float, vmax: float) -> None:
    """单帧深度存为 16bit 无损 PNG（保留线性相对深度，便于后续 3D/特效复用）。"""
    span = max(vmax - vmin, 1e-6)
    quantized = np.clip((depth - vmin) / span, 0.0, 1.0)
    cv2.imwrite(path, (quantized * 65535.0).astype(np.uint16))
