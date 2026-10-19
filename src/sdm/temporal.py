"""时序一致性处理：缓解逐帧独立推理带来的深度闪烁/抖动。

核心手段（按性价比排序）：
1. 全序列统一归一化范围（在 video_pipeline 中实现，本模块不负责）；
2. 沿时间轴的轻度低通滤波（本模块提供）。
"""

from __future__ import annotations

import numpy as np


def smooth_along_time(depth_seq: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """一阶指数平滑（EMA），alpha 越小越平滑但拖影越明显。

    Args:
        depth_seq: [N, H, W] 浮点深度序列。
        alpha: 当前帧权重 (0, 1]；1.0 等于不平滑。
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha 必须落在 (0, 1]")
    out = depth_seq.copy()
    for i in range(1, out.shape[0]):
        out[i] = alpha * depth_seq[i] + (1.0 - alpha) * out[i - 1]
    return out


def temporal_gradient(depth_seq: np.ndarray) -> np.ndarray:
    """相邻帧平均绝对差，用于量化闪烁程度（评估平滑参数是否有效）。"""
    if depth_seq.shape[0] < 2:
        return np.zeros(1)
    return np.abs(np.diff(depth_seq, axis=0)).mean(axis=(1, 2))
