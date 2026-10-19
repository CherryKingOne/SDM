"""可视化辅助：原图与深度并排对比图，便于快速判断某个规格的效果。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def side_by_side(frame_bgr: np.ndarray, depth_gray: np.ndarray, *, out_path: str | Path) -> Path:
    """把 RGB 帧与灰度深度左右拼接并保存，中间画一条分隔线。"""
    depth_bgr = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
    divider = np.full((frame_bgr.shape[0], 4, 3), 255, dtype=np.uint8)
    canvas = np.concatenate([frame_bgr, divider, depth_bgr], axis=1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path
