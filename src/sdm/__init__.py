"""SDM —— 基于 Depth Anything V2 的视频深度估计工具包。

对外顶层 API：
    - ensure_model:     检测本地权重，缺失则自动下载到 models/
    - load_depth_model: 通过 transformers 加载指定规格的 DA V2 模型
    - DepthEstimator:   单帧/批量深度推理
    - estimate_video_depth: 视频级深度估计管线
"""

from sdm.estimator import DepthEstimator
from sdm.model_loader import (
    MODELS,
    available_devices,
    ensure_model,
    load_depth_model,
)
from sdm.video_pipeline import estimate_video_depth

__all__ = [
    "MODELS",
    "DepthEstimator",
    "available_devices",
    "ensure_model",
    "estimate_video_depth",
    "load_depth_model",
]
