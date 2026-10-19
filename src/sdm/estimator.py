"""单帧/批量深度推理封装。

对 transformers 的 `DepthAnythingForDepthEstimation` 做一层薄封装：
预处理 → 前向 → 上采样回原分辨率，输出 [N, H, W] float32 相对深度。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sdm.model_loader import load_depth_model


def resolve_backend(
    requested: str | None,
    *,
    trt_runtime_available: bool,
    device: str | None,
) -> tuple[str, str | None]:
    """把（请求后端, 环境）解析为（实际后端, 警告信息）。纯函数，便于离线单测。

    规则（与“不可用则自动回退 torch 并告警”一致）：
      - torch / None → torch（无警告）
      - tensorrt 但运行时缺失 → 回退 torch + 警告
      - tensorrt 但设备非 cuda → 回退 torch + 警告
      - tensorrt 且运行时/设备均就绪 → tensorrt
    """
    req = (requested or "torch").strip().lower()
    if req == "torch":
        return "torch", None
    if req == "tensorrt":
        if not trt_runtime_available:
            return "torch", "请求了 TensorRT 但运行时不可用（需 NVIDIA GPU + tensorrt，uv sync --group trt），已回退 torch。"
        if (device or "cpu") != "cuda":
            return "torch", "TensorRT 仅在 --device cuda 时可用，已回退 torch。"
        return "tensorrt", None
    raise ValueError(f"未知后端：{requested}（可选 torch / tensorrt）")


class DepthEstimator:
    """封装 DA V2 的加载与推理，供视频管线复用同一份权重。"""

    backend = "torch"

    def __init__(
        self,
        model: str = "small",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        models_dir=None,
        allow_download: bool = True,
    ):
        self.net, self.processor, self.device = load_depth_model(
            model,
            device=device,
            dtype=dtype,
            models_dir=models_dir,
            allow_download=allow_download,
        )
        self.dtype = next(self.net.parameters()).dtype

    @torch.inference_mode()
    def __call__(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        """推理一批 BGR 帧（cv2 格式），返回 [N, H, W] float32 深度。

        深度为模型原生的相对深度（disparity，值越大越近），
        本方法不做归一化，归一化留给 postprocess / 管线统一处理。
        """
        # cv2 是 BGR，processor 内部按 RGB 处理
        images = [Image.fromarray(f[..., ::-1]) for f in frames_bgr]
        heights = [f.shape[0] for f in frames_bgr]
        widths = [f.shape[1] for f in frames_bgr]

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device, self.dtype) for k, v in inputs.items()}
        logits = self.net(**inputs)["predicted_depth"].float()

        # 不同 transformers 版本输出维度不一致：v5 为 [N, h, w]，v4 为 [N, 1, h, w]，
        # 统一成 [N, 1, h, w] 再上采样回原分辨率（processor 会先把帧拉到 14 的倍数）。
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)

        if len(set(zip(heights, widths))) == 1:
            # 整批同尺寸（视频场景常态）：一次上采样，避免逐帧循环
            up = F.interpolate(logits, size=(heights[0], widths[0]), mode="bilinear", align_corners=False)
            return up.squeeze(1).cpu().numpy()

        depths = []
        for i in range(len(frames_bgr)):
            d = F.interpolate(
                logits[i : i + 1],
                size=(heights[i], widths[i]),
                mode="bilinear",
                align_corners=False,
            )
            depths.append(d.squeeze().cpu().numpy())
        return np.stack(depths)
