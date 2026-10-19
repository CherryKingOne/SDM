"""TensorRT 加速后端：ONNX 导出 → engine 构建 → 推理。

设计要点
--------
- **仅服务器生效**：TensorRT 需要 NVIDIA GPU + CUDA + `tensorrt` 运行时（`uv sync --group trt`）。
  本机（macOS/CPU）无法运行，因此本模块对 `tensorrt`/`torch` 采用**惰性导入**：
  导入本模块、调用 `trt_runtime_available()` 都不需要装 TensorRT；真正构建/推理时才校验。
- **与 torch 后端同接口**：`TensorRTEstimator.__call__(frames_bgr) -> [N, H, W] float32`
  相对深度，与 `sdm.estimator.DepthEstimator` 完全一致，可被 video_pipeline 无缝替换。
- **预处理仍用 HF processor**：只有主干前向交给 TRT engine；engine 输入是 processor 产出的
  `pixel_values`（float32）。这样归一化/resize 逻辑与 torch 后端保持逐位一致。
- **engine 与分辨率绑定**：DA V2 是 ViT+DPT，位置编码插值对"全动态空间尺寸"在 TRT 里不稳，
  故 engine 按 (规格, 精度, 高, 宽, TRT 版本) 缓存；批维走动态 profile（min/opt/max），
  换分辨率会命中不同缓存并重建。engine **不可跨 GPU 架构 / TRT 版本复用**，需目标机构建。

后端选择与回退由 `sdm.estimator.resolve_backend()` + `video_pipeline` 负责：TensorRT 不可用时
打警告并自动回退 torch，本模块只负责"要么干成、要么抛清晰错误"。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from sdm.model_loader import DEFAULT_MODELS_DIR, ensure_model, load_depth_model

TRT_IMPORT_HINT = (
    "未检测到 TensorRT 运行时（import tensorrt 失败）。TensorRT 后端需在带 NVIDIA GPU 的服务器上安装：\n"
    "  uv sync --group trt\n"
    "  uv pip install \"tensorrt>=10,<11\" --extra-index-url https://pypi.nvidia.com/\n"
    "本地 / 无 GPU 环境请改用 --backend torch（默认）。"
)


def trt_runtime_available() -> bool:
    """当前环境能否 `import tensorrt`（不实际导入，避免无 GPU 时触发加载副作用）。"""
    return importlib.util.find_spec("tensorrt") is not None


def _require_tensorrt():
    """惰性导入 tensorrt；缺失时给可操作提示，由上层回退逻辑捕获。"""
    if not trt_runtime_available():
        raise RuntimeError(TRT_IMPORT_HINT)
    import tensorrt as trt  # noqa: PLC0415  仅服务器可用，延迟导入

    return trt


def trt_dir(models_dir: Path | str | None = None) -> Path:
    """TRT 产物（onnx / engine）目录：models/trt，与 HF 权重目录隔离，避免被误当本地模型。"""
    base = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    return base / "trt"


def _sanitize(tag: str) -> str:
    return tag.replace("/", "_").replace(" ", "")


def engine_tag(spec_name: str, precision: str, height: int, width: int, trt_version: str) -> str:
    """engine/onnx 文件名标签：规格 + 精度 + 空间尺寸 + TRT 版本（不含批维，批维在 profile 里动态）。"""
    return _sanitize(f"{spec_name}__{precision}__{height}x{width}__trt{trt_version}")


def export_model_to_onnx(net, sample_pixel_values, onnx_path: Path, *, opset: int = 17) -> Path:
    """把 DA V2 主干导出为 ONNX：输入 pixel_values，输出 predicted_depth，仅批维动态。

    transformers 模型 forward 返回 ModelOutput（dict -like），直接导出不友好，故包一层只返回张量。
    导出在 CPU 上以 fp32 进行即可（与设备无关），engine 构建才需要 GPU。
    """
    import torch  # noqa: PLC0415

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values)["predicted_depth"]

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = _Wrap(net).eval()
    dummy = sample_pixel_values.detach().float().cpu()
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(onnx_path),
        opset=opset,
        input_names=["pixel_values"],
        output_names=["predicted_depth"],
        dynamic_axes={"pixel_values": {0: "batch"}, "predicted_depth": {0: "batch"}},
        dynamo=False,  # 经典导出器对 TRT OnnxParser 兼容性最好
    )
    return onnx_path


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    fp16: bool = True,
    min_batch: int = 1,
    opt_batch: int = 4,
    max_batch: int = 16,
    spatial: tuple[int, int],
    workspace_gb: int = 4,
    logger=print,
) -> Path:
    """从 ONNX 构建并序列化 TensorRT engine（固定空间尺寸、批维动态）。需 GPU + tensorrt。"""
    trt = _require_tensorrt()
    h, w = spatial
    logger(f"[trt] 构建 engine：{onnx_path.name} → {engine_path.name}（首次较慢，opt_batch={opt_batch}）")
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX 解析失败：{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb) << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    in_name = network.get_input(0).name
    profile.add_shape(
        in_name,
        (min_batch, 3, h, w),
        (opt_batch, 3, h, w),
        (max_batch, 3, h, w),
    )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine 构建失败（见上方 TRT 日志）")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    logger(f"[trt] engine 就绪：{engine_path}")
    return engine_path


def ensure_engine(
    model: str = "small",
    *,
    precision: str = "fp16",
    models_dir: Path | str | None = None,
    allow_download: bool = True,
    input_hw: tuple[int, int] | None = None,
    opt_batch: int = 4,
    max_batch: int = 16,
    logger=print,
) -> Path:
    """保证目标规格 + 分辨率的 engine 就绪：已缓存直接返回，否则导出 ONNX 再构建。

    这是 `build_trt_engine.py` 与 `TensorRTEstimator` 的共用入口，保证缓存键与构建流程单一。
    需要 GPU + tensorrt（构建阶段）；ONNX 导出阶段用 CPU。
    """
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import AutoImageProcessor  # noqa: PLC0415

    trt = _require_tensorrt()
    source = ensure_model(model, models_dir=models_dir, allow_download=allow_download, logger=logger)
    processor = AutoImageProcessor.from_pretrained(source)

    src_h, src_w = input_hw if input_hw else (518, 518)
    probe = processor(images=[Image.fromarray(np.zeros((src_h, src_w, 3), np.uint8))], return_tensors="pt")
    pixel_values = probe["pixel_values"]
    sh, sw = int(pixel_values.shape[2]), int(pixel_values.shape[3])

    tag = engine_tag(Path(source).name, precision, sh, sw, trt.__version__)
    d = trt_dir(models_dir)
    engine_path = d / f"{tag}.engine"
    onnx_path = d / f"{tag}.onnx"
    if engine_path.exists():
        return engine_path

    # 仅在需要构建时才加载 torch 权重（CPU, fp32，导出用）
    net, _, _ = load_depth_model(
        model, device="cpu", dtype=torch.float32,
        models_dir=models_dir, allow_download=allow_download, logger=logger,
    )
    net.eval()
    if not onnx_path.exists():
        logger(f"[trt] 导出 ONNX：{onnx_path}")
        export_model_to_onnx(net, pixel_values, onnx_path)
    del net  # 释放权重，engine 阶段不吃 torch 显存
    build_engine(
        onnx_path, engine_path,
        fp16=(precision == "fp16"), opt_batch=opt_batch, max_batch=max_batch,
        spatial=(sh, sw), logger=logger,
    )
    return engine_path


class TensorRTEstimator:
    """TensorRT 推理封装，接口与 `DepthEstimator` 对齐（供两趟管线直接替换）。"""

    backend = "tensorrt"

    def __init__(
        self,
        model: str = "small",
        *,
        device: str | None = None,
        dtype=None,  # 保留形参以与 DepthEstimator 构造签名兼容；TRT 用 engine 精度，忽略此参数
        models_dir: Path | str | None = None,
        allow_download: bool = True,
        precision: str = "fp16",
        input_size: tuple[int, int] | None = None,
        opt_batch: int = 4,
        max_batch: int = 16,
        logger=print,
    ):
        import torch  # noqa: PLC0415
        from transformers import AutoImageProcessor  # noqa: PLC0415

        if (device or "cpu") != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("TensorRT 后端需要 CUDA 设备（请显式 --device cuda）")
        self._torch = torch
        trt = _require_tensorrt()
        self._trt = trt
        self.device = torch.device("cuda")
        self.precision = precision
        self.logger = logger

        engine_path = ensure_engine(
            model, precision=precision, models_dir=models_dir, allow_download=allow_download,
            input_hw=input_size, opt_batch=opt_batch, max_batch=max_batch, logger=logger,
        )
        source = ensure_model(model, models_dir=models_dir, allow_download=allow_download, logger=logger)
        self.processor = AutoImageProcessor.from_pretrained(source)

        logger_ = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger_).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"反序列化 engine 失败：{engine_path}（GPU 架构或 TRT 版本变了，删除后重建）")
        self.engine_path = engine_path
        self.context = self.engine.create_execution_context()

        self.input_name = self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        if self.input_name is None or self.output_name is None:
            raise RuntimeError("engine 输入/输出张量不符合预期")

    def _torch_dtype(self, trt_dtype):
        trt = self._trt
        return {
            trt.float32: self._torch.float32,
            trt.float16: self._torch.float16,
            trt.int32: self._torch.int32,
        }.get(trt_dtype, self._torch.float32)

    def __call__(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        """一批 BGR 帧 → [N, H, W] float32 相对深度（值越大越近，不做归一化，交给管线）。"""
        import torch.nn.functional as F  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        torch = self._torch
        trt = self._trt
        heights = [f.shape[0] for f in frames_bgr]
        widths = [f.shape[1] for f in frames_bgr]
        images = [Image.fromarray(f[..., ::-1]) for f in frames_bgr]

        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
        # engine 空间尺寸固定：非目标分辨率会在此拦住，提示重建而非静默算错
        exp_h, exp_w = int(self.engine.get_tensor_shape(self.input_name)[2]), \
            int(self.engine.get_tensor_shape(self.input_name)[3])
        if int(pixel_values.shape[2]) != exp_h or int(pixel_values.shape[3]) != exp_w:
            raise RuntimeError(
                f"帧经 processor 得到 {tuple(pixel_values.shape[2:])}，与 engine 期望 "
                f"({exp_h},{exp_w}) 不符；请用对应分辨率重建 engine"
            )
        pixel_values = pixel_values.to(self.device, torch.float32).contiguous()

        self.context.set_input_shape(self.input_name, tuple(pixel_values.shape))
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        out_dtype = self._torch_dtype(self.engine.get_tensor_dtype(self.output_name))
        output = torch.empty(out_shape, dtype=out_dtype, device=self.device)

        self.context.set_tensor_address(self.input_name, pixel_values.data_ptr())
        self.context.set_tensor_address(self.output_name, output.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        with torch.inference_mode():
            if not self.context.execute_async_v3(stream):
                raise RuntimeError("TensorRT 执行失败")
        torch.cuda.synchronize()

        logits = output.float()
        if logits.dim() == 3:  # v5 [N,h,w] → v4 风格 [N,1,h,w]，与 DepthEstimator 对齐
            logits = logits.unsqueeze(1)

        if len(set(zip(heights, widths))) == 1:
            up = F.interpolate(logits, size=(heights[0], widths[0]), mode="bilinear", align_corners=False)
            return up.squeeze(1).cpu().numpy()

        depths = []
        for i in range(len(frames_bgr)):
            d = F.interpolate(
                logits[i : i + 1], size=(heights[i], widths[i]),
                mode="bilinear", align_corners=False,
            )
            depths.append(d.squeeze().cpu().numpy())
        return np.stack(depths)
