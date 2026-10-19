"""在服务器上一性构建 TensorRT engine（ONNX 导出 → engine 序列化 → 缓存到 models/trt）。

TensorRT 后端（`--backend tensorrt`）依赖预先构建好的 engine。engine **不可跨 GPU 架构 /
TRT 版本复用**，因此必须在使用它的目标机上跑本脚本（或首次推理时由管线自动触发构建，见 sdm.tensorrt）。

用法：
    uv run python scripts/build_trt_engine.py --model small --device cuda
    uv run python scripts/build_trt_engine.py --model large --precision fp16 \
        --height 480 --width 720 --opt-batch 4 --max-batch 16
    uv run python scripts/build_trt_engine.py --model base --force   # 忽略缓存重建

说明：
  - engine 与「分辨率」绑定：--height/--width 传目标视频源分辨率（缺省 518×518）。
    processor 会把它 resize 到 14 的倍数，engine 按该尺寸构建；换分辨率需另建。
  - 本脚本需要 GPU + tensorrt：`uv sync --group trt` 后再
    `uv pip install "tensorrt>=10,<11" --extra-index-url https://pypi.nvidia.com/`。无 TensorRT 时会清晰报错。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sdm.model_loader import MODELS, ensure_model  # noqa: E402
from sdm.tensorrt import ensure_engine, trt_dir, trt_runtime_available  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="构建 Depth Anything V2 的 TensorRT engine")
    p.add_argument("--model", "-m", default="small", choices=sorted(MODELS))
    p.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--height", type=int, default=None, help="目标视频高（缺省按 518）")
    p.add_argument("--width", type=int, default=None, help="目标视频宽（缺省按 518）")
    p.add_argument("--opt-batch", type=int, default=4, help="engine 优化批大小（≈ --batch-size）")
    p.add_argument("--max-batch", type=int, default=16, help="engine 支持的最大批")
    p.add_argument("--models-dir", default=None, help="本地权重根目录，默认项目 models/")
    p.add_argument("--force", action="store_true", help="忽略已有缓存，删除后重建")
    args = p.parse_args(argv)

    if not trt_runtime_available():
        print("[trt] 未检测到 TensorRT 运行时。请在带 NVIDIA GPU 的服务器执行：\n"
              "      uv sync --group trt\n"
              "      uv pip install \"tensorrt>=10,<11\" --extra-index-url https://pypi.nvidia.com/")
        return 2

    models_dir = Path(args.models_dir) if args.models_dir else None
    if models_dir is not None and not models_dir.is_absolute():
        models_dir = PROJECT_ROOT / models_dir

    input_hw = None
    if args.height or args.width:
        input_hw = (args.height or 518, args.width or 518)

    if args.force:
        # 删掉同规格的历史产物，ensure_engine 遂重新导出+构建
        name = Path(ensure_model(args.model, models_dir=models_dir, logger=lambda *_: None)).name
        for f in trt_dir(models_dir).glob(f"{name}*"):
            f.unlink(missing_ok=True)
            print(f"[trt] 清理缓存：{f.name}")

    engine = ensure_engine(
        args.model, precision=args.precision, models_dir=models_dir,
        input_hw=input_hw, opt_batch=args.opt_batch, max_batch=args.max_batch,
    )
    print(f"[trt] 完成：{engine}")
    print("[trt] 现在可运行：uv run python scripts/depth_video.py -i <视频> "
          f"-m {args.model} --device cuda --backend tensorrt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
