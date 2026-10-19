"""主入口：输入视频 → 输出深度视频。

用法示例：
    uv run python scripts/depth_video.py --input data/inputs/clip.mp4
    uv run python scripts/depth_video.py --input data/inputs/clip.mp4 \
        --model large --stride 2 --colormap inferno --png16 data/outputs/clip_depth_frames
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml

# 让脚本在未安装包时也能直接运行（uv run python scripts/... 场景）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sdm.model_loader import MODELS, TRANSFORMERS_UNAVAILABLE  # noqa: E402
from sdm.video_pipeline import estimate_video_depth  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"

# 可选规格：可加载的三个 + giant（选中时报清晰错误）
MODEL_CHOICES = sorted(MODELS) + sorted(TRANSFORMERS_UNAVAILABLE)

COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
}


def load_config(path: str | Path) -> dict:
    """读取 yaml 默认配置；文件不存在时返回空 dict。"""
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # 先单独解析 --config，用它的值作为其余参数的默认（命令行显式传入仍可覆盖）
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_CONFIG))
    known, _ = pre.parse_known_args(argv)
    cfg = load_config(known.config)

    p = argparse.ArgumentParser(description="Depth Anything V2 视频深度估计", parents=[pre])
    p.add_argument("--input", "-i", required=True, help="输入视频路径")
    p.add_argument("--output", "-o", help="输出视频路径，默认 data/outputs/<名称>_depth.mp4")
    p.add_argument(
        "--model", "-m", default=cfg.get("model", "small"), choices=MODEL_CHOICES,
        help="模型规格（giant 无 transformers 格式权重，不可用）",
    )
    p.add_argument("--device", default=cfg.get("device"), choices=["cuda", "mps", "cpu"],
                   help="计算设备；不传默认 CPU，--device cuda 用 GPU")
    p.add_argument(
        "--backend", default=cfg.get("backend", "torch"), choices=["torch", "tensorrt"],
        help="推理后端；tensorrt 需 --device cuda + 已装 TensorRT，否则自动回退 torch",
    )
    p.add_argument("--models-dir", default=cfg.get("models_dir"), help="本地权重根目录，默认项目 models/")
    p.add_argument("--batch-size", type=int, default=cfg.get("batch_size", 4))
    p.add_argument("--stride", type=int, default=cfg.get("stride", 1), help="抽帧步长，>1 用于快速预览")
    p.add_argument(
        "--output-fps", type=int, default=cfg.get("output_fps"), choices=[24, 30, 60],
        help="输出视频目标帧率（对已算深度序列最近邻重采样）；不选则跟随源帧率",
    )
    p.add_argument("--max-frames", type=int, default=cfg.get("max_frames"))
    p.add_argument("--smooth-alpha", type=float, default=cfg.get("smooth_alpha"), help="时序 EMA 平滑强度 (0,1]")
    p.add_argument("--dark-near", action="store_true", help="反转为近暗远亮")
    p.add_argument("--colormap", default=cfg.get("colormap"), choices=sorted(COLORMAPS), help="额外输出伪彩视频")
    p.add_argument("--png16", default=cfg.get("png16_dir"), help="额外导出 16bit 深度帧序列到该目录")
    p.add_argument("--summary-json", default=None, help="把运行摘要写入该 json 文件")
    p.add_argument(
        "--no-download", action="store_true",
        help="禁止自动下载：本地 models/ 里缺权重时直接报错（默认会自动下载）",
    )
    p.add_argument(
        "--no-progress", action="store_true",
        help="关闭 tqdm 进度条（默认仅在终端显示；非终端环境自动静默）",
    )
    args = p.parse_args(argv)
    if not 0 < (args.smooth_alpha or 1) <= 1:
        p.error("--smooth-alpha 必须落在 (0, 1]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src = Path(args.input)
    out = Path(args.output) if args.output else PROJECT_ROOT / "data" / "outputs" / f"{src.stem}_depth.mp4"
    # 配置里的相对权重目录统一按项目根解析，避免受当前工作目录影响
    models_dir = Path(args.models_dir) if args.models_dir else None
    if models_dir is not None and not models_dir.is_absolute():
        models_dir = PROJECT_ROOT / models_dir

    try:
        summary = estimate_video_depth(
            src,
            out,
            model=args.model,
            device=args.device,
            models_dir=models_dir,
            batch_size=args.batch_size,
            stride=args.stride,
            output_fps=args.output_fps,
            max_frames=args.max_frames,
            smooth_alpha=args.smooth_alpha,
            near_is_bright=not args.dark_near,
            colormap=COLORMAPS[args.colormap] if args.colormap else None,
            png16_dir=args.png16,
            allow_download=not args.no_download,
            backend=args.backend,
            show_progress=False if args.no_progress else None,
        )
    except KeyError as exc:  # 规格不可用：给一行清晰提示，不刷 traceback
        print(f"[sdm] {exc.args[0] if exc.args else exc}")
        return 2
    except RuntimeError as exc:  # 下载失败/禁用下载：同样只要一行可操作的提示
        print(f"[sdm] {exc}")
        return 3
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
