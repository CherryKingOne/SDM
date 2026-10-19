"""批量处理：把一个目录下的所有视频逐个转成深度视频。

用法：
    uv run python scripts/batch_infer.py --input-dir data/inputs --model small --stride 2

只扫描 data/inputs（输入区），结果写 data/outputs（输出区）——两者必须分离，
否则下一次批量会把自己的输出当输入再算一遍（得到"深度的深度"）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sdm.model_loader import MODELS  # noqa: E402
from sdm.video_pipeline import estimate_video_depth  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="批量视频深度估计")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", default=None, help="默认 data/outputs")
    p.add_argument("--model", default="small", choices=sorted(MODELS))
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                   help="计算设备；不传默认 CPU")
    p.add_argument("--backend", default="torch", choices=["torch", "tensorrt"],
                   help="推理后端；tensorrt 需 --device cuda + 已装 TensorRT，否则自动回退 torch")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--output-fps", type=int, default=None, choices=[24, 30, 60],
                   help="输出视频目标帧率；不选则跟随源帧率")
    p.add_argument("--max-frames", type=int, default=None, help="每个视频只处理前 N 帧（冒烟测试用）")
    p.add_argument("--smooth-alpha", type=float, default=None)
    p.add_argument("--no-download", action="store_true", help="禁止自动下载缺失权重")
    p.add_argument("--no-progress", action="store_true", help="关闭 tqdm 进度条")
    args = p.parse_args(argv)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "outputs"
    videos = sorted(f for f in in_dir.iterdir() if f.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        print(f"[sdm] {in_dir} 下没有找到视频文件")
        return 1

    records = []
    failed = 0
    for i, video in enumerate(videos, 1):
        print(f"\n[sdm] ({i}/{len(videos)}) {video.name}")
        try:
            records.append(
                estimate_video_depth(
                    video,
                    out_dir / f"{video.stem}_depth.mp4",
                    model=args.model,
                    device=args.device,
                    batch_size=args.batch_size,
                    stride=args.stride,
                    output_fps=args.output_fps,
                    max_frames=args.max_frames,
                    smooth_alpha=args.smooth_alpha,
                    allow_download=not args.no_download,
                    backend=args.backend,
                    show_progress=False if args.no_progress else None,
                )
            )
        except Exception as exc:  # 单个失败不中断整批
            failed += 1
            print(f"[fail] {video.name}: {exc}")

    report = out_dir / "batch_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"\n[sdm] 成功 {len(records)} / 失败 {failed}，报告：{report}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
