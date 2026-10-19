"""评估入口：把预测深度（16bit PNG 序列）与 GT 深度比对，输出标准指标。

用法：
    uv run python evaluation/eval.py --pred data/outputs/clip_depth_frames --gt data/gt/clip --align

说明：DA V2 输出的是相对深度（无量纲），与米制 GT 比较时需加 --align 做尺度+偏移对齐。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.metrics import align_scale_shift, depth_errors  # noqa: E402


def load_pair(pred_path: Path, gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pred = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
    if pred is None or gt is None:
        raise FileNotFoundError(f"读取失败：{pred_path} / {gt_path}")
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    if pred.shape != gt.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
    return pred, gt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="深度指标评估")
    p.add_argument("--pred", required=True, help="预测深度帧目录（16bit PNG）")
    p.add_argument("--gt", required=True, help="GT 深度帧目录")
    p.add_argument("--align", action="store_true", help="对相对深度做 scale-and-shift 对齐")
    p.add_argument("--out", default=None, help="指标结果 json 输出路径")
    args = p.parse_args(argv)

    pred_dir, gt_dir = Path(args.pred), Path(args.gt)
    pred_files = sorted(pred_dir.glob("*.png"))
    if not pred_files:
        print(f"[eval] {pred_dir} 下没有 PNG 深度帧")
        return 1

    accum: dict[str, list[float]] = {}
    for pf in pred_files:
        gf = gt_dir / pf.name
        if not gf.exists():
            gf = gt_dir / pf.name.split("_")[-1]  # 兼容 depth_000123.png ↔ 000123.png
        if not gf.exists():
            continue
        pred, gt = load_pair(pf, gf)
        if args.align:
            pred = align_scale_shift(pred, gt)
        for k, v in depth_errors(pred, gt).items():
            accum.setdefault(k, []).append(v)

    if not accum:
        print("[eval] 没有匹配到任何帧对")
        return 1
    result = {k: float(np.mean(v)) for k, v in accum.items()}
    result["_frames_evaluated"] = len(next(iter(accum.values())))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
