"""下载 Depth Anything V2 权重到本地 models/ 目录（供 transformers 离线加载）。

只下载 **`-hf` 后缀仓库**（含 config.json + model.safetensors，transformers 可直读）；
不带后缀的同名仓库只放原版 .pth，下载了也用不了。

用法：
    uv run python scripts/download_models.py                 # 只下 small
    uv run python scripts/download_models.py --sizes small base large
    uv run python scripts/download_models.py --all           # 三个可用规格（large 约 1.3GB）

国内网络需代理或镜像，见 docs/usage.md 「国内网络环境」。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sdm.model_loader import (  # noqa: E402
    MODELS,
    TRANSFORMERS_UNAVAILABLE,
    _is_loadable_hf_dir,
    ensure_model,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="下载 DA V2 的 transformers 格式权重到 models/")
    p.add_argument("--sizes", nargs="*", default=["small"], choices=sorted(MODELS))
    p.add_argument("--all", action="store_true", help="下载全部可用规格")
    p.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = p.parse_args(argv)

    for name, hint in TRANSFORMERS_UNAVAILABLE.items():
        print(f"[skip] {name}: {hint}")

    targets = sorted(MODELS) if args.all else args.sizes
    models_root = PROJECT_ROOT / "models"
    for size in targets:
        repo_id, local_name = MODELS[size]
        dest = models_root / local_name
        if _is_loadable_hf_dir(dest) and not args.force:
            print(f"[skip] {size} 已存在：{dest}")
            continue
        print(f"[download] {repo_id} -> {dest}")
        try:
            # 与运行时自动下载走同一套逻辑（同一 patterns、同一完整性校验）
            ensure_model(size, models_dir=models_root, force=args.force, logger=lambda *_: None)
        except Exception as exc:  # 网络/权限问题逐个报告，不影响其余规格
            print(f"[fail] {size}: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
