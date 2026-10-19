"""Depth Anything V2 模型加载封装（基于 HuggingFace transformers）。

规格名称（small/base/large）→ HF **`-hf`** 仓库 ID 的映射；
权重可先由 scripts/download_models.py 下载到本地 `models/` 目录，
存在本地副本（含 config.json）时优先从本地加载，否则回退到 HF 在线仓库。

设备/精度自适应：**默认 CPU**；仅当显式 `--device cuda/mps` 时才用对应设备。
（GPU 上默认 fp16。可用 `available_devices()` 查询本机支持哪些设备。）

推理后端与设备解耦：`--backend torch`（默认）走 transformers 原生前向；
`--backend tensorrt` 需 `--device cuda` + 已安装 TensorRT，见 `sdm.tensorrt`。
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# 规格简称 → (HF 仓库 ID, 本地目录名)
# ⚠必须用带 `-hf` 后缀的仓库：它们的 library_name=transformers，含 config.json + model.safetensors；
# 不带后缀的同名仓库只放了原版 .pth（如 depth_anything_v2_vits.pth），transformers 无法加载。
MODELS: dict[str, tuple[str, str]] = {
    "small": ("depth-anything/Depth-Anything-V2-Small-hf", "Depth-Anything-V2-Small-hf"),
    "base": ("depth-anything/Depth-Anything-V2-Base-hf", "Depth-Anything-V2-Base-hf"),
    "large": ("depth-anything/Depth-Anything-V2-Large-hf", "Depth-Anything-V2-Large-hf"),
}

# HF 上不存在 `-hf` 版（因此无法用 transformers 直接加载）的规格，仅用于给出清晰报错
TRANSFORMERS_UNAVAILABLE: dict[str, str] = {
    "giant": "depth-anything/Depth-Anything-V2-Giant 只有原版 .pth，HF 上没有 -hf 版；"
             "要跑 Giant 需改用官方仓库代码（depth_anything_v2.dpt），或换 large 规格",
}

# 项目默认权重目录：仓库根下的 models/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

# 自动下载与 scripts/download_models.py 共用，避免两处 pattern 飘移
DOWNLOAD_ALLOW_PATTERNS = ["*.json", "*.txt", "model*.safetensors*"]


def available_devices() -> list[str]:
    """返回当前环境可用的计算设备，按优先级排序。"""
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def pick_device(device: str | None = None) -> torch.device:
    """不指定则用 **CPU**（服务器默认行为）；显式指定则校验可用后使用。

    历史版本会在 device=None 时自动优选 CUDA，现改为默认 CPU：
    要用 GPU 必须显式传 `--device cuda`（或 mps），避免在共享服务器上误占卡。
    """
    if device is None:
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 cuda，但当前环境不可用")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("指定了 mps，但当前环境不可用")
    return torch.device(device)


def _is_loadable_hf_dir(path: Path) -> bool:
    """目录是否构成一个可直接 from_pretrained 的 HF 模型（config + 至少一个权重文件）。

    只查 config.json 会把“权重没下完”的半成品目录误判为可用，从而报错而不是回退在线仓库。
    """
    if not (path / "config.json").is_file():
        return False
    return bool(list(path.glob("*.safetensors")) or list(path.glob("*.bin")))


def local_model_dir(model: str, models_dir: Path | str | None = None) -> Path | None:
    """得到某个规格的本地存放目录；无法确定（自定义仓库 ID）时返回 None。"""
    base = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    if model in MODELS:
        return base / MODELS[model][1]
    if model.count("/") == 1:
        return base / model.split("/")[-1]
    return None


def resolve_source(model: str, models_dir: Path | str | None = None) -> str:
    """把模型名解析为 from_pretrained 可用的来源（本地路径或 HF 仓库 ID）。

    接受三种形式，优先级从高到低：规格简称（small/base/large）、含 config.json 的本地目录、
    完整 HF 仓库 ID（形如 `org/name`）。
    """
    if model in TRANSFORMERS_UNAVAILABLE:
        # 必须在此拦住：否则会被当成仓库 ID 传给 transformers，报出一堆难懂的 OSError
        raise KeyError(
            f"规格 '{model}' 无法用 transformers 加载：{TRANSFORMERS_UNAVAILABLE[model]}"
        )
    if model in MODELS:
        repo_id, local_name = MODELS[model]
        base = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
        local_path = base / local_name
        # 原版 .pth 目录、以及权重未下完的半成品目录都不算可用，必须回退到在线仓库
        if _is_loadable_hf_dir(local_path):
            return str(local_path)
        return repo_id

    # 非简称：允许直接传本地目录或完整仓库 ID
    local = Path(model)
    if local.is_dir():
        return str(local)
    if model.count("/") == 1:
        return model
    raise KeyError(
        f"无法识别的模型 '{model}'：可选 {sorted(MODELS)}，或传含 config.json 的本地目录 / '组织/名称' 仓库 ID"
    )


def ensure_model(
    model: str,
    *,
    models_dir: Path | str | None = None,
    allow_download: bool = True,
    force: bool = False,
    logger=print,
) -> str:
    """保证模型可用：本地已有则直接用，缺失则自动下载到项目 `models/` 并返回本地路径。

    下载目标是仓库而非 HF 全局缓存，目的：与 `.gitignore` 的 `models/` 约定一致、
    多人共享项目时位置可预期、下载不完整时 `_is_loadable_hf_dir` 能兜住重试。

    Args:
        force: 本地已完整也重新拉取（snapshot_download 本身可断点续传）。
    """
    source = resolve_source(model, models_dir)
    local_ready = Path(source).is_dir()
    if local_ready and not force:
        return source

    if local_ready:
        # force 重新下载：source 是本地路径，仓库 ID 需从映射表取回
        if model not in MODELS:
            raise RuntimeError(f"{source} 已是完整本地模型，无法确定其来源仓库，不能配合 force 使用")
        repo_id = MODELS[model][0]
    else:
        repo_id = source
    if not allow_download:
        raise RuntimeError(
            f"未找到 '{model}' 的本地权重（应为 models/{repo_id.split('/')[-1]}/），"
            f"且已禁用自动下载。请先执行：uv run python scripts/download_models.py"
        )

    dest = local_model_dir(model, models_dir)
    if dest is None:  # resolve_source 已限制形式，理论不会到这里
        raise RuntimeError(f"无法为 '{model}' 确定本地存放目录")

    logger(f"[sdm] 未检测到 {model} 权重" if not force else f"[sdm] 强制重新下载 {model} 权重")
    logger(f"[sdm] 开始下载 {repo_id} → {dest}")
    logger("[sdm] 首次下载需联网，国内网络请预先设置代理或镜像（见 docs/usage.md）")

    from huggingface_hub import snapshot_download

    try:
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            # 只取 transformers 格式文件，不拉原版 .pth
            allow_patterns=DOWNLOAD_ALLOW_PATTERNS,
        )
    except Exception as exc:
        # 不直接抛原始堆栈：网络/证书类错误在这里最难诊断，附上已实测可用的两种绕法
        raise RuntimeError(
            f"下载 {repo_id} 失败：{exc}\n"
            f"可改用以下任一方式手动下载后重试：\n"
            f"  1) 代理：export https_proxy=http://127.0.0.1:10808 http_proxy=http://127.0.0.1:10808\n"
            f"  2) 镜像：export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1\n"
            f"     然后：uv run python scripts/download_models.py"
        ) from exc

    if not _is_loadable_hf_dir(dest):
        raise RuntimeError(
            f"下载结束但 {dest} 仍缺少权重文件（可能中断）；"
            f"请重跑 scripts/download_models.py 或删除该目录后重试"
        )
    logger(f"[sdm] 下载完成：{dest}")
    return str(dest)


def load_depth_model(
    model: str = "small",
    *,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    models_dir: Path | str | None = None,
    allow_download: bool = True,
    logger=print,
):
    """加载 DA V2 模型与 image processor（本地缺失时自动下载）。

    Args:
        model: 规格简称 small/base/large（也接受自定义本地路径）。
        device: 强制指定设备（cuda/mps/cpu），默认自动选择。
        dtype: 强制指定精度；默认 GPU 用 fp16、CPU 用 fp32。
        models_dir: 本地权重根目录，默认项目 models/。
        allow_download: 本地无权重时是否自动联网下载（默认 True）。

    Returns:
        (model, processor, used_device) 三元组。
    """
    source = ensure_model(model, models_dir=models_dir, allow_download=allow_download, logger=logger)
    used_device = pick_device(device)
    if dtype is None:
        dtype = torch.float16 if used_device.type in ("cuda", "mps") else torch.float32

    processor = AutoImageProcessor.from_pretrained(source)
    # 兼容不同 transformers 大版本：v5 用 dtype=，v4 用 torch_dtype=
    try:
        net = AutoModelForDepthEstimation.from_pretrained(
            source, dtype=dtype, low_cpu_mem_usage=True
        )
    except TypeError:
        net = AutoModelForDepthEstimation.from_pretrained(
            source, torch_dtype=dtype, low_cpu_mem_usage=True
        )
    net.to(used_device).eval()
    return net, processor, used_device
