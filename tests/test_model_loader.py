"""验证模型规格映射、本地权重优先解析与设备选择（不依赖网络）。"""

import json
import os
from pathlib import Path

import pytest

from sdm.model_loader import (
    MODELS,
    TRANSFORMERS_UNAVAILABLE,
    available_devices,
    ensure_model,
    pick_device,
    resolve_source,
)


def test_transformers_sizes_registered_as_hf_repos():
    assert set(MODELS) == {"small", "base", "large"}
    for size, (repo_id, dir_name) in MODELS.items():
        # 必须是 -hf 仓库：不带后缀的只放原版 .pth，transformers 读不了
        assert repo_id == f"depth-anything/Depth-Anything-V2-{size.capitalize()}-hf"
        assert dir_name == repo_id.split("/")[-1]


def test_giant_is_declared_unavailable_with_hint():
    assert "giant" in TRANSFORMERS_UNAVAILABLE
    with pytest.raises(KeyError) as exc:
        resolve_source("giant")
    assert "transformers" in str(exc.value)


def test_resolve_source_falls_back_to_hub_id(tmp_path):
    assert resolve_source("small", tmp_path) == "depth-anything/Depth-Anything-V2-Small-hf"


def test_resolve_source_prefers_local_weights(tmp_path):
    local = tmp_path / "Depth-Anything-V2-Small-hf"
    local.mkdir()
    (local / "config.json").write_text(json.dumps({"model_type": "depth_anything"}))
    (local / "model.safetensors").write_text("fake")
    assert resolve_source("small", tmp_path) == str(local)


def test_incomplete_download_falls_back_to_hub(tmp_path):
    """有 config.json 但权重没下完的半成品目录，不能当本地模型用。"""
    local = tmp_path / "Depth-Anything-V2-Small-hf"
    local.mkdir()
    (local / "config.json").write_text("{}")
    assert resolve_source("small", tmp_path) == "depth-anything/Depth-Anything-V2-Small-hf"


def test_pth_only_dir_is_not_treated_as_loadable(tmp_path):
    """只有原版 .pth（无 config.json）的目录必须回退到 HF 仓库，不能误当本地模型。"""
    local = tmp_path / "Depth-Anything-V2-Small-hf"
    local.mkdir()
    (local / "depth_anything_v2_vits.pth").write_text("fake")
    assert resolve_source("small", tmp_path) == "depth-anything/Depth-Anything-V2-Small-hf"


def test_resolve_source_unknown_size():
    with pytest.raises(KeyError):
        resolve_source("tiny")


def test_resolve_source_passes_through_custom_repo_id():
    assert resolve_source("depth-anything/Custom-Model-hf") == "depth-anything/Custom-Model-hf"


def test_resolve_source_accepts_custom_local_dir(tmp_path):
    custom = tmp_path / "my-da-copy"
    custom.mkdir()
    (custom / "config.json").write_text("{}")
    assert resolve_source(str(custom)) == str(custom)


def test_giant_never_reaches_transformers_as_repo_id():
    """giant 必须在 resolve_source 就被拦住，而不是报难懂的 OSError。"""
    from sdm.model_loader import load_depth_model

    with pytest.raises(KeyError, match="giant"):
        load_depth_model("giant", device="cpu")


def test_available_devices_always_ends_with_cpu():
    assert available_devices()[-1] == "cpu"


def test_pick_device_defaults_to_cpu():
    """不传 device 时默认 CPU（不再自动优选 cuda）；要用 GPU 必须显式 --device cuda。"""
    assert pick_device(None).type == "cpu"
    assert pick_device().type == "cpu"


def test_pick_device_rejects_unavailable_cuda():
    try:
        import torch

        if torch.cuda.is_available():
            pytest.skip("本机有 cuda，无法测拒绝分支")
    except ImportError:  # pragma: no cover
        pytest.skip("未安装 torch")
    with pytest.raises(RuntimeError):
        pick_device("cuda")


# ---------------- 本地缺权重时自动安装（ensure_model）----------------


class FakeHub:
    """替身 snapshot_download：可模拟成功 / 网络失败 / 下完仍无权重（中断）。"""

    def __init__(self, *, fail=False, write_weights=True):
        self.fail, self.write_weights = fail, write_weights
        self.calls = []

    def __call__(self, *, repo_id, local_dir, allow_patterns=None, **_kw):
        self.calls.append({"repo_id": repo_id, "local_dir": local_dir, "patterns": allow_patterns})
        if self.fail:
            raise OSError("Connect timeout against huggingface.co")
        dest = Path(local_dir)
        (dest / "config.json").write_text("{}")
        if self.write_weights:
            (dest / "model.safetensors").write_text("fake")
        return str(dest)


def test_ensure_model_uses_local_without_network(tmp_path, monkeypatch):
    local = tmp_path / "Depth-Anything-V2-Small-hf"
    local.mkdir()
    (local / "config.json").write_text("{}")
    (local / "model.safetensors").write_text("fake")
    hub = FakeHub()
    monkeypatch.setattr("huggingface_hub.snapshot_download", hub)

    assert ensure_model("small", models_dir=tmp_path) == str(local)
    assert hub.calls == []  # 本地已有就不能触网


def test_ensure_model_downloads_when_missing(tmp_path, monkeypatch):
    hub = FakeHub()
    monkeypatch.setattr("huggingface_hub.snapshot_download", hub)

    got = ensure_model("small", models_dir=tmp_path)

    dest = tmp_path / "Depth-Anything-V2-Small-hf"
    assert got == str(dest)
    assert (dest / "model.safetensors").exists()
    # 必须用 -hf 仓库，且不能拉原版 .pth
    assert hub.calls[0]["repo_id"] == "depth-anything/Depth-Anything-V2-Small-hf"
    assert not any(p.endswith(".pth") for p in hub.calls[0]["patterns"])


def test_ensure_model_respects_no_download(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", FakeHub())

    with pytest.raises(RuntimeError, match="已禁用自动下载"):
        ensure_model("small", models_dir=tmp_path, allow_download=False)


def test_ensure_model_network_failure_gives_actionable_hint(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", FakeHub(fail=True))

    with pytest.raises(RuntimeError) as exc:
        ensure_model("small", models_dir=tmp_path)
    msg = str(exc.value)
    assert "https_proxy" in msg and "HF_ENDPOINT" in msg  # 代理与镜像两条出路


def test_ensure_model_detects_truncated_download(tmp_path, monkeypatch):
    """下载函数返回了但没权重文件（中断）：不能假称就绪。"""
    monkeypatch.setattr("huggingface_hub.snapshot_download", FakeHub(write_weights=False))

    with pytest.raises(RuntimeError, match="中断"):
        ensure_model("small", models_dir=tmp_path)


def test_ensure_model_skips_download_for_giant(tmp_path, monkeypatch):
    hub = FakeHub()
    monkeypatch.setattr("huggingface_hub.snapshot_download", hub)

    with pytest.raises(KeyError, match="giant"):
        ensure_model("giant", models_dir=tmp_path)
    assert hub.calls == []  # 不可能尝试去下不存在的 -hf 仓库


@pytest.mark.skipif(not os.getenv("SDM_MODEL_TEST"), reason="需真实权重；设 SDM_MODEL_TEST=1 启用")
@pytest.mark.parametrize("size", sorted(MODELS))
def test_real_loading_and_inference(size):  # pragma: no cover
    """对每个**已下载到本地**的规格：验证 transformers 能加载，且前向输出是合法 2D 深度图。

    未下载的规格直接 skip（不让普通 `pytest` 因为联网拉 1.3GB 权重而变慢）。
    """
    import numpy as np

    from sdm import DepthEstimator

    source = resolve_source(size)
    if not Path(source).is_dir():
        pytest.skip(f"{size} 权重不在本地，先跑：uv run python scripts/download_models.py --sizes {size}")

    est = DepthEstimator(size)  # 设备自动（cuda>mps>cpu）
    frame = np.zeros((240, 320, 3), np.uint8)
    frame[60:180, 60:180] = 200
    depth = est([frame, frame[::-1]])

    assert depth.shape == (2, 240, 320)  # 上采样必须回到原分辨率
    assert depth.dtype == np.float32
    assert np.isfinite(depth).all()
    assert depth.max() > depth.min()  # 深度非退化
