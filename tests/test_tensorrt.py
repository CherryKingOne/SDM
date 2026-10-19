"""TensorRT 后端的离线可测部分：导入门控、后端回退决策、命名与参数校验。

本套测试**不需要 GPU / TensorRT**：
  - 导入 `sdm.tensorrt` 不应触发 `import tensorrt`（惰性导入）；
  - 后端选择用纯函数 `resolve_backend` 覆盖各回退分支；
  - 需要真实 TensorRT 运行时的调用，用 skipif 在无 TRT 环境（如本地 Mac）自动跳过。
"""

import pytest

from sdm.estimator import resolve_backend
from sdm.tensorrt import (
    TensorRTEstimator,
    build_engine,
    engine_tag,
    ensure_engine,
    trt_runtime_available,
    trt_dir,
)

needs_trt = pytest.mark.skipif(not trt_runtime_available(), reason="需要 TensorRT 运行时（服务器）")


# ---------------- 后端回退决策（纯函数）----------------

def test_resolve_backend_torch_default():
    assert resolve_backend("torch", trt_runtime_available=False, device="cpu") == ("torch", None)
    assert resolve_backend(None, trt_runtime_available=False, device="cpu") == ("torch", None)


def test_resolve_backend_tensorrt_without_runtime_falls_back():
    eff, warn = resolve_backend("tensorrt", trt_runtime_available=False, device="cuda")
    assert eff == "torch"
    assert warn and "回退" in warn


def test_resolve_backend_tensorrt_on_non_cuda_falls_back():
    eff, warn = resolve_backend("tensorrt", trt_runtime_available=True, device="cpu")
    assert eff == "torch"
    assert warn and "cuda" in warn


def test_resolve_backend_tensorrt_ready():
    assert resolve_backend("tensorrt", trt_runtime_available=True, device="cuda") == ("tensorrt", None)


def test_resolve_backend_unknown_raises():
    with pytest.raises(ValueError):
        resolve_backend("openvino", trt_runtime_available=False, device="cpu")


# ---------------- 命名与目录 ----------------

def test_engine_tag_encodes_spec_precision_size_version():
    tag = engine_tag("Depth-Anything-V2-Small-hf", "fp16", 392, 518, "10.2.0")
    assert tag == "Depth-Anything-V2-Small-hf__fp16__392x518__trt10.2.0"
    assert "/" not in tag and " " not in tag


def test_trt_dir_isolated_from_weights(tmp_path):
    assert trt_dir(tmp_path) == tmp_path / "trt"


# ---------------- 无 TensorRT 时的清晰失败（本地 Mac 走这条）----------------

@pytest.mark.skipif(trt_runtime_available(), reason="本机有 TensorRT，跳过“缺失即报错”分支")
def test_build_engine_raises_actionable_error_without_trt(tmp_path):
    with pytest.raises(RuntimeError, match="tensorrt"):
        build_engine(tmp_path / "x.onnx", tmp_path / "x.engine", spatial=(518, 518))


@pytest.mark.skipif(trt_runtime_available(), reason="本机有 TensorRT，跳过“缺失即报错”分支")
def test_ensure_engine_raises_without_trt(tmp_path):
    with pytest.raises(RuntimeError, match="tensorrt"):
        ensure_engine("small", models_dir=tmp_path)


def test_tensorrt_estimator_requires_cuda():
    """非 cuda 设备（含本地 CPU）构造 TensorRTEstimator 应清晰报错，交由上层回退 torch。"""
    with pytest.raises(RuntimeError, match="CUDA"):
        TensorRTEstimator("small", device="cpu")


# ---------------- 真实运行时冒烟（仅服务器执行）----------------

@needs_trt
def test_ensure_engine_builds_file(tmp_path):  # pragma: no cover
    from pathlib import Path

    engine = ensure_engine("small", models_dir=tmp_path, precision="fp16", input_hw=(384, 512))
    assert Path(engine).is_file() and engine.suffix == ".engine"
