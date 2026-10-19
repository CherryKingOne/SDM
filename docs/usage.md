# 使用说明

## 1. 环境

项目用 **uv** 管理（不用 conda，也不用 requirements.txt）：

```bash
uv sync --group dev
```

- 解释器：`.python-version` 固定 3.12，uv 自动创建 `.venv`
- 依赖声明：`pyproject.toml` → `transformers / torch / opencv-python / imageio[ffmpeg] / pillow / pyyaml / safetensors`
- 锁定版本：`uv.lock`（提交到 git，保证复现）
- 加依赖：`uv add <pkg>`；看环境：`uv run python -c "import torch; print(torch.__version__)"`

## 2. 权重获取

### ⚠ 必须下带 `-hf` 后缀的仓库

实测两个名字极像的仓库，只有一个能被 transformers 加载：

| HF 仓库 | library_name | 内容 | `from_pretrained` 可用 |
|---|---|---|---|
| `depth-anything/Depth-Anything-V2-Small` | `depth-anything-v2` | 只有 `depth_anything_v2_vits.pth` | ❌ |
| `depth-anything/Depth-Anything-V2-Small-hf` | `transformers` | `config.json` + `model.safetensors` + `preprocessor_config.json` | ✅ |

本项目 `MODELS` 已统一指向 `-hf` 仓库。若你手工下成了 `.pth`，它对本项目无用（加载靠官方仓库代码）。

### 下载方式

```bash
# A. 不管它，直接跑。本地缺失时会自动下载到 models/ 并继续推理
uv run python scripts/depth_video.py -i data/inputs/clip.mp4 -m large

# B. 预先批量拉齐（适合开工前一次性准备、或给离线机器拷过去）
uv run python scripts/download_models.py --sizes small base
uv run python scripts/download_models.py --all      # 三个可用规格（large 约 1.3GB）

# C. 禁用自动下载：缺权重直接报错（CI/离线环境用，避免任务中途卡在下载上）
uv run python scripts/depth_video.py -i data/inputs/clip.mp4 --no-download
```

自动安装由 `sdm.ensure_model()` 完成，落点是**项目 `models/`**而不是 `~/.cache/huggingface`，
这样目录位置可预期、与 `.gitignore` 约定一致。判定“已安装”的标准是 `config.json` **加上**至少一个
`*.safetensors`/`*.bin`：只有 config.json 的半成品会被当成未安装重下，不会误加载。

`sdm.model_loader.resolve_source()` 的优先级：`models/<规格>-hf/` 完整 → 用本地；否则自动下载后再用本地。

### 国内网络环境（两个可行方案）

实测：`huggingface.co` 直连超时。

**方案一（本地代理，已实测跑通）**：

```bash
export https_proxy=http://127.0.0.1:10808 http_proxy=http://127.0.0.1:10808
uv run python scripts/download_models.py --sizes small
```

**方案二（镜像）**：`hf-mirror.com` 可达，但**仅设 `HF_ENDPOINT` 仍会失败**——`hf_xet`
传输层会去连未被镜像的 xethub 域名，需同时禁用它：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
uv run python scripts/download_models.py --sizes small
```

两种方式都只依赖环境变量，代码无需改动；权重一旦落到 `models/`，后续推理完全离线。

## 3. 规格与硬件建议（Mac 场景）

| 规格 | HF 仓库 | 本地权重大小 | Apple Silicon(MPS) 建议 |
|---|---|---|---|
| small | `Depth-Anything-V2-Small-hf` | ~100MB | 放心用，逐帧 + 高清也可 |
| base | `Depth-Anything-V2-Base-hf` | ~400MB | 常规分辨率可用 |
| large | `Depth-Anything-V2-Large-hf` | ~1.3GB | 建议 `--stride 2` 抽帧，控制分辨率 |
| giant | 无 `-hf` 版 | — | ❌ 不能用于本项目（需官方仓库代码） |

### 实测性能（MacBook Air，Apple Silicon / MPS fp16）

| 场景 | small | base | large |
|---|---|---|---|
| 8 帧 @512×384（含加载） | 1.00s | 1.01s | 4.45s |
| 20 帧 @720×480（`--stride 2`） | — | — | 9.66s（峰值内存≈2.4GB） |

同一帧的跨规格一致性（归一化深度 Pearson r）：base vs large **0.976**、small vs large 0.900、small vs base 0.839
——小模型在合成/弱纹理场景上偏差明显更大，选型时优先 base 而非 small。

时序平滑实测（large + 720×480，20 帧）：帧间平均绝对差由 0.0236 降到 0.0181（**-23%**），
代价是运动边缘会有轻微拖影，建议 `--smooth-alpha 0.4~0.7` 取中。

- 设备**默认 CPU**；要用 GPU 必须显式 `--device cuda`（或 `mps`），GPU/MPS 上默认 fp16 省内存。
- 推理后端 `--backend torch`（默认）/ `tensorrt`。后者面向 NVIDIA GPU 服务器，需 `--device cuda` + 已装
  TensorRT；不可用时**自动回退 torch 并打警告**。集成方式 ONNX→engine，预构建见 `scripts/build_trt_engine.py`。
- 处理速度瓶颈通常在逐帧前向：优先 `--stride`（抽帧）与降分辨率，而不是换更小的模型。

## 4. 关键参数

| 参数 | 作用 | 建议 |
|---|---|---|
| `--stride N` | 每 N 帧处理一次，输出帧率同比降低 | 快速预览用 2~5；交付用 1 |
| `--batch-size N` | 单次前向帧数 | GPU 8~16；MPS/CPU 1~4 |
| `--smooth-alpha a` | 沿时间的因果 EMA（a 越小越平滑、越拖影） | 有闪烁时试 0.4~0.7 |
| `--dark-near` | 反转明暗（默认近=亮） | 与下游约定对齐时使用 |
| `--colormap inferno` | 额外输出伪彩 mp4 | 仅预览用，交付建议保留灰度 |
| `--png16 DIR` | 额外输出 16bit 无损深度帧 | 需要再量化/3D 复用时开启，建议 `data/outputs/<名>_depth_frames` |
| `--max-frames N` | 只处理前 N 帧 | 调参、看效果 |

`config/default.yaml` 提供以上选项的默认值，命令行显式传参可覆盖。

## 5. 输出说明

- 主输出：`data/outputs/<名称>_depth.mp4`，灰度 8bit（写盘时灰度会复制成 3 通道 BGR，因为 mp4 编码器不接受单通道帧），与输入同分辨率/（按 stride 调整后的）帧率。
- 8bit mp4 是有量化损失的展示格式。**若要把深度喂回 3D/特效流程，请用 `--png16` 的 16bit 序列**，
  并在 `depth_range`（摘要 JSON 中的 vmin/vmax）上做反量化还原线性相对深度。
- 摘要 JSON 字段：`frames / fps_in / fps_out / depth_range / device / elapsed_sec`。

## 6. 常见现象与处理

| 现象 | 原因 | 处理 |
|---|---|---|
| 深度视频"呼吸/闪烁" | 逐帧独立推理 + 逐帧归一化 | 已默认全序列统一范围；仍抖则加 `--smooth-alpha 0.5` |
| 整体对比度很平 | 场景深度范围大，百分位被拉伸 | 换 `--dark-near` 无效时，考虑分段处理或改分位数 |
| 前景/背景判定相反 | 明暗约定不同 | 用 `--dark-near` 翻转 |
| 想要米制深度 | DA V2 这四个头是相对深度 | 换 `Depth-Anything-V2-Metric-*` 权重并在 `MODELS` 里登记 |
| MPS 报显存/算子错误 | 大模型 + 高分辨率 | `--device cpu`，或降分辨率/`--stride`/换 small |

## 7. 测试

```bash
uv run pytest -q                 # 离线：纯数值 + 桩 estimator 端到端（不需权重/GPU/TensorRT）
SDM_MODEL_TEST=1 uv run pytest -q  # 逐个实测本地已下载的规格（缺的自动 skip）
```

TensorRT 相关测试（`tests/test_tensorrt.py`）全部离线可跑：后端回退决策、导入门控、命名/参数校验不需 GPU；
真正依赖 `import tensorrt` 的冒烟用例在无 TRT 环境（如本地 Mac）自动 skip。
