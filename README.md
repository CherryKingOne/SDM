# SDM — 视频深度估计

用 [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)（Small / Base / Large）
从视频中**逐帧估计深度**，输出**深度视频**（灰度 mp4 / 可选伪彩 mp4 / 可选 16bit 无损深度帧序列）。
模型统一通过 Hugging Face **`transformers`** 的 `AutoModelForDepthEstimation` 加载，本地无权重时会自动下载安装。

- 依赖用 **uv** 管理（不用 conda、不用 requirements.txt）
- 设备自动选择 `cuda → mps → cpu`，GPU/MPS 默认 fp16
- 长视频两趟（two-pass）流式管线，不把整段 float 深度堆进内存
- 全序列统一归一化，抑制深度视频“闪烁/呼吸”

> 详细参数含义、性能实测与常见问题见 [docs/usage.md](docs/usage.md)；设计决策见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。

---

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [命令与参数参考](#命令与参数参考)
  - [depth_video.py（主入口）](#1-depth_videopy主入口)
  - [download_models.py（预下载权重）](#2-download_modelspy预下载权重)
  - [batch_infer.py（批量处理目录）](#3-batch_inferpy批量处理目录)
  - [eval.py（指标评估）](#4-evalpy指标评估)
- [常用配方](#常用配方)
- [在 Python 代码里使用](#在-python-代码里使用)
- [输出说明](#输出说明)
- [输出帧率选择（--output-fps）](#输出帧率--output-fps)
- [模型规格选择](#模型规格选择)
- [重要：输出是相对深度](#重要输出是相对深度)
- [目录约定](#目录约定)
- [测试](#测试)
- [常见问题](#常见问题)

---

## 安装

需要 [uv](https://docs.astral.sh/uv/)（`pip install uv` 或 `brew install uv`）。Python 版本由 `.python-version` 固定为 **3.12**。

```bash
# 克隆仓库
git clone https://github.com/CherryKingOne/SDM.git
cd SDM

# 创建 .venv 并安装依赖（含开发依赖 pytest）
uv sync --group dev
```

之后所有命令都用 `uv run` 前缀执行，**无需手动激活虚拟环境**：

```bash
uv run python ...          # 自动使用 .venv 里的解释器
uv add <pkg>               # 新增依赖（自动更新 pyproject.toml + uv.lock）
```

> 模型权重**不需要**手动安装：首次运行时会自动下载到项目 `models/`。见[快速开始](#快速开始)。

---

## 快速开始

仓库 `data/inputs/` 自带示例视频（`sample.mp4`、`long_clip.mp4`），克隆后即可直接跑：

```bash
# 1) 最简：小模型、全部帧、输出灰度深度视频
uv run python scripts/depth_video.py --input data/inputs/sample.mp4 --model small
#    → 输出写到 data/outputs/sample_depth.mp4，并在终端打印运行摘要 JSON

# 2) 大模型 + 抽帧预览 + 伪彩 + 16bit 无损序列 + 时序平滑
uv run python scripts/depth_video.py \
    -i data/inputs/sample.mp4 -m large \
    --stride 2 --colormap inferno \
    --png16 data/outputs/sample_depth_frames \
    --smooth-alpha 0.5

# 3) 预先批量下载权重（适合开工前一次性拉齐 / 拷给离线机器）
uv run python scripts/download_models.py --sizes small base large
uv run python scripts/download_models.py --all      # 全部可用规格（large 约 1.3GB）

# 4) 禁止自动下载：本地缺权重直接报错（CI / 离线环境）
uv run python scripts/depth_video.py -i data/inputs/sample.mp4 --no-download

# 5) 批量处理整个目录（只扫 data/inputs，结果写 data/outputs）
uv run python scripts/batch_infer.py --input-dir data/inputs --model small --stride 2
```

运行时若本地无权重，会**自动检测并下载**到 `models/`，完成后继续推理：

```
[sdm] 未检测到 small 权重
[sdm] 开始下载 depth-anything/Depth-Anything-V2-Small-hf → models/Depth-Anything-V2-Small-hf
[sdm] 下载完成：models/Depth-Anything-V2-Small-hf
[sdm] model=small device=mps 预计帧数≈8
  pass1 4 帧完成
  深度范围（0.5~99.5 百分位）：[-0.5123, 2.1874]
  pass2 8/8 帧完成
{ "model": "small", "device": "mps", "frames": 8, ... }
```

### 国内网络下载权重

直连 `huggingface.co` 可能超时，两种方式都只靠环境变量，代码无需改动：

```bash
# 方案一：本地代理
export https_proxy=http://127.0.0.1:10808 http_proxy=http://127.0.0.1:10808
uv run python scripts/download_models.py --sizes small

# 方案二：镜像（需同时禁用 hf_xet 传输层，否则仍会去连未镜像的域名而失败）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
uv run python scripts/download_models.py --sizes small
```

权重一旦落到 `models/`，后续推理完全离线。

---

## 命令与参数参考

### 1. `depth_video.py`（主入口）

单个视频 → 深度视频。默认值来自 [`config/default.yaml`](config/default.yaml)，**命令行显式传参优先级更高**。

```bash
uv run python scripts/depth_video.py --input <视频> [选项]
```

| 参数 | 简写 | 默认 | 说明 |
|---|---|---|---|
| `--input` | `-i` | （必填） | 输入视频路径 |
| `--output` | `-o` | `data/outputs/<名>_depth.mp4` | 输出视频路径 |
| `--model` | `-m` | `small` | 模型规格 `small` / `base` / `large`（`giant` 无 transformers 格式，选中会清晰报错） |
| `--device` | | 自动 | `cuda` / `mps` / `cpu`，默认按 cuda→mps→cpu 自动选 |
| `--models-dir` | | `models` | 本地权重根目录 |
| `--batch-size` | | `4` | 单次前向推理帧数（GPU 可 8~16；MPS/CPU 建议 1~4） |
| `--stride` | | `1` | 抽帧步长，>1 加速预览，输出帧率同比降低 |
| `--output-fps` | | 跟随源 | 输出 mp4 目标帧率，可选 `24` / `30` / `60`（详见[输出帧率](#输出帧率--output-fps)）；不选则 = 源帧率/stride |
| `--max-frames` | | 全部 | 只处理前 N 帧（调参、看效果） |
| `--smooth-alpha` | | 关 | 时序因果 EMA 平滑，取值 `(0, 1]`，越小越平滑（如 `0.5`） |
| `--dark-near` | | 关 | 反转为“近暗远亮”（默认近=亮） |
| `--colormap` | | 关 | 额外输出伪彩 mp4：`inferno`/`magma`/`viridis`/`turbo`/`jet`/`hot` |
| `--png16` | | 关 | 额外导出 16bit 无损深度帧序列到指定目录 |
| `--summary-json` | | 关 | 把运行摘要 JSON 另存到指定文件 |
| `--no-download` | | 关 | 禁止自动下载，本地缺权重直接报错 |
| `--config` | | `config/default.yaml` | 指定默认配置文件 |

**示例：**

```bash
# 冒烟测试：只跑前 10 帧快速看效果
uv run python scripts/depth_video.py -i data/inputs/long_clip.mp4 -m small --max-frames 10

# 交付用：base 模型、全帧、强制 CPU
uv run python scripts/depth_video.py -i data/inputs/sample.mp4 -m base --device cpu -o data/outputs/mine.mp4

# 把摘要写入文件，方便脚本化 / 批量收集
uv run python scripts/depth_video.py -i data/inputs/sample.mp4 --summary-json data/outputs/sample_summary.json
```

### 2. `download_models.py`（预下载权重）

把 DA V2 的 **`-hf` 格式**权重下载到本地 `models/`，供离线加载。

```bash
uv run python scripts/download_models.py [选项]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--sizes` | `small` | 要下载的规格，可多选：`small` / `base` / `large` |
| `--all` | 关 | 下载全部可用规格（large 约 1.3GB） |
| `--force` | 关 | 已存在也重新下载 |

**示例：**

```bash
uv run python scripts/download_models.py                      # 只下 small
uv run python scripts/download_models.py --sizes small base   # 下多个
uv run python scripts/download_models.py --all                # 全部可用规格
uv run python scripts/download_models.py --sizes large --force # 强制重下
```

### 3. `batch_infer.py`（批量处理目录）

把一个目录里的所有视频逐个转成深度视频，汇总写 `data/outputs/batch_report.json`。单个失败不中断整批。

```bash
uv run python scripts/batch_infer.py --input-dir <目录> [选项]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input-dir` | （必填） | 输入视频目录（只扫这里） |
| `--output-dir` | `data/outputs` | 输出目录 |
| `--model` | `small` | `small` / `base` / `large` |
| `--device` | 自动 | `cuda` / `mps` / `cpu` |
| `--batch-size` | `4` | 单次前向帧数 |
| `--stride` | `1` | 抽帧步长 |
| `--output-fps` | 跟随源 | 输出目标帧率 `24`/`30`/`60` |
| `--max-frames` | 全部 | 每个视频只处理前 N 帧（冒烟测试） |
| `--smooth-alpha` | 关 | 时序 EMA 平滑强度 |
| `--no-download` | 关 | 禁止自动下载缺失权重 |

> 识别的视频后缀：`.mp4 .mov .mkv .avi .webm .m4v`。

**示例：**

```bash
# 处理整个输入目录
uv run python scripts/batch_infer.py --input-dir data/inputs --model small

# 快速预览整批：抽帧 + 每个视频只前 30 帧
uv run python scripts/batch_infer.py -i data/inputs -m base --stride 3 --max-frames 30
```

### 4. `eval.py`（指标评估）

把预测深度（`--png16` 导出的 16bit PNG 序列）与 GT 深度比对，输出 AbsRel / SqRel / RMSE / log10 / δ1–δ3 等标准指标。

```bash
uv run python evaluation/eval.py --pred <预测帧目录> --gt <GT帧目录> [--align]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--pred` | （必填） | 预测深度帧目录（16bit PNG） |
| `--gt` | （必填） | GT 深度帧目录 |
| `--align` | 关 | 对相对深度做 scale-and-shift 对齐（与米制 GT 比较时**必开**） |
| `--out` | 打印 | 指标结果 JSON 输出路径 |

**示例：**

```bash
# 先生成 16bit 无损序列
uv run python scripts/depth_video.py -i data/inputs/sample.mp4 --png16 data/outputs/sample_depth_frames
# 再对齐评估
uv run python evaluation/eval.py \
    --pred data/outputs/sample_depth_frames \
    --gt data/gt/sample --align
```

---

## 输出帧率（`--output-fps`）

输出的深度 mp4 可以指定一个**目标播放帧率**，只从 `24 / 30 / 60` 三个预设里选（上限锁 60）：

| 取值 | 适用 | 说明 |
|---|---|---|
| 不指定（默认） | 保持原样 | 输出帧率 = 源帧率 / `stride`，不额外重采样 |
| `24` | 电影感 / 减小体积 | 每秒 24 帧，文件更小 |
| `30` | 通用网络视频 | 日常流畅，最常见 |
| `60` | 顺滑预览 | 每秒 60 帧，动态更连贯 |

```bash
# 把输出统一到 30fps
uv run python scripts/depth_video.py -i data/inputs/sample.mp4 --output-fps 30

# 输出 60fps（源若是 30fps，会复制帧填充，见下方注意）
uv run python scripts/depth_video.py -i data/inputs/sample.mp4 --output-fps 60
```

### ⚠ 关于帧率的两个真相

1. **`--output-fps` 只改播放帧率，不改模型算了多少帧。**
   模型对哪些帧做深度推断完全由 `--stride` 决定；`--output-fps` 是在写 mp4 时对已算好的深度序列做**最近邻时间重采样**：
   - 目标 **低于** 已算帧率 → **丢帧**（输出更少帧，时长不变）；
   - 目标 **高于** 已算帧率 → **复制帧**（同一帧重复写，让容器报告更高 fps）。
   复制帧**不会新增真实动作**，只是把同一张深度图多显示几帧——所以追求真实高帧率应当用更小的 `stride`（多算帧），而不是只调大 `--output-fps`。
2. **没有真正的“补帧/插帧”。** 想从 30fps 得到丝般顺滑的 60fps，需要光流帧插值（如 RIFE），本项目未内置；当前上限 60 已避开 120/240 那种“假高帧”的大倍数复制。

> 想**减少计算量**用 `--stride`（少算帧）；想**统一输出规格/交付帧率**用 `--output-fps`。二者可组合：例如 `--stride 2 --output-fps 30` 表示隔帧算深度、再重采样到 30fps 播放。

摘要 JSON 里会同时给出 `frames`（模型实际算的深度帧数）与 `output_frames`（重采样后写入 mp4 的帧数），以及 `fps_in / fps_proc / fps_out` 便于核对。

---

## 常用配方

```bash
# 想快速看效果（低帧率预览，最省时间）
uv run python scripts/depth_video.py -i data/inputs/long_clip.mp4 -m small --stride 5 --colormap inferno

# 想要可用于 3D/特效的高质量深度（16bit 无损）
uv run python scripts/depth_video.py -i data/inputs/sample.mp4 -m large \
    --png16 data/outputs/sample_depth_frames

# 深度视频有闪烁 → 加时序平滑
uv run python scripts/depth_video.py -i data/inputs/sample.mp4 -m base --smooth-alpha 0.5

# 前景/背景明暗反了 → 翻转
uv run python scripts/depth_video.py -i data/inputs/sample.mp4 --dark-near

# MPS 显存/算子报错 → 降负担
uv run python scripts/depth_video.py -i data/inputs/sample.mp4 -m small --stride 2 --batch-size 1 --device cpu
```

---

## 在 Python 代码里使用

包名 `sdm`（src-layout），对外顶层 API 从 `sdm` 直接导入：

```python
from sdm import estimate_video_depth, load_depth_model, ensure_model, DepthEstimator, available_devices

# A) 视频级管线：输入视频 → 深度视频，返回运行摘要 dict
summary = estimate_video_depth(
    "data/inputs/sample.mp4",
    "data/outputs/sample_depth.mp4",
    model="base",
    stride=2,
    smooth_alpha=0.5,
    near_is_bright=True,      # 近=亮
    # colormap=cv2.COLORMAP_INFERNO,        # 额外伪彩
    # png16_dir="data/outputs/frames",      # 额外 16bit 无损序列
    # device="mps", batch_size=4, allow_download=True,
)
print(summary["frames"], summary["depth_range"], summary["elapsed_sec"])

# B) 只做单帧 / 批量推理（复用同一份权重）
est = DepthEstimator("small")             # 权重加载一次
depth = est([frame_bgr])                  # 传入 cv2 BGR 帧列表 → [N, H, W] float32 相对深度（值越大越近）
# 注意：__call__ 不做归一化，归一化/伪彩由 sdm.postprocess 或管线统一处理

# C) 查询可用设备 / 手动准备权重
print(available_devices())                # 例如 ['mps', 'cpu']
path = ensure_model("large")              # 检测本地 large 权重，缺失则下载到 models/，返回其路径
```

`estimate_video_depth` 的完整关键字参数：`model, device, models_dir, batch_size, stride, max_frames, smooth_alpha, near_is_bright, colormap, png16_dir, allow_download, logger`。

---

## 输出说明

- **主输出**：`data/outputs/<名称>_depth.mp4`，8bit 灰度，与输入同分辨率 /（按 stride 调整后的）帧率。
  （写盘时灰度会复制成 3 通道 BGR，因为 mp4 编码器不接受单通道帧，视觉仍是灰度。）
- **伪彩 mp4**：加 `--colormap` 时额外产出 `<名称>_depth_color.mp4`，仅建议预览用。
- **16bit 无损序列**：加 `--png16 DIR` 时导出 `DIR/depth_<帧号:06d>.png`。8bit mp4 有量化损失，**要喂回 3D/特效流程请用 16bit 序列**，并在摘要 JSON 的 `depth_range`（vmin/vmax）上反量化还原线性相对深度。
- **摘要 JSON 字段**：`model / device / input / output / frames / fps_in / fps_out / depth_range / elapsed_sec`。

---

## 模型规格选择

| 规格 | HF 仓库 | 参数量 | 本地权重大小 | 适用 |
|---|---|---|---|---|
| `small` | `Depth-Anything-V2-Small-hf` | ~25M | ~100MB | 视频逐帧推荐起点，Mac(MPS)/CPU 也快 |
| `base` | `Depth-Anything-V2-Base-hf` | ~98M | ~400MB | 精度/速度均衡，**弱纹理/合成场景更稳** |
| `large` | `Depth-Anything-V2-Large-hf` | ~335M | ~1.3GB | 高精度，建议 `--stride 2` 并控制分辨率 |
| `giant` | — | ~1.3B | — | ❌ HF 无 `-hf` 版，不能用 transformers 加载 |

设备按 `cuda → mps → cpu` 自动选择，GPU/MPS 默认 fp16，可用 `--device` 覆盖。
瓶颈通常在逐帧前向：优先 `--stride`（抽帧）与降分辨率，而不是换更小模型。

### ⚠ 仓库后缀是个坑

HF 上 `depth-anything/Depth-Anything-V2-Small`（**不带 `-hf`**）只放原版 `.pth`，`from_pretrained` 读不了；
必须用带 `-hf` 后缀的仓库（含 `config.json` + `model.safetensors`）。两者名字极像，本项目已统一使用 `-hf` 仓库。

---

## 重要：输出是相对深度

DA V2 这四个意识头模型输出的是 **relative depth（disparity，值越大越近）**，**没有米制尺度**。因此：

- 管线对**整段视频**统一取 0.5~99.5 百分位作为归一化范围，而不是逐帧各自 min-max——逐帧归一化会让每帧对比度基准不同，是深度视频闪烁的主要来源。
- 如需**绝对深度（米）**，请改用 `Depth-Anything-V2-Metric-*` 系列权重，并在 `sdm.model_loader.MODELS` 里登记。
- 与米制 GT 评估时请用 `eval.py --align` 做 scale-and-shift 对齐。

---

## 目录约定

```
data/inputs/    放你要处理的原始视频（输入区，只读，gitignore）
data/outputs/   放模型算出的深度视频/帧序列（输出区，可随时清空重跑，gitignore）
models/         模型权重（gitignore，缺失时自动下载到这里）
src/sdm/        核心包：model_loader / estimator / video_pipeline / postprocess / temporal / utils
scripts/        CLI 入口：depth_video / download_models / batch_infer
evaluation/     指标评估：eval / metrics
tests/          离线可跑的测试
config/         default.yaml 运行默认配置
docs/           usage.md 详细使用与排障
```

**输入与输出必须分目录**：`batch_infer.py` 只扫 `data/inputs/`。若输出与输入同目录，下次批量会把自己的输出再当输入算一遍（得到“深度的深度”），结果错且难以察觉。

---

## 测试

```bash
uv run pytest -q                    # 全部离线可跑（管线用桩 estimator，不需权重）
SDM_MODEL_TEST=1 uv run pytest -q   # 额外跑真实权重加载 + 前向推理验证（缺的自动 skip）
```

---

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 深度视频“呼吸/闪烁” | 逐帧独立推理 + 逐帧归一化 | 已默认全序列统一范围；仍抖则加 `--smooth-alpha 0.5` |
| 前景/背景判定相反 | 明暗约定不同 | 用 `--dark-near` 翻转 |
| 整体对比度很平 | 场景深度范围大，百分位被拉伸 | 考虑分段处理或调整分位数 |
| 想要米制深度 | 这四个头是相对深度 | 换 `Depth-Anything-V2-Metric-*` 权重并登记 |
| MPS 报显存/算子错误 | 大模型 + 高分辨率 | `--device cpu`，或降分辨率 / `--stride` / 换 `small` |
| 下载超时 | 直连 huggingface.co 受限 | 用[代理或镜像](#国内网络下载权重)两种环境变量方案 |
| `from_pretrained` 找不到配置 | 下成了不带 `-hf` 的仓库 | 用 `download_models.py` 重下 `-hf` 规格 |

更多参数含义、性能实测数据见 [docs/usage.md](docs/usage.md)。
