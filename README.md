# SDM — 视频深度估计

用 [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)（Small / Base / Large / Giant）
从视频中逐帧估计深度，输出**深度视频**（灰度 mp4 / 可选伪彩 mp4 / 可选 16bit 无损深度帧序列）。
模型统一通过 Hugging Face **`transformers`** 的 `AutoModelForDepthEstimation` 加载。

> 项目结构详见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。

## 环境：uv（不使用 conda / requirements.txt）

依赖全部声明在 [pyproject.toml](pyproject.toml)，版本锁定在 `uv.lock`。

```bash
uv sync --group dev          # 创建 .venv 并安装依赖（含 pytest）
uv run python ...            # 所有命令统一用 uv run 前缀，无需手动 source 虚拟环境
```

## 快速开始

```bash
# 1) 直接跑就行——本地没权重会自动检测并安装到 models/
uv run python scripts/depth_video.py --input data/inputs/clip.mp4 --model small

# 2) 也可预先批量下载（适合开工前一次性拉齐 / 离线机器）
uv run python scripts/download_models.py --sizes small base large
uv run python scripts/download_models.py --all
#    国内网络需先走代理：
#    export https_proxy=http://127.0.0.1:10808 http_proxy=http://127.0.0.1:10808

# 3) 禁止自动下载（例如 CI、离线环境，缺权重则直接报错）
uv run python scripts/depth_video.py -i data/inputs/clip.mp4 --no-download

# 4) 常用可选项：抽帧预览 + 伪彩 + 16bit 无损序列 + 时序平滑
uv run python scripts/depth_video.py -i data/inputs/clip.mp4 -m large \
    --stride 2 --colormap inferno --png16 data/outputs/clip_depth_frames --smooth-alpha 0.5

# 5) 批量处理整个目录（只扫 data/inputs，结果写 data/outputs）
uv run python scripts/batch_infer.py --input-dir data/inputs --model small
```

## 模型自动安装

运行时若本地无权重，会**自动检测并下载安装**到项目 `models/`（不是 HF 全局缓存），完成后继续推理：

```
[sdm] 未检测到 small 权重
[sdm] 开始下载 depth-anything/Depth-Anything-V2-Small-hf → models/Depth-Anything-V2-Small-hf
[sdm] 下载完成：models/Depth-Anything-V2-Small-hf
```

实现集中在 `sdm.ensure_model()`，下载脚本与运行时走同一套逻辑（同一文件过滤、同一完整性校验）：

- 本地已完整 → 直接用，**绝不触网**
- 只有 `config.json` 而权重没下完 → 视为未安装，重试下载（不会误加载）
- 下载失败 → 报错里直接给出代理/镜像两种可用方案，不刷原始堆栈
- `--no-download` → 缺权重就报错（适合 CI 与离线机器）
- `giant` → 在触网前就被拦住（HF 无此格式仓库）

输出默认落在 `data/outputs/<原名>_depth.mp4`，并打印 JSON 运行摘要（帧数、深度范围、耗时）。

## 目录约定

```
data/inputs/    放你要处理的原始视频（输入区，只读）
data/outputs/   放模型算出的深度视频/帧序列（输出区，可随时清空重跑）
models/         模型权重
```

输入与输出必须分目录：`batch_infer.py` 只扫 `data/inputs/`，否则它会把自己的输出再当输入算一遍（得到“深度的深度”）。

## 在代码里使用

```python
from sdm import estimate_video_depth, load_depth_model, DepthEstimator

# 视频级管线
summary = estimate_video_depth("data/inputs/clip.mp4", "data/outputs/depth.mp4", model="small")

# 或只做单帧/批量推理
est = DepthEstimator("base")
depth = est([frame_bgr])          # -> [1, H, W] float32 相对深度（值越大越近）
```

## 模型规格选择

| 规格 | HF 仓库 | 参数量 | 说明 |
|---|---|---|---|
| small | `depth-anything/Depth-Anything-V2-Small-hf` | ~25M | 视频逐帧推荐起点，Mac(MPS)/CPU 也快 |
| base | `depth-anything/Depth-Anything-V2-Base-hf` | ~98M | 精度/速度均衡 |
| large | `depth-anything/Depth-Anything-V2-Large-hf` | ~335M | 高精度（权重~1.3GB） |
| giant | — | ~1.3B | ❌ **HF 上没有 `-hf` 版**，不能用 transformers 加载 |

设备按 `cuda → mps → cpu` 自动选择，GPU 上默认 fp16；可用 `--device` 覆盖。

### ⚠ 仓库后缀是个坑

HF 上 `depth-anything/Depth-Anything-V2-Small`（**不带 `-hf`**）只放了原版 `depth_anything_v2_vits.pth`，
`library_name=depth-anything-v2`；而 `transformers` 需要的是带 `-hf` 后缀的仓库
（含 `config.json` + `model.safetensors`，`library_name=transformers`）。两者名字极像，
下错了 `from_pretrained` 会报找不到配置。本项目已统一使用 `-hf` 仓库。

## 重要说明：输出是相对深度

DA V2 这四个意识头模型输出的是 **relative depth（disparity，值越大越近）**，没有米制尺度。
因此：
- 管线对**整段视频**统一取 0.5~99.5 百分位作为归一化范围，而不是逐帧各自 min-max——
  逐帧归一化会让每帧对比度基准不同，是深度视频闪烁的主要来源；
- 如需绝对深度（米），请改用 `Depth-Anything-V2-Metric-*` 系列权重；
- 与米制 GT 评估时请用 `--align` 做 scale-and-shift 对齐。

## 测试

```bash
uv run pytest -q               # 全部离线可跑（管线用桩 estimator，不需权重）
SDM_MODEL_TEST=1 uv run pytest -q   # 额外跑真实权重加载 + 前向推理验证
```

## 评估

```bash
uv run python evaluation/eval.py --pred data/outputs/clip_depth_frames --gt data/gt/clip --align
```
