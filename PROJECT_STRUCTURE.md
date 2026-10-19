# SDM — 基于 Depth Anything V2 的视频深度估计项目目录结构设计方案

> 目标：使用 Hugging Face `transformers` 库加载 **Depth Anything V2**（Small / Base / Large / Giant）
> 系列模型，对输入视频逐帧推理深度图，并输出深度视频（灰度 / 可选伪彩 / 可选 16bit PNG 序列）。
>
> 状态：目录结构与代码骨架已按本文档实现（uv 管理依赖）。

## 一、模型选型（transformers 原生支持）

Depth Anything V2 已合入 `transformers`，无需自定义 modeling 代码。
**必须用带 `-hf` 后缀的仓库**（含 `config.json` + `model.safetensors`）；不带后缀的同名仓库
只放原版 `.pth`（`library_name=depth-anything-v2`），`from_pretrained` 读不了。

| 规格 | HF 仓库 ID | 参数量 | 适用场景 |
|---|---|---|---|
| Small | `depth-anything/Depth-Anything-V2-Small-hf` | ~25M | 实时/低显存（视频逐帧推荐起点） |
| Base | `depth-anything/Depth-Anything-V2-Base-hf` | ~98M | 精度/速度均衡 |
| Large | `depth-anything/Depth-Anything-V2-Large-hf` | ~335M | 高精度 |
| Giant | — | ~1.3B | ❌ HF 无 `-hf` 版，不能用 transformers 加载（代码里有清晰报错） |

```python
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)  # 本地目录同样适用

# 注意（transformers 5.x 实测）：predicted_depth 是 [N, H, W]，无通道维；
# 且 processor 会把帧 resize 到 14 的倍数（如 384×512 → 392×518），需自行上采样回原尺寸。
```

## 二、依赖管理：uv（不使用 conda / requirements.txt）

| 事项 | 做法 |
|---|---|
| 依赖唯一声明处 | `pyproject.toml` 的 `[project.dependencies]` |
| 开发依赖 | `[dependency-groups].dev`（pytest），**不放进主依赖** |
| 版本锁定 | `uv.lock`（提交进 git 保证复现） |
| Python 版本 | `.python-version` 固定 3.12 |
| 安装 / 同步 | `uv sync --group dev`（自动创建 `.venv`） |
| 运行 | 一律 `uv run <命令>`，无需手动激活虚拟环境 |
| 新增依赖 | `uv add <pkg>`（自动更新 pyproject + lock） |
| 不使用的东西 | 无 `requirements.txt`、无 `environment.yml`、不用 conda |

## 三、完整目录结构树

```
SDM/
├── README.md                        # 项目说明、快速开始、命令示例
├── PROJECT_STRUCTURE.md             # 本文档：目录结构与设计方案
├── pyproject.toml                   # ★ 唯一依赖声明（uv 管理）
├── uv.lock                          # ★ 精确版本锁定（提交 git）
├── .python-version                  # Python 3.12
├── .venv/                           # uv 创建的环境（gitignore）
├── .gitignore                       # 忽略 .venv/、models/ 权重、data/inputs 与 data/outputs 内容
│
├── config/
│   └── default.yaml                 # 运行默认配置：规格、batch、stride、平滑、输出格式
│
├── src/
│   └── sdm/                         # ★ 核心 Python 包（uv src-layout）
│       ├── __init__.py              # 导出顶层 API（DepthEstimator / estimate_video_depth 等）
│       ├── model_loader.py          # ★ transformers 加载封装：规格名 → HF 仓库 ID / 本地路径，
│       │                            #   设备**默认 CPU**（--device cuda/mps 显式启用），本地权重优先
│       ├── estimator.py             # 单帧/批量推理 + resolve_backend（torch/tensorrt 选择与回退）
│       ├── tensorrt.py              # 可选 TensorRT 后端（服务器）：惰性导入→ONNX导出→engine构建→推理
│       ├── video_pipeline.py        # ★ 视频级管线（两趟）：抽帧推理落盘 → 统一归一化 → 写视频
│       ├── postprocess.py           # 稳健归一化、伪彩 colormap、16bit 无损 PNG 导出
│       ├── temporal.py              # 时序平滑（EMA）与闪烁度量（temporal_gradient）
│       └── utils/
│           ├── __init__.py
│           ├── video_io.py          # 视频元信息、生成器式抽帧、mp4 写入器
│           └── visualize.py         # 原图/深度 side-by-side 对比图
│
├── models/                          # ★ 模型权重目录（gitignore）
│   ├── Depth-Anything-V2-Small-hf/  # 存在则优先本地加载，跳过联网
│   │   ├── config.json              #   architectures: DepthAnythingForDepthEstimation
│   │   ├── preprocessor_config.json #   image processor 配置
│   │   └── model.safetensors        #   权重（-hf 仓库专供 transformers）
│   ├── Depth-Anything-V2-Base-hf/
│   └── Depth-Anything-V2-Large-hf/
│
├── scripts/
│   ├── download_models.py           # 下载指定/全部规格权重到 models/
│   ├── depth_video.py               # ★ 主入口：视频 → 深度视频（读取 config 默认值，CLI 可覆盖）
│   ├── build_trt_engine.py          # 服务器一次性预构建 TensorRT engine（ONNX → .engine）
│   └── batch_infer.py               # 批量处理目录，产出 batch_report.json
│
├── data/
│   ├── inputs/                      # 【输入区】待处理的原始视频（gitignore，保留目录）
│   └── outputs/                     # 【输出区】生成结果（gitignore，可随时清空重跑）
│       ├── <名>_depth.mp4           #   灰度深度视频（主输出）
│       ├── <名>_depth_color.mp4     #   可选：伪彩预览
│       ├── <名>_depth_frames/       #   可选：16bit 无损深度 PNG 序列
│       └── batch_report.json        #   批量任务的汇总报告
│
├── evaluation/
│   ├── metrics.py                   # AbsRel/SqRel/RMSE/log10/δ1-δ3 + scale-and-shift 对齐
│   └── eval.py                      # 评估入口：pred PNG 序列 vs GT
│
├── tests/                           # 全部可离线运行（管线用桩 estimator 替换真实模型）
│   ├── test_model_loader.py         # 规格映射、本地权重优先解析、设备选择
│   ├── test_postprocess.py          # 归一化边界/明暗反转/16bit 无损往返/退化范围
│   ├── test_temporal.py             # EMA 降低帧间闪烁、alpha 参数校验
│   └── test_pipeline.py             # ★ 端到端：合成视频 → 深度 mp4/伪彩/PNG16/帧率与分批
│
└── docs/
    └── usage.md                     # 规格与硬件建议、参数含义、输出说明、常见问题
```

## 四、核心数据流

```
输入视频 (mp4)
   │  utils/video_io.iter_frames：生成器抽帧（stride 可抽稀，保持原分辨率）
   ▼
帧批次 [B, H, W, 3] BGR
   │  estimator.DepthEstimator：processor → model(**inputs) → 双线性上采样回原尺寸
   │  pass 1：每批 float16 深度存临时 .npy，同时累计子采样统计量（不把整段堆内存）
   ▼
深度序列 [N, H, W]（relative depth / disparity，值越大越近）
   │  全序列 0.5~99.5 百分位统一范围（抑制闪烁的关键）→ 可选因果 EMA 时序平滑
   │  postprocess：normalize_to_uint8 / colorize / save_depth_png16
   ▼
深度视频（灰度 mp4 [+ 伪彩]）+ 可选 16bit PNG 序列 + 运行摘要 JSON   ──→ data/outputs/
```

### 输入/输出为何分目录

`data/inputs/`（只进）与 `data/outputs/`（只出）必须分开，哪怕同在 `data/` 下：

1. **避免自我循环**：`batch_infer.py` 扫描输入目录下所有 `.mp4`。若输出与输入同目录，下次批量会
   把自己的输出再当输入算一遍（得到“深度的深度”），结果错且难以察觉。
2. **可重生 vs 需保留**：`outputs/` 可整体删除重跑，`inputs/` 是不可再生的素材，混合后清理必需靠文件名猜。
3. **交付取用**：给 3D/剪辑流程打包时只需 `outputs/`（尤其 16bit 序列），不拖着原视频走。

## 五、关键设计决策

| 决策点 | 当前实现 | 理由 / 后续可选 |
|---|---|---|
| 模型加载 | `transformers.AutoModelForDepthEstimation` + **`-hf` 仓库** | 官方已支持，无需 remote code；不引入 diffusers |
| 依赖管理 | uv + `pyproject.toml` + `uv.lock` | 不用 conda、不用 requirements.txt |
| 包布局 | `src/sdm`（src-layout） | 避免仓库根被误当作包导入 |
| 数据布局 | `data/inputs`（入）+ `data/outputs`（出），根目录不设 `outputs/` | 同在 data/ 下但仍分离，避免批量任务自吞输出 |
| 权重位置 | 本地 `models/<规格>/` 优先，缺失则回退 HF 仓库 ID | 离线可跑、大模型可校验 |
| 深度尺度 | 相对深度（模型原生输出）+ 全序列统一归一化 | 逐帧各自 min-max 是闪烁主因；需米制换 `_metric` 权重 |
| 精度/设备 | **默认 CPU**；`--device cuda/mps` 显式启用，GPU 默认 fp16 | 共享服务器不自动占卡；要 GPU 明说 |
| 推理后端 | `--backend torch`（默认）/ `tensorrt`；不可用自动回退 torch+警告 | TRT 走 ONNX→engine，仅 `--device cuda`+Linux；`tensorrt` 不入共享 lock |
| 内存策略 | 两趟管线 + 临时批次文件流式读写 | 长视频不把 float 深度全放内存 |
| 输出格式 | 8bit 灰度 mp4（主）+ 伪彩 mp4 + 16bit PNG 序列 | 需 EXR 浮点可在此扩展 |
| 时序处理 | 统一范围 + 可选因果 EMA；`temporal_gradient` 量化闪烁 | 更强方案（光流引导一致性）待定 |
