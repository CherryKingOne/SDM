"""视频级深度估计管线：抽帧 → 分批推理 → 全序列统一归一化 → 写深度视频。

采用两趟（two-pass）设计，避免把整段 float 深度堆进内存：
  pass 1：逐批推理，每批深度以 float16 存为临时 .npy；同时累计子采样统计量；
  pass 2：按批次顺序读回，做可选时序平滑（因果 EMA，流式）+ 全序列统一归一化，
          写出 mp4（及可选伪彩 mp4 / 16bit PNG 序列）。

关键点：归一化范围取自**整段视频**的稳健分位数，而不是逐帧各自 min-max——
后者会让每帧对比度基准不同，是深度视频闪烁的主要来源。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from sdm.estimator import DepthEstimator
from sdm.postprocess import colorize, normalize_to_uint8, robust_range, save_depth_png16
from sdm.utils.video_io import iter_frames, open_video_writer, probe_video

# 统计量子采样：每多少帧取一帧、空间上每多少像素取一个
_STAT_FRAME_STEP = 2
_STAT_SPATIAL_STEP = 8


def estimate_video_depth(
    input_video: str | Path,
    output_video: str | Path,
    *,
    model: str = "small",
    device: str | None = None,
    models_dir: str | Path | None = None,
    batch_size: int = 4,
    stride: int = 1,
    output_fps: float | None = None,
    max_frames: int | None = None,
    smooth_alpha: float | None = None,
    near_is_bright: bool = True,
    colormap: int | None = None,
    png16_dir: str | Path | None = None,
    allow_download: bool = True,
    show_progress: bool | None = None,
    logger=print,
) -> dict:
    """从输入视频估计深度并写出深度视频。

    Args:
        input_video / output_video: 输入、输出视频路径。
        model: 规格简称 small/base/large/giant。
        device: cuda/mps/cpu，默认自动选择。
        models_dir: 本地权重根目录，默认项目 models/。
        batch_size: 每次前向推理的帧数。
        stride: 抽帧步长（>1 输出低帧率深度视频，适合快速预览）。
        output_fps: 指定输出 mp4 的目标帧率（如 24/30/60）。内部对已算好的深度序列做
            最近邻时间重采样：目标<已算帧率则丢帧、目标>则复制帧（不新增真实动作）。
            默认 None 表示跟随源帧率（fps_in/stride）。注意这**不改变模型算了多少帧**，
            只改变输出视频的播放帧率；想减少算力请用 stride。
        max_frames: 限制处理帧数（调试用）。
        smooth_alpha: 给定时沿时间轴做因果 EMA 平滑（0<alpha<=1，如 0.5）。
        near_is_bright: True=近亮远暗；False=反转。
        colormap: 给定 cv2.COLORMAP_* 则额外输出伪彩视频。
        png16_dir: 给定目录则额外导出 16bit 无损深度帧序列。
        allow_download: 本地无权重时自动下载到 models/（默认 True）。
        show_progress: True/False 强制开关 tqdm 进度条；None（默认）= 仅当 stderr 是
            TTY 时显示，因此在管道/测试环境自动静默。
        logger: 接收一次性信息行的回调（默认 print）；进度条激活时改用 tqdm.write 以免打断条。

    Returns:
        运行摘要 dict（帧数、深度范围、fps、输出路径、耗时等）。
    """
    t0 = time.time()
    input_video = Path(input_video)
    if not input_video.exists():
        raise FileNotFoundError(input_video)
    if batch_size < 1:
        raise ValueError("batch_size 必须 >= 1")

    if show_progress is None:
        try:
            show_progress = sys.stderr.isatty()
        except Exception:  # noqa: BLE001  某些环境无 stderr/非 tty，一律按关闭处理
            show_progress = False
    _log = (lambda m: tqdm.write(str(m))) if show_progress else logger

    meta = probe_video(input_video)
    proc_fps = float(meta["fps"]) / max(stride, 1)  # 已计算深度帧的时间率
    expected = _expected_frames(meta["frame_count"], stride=stride, max_frames=max_frames)

    estimator = DepthEstimator(
        model, device=device, models_dir=models_dir, allow_download=allow_download
    )
    _log(f"[sdm] model={model} device={estimator.device} 预计帧数≈{expected}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="sdm_depth_"))
    frame_ids: list[int] = []          # 与临时批次文件顺序一致的原视频帧号
    stat_pool: list[np.ndarray] = []   # 子采样深度值，用于全局分位数
    n = 0
    try:
        # ---------- pass 1：推理，按批落盘 ----------
        buf: list[np.ndarray] = []
        buf_ids: list[int] = []
        pbar = tqdm(total=(expected or None), desc="pass1 推理", unit="帧",
                    disable=not show_progress, dynamic_ncols=True)
        try:
            for idx, frame in iter_frames(input_video, max_frames=max_frames, stride=stride):
                buf.append(frame)
                buf_ids.append(idx)
                if len(buf) >= batch_size:
                    cnt = len(buf)
                    n += _store_batch(tmp_dir, estimator, buf, buf_ids, frame_ids, stat_pool, n)
                    buf, buf_ids = [], []
                    pbar.update(cnt)
            if buf:
                cnt = len(buf)
                n += _store_batch(tmp_dir, estimator, buf, buf_ids, frame_ids, stat_pool, n)
                pbar.update(cnt)
        finally:
            pbar.close()

        if n == 0:
            raise RuntimeError(f"视频没有可处理的帧：{input_video}")

        # ---------- 全序列稳健范围 ----------
        sample = np.concatenate(stat_pool)
        vmin, vmax = robust_range(sample)
        _log(f"  深度范围（0.5~99.5 百分位）：[{vmin:.4f}, {vmax:.4f}]")

        # ---------- pass 2：归一化并写出 ----------
        out_video = Path(output_video)
        size_wh = (int(meta["width"]), int(meta["height"]))
        # 输出帧率：显式指定则重采样，否则跟随已算帧率
        out_fps = float(output_fps) if output_fps else proc_fps
        emit_counts, n_out = _compute_emit_counts(n, proc_fps, out_fps)
        _log(f"  输出帧率：{out_fps:g} fps（源已算 {proc_fps:g} fps，重采样后共 {n_out} 帧）")
        writer = open_video_writer(out_video, size_wh, out_fps)
        color_path = out_video.with_suffix("").as_posix() + "_color.mp4"
        color_writer = open_video_writer(color_path, size_wh, out_fps) if colormap is not None else None
        if png16_dir is not None:
            Path(png16_dir).mkdir(parents=True, exist_ok=True)

        prev: np.ndarray | None = None  # 因果 EMA 的上一帧状态
        written = 0
        pbar2 = tqdm(total=n, desc="pass2 写出", unit="帧",
                     disable=not show_progress, dynamic_ncols=True)
        try:
            for batch_file in sorted(tmp_dir.glob("b*.npy")):
                chunk = np.load(batch_file).astype(np.float32)
                offset = int(batch_file.stem[1:])
                for j in range(chunk.shape[0]):
                    d = chunk[j]
                    if smooth_alpha is not None:
                        d = d if prev is None else smooth_alpha * d + (1.0 - smooth_alpha) * prev
                        prev = d
                    gray = normalize_to_uint8(d, vmin, vmax, near_is_bright=near_is_bright)
                    # mp4 编码器需要 3 通道：灰度复制成 BGR（三通道相同，视觉上仍是灰度）。
                    # 直接写单通道数组会让 VideoWriter 产出无法回读的文件。
                    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    si = offset + j
                    # 目标帧率高于已算帧率时同一帧多写几次；低于时部分帧 emit=0 被跳过
                    for _ in range(emit_counts[si]):
                        writer.write(gray_bgr)
                        if color_writer is not None:
                            color_writer.write(colorize(gray, colormap))
                    if png16_dir is not None:
                        save_depth_png16(
                            str(Path(png16_dir) / f"depth_{frame_ids[si]:06d}.png"),
                            d, vmin, vmax,
                        )
                    written += 1
                    pbar2.update(1)
        finally:
            pbar2.close()
            writer.release()
            if color_writer is not None:
                color_writer.release()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "model": model,
        "device": str(estimator.device),
        "input": str(input_video),
        "output": str(output_video),
        "frames": n,
        "output_frames": n_out,
        "fps_in": meta["fps"],
        "fps_proc": proc_fps,
        "fps_out": out_fps,
        "depth_range": (vmin, vmax),
        "elapsed_sec": round(time.time() - t0, 2),
    }


def _store_batch(
    tmp_dir: Path,
    estimator: DepthEstimator,
    frames: list[np.ndarray],
    frame_ids: list[int],
    id_registry: list[int],
    stat_pool: list[np.ndarray],
    written: int,
) -> int:
    """推理一批、存为临时 float16 文件，并累计帧号与统计采样。返回本批帧数。"""
    batch = estimator(frames)  # [B, H, W] float32
    np.save(tmp_dir / f"b{written:08d}.npy", batch.astype(np.float16))
    id_registry.extend(frame_ids)
    for j in range(batch.shape[0]):
        if (written + j) % _STAT_FRAME_STEP == 0:
            stat_pool.append(batch[j, ::_STAT_SPATIAL_STEP, ::_STAT_SPATIAL_STEP].ravel())
    return batch.shape[0]


def _expected_frames(frame_count: int, *, stride: int, max_frames: int | None) -> int:
    total = max(int(frame_count) // max(stride, 1), 0)
    if max_frames is not None:
        total = min(total, max_frames)
    return total


def _compute_emit_counts(n_src: int, src_fps: float, dst_fps: float) -> tuple[list[int], int]:
    """为最近邻帧率重采样预计算：每个源帧应写入输出的次数。

    保持总时长不变：duration = n_src / src_fps = n_dst / dst_fps。
    对第 k 个输出帧，映射到最近的源帧 src = round((k+0.5)*src_fps/dst_fps - 0.5)，
    于是 dst>src 时某些源帧被多写（复制）、dst<src 时某些源帧写 0 次（丢帧）。

    Returns:
        (emit_counts, n_dst)，emit_counts 长度 = n_src，n_dst = 输出总帧数。
    """
    if n_src <= 0:
        return [], 0
    if dst_fps is None or abs(src_fps - dst_fps) < 1e-6:
        return [1] * n_src, n_src
    n_dst = max(1, round(n_src * dst_fps / src_fps))
    counts = [0] * n_src
    for k in range(n_dst):
        j = int(round((k + 0.5) * src_fps / dst_fps - 0.5))
        j = min(max(j, 0), n_src - 1)
        counts[j] += 1
    return counts, n_dst
