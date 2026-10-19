"""视频读写工具（OpenCV）。"""

from __future__ import annotations

from pathlib import Path

import cv2


def probe_video(path: str | Path) -> dict:
    """读取视频元信息：帧数、fps、宽高。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")
    meta = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return meta


def iter_frames(path: str | Path, *, max_frames: int | None = None, stride: int = 1):
    """按 stride 抽帧，产出 (索引, BGR ndarray)。避免一次性读入全部帧占内存。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")
    idx, emitted = 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, frame
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def open_video_writer(path: str | Path, size_wh: tuple[int, int], fps: float):
    """创建灰度/彩色 mp4 写入器（调用方逐帧 write，最后 release）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), tuple(size_wh)
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频：{path}")
    return writer
