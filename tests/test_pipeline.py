"""端到端管线测试：用桩 estimator 替换真实模型，验证管线编排本身（不需联网/GPU）。"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import sdm.video_pipeline as vp  # noqa: E402


class StubEstimator:
    """把帧的亮度图当深度输出，并记录每批帧数以验证分批逻辑。"""

    batch_sizes: list[int] = []

    def __init__(self, model="small", *, device=None, dtype=None, models_dir=None, allow_download=True):
        self.device = "cpu"

    def __call__(self, frames):
        StubEstimator.batch_sizes.append(len(frames))
        return np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])


@pytest.fixture()
def tiny_video(tmp_path):
    """生成 8 帧 64x48 的合成视频，亮度逐帧递增，保证深度非退化。"""
    path = tmp_path / "clip.mp4"
    size = (64, 48)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, size)
    assert writer.isOpened(), "本机 OpenCV 无法写 mp4，测试环境不支持"
    for i in range(8):
        frame = np.full((48, 64, 3), 10 + i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert path.exists()
    return path


@pytest.fixture(autouse=True)
def patch_estimator(monkeypatch):
    StubEstimator.batch_sizes = []
    monkeypatch.setattr(vp, "DepthEstimator", StubEstimator)
    return StubEstimator


def test_pipeline_writes_depth_video(tiny_video, tmp_path):
    out = tmp_path / "depth.mp4"
    summary = vp.estimate_video_depth(tiny_video, out, batch_size=3, logger=lambda *_: None)

    assert out.exists()
    assert summary["frames"] == 8
    assert summary["depth_range"][1] > summary["depth_range"][0]
    # 分批正确：3 + 3 + 2
    assert StubEstimator.batch_sizes == [3, 3, 2]

    cap = cv2.VideoCapture(str(out))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    assert len(frames) == 8
    assert frames[0].shape[:2] == (48, 64)
    # 亮度递增 → 深度值整体递增（近亮约定）
    assert frames[-1].mean() > frames[0].mean()


def test_pipeline_batch_size_one_and_larger_than_video(tiny_video, tmp_path):
    assert vp.estimate_video_depth(tiny_video, tmp_path / "a.mp4", batch_size=1, logger=lambda *_: None)["frames"] == 8
    assert vp.estimate_video_depth(tiny_video, tmp_path / "b.mp4", batch_size=64, logger=lambda *_: None)["frames"] == 8


def test_pipeline_stride_halves_frames_and_fps(tiny_video, tmp_path):
    summary = vp.estimate_video_depth(tiny_video, tmp_path / "s.mp4", stride=2, logger=lambda *_: None)
    assert summary["frames"] == 4
    assert summary["fps_out"] == pytest.approx(12.5)


def test_pipeline_exports_png16_and_color(tiny_video, tmp_path):
    png_dir = tmp_path / "png16"
    summary = vp.estimate_video_depth(
        tiny_video, tmp_path / "c.mp4", colormap=cv2.COLORMAP_INFERNO, png16_dir=png_dir,
        logger=lambda *_: None,
    )
    assert len(list(png_dir.glob("depth_*.png"))) == summary["frames"]
    assert (tmp_path / "c_color.mp4").exists()
    first = cv2.imread(str(sorted(png_dir.glob("*.png"))[0]), cv2.IMREAD_UNCHANGED)
    assert first.dtype == np.uint16


def test_pipeline_rejects_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        vp.estimate_video_depth(tmp_path / "nope.mp4", tmp_path / "out.mp4")


def test_pipeline_rejects_bad_batch_size(tiny_video, tmp_path):
    with pytest.raises(ValueError):
        vp.estimate_video_depth(tiny_video, tmp_path / "out.mp4", batch_size=0)
