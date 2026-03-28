"""Tests for video and screenshot rendering."""

import os
from pathlib import Path

import numpy as np
import pytest

from scatterview.core.data_loader import SimulationData, load
from scatterview.core.interpolation import TrajectoryInterpolator
from scatterview.rendering.engine import RenderEngine


SAMPLE_CSV = Path(__file__).resolve().parents[2] / "data" / "ScatterParts.csv"


@pytest.fixture
def engine():
    data = load(SAMPLE_CSV)
    interp = TrajectoryInterpolator(data)
    return RenderEngine(data, interp, size=(320, 240))


class TestScreenshot:
    def test_screenshot_creates_file(self, engine, tmp_path):
        out = tmp_path / "shot.png"
        engine.screenshot(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_screenshot_custom_size(self, engine, tmp_path):
        out = tmp_path / "shot.png"
        engine.screenshot(out, size=(640, 480))
        assert out.exists()


class TestRenderVideo:
    def test_render_creates_file(self, engine, tmp_path):
        out = tmp_path / "test.mp4"
        engine.render_video(out, duration=1.0, fps=5, size=(320, 240))
        assert out.exists()
        assert out.stat().st_size > 1000  # not an empty/corrupt file

    def test_render_time_range(self, engine, tmp_path):
        """Rendering a sub-range should produce a valid file."""
        t_min = float(engine._t_min)
        t_max = float(engine._t_max)
        t_mid = (t_min + t_max) / 2

        out = tmp_path / "partial.mp4"
        engine.render_video(
            out, duration=1.0, fps=5, size=(320, 240),
            t_start=t_min, t_end=t_mid,
        )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_render_frame_count(self, engine, tmp_path):
        """Verify the correct number of frames are rendered via callback."""
        out = tmp_path / "count.mp4"
        reported = []

        def on_progress(current, total):
            reported.append((current, total))

        engine.render_video(
            out, duration=2.0, fps=10, size=(320, 240),
            progress_callback=on_progress,
        )
        # 2s * 10fps = 20 frames
        assert len(reported) == 20
        assert reported[-1] == (20, 20)
        assert all(t == 20 for _, t in reported)

    def test_render_progress_cancel(self, engine, tmp_path):
        """Cancellation via InterruptedError should not crash."""
        out = tmp_path / "cancel.mp4"
        rendered_frames = []

        def cancel_after_3(current, total):
            rendered_frames.append(current)
            if current >= 3:
                raise InterruptedError("cancelled")

        with pytest.raises(InterruptedError):
            engine.render_video(
                out, duration=2.0, fps=10, size=(320, 240),
                progress_callback=cancel_after_3,
            )
        assert len(rendered_frames) == 3
