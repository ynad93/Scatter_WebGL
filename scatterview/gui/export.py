"""Export utilities for screenshots and video rendering."""

from __future__ import annotations

from pathlib import Path

from .. import defaults as _D


def render_video_headless(
    engine,
    filepath: str | Path,
    duration: float = _D.VIDEO_DURATION,
    fps: int = _D.VIDEO_FPS,
    size: tuple[int, int] = (_D.VIDEO_WIDTH, _D.VIDEO_HEIGHT),
    codec: str = "libx264",
    codec_options: dict | None = None,
) -> None:
    """Render video without GUI (headless batch mode).

    Args:
        engine: RenderEngine instance.
        filepath: Output path (.mp4 or .gif).
        duration: Duration in seconds.
        fps: Frames per second.
        size: Render resolution (width, height).
        codec: ffmpeg codec name (e.g. "libx264", "h264_nvenc").
        codec_options: ffmpeg stream options dict.
    """
    engine.render_video(
        filepath, duration=duration, fps=fps, size=size,
        codec=codec, codec_options=codec_options,
    )


def screenshot_headless(
    engine,
    filepath: str | Path,
    sim_time: float | None = None,
    size: tuple[int, int] = (_D.VIDEO_WIDTH, _D.VIDEO_HEIGHT),
) -> None:
    """Take a screenshot without GUI.

    Args:
        engine: RenderEngine instance.
        filepath: Output path (.png).
        sim_time: Simulation time to capture. Uses current time if None.
        size: Render resolution.
    """
    if sim_time is not None:
        engine.sim_time = sim_time
    engine.screenshot(filepath, size=size)
