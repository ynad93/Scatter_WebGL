"""Export utilities for screenshots and video rendering."""

from __future__ import annotations

from pathlib import Path


def render_video_headless(
    engine,
    filepath: str | Path,
    duration: float = 10.0,
    fps: int = 30,
    size: tuple[int, int] = (1920, 1080),
) -> None:
    """Render video without GUI (headless batch mode).

    Args:
        engine: RenderEngine instance.
        filepath: Output path (.mp4 or .gif).
        duration: Duration in seconds.
        fps: Frames per second.
        size: Render resolution (width, height).
    """
    engine.render_video(filepath, duration=duration, fps=fps, size=size)


def screenshot_headless(
    engine,
    filepath: str | Path,
    sim_time: float | None = None,
    size: tuple[int, int] = (1920, 1080),
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
