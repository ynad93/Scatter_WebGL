"""Command-line interface for ScatterView."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import defaults as D


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="scatterview",
        description="ScatterView: N-body simulation visualization tool",
    )
    parser.add_argument(
        "datafile",
        type=str,
        help="Path to simulation data file (CSV or HDF5)",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default=None,
        choices=["csv", "hdf5"],
        help="Data format (auto-detected from extension if not specified)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for batch rendering (e.g., movie.mp4, frame.png). "
             "If not specified, launches interactive mode.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="tracking",
        choices=["manual", "tracking", "event-track"],
        help="Camera mode (default: tracking)",
    )
    parser.add_argument(
        "--trail-length",
        type=float,
        default=None,
        help=f"Trail length as fraction of total time range (default: {D.TRAIL_LENGTH_FRAC})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=D.VIDEO_DURATION,
        help=f"Video duration in seconds (default: {D.VIDEO_DURATION})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=D.VIDEO_FPS,
        help=f"Video frames per second (default: {D.VIDEO_FPS})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=D.WINDOW_WIDTH,
        help=f"Window/render width (default: {D.WINDOW_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=D.WINDOW_HEIGHT,
        help=f"Window/render height (default: {D.WINDOW_HEIGHT})",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Target particle ID for tracking camera modes",
    )
    parser.add_argument(
        "--detect-events",
        action="store_true",
        help="Run event detection and print results",
    )

    args = parser.parse_args(argv)

    # Load data
    from .core.data_loader import load

    print(f"Loading {args.datafile}...")
    data = load(args.datafile, fmt=args.format)
    print(
        f"Loaded {len(data.particle_ids)} particles, "
        f"{len(data.times)} timesteps "
        f"(t={data.times[0]:.4g} to {data.times[-1]:.4g})"
    )

    # Build interpolator
    from .core.interpolation import TrajectoryInterpolator

    print("Building spline interpolation...")
    interpolator = TrajectoryInterpolator(data)

    # Event detection
    if args.detect_events:
        from .core.event_detection import EventDetector

        print("Detecting events...")
        detector = EventDetector(data, interpolator)
        events = detector.detect_all()
        if events:
            print(f"\nFound {len(events)} events:")
            for e in events:
                print(f"  [{e.event_type}] t={e.time:.4g}: {e.description}")
        else:
            print("No events detected.")
        print()

    # Shared engine + camera setup
    def _build_engine_and_camera():
        from .rendering.engine import RenderEngine
        from .core.camera import CameraController, CameraMode

        engine = RenderEngine(
            data, interpolator,
            size=(args.width, args.height),
        )
        if args.trail_length is not None:
            engine.set_trail_length(args.trail_length)

        cam = CameraController(engine.view, masses=data.masses)
        mode_map = {
            "manual": CameraMode.MANUAL,
            "tracking": CameraMode.TARGET_COMOVING,
            "event-track": CameraMode.EVENT_TRACK,
        }
        cam.mode = mode_map.get(args.camera, CameraMode.MANUAL)
        if args.target is not None:
            cam.target_particle = args.target
        engine.set_camera_controller(cam)
        return engine, cam

    # Batch rendering
    if args.output:
        output = Path(args.output)
        print("Initializing rendering engine...")
        engine, cam = _build_engine_and_camera()

        if output.suffix.lower() == ".png":
            print(f"Saving screenshot to {output}...")
            engine.screenshot(output, size=(args.width, args.height))
        else:
            print(f"Rendering video to {output}...")
            engine.render_video(
                output,
                duration=args.duration,
                fps=args.fps,
                size=(args.width, args.height),
            )
        return

    # Interactive mode — QApplication must exist before VisPy canvas.
    # Must hold a reference so it isn't garbage collected.
    try:
        from PyQt6 import QtWidgets
        _app = QtWidgets.QApplication.instance()
        if _app is None:
            _app = QtWidgets.QApplication([])
    except ImportError:
        _app = None

    print("Launching interactive viewer...")
    engine, cam = _build_engine_and_camera()

    # Try to launch with GUI controls
    if _app is not None:
        try:
            from .gui.controls import ControlPanel

            panel = ControlPanel(engine, cam)
            panel.show()
            return
        except ImportError:
            pass

    # Fall back to standalone VisPy mode
    engine.show()


if __name__ == "__main__":
    main()
