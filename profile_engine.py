"""Comprehensive profiling of ScatterView rendering engine.

Monkey-patches timing wrappers around key methods, steps through
N frames headlessly, and reports per-component breakdown + cProfile stats.

Usage:
    python profile_engine.py [datafile] [--frames N] [--warmup N]
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from collections import defaultdict
from io import StringIO

import numpy as np


# ---------------------------------------------------------------------------
# Timing collector
# ---------------------------------------------------------------------------

class TimingCollector:
    """Collects per-frame, per-component timing in nanoseconds."""

    def __init__(self):
        # label -> list of per-frame totals (ns)
        self._totals: dict[str, list[int]] = defaultdict(list)
        # label -> accumulator for current frame
        self._accum: dict[str, int] = defaultdict(int)
        # per-frame trail cache stats: list of (total_pts, max_per_particle, n_dirty)
        self._trail_stats: list[tuple[int, int, int]] = []
        self._frame_times: list[int] = []

    def record(self, label: str, elapsed_ns: int) -> None:
        self._accum[label] += elapsed_ns

    def begin_frame(self) -> None:
        self._accum.clear()

    def end_frame(self, total_ns: int) -> None:
        self._frame_times.append(total_ns)
        for label, ns in self._accum.items():
            self._totals[label].append(ns)
        # Ensure all labels have entries for every frame (0 if not called)
        for label in list(self._totals.keys()):
            if len(self._totals[label]) < len(self._frame_times):
                self._totals[label].append(0)

    def add_trail_stats(self, total_pts: int, max_per_particle: int, n_dirty: int) -> None:
        self._trail_stats.append((total_pts, max_per_particle, n_dirty))


def _ms(ns_array: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, median, p95, max) in milliseconds."""
    ms = ns_array / 1e6
    return float(ms.mean()), float(np.median(ms)), float(np.percentile(ms, 95)), float(ms.max())


# ---------------------------------------------------------------------------
# Monkey-patch wrappers
# ---------------------------------------------------------------------------

def _wrap(obj, attr: str, label: str, collector: TimingCollector):
    """Replace obj.attr with a timing wrapper. Returns original for restore."""
    original = getattr(obj, attr)

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter_ns()
        result = original(*args, **kwargs)
        t1 = time.perf_counter_ns()
        collector.record(label, t1 - t0)
        return result

    setattr(obj, attr, wrapper)
    return original


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile ScatterView rendering engine")
    parser.add_argument("datafile", nargs="?", default="cluster_sim_data.csv",
                        help="Path to simulation data file")
    parser.add_argument("--frames", type=int, default=200, help="Number of measured frames")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup frames")
    args = parser.parse_args()

    N_FRAMES = args.frames
    N_WARMUP = args.warmup

    # ---- Phase 1: Setup ----
    print(f"Loading {args.datafile}...")
    from scatterview.core.data_loader import load
    data = load(args.datafile)
    print(f"  {len(data.particle_ids)} particles, {len(data.times)} timesteps")

    from scatterview.core.interpolation import TrajectoryInterpolator
    print("Building spline interpolation...")
    interpolator = TrajectoryInterpolator(data)

    from scatterview.core.time_mapping import TimeMapping
    time_mapping = TimeMapping(data.times, gamma=0.0)

    from scatterview.rendering.engine import RenderEngine
    from scatterview.core.camera import CameraController, CameraMode

    engine = RenderEngine(data, interpolator, time_mapping, size=(1280, 720))
    cam = CameraController(engine.view, masses=data.masses)
    cam.mode = CameraMode.AUTO_FRAME
    engine.set_camera_controller(cam)

    n_particles = len(data.particle_ids)
    frame_times = time_mapping.get_frame_times(N_WARMUP + N_FRAMES)

    # ---- Phase 2: Warmup ----
    print(f"Warmup: {N_WARMUP} frames...")
    for i in range(N_WARMUP):
        t = float(frame_times[i])
        engine._current_sim_time = t
        engine._anim_time = float(time_mapping.sim_to_anim(t))
        engine._update_frame()

    # Force a render to ensure GL context is warm
    engine._canvas.render(size=(1280, 720))

    # ---- Phase 3: Apply wrappers ----
    collector = TimingCollector()
    originals = {}

    originals["evaluate_batch"] = _wrap(engine._interp, "evaluate_batch", "evaluate_batch", collector)
    originals["_get_particle_attrs"] = _wrap(engine, "_get_particle_attrs", "get_particle_attrs", collector)

    if engine._particle_visual is not None:
        originals["particle_set_data"] = _wrap(engine._particle_visual, "set_data", "particle_set_data", collector)

    originals["_update_trails"] = _wrap(engine, "_update_trails", "update_trails", collector)
    originals["_trail_append"] = _wrap(engine, "_trail_append", "trail_append", collector)

    if engine._trail_line is not None:
        originals["trail_line_set_data"] = _wrap(engine._trail_line, "set_data", "trail_line_set_data", collector)

    if engine._camera_controller is not None:
        originals["camera_update"] = _wrap(engine._camera_controller, "update", "camera_update", collector)

    originals["canvas_update"] = _wrap(engine._canvas, "update", "canvas_update", collector)

    # ---- Phase 4: Timed run ----
    print(f"Profiling: {N_FRAMES} frames...")
    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(N_WARMUP, N_WARMUP + N_FRAMES):
        t = float(frame_times[i])
        engine._current_sim_time = t
        engine._anim_time = float(time_mapping.sim_to_anim(t))

        collector.begin_frame()
        frame_t0 = time.perf_counter_ns()

        engine._update_frame()
        engine._canvas.render(size=(1280, 720))

        frame_t1 = time.perf_counter_ns()

        # Trail cache stats
        total_pts = 0
        max_pp = 0
        for pos_list in engine._trail_pos.values():
            n = len(pos_list)
            total_pts += n
            if n > max_pp:
                max_pp = n
        n_dirty = len(engine._trail_dirty)
        collector.add_trail_stats(total_pts, max_pp, n_dirty)

        collector.end_frame(frame_t1 - frame_t0)

        if (i - N_WARMUP + 1) % 50 == 0:
            elapsed_ms = (frame_t1 - frame_t0) / 1e6
            print(f"  Frame {i - N_WARMUP + 1}/{N_FRAMES}  ({elapsed_ms:.1f} ms)")

    profiler.disable()

    # ---- Phase 5: Report ----
    print()
    print(f"ScatterView Profiler -- {N_FRAMES} frames, {n_particles} particles")
    print("=" * 72)
    print()

    frame_ns = np.array(collector._frame_times)
    mean_f, med_f, p95_f, max_f = _ms(frame_ns)

    # Component table
    labels_order = [
        ("evaluate_batch", "evaluate_batch", ""),
        ("get_particle_attrs", "get_particle_attrs", ""),
        ("particle_set_data", "particle_set_data", ""),
        ("update_trails", "update_trails", ""),
        ("trail_append", "  trail_append (sum)", ""),
        ("gpu_array_build", "  gpu_array_build", "derived"),
        ("trail_line_set_data", "  trail_line_set_data", ""),
        ("camera_update", "camera_update", ""),
        ("canvas_update", "canvas_update", ""),
    ]

    # Compute derived: gpu_array_build = update_trails - trail_append - trail_line_set_data
    ut = np.array(collector._totals.get("update_trails", [0] * N_FRAMES))
    ta = np.array(collector._totals.get("trail_append", [0] * N_FRAMES))
    tl = np.array(collector._totals.get("trail_line_set_data", [0] * N_FRAMES))
    gpu_build = np.maximum(ut - ta - tl, 0)
    collector._totals["gpu_array_build"] = gpu_build.tolist()

    header = f"{'Component':<24} {'Mean':>8} {'Median':>8} {'P95':>8} {'Max':>8} {'%':>7}"
    print("FRAME TIMING (excluding warmup)")
    print("-" * 72)
    print(header)
    print("-" * 72)

    for key, display, note in labels_order:
        vals = collector._totals.get(key)
        if vals is None or len(vals) == 0:
            continue
        arr = np.array(vals[:N_FRAMES])
        if len(arr) < N_FRAMES:
            arr = np.pad(arr, (0, N_FRAMES - len(arr)))
        mn, md, p95, mx = _ms(arr)
        pct = (arr.mean() / frame_ns.mean()) * 100 if frame_ns.mean() > 0 else 0
        print(f"{display:<24} {mn:7.2f}ms {md:7.2f}ms {p95:7.2f}ms {mx:7.2f}ms {pct:6.1f}%")

    print("-" * 72)
    print(f"{'TOTAL FRAME':<24} {mean_f:7.2f}ms {med_f:7.2f}ms {p95_f:7.2f}ms {max_f:7.2f}ms")

    # Unaccounted
    accounted_labels = ["evaluate_batch", "get_particle_attrs", "particle_set_data",
                        "update_trails", "camera_update", "canvas_update"]
    accounted = sum(
        np.array(collector._totals.get(l, [0] * N_FRAMES)).mean()
        for l in accounted_labels
    )
    unaccounted_ns = frame_ns.mean() - accounted
    unaccounted_pct = (unaccounted_ns / frame_ns.mean()) * 100 if frame_ns.mean() > 0 else 0
    print(f"{'Unaccounted':<24} {unaccounted_ns / 1e6:7.2f}ms {'':>8} {'':>8} {'':>8} {unaccounted_pct:6.1f}%")

    # Trail cache stats
    print()
    print("TRAIL CACHE STATS")
    print("-" * 72)
    if collector._trail_stats:
        stats = np.array(collector._trail_stats)
        total_pts = stats[:, 0]
        max_pp = stats[:, 1]
        n_dirty = stats[:, 2]
        n_cached = len(engine._trail_pos)
        mean_pp = total_pts.mean() / max(n_cached, 1)
        print(f"  Particles with cache: {n_cached}")
        print(f"  Total trail points (mean/max): {int(total_pts.mean())} / {int(total_pts.max())}")
        print(f"  Per particle (mean/max): {int(mean_pp)} / {int(max_pp.max())}")
        print(f"  Dirty rebuilds/frame (mean): {n_dirty.mean():.1f}")

    # cProfile stats
    prof_file = "profile_200frames.prof"
    profiler.dump_stats(prof_file)
    print()
    print(f"cProfile stats saved to: {prof_file}")
    print()
    print("TOP 30 BY CUMULATIVE TIME")
    print("=" * 72)

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(30)
    print(stream.getvalue())

    # Also print top 30 by total time
    print()
    print("TOP 30 BY TOTAL TIME")
    print("=" * 72)
    stream2 = StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.sort_stats("tottime")
    stats2.print_stats(30)
    print(stream2.getvalue())


if __name__ == "__main__":
    main()
