"""Comprehensive profiling of ScatterView startup and rendering.

Two phases:
  1. STARTUP — data loading, spline construction, trail precomputation
     (with multiprocessing worker-count sweep).
  2. RUNTIME — per-frame breakdown of spline evaluation, trail windowing,
     particle attrs, camera update, and GPU upload.

Usage:
    python profile_engine.py <datafile> [--frames N] [--warmup N] [--workers W1,W2,...]
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
# Timing collector (runtime phase)
# ---------------------------------------------------------------------------

class TimingCollector:
    """Collects per-frame, per-component timing in nanoseconds."""

    def __init__(self):
        self._totals: dict[str, list[int]] = defaultdict(list)
        self._accum: dict[str, int] = defaultdict(int)
        self._frame_times: list[int] = []

    def record(self, label: str, elapsed_ns: int) -> None:
        self._accum[label] += elapsed_ns

    def begin_frame(self) -> None:
        self._accum.clear()

    def end_frame(self, total_ns: int) -> None:
        self._frame_times.append(total_ns)
        for label, ns in self._accum.items():
            self._totals[label].append(ns)
        for label in list(self._totals.keys()):
            if len(self._totals[label]) < len(self._frame_times):
                self._totals[label].append(0)


def _ms(ns_array: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, median, p95, max) in milliseconds."""
    ms = ns_array / 1e6
    return float(ms.mean()), float(np.median(ms)), float(np.percentile(ms, 95)), float(ms.max())


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
# Startup profiling
# ---------------------------------------------------------------------------

def profile_startup(datafile: str, worker_counts: list[int],
                    trail_length: float | None = None,
                    target: str | None = None,
                    n_framed: int | None = None) -> tuple:
    """Profile every stage of startup, return (data, interp, engine, cam, timings)."""
    import multiprocessing

    timings = {}

    # Stage 1: Data loading
    print(f"Loading {datafile}...")
    t0 = time.perf_counter()
    from scatterview.core.data_loader import load
    data = load(datafile)
    t1 = time.perf_counter()
    timings["load"] = t1 - t0
    n_particles = len(data.particle_ids)
    n_times = len(data.times)
    print(f"  {n_particles} particles, {n_times} timesteps")
    print(f"  velocities: {data.velocities is not None}, masses: {data.masses is not None}")
    print(f"  => {timings['load']:.3f}s")

    # Stage 2: Spline construction (broken into sub-stages)
    from scatterview.core.interpolation import TrajectoryInterpolator

    print("Building splines...")
    t2 = time.perf_counter()
    interp = TrajectoryInterpolator.__new__(TrajectoryInterpolator)
    interp._data = data
    interp._particle_splines = {}
    interp._build_splines()
    t3 = time.perf_counter()
    interp._build_batch_eval()
    t4 = time.perf_counter()
    timings["build_splines"] = t3 - t2
    timings["build_batch_eval"] = t4 - t3
    timings["splines_total"] = t4 - t2
    print(f"  _build_splines:    {timings['build_splines']:.3f}s")
    print(f"  _build_batch_eval: {timings['build_batch_eval']:.3f}s")
    print(f"  => {timings['splines_total']:.3f}s total")

    # Measure per-particle trail eval cost
    pid = int(data.particle_ids[0])
    t_end = float(data.times[-1])
    full_time_range = t_end - float(data.times[0])
    eval_times = []
    for _ in range(10):
        tt0 = time.perf_counter()
        interp.evaluate_trail(pid, t_end, full_time_range)
        tt1 = time.perf_counter()
        eval_times.append(tt1 - tt0)
    per_particle_ms = np.median(eval_times) * 1000
    timings["per_particle_trail_ms"] = per_particle_ms

    # Stage 3: Trail precomputation — sweep worker counts
    n_cpu = multiprocessing.cpu_count()
    print(f"Trail precomputation ({n_particles} particles, {n_cpu} cores available)...")
    print(f"  per-particle trail eval: {per_particle_ms:.1f} ms")

    trail_timings = {}
    for nw in worker_counts:
        t5 = time.perf_counter()
        precomp = interp.precompute_all_trails(n_workers=nw)
        t6 = time.perf_counter()
        trail_timings[nw] = t6 - t5
        label = f"{nw} workers" if nw > 1 else "1 worker (serial)"
        print(f"  {label:<25}: {trail_timings[nw]:.3f}s")

    best_nw = min(trail_timings, key=trail_timings.get)
    timings["trails_best"] = trail_timings[best_nw]
    timings["trails_best_nw"] = best_nw
    timings["trail_timings"] = trail_timings
    timings["trail_points"] = len(precomp.times)
    timings["trail_mb"] = precomp.positions.nbytes / 1e6

    # Stage 4: Engine + camera construction (including star field, visuals)
    from scatterview.rendering.engine import RenderEngine
    from scatterview.core.camera import CameraController, CameraMode

    print("Building engine + camera...")
    # Rebuild interp properly for the engine
    interp_full = TrajectoryInterpolator(data)

    t7 = time.perf_counter()
    engine = RenderEngine(data, interp_full, size=(1280, 720))
    t8 = time.perf_counter()
    cam = CameraController(engine.view, masses=data.masses, particle_ids=data.particle_ids)
    cam.mode = CameraMode.TARGET_COMOVING
    engine.set_camera_controller(cam)
    t9 = time.perf_counter()
    timings["engine_init"] = t8 - t7
    timings["camera_init"] = t9 - t8
    print(f"  RenderEngine.__init__: {timings['engine_init']:.3f}s (includes trail precomp)")
    print(f"  CameraController:      {timings['camera_init']:.3f}s")

    # Apply benchmark overrides
    if trail_length is not None:
        engine.set_trail_length(trail_length)
        print(f"  [override] trail_length = {trail_length}")
    if target is not None:
        # Resolve string label to integer pid
        target_pid = None
        if data.id_labels is not None:
            for int_id, label in data.id_labels.items():
                if label == target:
                    target_pid = int_id
                    break
        if target_pid is None:
            try:
                target_pid = int(target)
            except ValueError:
                raise ValueError(f"Target '{target}' not found in particle labels: "
                                 f"{list(data.id_labels.values()) if data.id_labels else 'no labels'}")
        cam.target_particle = target_pid
        print(f"  [override] target = {target} (pid={target_pid})")
    if n_framed is not None:
        cam.n_framed = n_framed
        print(f"  [override] n_framed = {n_framed}")

    # Summary
    total = timings["load"] + timings["splines_total"] + timings["engine_init"] + timings["camera_init"]
    timings["startup_total"] = total

    print()
    print("=" * 72)
    print("STARTUP SUMMARY")
    print("=" * 72)
    print(f"{'Stage':<30} {'Time':>10} {'%':>7}")
    print("-" * 72)
    stages = [
        ("Data loading", timings["load"]),
        ("Spline construction", timings["splines_total"]),
        ("Engine init (trails+visuals)", timings["engine_init"]),
        ("Camera init", timings["camera_init"]),
    ]
    for name, t in stages:
        pct = t / total * 100
        print(f"{name:<30} {t:>9.3f}s {pct:>6.1f}%")
    print("-" * 72)
    print(f"{'TOTAL':<30} {total:>9.3f}s")
    print()

    if len(trail_timings) > 1:
        print("MULTIPROCESSING SCALING (trail precomputation only)")
        print("-" * 72)
        serial = trail_timings.get(1, trail_timings[min(trail_timings)])
        for nw in sorted(trail_timings):
            t = trail_timings[nw]
            speedup = serial / t if t > 0 else 0
            marker = " <-- best" if nw == best_nw else ""
            print(f"  {nw:>3} workers: {t:>8.3f}s  ({speedup:>5.1f}x vs serial){marker}")
        print()

    return data, interp_full, engine, cam, timings


# ---------------------------------------------------------------------------
# Runtime profiling
# ---------------------------------------------------------------------------

def profile_runtime(engine, cam, data, n_frames: int, n_warmup: int):
    """Profile per-frame rendering costs."""
    from scatterview import defaults as D

    n_particles = len(data.particle_ids)
    TIMER_HZ = 60.0
    anim_step = D.ANIM_SPEED / TIMER_HZ
    t_min = float(data.times[0])
    t_range = float(data.times[-1] - data.times[0])

    engine._playing = True

    # Warmup
    print(f"Warmup: {n_warmup} frames...")
    anim_time = 0.0
    for _ in range(n_warmup):
        engine._current_sim_time = t_min + anim_time * t_range
        engine._anim_time = anim_time
        engine._update_frame()
        anim_time += anim_step

    engine._canvas.render(size=(1280, 720))

    # Apply timing wrappers
    collector = TimingCollector()
    originals = {}
    originals["evaluate_batch"] = _wrap(engine._interp, "evaluate_batch", "evaluate_batch", collector)
    originals["_get_particle_attrs"] = _wrap(engine, "_get_particle_attrs", "get_particle_attrs", collector)
    if engine._particle_visual is not None:
        originals["particle_set_data"] = _wrap(engine._particle_visual, "set_data", "particle_set_data", collector)
    originals["_update_trails"] = _wrap(engine, "_update_trails", "update_trails", collector)
    if engine._trail_line is not None:
        originals["trail_line_set_data"] = _wrap(engine._trail_line, "set_data", "trail_line_set_data", collector)
    if engine._camera_controller is not None:
        originals["camera_update"] = _wrap(engine._camera_controller, "update", "camera_update", collector)
    originals["canvas_update"] = _wrap(engine._canvas, "update", "canvas_update", collector)
    originals["canvas_render"] = _wrap(engine._canvas, "render", "canvas_render", collector)
    if engine._stars_enabled and engine._star_visual is not None:
        originals["_update_stars"] = _wrap(engine, "_update_stars", "update_stars", collector)
    originals["_update_light"] = _wrap(engine, "_update_light_direction", "update_light", collector)

    # Timed run
    anim_coverage_start = anim_time
    print(f"Profiling: {n_frames} frames...")
    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(n_frames):
        sim_time = t_min + anim_time * t_range
        engine._current_sim_time = sim_time
        engine._anim_time = anim_time

        collector.begin_frame()
        frame_t0 = time.perf_counter_ns()

        engine._update_frame()
        engine._canvas.render(size=(1280, 720))

        frame_t1 = time.perf_counter_ns()
        collector.end_frame(frame_t1 - frame_t0)

        anim_time += anim_step
        if anim_time > 1.0:
            anim_time = 0.0

        if (i + 1) % 50 == 0:
            elapsed_ms = (frame_t1 - frame_t0) / 1e6
            print(f"  Frame {i + 1}/{n_frames}  ({elapsed_ms:.1f} ms)")

    profiler.disable()
    anim_coverage_end = anim_time

    # Report
    print()
    print("=" * 72)
    print(f"RUNTIME PROFILER — {n_frames} frames, {n_particles} particles")
    print(f"  Timer: {TIMER_HZ:.0f} Hz, anim_speed: {D.ANIM_SPEED}")
    print(f"  Sim coverage: {anim_coverage_start:.4f} -> {anim_coverage_end:.4f}"
          f" ({(anim_coverage_end - anim_coverage_start) * 100:.1f}%)")
    print("=" * 72)
    print()

    frame_ns = np.array(collector._frame_times)
    mean_f, med_f, p95_f, max_f = _ms(frame_ns)

    labels_order = [
        ("evaluate_batch",     "evaluate_batch"),
        ("get_particle_attrs", "get_particle_attrs"),
        ("particle_set_data",  "particle_set_data"),
        ("update_trails",      "update_trails"),
        ("trail_line_set_data","  trail_line_set_data"),
        ("camera_update",      "camera_update"),
        ("update_light",       "update_light"),
        ("update_stars",       "update_stars"),
        ("canvas_update",      "canvas_update"),
        ("canvas_render",      "canvas_render*"),
    ]

    header = f"{'Component':<24} {'Mean':>8} {'Median':>8} {'P95':>8} {'Max':>8} {'%':>7}"
    print("PER-FRAME TIMING")
    print("-" * 72)
    print(header)
    print("-" * 72)

    accounted_total = 0
    for key, display in labels_order:
        vals = collector._totals.get(key)
        if vals is None or len(vals) == 0:
            continue
        arr = np.array(vals[:n_frames])
        if len(arr) < n_frames:
            arr = np.pad(arr, (0, n_frames - len(arr)))
        mn, md, p95, mx = _ms(arr)
        pct = (arr.mean() / frame_ns.mean()) * 100 if frame_ns.mean() > 0 else 0
        accounted_total += arr.mean()
        print(f"{display:<24} {mn:7.2f}ms {md:7.2f}ms {p95:7.2f}ms {mx:7.2f}ms {pct:6.1f}%")

    print("-" * 72)
    print(f"{'TOTAL FRAME':<24} {mean_f:7.2f}ms {med_f:7.2f}ms {p95_f:7.2f}ms {max_f:7.2f}ms")

    unaccounted_ns = frame_ns.mean() - accounted_total
    unaccounted_pct = (unaccounted_ns / frame_ns.mean()) * 100 if frame_ns.mean() > 0 else 0
    print(f"{'Unaccounted':<24} {unaccounted_ns / 1e6:7.2f}ms {'':>8} {'':>8} {'':>8} {unaccounted_pct:6.1f}%")
    print()
    print("* canvas_render uses offscreen FBO + glReadPixels; real GUI rendering")
    print("  is cheaper (no pixel readback).")

    # Trail cache
    print()
    print("TRAIL CACHE")
    print("-" * 72)
    precomp = engine._precomp
    n_with_trails = sum(
        1 for i in range(n_particles)
        if precomp.offsets[i + 1] > precomp.offsets[i]
    )
    print(f"  Particles with trails: {n_with_trails}/{n_particles}")
    print(f"  Total precomputed points: {len(precomp.times):,}")
    print(f"  Packed positions: {precomp.positions.nbytes / 1e6:.1f} MB")

    # FPS estimate
    cpu_only_ms = mean_f - _ms(np.array(collector._totals.get("canvas_render", [0])))[0]
    print()
    print("FPS ESTIMATE")
    print("-" * 72)
    print(f"  CPU per frame (no readback): {cpu_only_ms:.2f} ms => {1000/cpu_only_ms:.0f} FPS ceiling")
    print(f"  With offscreen render:       {mean_f:.2f} ms => {1000/mean_f:.0f} FPS")

    # cProfile dump
    prof_file = "profile_runtime.prof"
    profiler.dump_stats(prof_file)
    print()
    print(f"cProfile stats saved to: {prof_file}")
    print()
    print("TOP 20 BY CUMULATIVE TIME")
    print("=" * 72)
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    print(stream.getvalue())

    print("TOP 20 BY TOTAL TIME")
    print("=" * 72)
    stream2 = StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.sort_stats("tottime")
    stats2.print_stats(20)
    print(stream2.getvalue())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile ScatterView startup and rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("datafile", help="Path to simulation data file")
    parser.add_argument("--frames", type=int, default=200,
                        help="Number of measured runtime frames (default: 200)")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Number of warmup frames (default: 50)")
    parser.add_argument("--workers", type=str, default=None,
                        help="Comma-separated worker counts for trail precomp sweep "
                             "(default: 1,2,4,8,cpu_count)")
    parser.add_argument("--startup-only", action="store_true",
                        help="Only profile startup, skip runtime")
    parser.add_argument("--runtime-only", action="store_true",
                        help="Only profile runtime (fastest startup)")
    parser.add_argument("--trail-length", type=float, default=None,
                        help="Trail length as fraction of total time range")
    parser.add_argument("--target", type=str, default=None,
                        help="Target particle ID or label (e.g. 'IMBH')")
    parser.add_argument("--n-framed", type=int, default=None,
                        help="Number of nearest neighbors for camera framing")
    args = parser.parse_args()

    import multiprocessing
    n_cpu = multiprocessing.cpu_count()

    if args.workers:
        worker_counts = [int(x.strip()) for x in args.workers.split(",")]
    elif args.runtime_only:
        worker_counts = [1]
    else:
        worker_counts = sorted(set([1, 2, 4, min(8, n_cpu), n_cpu]))

    # Phase 1: Startup
    print()
    print("=" * 72)
    print("PHASE 1: STARTUP PROFILING")
    print("=" * 72)
    print()
    data, interp, engine, cam, startup_timings = profile_startup(
        args.datafile, worker_counts,
        trail_length=args.trail_length,
        target=args.target,
        n_framed=args.n_framed,
    )

    if args.startup_only:
        return

    # Phase 2: Runtime
    print()
    print("=" * 72)
    print("PHASE 2: RUNTIME PROFILING")
    print("=" * 72)
    print()
    profile_runtime(engine, cam, data, args.frames, args.warmup)


if __name__ == "__main__":
    main()
