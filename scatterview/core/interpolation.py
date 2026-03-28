"""Cubic spline interpolation for particle trajectories.

Replaces the linear interpolation used in the original ScatterView
(THREE.LinearInterpolant) with scipy cubic splines for smoother,
more physically accurate trajectory rendering.
"""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline

from .. import defaults as D
from .data_loader import SimulationData


# ---- Module-level worker for multiprocessing (fork inherits parent memory) ----

_worker_interp = None  # TrajectoryInterpolator, set by pool initializer or before fork


def _init_trail_worker(interp) -> None:
    global _worker_interp
    _worker_interp = interp


def _eval_trail_worker(args: tuple) -> tuple[int, np.ndarray, np.ndarray] | None:
    """Evaluate one particle's trail at simulation timesteps + 1-pass refinement."""
    pid, t_end, trail_length = args
    result = _worker_interp.evaluate_trail(pid, t_end, trail_length)
    if result is None:
        return None
    pos, times = result
    return (pid, pos, times)


@dataclass
class PrecomputedTrails:
    """Packed pre-computed trail data for all particles.

    Trails are stored in contiguous arrays with a per-particle offset table
    so that each particle's trail segment can be extracted via slicing.
    """
    times: np.ndarray       # (total_pts,) float64 — packed trail times
    positions: np.ndarray   # (total_pts, 3) float32 — packed trail positions
    offsets: np.ndarray     # (n_particles + 1,) int64 — start index per particle
    pid_to_idx: dict[int, int]  # particle ID → index into offsets


class TrajectoryInterpolator:
    """Manages cubic spline interpolation for all particle trajectories.

    For each particle, builds a single 3D spline object (x, y, z evaluated
    together). If velocity data is available, uses CubicHermiteSpline
    (physically superior for orbital mechanics). Otherwise uses natural
    CubicSpline.

    Handles particles that appear/disappear by maintaining separate
    splines per continuous time segment.
    """

    def __init__(self, data: SimulationData):
        self._data = data
        self._particle_splines: dict[int, list[_SegmentSpline]] = {}
        self._build_splines()
        self._build_batch_eval()

    def _build_splines(self) -> None:
        """Pre-compute all splines at load time."""
        data = self._data
        has_vel = data.velocities is not None

        for pid in data.particle_ids:
            pid_key = int(pid)
            pos = data.positions[pid_key]  # (T_i, 3)
            vel = data.velocities[pid_key] if has_vel else None
            intervals = data.valid_intervals.get(pid_key, [])

            # Get the times for this particle's valid positions
            p_times = self._get_particle_times(pid_key)

            if len(p_times) == 0:
                continue
            if len(p_times) == 1:
                self._particle_splines[pid_key] = [
                    _SegmentSpline(
                        t_start=p_times[0], t_end=p_times[0],
                        spline=None, constant_pos=pos[0],
                    )
                ]
                continue

            # Build splines per continuous segment
            segments = self._split_into_segments(p_times, pos, vel, intervals)
            self._particle_splines[pid_key] = segments

    def _get_particle_times(self, pid_key: int) -> np.ndarray:
        """Get the time values corresponding to this particle's valid positions."""
        intervals = self._data.valid_intervals.get(pid_key, [])
        all_times = self._data.times

        # Collect times that fall within any valid interval
        mask = np.zeros(len(all_times), dtype=bool)
        for t_start, t_end in intervals:
            mask |= (all_times >= t_start) & (all_times <= t_end)

        return all_times[mask]

    def _split_into_segments(
        self,
        times: np.ndarray,
        pos: np.ndarray,
        vel: np.ndarray | None,
        intervals: list[tuple[float, float]],
    ) -> list[_SegmentSpline]:
        """Build separate splines for each continuous time segment."""
        segments = []

        for t_start, t_end in intervals:
            mask = (times >= t_start) & (times <= t_end)
            seg_times = times[mask]
            seg_pos = pos[: mask.sum()]  # positions are contiguous within valid data
            pos = pos[mask.sum():]  # advance past this segment

            if vel is not None:
                seg_vel = vel[: mask.sum()]
                vel = vel[mask.sum():]
            else:
                seg_vel = None

            if len(seg_times) == 0:
                continue
            if len(seg_times) == 1:
                segments.append(
                    _SegmentSpline(
                        t_start=seg_times[0], t_end=seg_times[0],
                        spline=None, constant_pos=seg_pos[0],
                    )
                )
                continue

            # Build single 3D spline (evaluates x, y, z together)
            if seg_vel is not None:
                spline = CubicHermiteSpline(seg_times, seg_pos, seg_vel)
            else:
                spline = CubicSpline(seg_times, seg_pos, bc_type="natural")

            segments.append(
                _SegmentSpline(
                    t_start=seg_times[0],
                    t_end=seg_times[-1],
                    spline=spline,
                    constant_pos=None,
                )
            )

        return segments

    def evaluate(self, time: float) -> dict[int, np.ndarray | None]:
        """Evaluate all particle positions at a given time.

        Args:
            time: Simulation time to evaluate at.

        Returns:
            Dict mapping particle ID -> (3,) position array, or None if
            the particle doesn't exist at this time.
        """
        result = {}
        for pid_key, segments in self._particle_splines.items():
            pos = self._evaluate_particle(segments, time)
            result[pid_key] = pos
        return result

    def _build_batch_eval(self) -> None:
        """Pre-extract spline coefficients for vectorized position evaluation.

        Scipy's CubicSpline stores polynomial coefficients internally as
        arrays of shape (4, M, 3) where:
          - 4 = cubic polynomial coefficients [c₃, c₂, c₁, c₀] (highest degree first)
          - M = number of intervals between breakpoints
          - 3 = spatial dimensions (x, y, z)

        For each interval k, the position at time t is evaluated via
        Horner's method: pos = ((c₃·dt + c₂)·dt + c₁)·dt + c₀
        where dt = t - breakpoint[k].

        Particles are split into two groups:
          - Single-segment particles (existed for the full simulation):
            can share breakpoints and be evaluated together with one
            searchsorted call + vectorized Horner evaluation.
          - Multi-segment particles (appeared/disappeared mid-simulation):
            evaluated individually per segment.
        """
        all_ids = self._data.particle_ids

        # --- Single-segment particles (vectorizable) ---
        single_seg_indices = []       # index in all_ids
        single_seg_pids = []          # particle ID
        single_seg_breakpoints = []   # spline knot times
        single_seg_coefficients = []  # polynomial coefficients (4, M, 3)
        single_seg_t_min = []         # earliest valid time
        single_seg_t_max = []         # latest valid time

        # --- Multi-segment particles (evaluated individually) ---
        multi_seg_indices = []
        multi_seg_pids = []

        for i, pid in enumerate(all_ids):
            pid_key = int(pid)
            segments = self._particle_splines.get(pid_key, [])
            if len(segments) == 1 and segments[0].spline:
                spline = segments[0].spline
                single_seg_indices.append(i)
                single_seg_pids.append(pid_key)
                single_seg_breakpoints.append(spline.x)
                single_seg_coefficients.append(spline.c)
                single_seg_t_min.append(spline.x[0])
                single_seg_t_max.append(spline.x[-1])
            else:
                multi_seg_indices.append(i)
                multi_seg_pids.append(pid_key)

        self._batch_idx = np.array(single_seg_indices, dtype=np.intp)
        self._batch_pids = np.array(single_seg_pids, dtype=int)
        self._batch_t_min = np.array(single_seg_t_min)
        self._batch_t_max = np.array(single_seg_t_max)
        self._slow_idx = np.array(multi_seg_indices, dtype=np.intp)
        self._slow_pids = np.array(multi_seg_pids, dtype=int)

        n_single = len(single_seg_breakpoints)
        if n_single == 0:
            self._shared_x = None
            self._batch_c_list = []
            return

        # Check if all single-segment particles share the same breakpoints.
        # This is the common case when all particles exist for the full
        # simulation and were sampled at the same times.  If so, a single
        # searchsorted call finds the interval for ALL particles at once.
        reference_breakpoints = single_seg_breakpoints[0]
        all_share_breakpoints = all(
            len(x) == len(reference_breakpoints) and np.array_equal(x, reference_breakpoints)
            for x in single_seg_breakpoints[1:]
        )

        if all_share_breakpoints:
            self._shared_x = reference_breakpoints
            self._batch_c_list = single_seg_coefficients
        else:
            self._shared_x = None
            self._batch_c_list = single_seg_coefficients
            self._batch_x_list = single_seg_breakpoints

    def evaluate_batch(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate all particle positions at a given time.

        For single-segment particles sharing breakpoints: one searchsorted
        call, then vectorized coefficient gather and Horner evaluation.
        Falls back to per-particle evaluation for multi-segment particles.

        Returns:
            positions: (n_particles, 3) array — NaN for inactive particles.
            mask: (n_particles,) bool — True where particle is active.
        """
        n = len(self._data.particle_ids)
        positions = np.full((n, 3), np.nan)
        mask = np.zeros(n, dtype=bool)

        # --- Fast path: single-segment particles ---
        n_batch = len(self._batch_t_min)
        if n_batch > 0:
            active = (time >= self._batch_t_min) & (time <= self._batch_t_max)

            if active.any():
                active_indices = np.where(active)[0]
                n_active = len(active_indices)
                dest_idx = self._batch_idx[active]

                if self._shared_x is not None:
                    breakpoints = self._shared_x
                    interval = max(0, min(
                        int(np.searchsorted(breakpoints, time, side='right') - 1),
                        len(breakpoints) - 2,
                    ))
                    time_offset = time - breakpoints[interval]

                    interval_coeffs = np.empty((n_active, 4, 3))
                    for i, j in enumerate(active_indices):
                        interval_coeffs[i] = self._batch_c_list[j][:, interval, :]

                    positions[dest_idx] = (
                        ((interval_coeffs[:, 0] * time_offset
                          + interval_coeffs[:, 1]) * time_offset
                         + interval_coeffs[:, 2]) * time_offset
                        + interval_coeffs[:, 3]
                    )
                else:
                    for i, j in enumerate(active_indices):
                        breakpoints = self._batch_x_list[j]
                        coeffs = self._batch_c_list[j]
                        interval = max(0, min(
                            int(np.searchsorted(breakpoints, time, side='right') - 1),
                            len(breakpoints) - 2,
                        ))
                        time_offset = time - breakpoints[interval]
                        positions[dest_idx[i]] = (
                            ((coeffs[0, interval] * time_offset
                              + coeffs[1, interval]) * time_offset
                             + coeffs[2, interval]) * time_offset
                            + coeffs[3, interval]
                        )

                mask[dest_idx] = True

        # --- Slow path: multi-segment particles ---
        for j in range(len(self._slow_pids)):
            pid_key = int(self._slow_pids[j])
            segments = self._particle_splines.get(pid_key, [])
            pos = self._evaluate_particle(segments, time)
            if pos is not None:
                idx = self._slow_idx[j]
                positions[idx] = pos
                mask[idx] = True

        return positions, mask

    def evaluate_trails_batch(
        self,
        pids: list[int],
        t_end: float,
        trail_length: float,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Evaluate trails for multiple particles at once.

        Uses each particle's simulation timesteps as the seed grid,
        then refines per-particle.

        Returns dict of pid -> (positions, times).
        """
        results = {}
        for pid in pids:
            result = self.evaluate_trail(pid, t_end, trail_length)
            if result is not None:
                results[pid] = result
        return results

    def precompute_all_trails(
        self,
        n_workers: int | None = None,
    ) -> PrecomputedTrails:
        """Pre-compute refined trails for every particle over the full simulation.

        Seeds each particle from the simulation's adaptive timesteps
        (which already concentrate at close encounters), then applies
        single-pass 3-degree angle refinement.  The result is stored in
        packed arrays so that render-time trail extraction is purely
        searchsorted + slice — zero spline evaluation per frame.

        Uses multiprocessing to evaluate particles in parallel (fork-based
        — workers inherit the interpolator via copy-on-write, no pickling).

        Args:
            n_workers: Number of worker processes. Defaults to CPU count.

        Returns:
            PrecomputedTrails with packed arrays and an offset table.
        """
        pids = [int(pid) for pid in self._data.particle_ids]
        t_start = self._data.times[0]
        t_end = self._data.times[-1]
        trail_length = t_end - t_start

        if n_workers is None:
            n_workers = min(multiprocessing.cpu_count(), 4)

        global _worker_interp
        _worker_interp = self  # set before fork so workers inherit it
        args = [(pid, float(t_end), trail_length) for pid in pids]

        if n_workers > 1 and len(pids) > 1:
            chunksize = max(1, len(pids) // n_workers)
            with multiprocessing.Pool(
                n_workers, initializer=_init_trail_worker, initargs=(self,),
            ) as pool:
                raw_results = pool.map(_eval_trail_worker, args, chunksize=chunksize)
        else:
            raw_results = [_eval_trail_worker(a) for a in args]

        # Pack results into contiguous arrays with offset table
        pid_to_idx: dict[int, int] = {}
        all_times: list[np.ndarray] = []
        all_pos: list[np.ndarray] = []
        offsets = [0]

        for i, pid in enumerate(pids):
            pid_to_idx[pid] = i
            result = raw_results[i]
            if result is not None:
                _, pos, times = result
                all_times.append(times)
                all_pos.append(pos.astype(np.float32))
                offsets.append(offsets[-1] + len(times))
            else:
                offsets.append(offsets[-1])

        if all_times:
            packed_times = np.concatenate(all_times)
            packed_pos = np.concatenate(all_pos)
        else:
            packed_times = np.empty(0, dtype=np.float64)
            packed_pos = np.empty((0, 3), dtype=np.float32)

        return PrecomputedTrails(
            times=packed_times,
            positions=packed_pos,
            offsets=np.array(offsets, dtype=np.int64),
            pid_to_idx=pid_to_idx,
        )

    def evaluate_trail(
        self, pid: int, t_end: float, trail_length: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Evaluate a particle's trajectory over a time range for trail rendering.

        Uses the simulation's own adaptive timesteps as the seed grid
        (the integrator already concentrates steps where dynamics are
        fast, e.g. close encounters).  Then applies iterative angle-based
        refinement to smooth any remaining sharp bends.

        Args:
            pid: Particle ID.
            t_end: End time (current time).
            trail_length: Duration of trail in time units.

        Returns:
            (positions, times) tuple or None if particle doesn't exist.
        """
        segments = self._particle_splines.get(pid, [])
        if not segments:
            return None

        t_start = max(t_end - trail_length, self._data.times[0])
        if t_start >= t_end:
            return None

        # Use the simulation's adaptive timesteps as the seed grid.
        p_times = self._get_particle_times(pid)
        mask = (p_times >= t_start) & (p_times <= t_end)
        times = p_times[mask]

        if len(times) < 2:
            return None

        positions = self._eval_times(segments, times)
        if positions is None:
            return None

        positions, times = self._refine_trail(segments, times, positions)
        return positions, times

    def evaluate_spline(self, pid: int, times: np.ndarray) -> np.ndarray | None:
        """Evaluate a single particle's spline at the given times."""
        segments = self._particle_splines.get(pid, [])
        if not segments:
            return None
        return self._eval_times(segments, times)

    def evaluate_single(self, pid: int, time: float) -> np.ndarray | None:
        """Evaluate one particle at one scalar time. Returns (3,) array or None."""
        return self._evaluate_particle(self._particle_splines.get(pid, []), time)

    def refine_trail(
        self, pid: int, times: np.ndarray, positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run angle-based refinement on a particle's trail segment."""
        segments = self._particle_splines.get(pid, [])
        return self._refine_trail(segments, times, positions)

    def _eval_times(
        self, segments: list, times: np.ndarray
    ) -> np.ndarray | None:
        """Evaluate spline at an array of times."""
        # Fast path: single segment covering the whole range
        if len(segments) == 1:
            seg = segments[0]
            if seg.spline and seg.t_start <= times[0] and seg.t_end >= times[-1]:
                return seg.spline(times)
            elif not seg.spline:
                return np.tile(seg.constant_pos, (len(times), 1))

        # Slow path: multiple segments or partial coverage.
        # Must return exactly len(times) rows so callers can pair
        # positions with times.  Fill gaps with NaN.
        positions = np.full((len(times), 3), np.nan)
        any_valid = False
        for i, t in enumerate(times):
            pos = self._evaluate_particle(segments, t)
            if pos is not None:
                positions[i] = pos
                any_valid = True
        return positions if any_valid else None

    def _refine_trail(
        self,
        segments: list,
        times: np.ndarray,
        positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single-pass angle-based densification of trail segments.

        For each pair of consecutive chords exceeding the angle threshold
        (~3 degrees), inserts ceil(angle)-1 evenly-spaced points into the
        adjacent segments.  Seeded from the simulation's adaptive timesteps,
        very few insertions are needed — the integrator already concentrates
        steps at close encounters and fast orbital phases.
        """
        if len(positions) < 3:
            return positions, times

        # Compute the chord (straight-line) vectors between consecutive points
        chord_vectors = np.diff(positions, axis=0)
        chord_lengths = np.linalg.norm(chord_vectors, axis=1)

        # Measure the deflection angle at each interior vertex:
        # the angle between consecutive chord vectors tells us how
        # sharply the trajectory bends at that point.
        chord_before = chord_vectors[:-1]
        chord_after = chord_vectors[1:]
        dot_products = np.einsum("ij,ij->i", chord_before, chord_after)
        norm_products = chord_lengths[:-1] * chord_lengths[1:]
        # Guard against zero-length chords (stationary particles)
        norm_products = np.maximum(norm_products, 1e-30)
        cos_deflection = np.clip(dot_products / norm_products, -1.0, 1.0)
        deflection_angles_deg = np.degrees(np.arccos(cos_deflection))

        # Assign each segment the max deflection angle from its two
        # endpoints.  Segment i connects vertex i to vertex i+1;
        # deflection angles are measured at interior vertices 1..N-2.
        n_segments = len(chord_lengths)
        segment_bend_angle = np.zeros(n_segments)
        segment_bend_angle[:-1] = np.maximum(segment_bend_angle[:-1], deflection_angles_deg)
        segment_bend_angle[1:] = np.maximum(segment_bend_angle[1:], deflection_angles_deg)

        # Insert ceil(angle)-1 new points into each segment that exceeds
        # the threshold.  This gives ~1 point per degree of bend, making
        # sharp slingshots visually smooth as polylines.
        points_to_insert = np.maximum(np.ceil(segment_bend_angle).astype(int) - 1, 0)
        points_to_insert[segment_bend_angle < D.REFINE_ANGLE_DEG] = 0

        if points_to_insert.sum() == 0:
            return positions, times

        new_times = []
        for i in range(n_segments):
            n_new = points_to_insert[i]
            if n_new > 0:
                t_a, t_b = times[i], times[i + 1]
                fractions = np.linspace(0, 1, n_new + 2)[1:-1]
                new_times.append(t_a + fractions * (t_b - t_a))

        if not new_times:
            return positions, times

        # Evaluate the spline at the new insertion times
        insertion_times = np.concatenate(new_times)
        insertion_positions = self._eval_times(segments, insertion_times)
        if insertion_positions is None:
            return positions, times

        # Merge original and inserted points, sorted by time
        merged_times = np.concatenate([times, insertion_times])
        merged_positions = np.concatenate([positions, insertion_positions], axis=0)
        sort_order = np.argsort(merged_times)
        return merged_positions[sort_order], merged_times[sort_order]

    def _evaluate_particle(
        self, segments: list[_SegmentSpline], time: float
    ) -> np.ndarray | None:
        """Evaluate a single particle's position at a given time."""
        for seg in segments:
            if seg.t_start <= time <= seg.t_end:
                return seg.spline(time) if seg.spline else seg.constant_pos

        if not segments:
            return None

        # Clamp near segment boundaries (within 1% of total span)
        total_span = segments[-1].t_end - segments[0].t_start
        clamp_tol = max(total_span * 0.01, 1e-10)

        if time < segments[0].t_start and (segments[0].t_start - time) <= clamp_tol:
            seg = segments[0]
            return seg.spline(seg.t_start) if seg.spline else seg.constant_pos

        if time > segments[-1].t_end and (time - segments[-1].t_end) <= clamp_tol:
            seg = segments[-1]
            return seg.spline(seg.t_end) if seg.spline else seg.constant_pos

        return None


class _SegmentSpline:
    """A spline covering one continuous time segment of a particle's trajectory."""

    __slots__ = ("t_start", "t_end", "spline", "constant_pos")

    def __init__(
        self,
        t_start: float,
        t_end: float,
        spline,
        constant_pos: np.ndarray | None,
    ):
        self.t_start = t_start
        self.t_end = t_end
        self.spline = spline  # single 3D CubicSpline/CubicHermiteSpline, or None
        self.constant_pos = constant_pos  # fallback for single-point segments
