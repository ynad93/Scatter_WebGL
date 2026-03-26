"""Cubic spline interpolation for particle trajectories.

Replaces the linear interpolation used in the original ScatterView
(THREE.LinearInterpolant) with scipy cubic splines for smoother,
more physically accurate trajectory rendering.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline

from .. import defaults as D
from .data_loader import SimulationData


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

            if len(p_times) < 2:
                # Not enough points for interpolation; store raw data
                self._particle_splines[pid_key] = [
                    _SegmentSpline(
                        t_start=p_times[0] if len(p_times) > 0 else 0.0,
                        t_end=p_times[0] if len(p_times) > 0 else 0.0,
                        spline=None,
                        constant_pos=pos[0] if len(pos) > 0 else np.zeros(3),
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

            if len(seg_times) < 2:
                segments.append(
                    _SegmentSpline(
                        t_start=seg_times[0] if len(seg_times) > 0 else t_start,
                        t_end=seg_times[0] if len(seg_times) > 0 else t_end,
                        spline=None,
                        constant_pos=seg_pos[0] if len(seg_pos) > 0 else np.zeros(3),
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
        """Pre-extract spline coefficients for vectorized evaluate_batch.

        For each particle with a single segment (the common case), caches
        the breakpoints and coefficients directly (avoiding scipy property
        access overhead per frame).
        """
        all_ids = self._data.particle_ids

        # Fast-path data: pre-extracted from scipy spline objects
        batch_idx = []
        batch_pids = []
        batch_x = []         # breakpoint arrays
        batch_c = []         # coefficient arrays (4, M, 3)
        batch_t_min = []
        batch_t_max = []

        # Slow-path particles (multi-segment or constant)
        slow_idx = []
        slow_pids = []

        for i, pid in enumerate(all_ids):
            pid_key = int(pid)
            segments = self._particle_splines.get(pid_key, [])
            if len(segments) == 1 and segments[0].spline:
                spline = segments[0].spline
                batch_idx.append(i)
                batch_pids.append(pid_key)
                batch_x.append(spline.x)
                batch_c.append(spline.c)
                batch_t_min.append(spline.x[0])
                batch_t_max.append(spline.x[-1])
            else:
                slow_idx.append(i)
                slow_pids.append(pid_key)

        self._batch_idx = np.array(batch_idx, dtype=np.intp)
        self._batch_pids = np.array(batch_pids, dtype=int)
        self._batch_x = batch_x
        self._batch_c = batch_c
        self._batch_t_min = np.array(batch_t_min)
        self._batch_t_max = np.array(batch_t_max)
        self._slow_idx = np.array(slow_idx, dtype=np.intp)
        self._slow_pids = np.array(slow_pids, dtype=int)

    def evaluate_batch(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate all particle positions at a given time.

        Uses direct polynomial evaluation with pre-cached coefficients
        for single-segment particles, falling back to per-particle
        evaluation for multi-segment particles.

        Returns (positions, ids, mask) for active particles.
        """
        all_ids = self._data.particle_ids
        n = len(all_ids)
        mask = np.zeros(n, dtype=bool)
        positions = np.empty((n, 3))
        active_ids = np.empty(n, dtype=int)
        count = 0

        # Fast path: direct Horner evaluation with cached coefficients
        for j in range(len(self._batch_x)):
            t_min = self._batch_t_min[j]
            t_max = self._batch_t_max[j]
            if time < t_min or time > t_max:
                continue
            x = self._batch_x[j]
            c = self._batch_c[j]
            k = min(int(np.searchsorted(x, time, side='right') - 1), len(x) - 2)
            dt = time - x[k]
            positions[count] = ((c[0, k] * dt + c[1, k]) * dt + c[2, k]) * dt + c[3, k]
            active_ids[count] = self._batch_pids[j]
            mask[self._batch_idx[j]] = True
            count += 1

        # Slow path: multi-segment particles
        for j in range(len(self._slow_pids)):
            pid_key = int(self._slow_pids[j])
            segments = self._particle_splines.get(pid_key, [])
            pos = self._evaluate_particle(segments, time)
            if pos is not None:
                positions[count] = pos
                active_ids[count] = pid_key
                mask[self._slow_idx[j]] = True
                count += 1

        return positions[:count], active_ids[:count], mask

    def evaluate_trails_batch(
        self,
        pids: list[int],
        t_end: float,
        trail_length: float,
        n_initial: int = D.TRAIL_INITIAL_POINTS,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Evaluate trails for multiple particles at once.

        Batch-evaluates the initial spline sample for all single-segment
        particles, then refines per-particle.

        Returns dict of pid -> (positions, times).
        """
        t_start = max(t_end - trail_length, self._data.times[0])
        if t_start >= t_end:
            return {}

        base_times = np.linspace(t_start, t_end, n_initial)
        results = {}

        # Partition particles by evaluation strategy
        batch_pids = []
        batch_splines = []
        slow_pids = []

        for pid in pids:
            segments = self._particle_splines.get(pid, [])
            if not segments:
                continue
            if (len(segments) == 1
                    and segments[0].spline is not None
                    and segments[0].t_start <= base_times[0]
                    and segments[0].t_end >= base_times[-1]):
                batch_pids.append(pid)
                batch_splines.append(segments[0].spline)
            else:
                slow_pids.append(pid)

        # Batch evaluation: one scipy call per particle (3D spline)
        if batch_pids:
            n_p = len(batch_pids)
            n_t = len(base_times)
            all_pos = np.empty((n_p, n_t, 3), dtype=np.float64)

            for i, spline in enumerate(batch_splines):
                all_pos[i] = spline(base_times)

            # Refine per-particle (curvature differs)
            for i, pid in enumerate(batch_pids):
                segments = self._particle_splines[pid]
                pos, t = self._refine_trail(segments, base_times.copy(), all_pos[i])
                results[pid] = (pos, t)

        for pid in slow_pids:
            segments = self._particle_splines.get(pid, [])
            positions = self._eval_times(segments, base_times)
            if positions is None:
                continue
            pos, t = self._refine_trail(segments, base_times.copy(), positions)
            results[pid] = (pos, t)

        return results

    def evaluate_trail(
        self, pid: int, t_end: float, trail_length: float,
        n_initial: int = D.TRAIL_INITIAL_POINTS,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Evaluate a particle's trajectory over a time range for trail rendering.

        1. Start with n_initial uniformly-spaced points across the window.
        2. Check consecutive chord angles via dot product.
        3. Where angle > 3 degrees, insert midpoints (doubling local density).
        4. Repeat until all angles are below threshold.

        Args:
            pid: Particle ID.
            t_end: End time (current time).
            trail_length: Duration of trail in time units.
            n_initial: Starting number of uniformly-spaced points.

        Returns:
            (positions, times) tuple or None if particle doesn't exist.
        """
        segments = self._particle_splines.get(pid, [])
        if not segments:
            return None

        t_start = max(t_end - trail_length, self._data.times[0])
        if t_start >= t_end:
            return None

        times = np.linspace(t_start, t_end, n_initial)
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

        # Slow path: multiple segments or partial coverage
        positions = []
        for t in times:
            pos = self._evaluate_particle(segments, t)
            if pos is not None:
                positions.append(pos)
        if not positions:
            return None
        return np.array(positions)

    def _refine_trail(
        self,
        segments: list,
        times: np.ndarray,
        positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Densify trail segments proportional to their deflection angle.

        For each pair of consecutive chords, measure the angle via dot
        product. Insert ceil(angle) evenly-spaced points into each
        segment adjacent to the bend (1 point per degree). Single pass.
        """
        if len(positions) < 3:
            return positions, times

        diffs = np.diff(positions, axis=0)
        lens = np.linalg.norm(diffs, axis=1)

        d1 = diffs[:-1]
        d2 = diffs[1:]
        dots = np.einsum("ij,ij->i", d1, d2)
        norms = lens[:-1] * lens[1:]
        norms = np.maximum(norms, 1e-30)
        cos_angle = np.clip(dots / norms, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(cos_angle))

        # For each segment, take the max angle from its two adjacent vertices.
        # Segment i sits between vertex i and vertex i+1;
        # angles are defined at interior vertices 1..N-2.
        n_segs = len(lens)
        seg_angle = np.zeros(n_segs)
        seg_angle[:-1] = np.maximum(seg_angle[:-1], angles_deg)
        seg_angle[1:] = np.maximum(seg_angle[1:], angles_deg)

        # Insert ceil(angle) - 1 points per segment (1 per degree)
        n_insert = np.maximum(np.ceil(seg_angle).astype(int) - 1, 0)

        if n_insert.sum() == 0:
            return positions, times

        # Build all insertion times
        all_insert_times = []
        for i in range(n_segs):
            k = n_insert[i]
            if k > 0:
                t_a, t_b = times[i], times[i + 1]
                fracs = np.linspace(0, 1, k + 2)[1:-1]
                all_insert_times.append(t_a + fracs * (t_b - t_a))

        if not all_insert_times:
            return positions, times

        insert_times = np.concatenate(all_insert_times)
        insert_pos = self._eval_times(segments, insert_times)
        if insert_pos is None:
            return positions, times

        all_times = np.concatenate([times, insert_times])
        all_pos = np.concatenate([positions, insert_pos], axis=0)
        order = np.argsort(all_times)
        return all_pos[order], all_times[order]

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
