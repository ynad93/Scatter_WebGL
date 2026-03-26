"""Event detection for N-body simulations.

Pre-scans trajectories at load time to identify interesting events:
close encounters, mergers, and ejections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_loader import SimulationData
from .interpolation import TrajectoryInterpolator


@dataclass
class Event:
    """A detected simulation event."""

    time: float
    event_type: str  # "close_encounter", "merger", "ejection"
    particle_ids: list[int]
    position: np.ndarray  # 3D position of the event
    interest_score: float  # higher = more interesting
    description: str = ""


class EventDetector:
    """Detects interesting events in N-body simulation data.

    Pre-scans all particle trajectories to find:
    - Close encounters (minimum separation below threshold)
    - Mergers (particle disappearance coinciding with close approach)
    - Ejections (particle escaping from the system)

    Args:
        data: Loaded simulation data.
        interpolator: Trajectory interpolator for fine-grained position queries.
        close_encounter_threshold: Distance threshold for close encounters.
            If None, uses 10% of the initial system extent.
        ejection_threshold: Distance from COM threshold for ejections.
            If None, uses 5x the initial system extent.
        n_sample_points: Number of time samples for scanning trajectories.
    """

    def __init__(
        self,
        data: SimulationData,
        interpolator: TrajectoryInterpolator,
        close_encounter_threshold: float | None = None,
        ejection_threshold: float | None = None,
        n_sample_points: int = 500,
    ):
        self._data = data
        self._interp = interpolator
        self._n_samples = n_sample_points

        # Compute default thresholds from initial data extent
        initial_extent = self._compute_initial_extent()
        self._close_threshold = (
            close_encounter_threshold
            if close_encounter_threshold is not None
            else initial_extent * 0.1
        )
        self._ejection_threshold = (
            ejection_threshold
            if ejection_threshold is not None
            else initial_extent * 5.0
        )

        self._events: list[Event] = []

    def _compute_initial_extent(self) -> float:
        """Compute the initial spatial extent of the system."""
        t0 = self._data.times[0]
        positions, _, _ = self._interp.evaluate_batch(t0)
        if len(positions) < 2:
            return 1.0
        center = positions.mean(axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        return float(np.max(distances)) or 1.0

    def detect_all(self) -> list[Event]:
        """Run all detection algorithms and return sorted events."""
        self._events = []
        self._detect_close_encounters()
        self._detect_mergers()
        self._detect_ejections()

        # Sort by interest score (highest first)
        self._events.sort(key=lambda e: e.interest_score, reverse=True)
        return self._events

    @property
    def events(self) -> list[Event]:
        return self._events

    def _detect_close_encounters(self) -> None:
        """Find times when pairs of particles come close together."""
        times = self._data.times
        sample_times = np.linspace(times[0], times[-1], self._n_samples)
        pids = [int(p) for p in self._data.particle_ids]

        # For each pair of particles, track minimum separation
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                pid_a, pid_b = pids[i], pids[j]
                min_sep = float("inf")
                min_time = times[0]
                min_pos = np.zeros(3)

                for t in sample_times:
                    pos_a = self._interp._evaluate_particle(
                        self._interp._particle_splines.get(pid_a, []), t
                    )
                    pos_b = self._interp._evaluate_particle(
                        self._interp._particle_splines.get(pid_b, []), t
                    )

                    if pos_a is None or pos_b is None:
                        continue

                    sep = np.linalg.norm(pos_a - pos_b)
                    if sep < min_sep:
                        min_sep = sep
                        min_time = t
                        min_pos = (pos_a + pos_b) / 2

                if min_sep < self._close_threshold:
                    score = self._close_threshold / max(min_sep, 1e-30)
                    self._events.append(
                        Event(
                            time=min_time,
                            event_type="close_encounter",
                            particle_ids=[pid_a, pid_b],
                            position=min_pos,
                            interest_score=min(score, 100.0),
                            description=(
                                f"Close encounter: particles {pid_a} & {pid_b} "
                                f"at separation {min_sep:.4g}"
                            ),
                        )
                    )

    def _detect_mergers(self) -> None:
        """Detect mergers: particle disappearance near another particle."""
        pids = [int(p) for p in self._data.particle_ids]
        times = self._data.times

        for pid in pids:
            intervals = self._data.valid_intervals.get(pid, [])
            if not intervals:
                continue

            # Check if particle ends before the simulation ends
            last_valid_time = intervals[-1][1]
            if last_valid_time >= times[-1]:
                continue  # Particle survives to the end

            # Get position at disappearance time
            pos_at_end = self._interp._evaluate_particle(
                self._interp._particle_splines.get(pid, []), last_valid_time
            )
            if pos_at_end is None:
                continue

            # Find nearest other particle at that time
            nearest_pid = None
            nearest_sep = float("inf")
            nearest_pos = np.zeros(3)

            for other_pid in pids:
                if other_pid == pid:
                    continue
                pos_other = self._interp._evaluate_particle(
                    self._interp._particle_splines.get(other_pid, []),
                    last_valid_time,
                )
                if pos_other is None:
                    continue
                sep = np.linalg.norm(pos_at_end - pos_other)
                if sep < nearest_sep:
                    nearest_sep = sep
                    nearest_pid = other_pid
                    nearest_pos = (pos_at_end + pos_other) / 2

            if nearest_pid is not None and nearest_sep < self._close_threshold * 2:
                self._events.append(
                    Event(
                        time=last_valid_time,
                        event_type="merger",
                        particle_ids=[pid, nearest_pid],
                        position=nearest_pos,
                        interest_score=50.0,  # Mergers are always high interest
                        description=(
                            f"Merger: particle {pid} merged with {nearest_pid} "
                            f"at separation {nearest_sep:.4g}"
                        ),
                    )
                )

    def _detect_ejections(self) -> None:
        """Detect particles being ejected from the system."""
        times = self._data.times
        pids = [int(p) for p in self._data.particle_ids]

        # Sample at fewer points for ejection detection
        sample_times = np.linspace(times[0], times[-1], min(self._n_samples, 200))

        for pid in pids:
            intervals = self._data.valid_intervals.get(pid, [])
            if not intervals:
                continue

            # Track distance from COM over time
            prev_dist = None
            increasing_count = 0
            ejection_start_time = None

            for t in sample_times:
                # Get all positions for COM
                all_pos, all_ids, _ = self._interp.evaluate_batch(t)
                if len(all_pos) < 2:
                    continue

                com = all_pos.mean(axis=0)

                # Get this particle's position
                pos = self._interp._evaluate_particle(
                    self._interp._particle_splines.get(pid, []), t
                )
                if pos is None:
                    continue

                dist = np.linalg.norm(pos - com)

                if prev_dist is not None and dist > prev_dist:
                    increasing_count += 1
                    if increasing_count == 1:
                        ejection_start_time = t
                else:
                    increasing_count = 0
                    ejection_start_time = None

                # Ejection: distance exceeds threshold AND monotonically increasing
                # for at least 10 consecutive samples
                if dist > self._ejection_threshold and increasing_count >= 10:
                    self._events.append(
                        Event(
                            time=ejection_start_time or t,
                            event_type="ejection",
                            particle_ids=[pid],
                            position=pos,
                            interest_score=30.0,
                            description=(
                                f"Ejection: particle {pid} at distance "
                                f"{dist:.4g} from COM"
                            ),
                        )
                    )
                    break  # Only report ejection once per particle

                prev_dist = dist
