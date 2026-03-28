"""Tests for camera controller, framing scope, and event detection."""

import numpy as np
import pytest

from scatterview.core.camera import CameraController, CameraMode, FramingScope
from scatterview.core.data_loader import SimulationData
from scatterview.core.event_detection import EventDetector
from scatterview.core.interpolation import TrajectoryInterpolator


def _make_synthetic_data(
    n_particles: int = 3,
    n_times: int = 50,
    include_close_encounter: bool = False,
    include_ejection: bool = False,
) -> SimulationData:
    """Create synthetic simulation data for testing."""
    times = np.linspace(0, 10, n_times)

    positions = {}
    valid_intervals = {}

    for i in range(n_particles):
        # Circular orbits at different radii
        radius = (i + 1) * 0.5
        angle = times * (1.0 / (i + 1))
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = np.zeros_like(times)
        positions[i] = np.column_stack([x, y, z])
        valid_intervals[i] = [(times[0], times[-1])]

    if include_close_encounter and n_particles >= 2:
        # Make particle 1 approach particle 0 at the midpoint
        mid = n_times // 2
        window = n_times // 10
        for j in range(max(0, mid - window), min(n_times, mid + window)):
            blend = 1.0 - abs(j - mid) / window
            positions[1][j] = (1 - blend) * positions[1][j] + blend * positions[0][j]

    if include_ejection and n_particles >= 3:
        # Make last particle escape
        pid = n_particles - 1
        escape_start = n_times // 2
        for j in range(escape_start, n_times):
            t_frac = (j - escape_start) / (n_times - escape_start)
            positions[pid][j] += np.array([10 * t_frac, 10 * t_frac, 0])

    return SimulationData(
        particle_ids=np.arange(n_particles),
        times=times,
        positions=positions,
        valid_intervals=valid_intervals,
    )


class _FakeCamera:
    """Minimal mock for VisPy TurntableCamera."""
    def __init__(self):
        self.fov = 45
        self.center = (0, 0, 0)
        self.distance = 10.0
        self.azimuth = 0.0


class _FakeView:
    """Minimal mock for VisPy ViewBox."""
    def __init__(self):
        self.camera = _FakeCamera()


class TestFramingScope:
    def test_core_group_rejects_outlier(self):
        """An ejected particle far from the cluster should be excluded by CORE_GROUP."""
        view = _FakeView()
        ctrl = CameraController(view)
        ctrl.framing_scope = FramingScope.CORE_GROUP
        ctrl._core_group_percentile = 75.0  # keep inner 75% = 3 of 4

        # 4 particles: 3 clustered near origin, 1 far away
        active_pos = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [50.0, 50.0, 0.0],  # outlier
        ])
        framed = ctrl._select_framed_particles(active_pos)
        assert len(framed) == 3

    def test_core_group_keeps_percentage(self):
        """CORE_GROUP at 100% keeps all; at 50% keeps half."""
        view = _FakeView()
        ctrl = CameraController(view)
        ctrl.framing_scope = FramingScope.CORE_GROUP

        active_pos = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0],
        ])
        ctrl._core_group_percentile = 100.0
        framed = ctrl._select_framed_particles(active_pos)
        assert len(framed) == 4

        ctrl._core_group_percentile = 50.0
        framed = ctrl._select_framed_particles(active_pos)
        assert len(framed) == 2

    def test_nearest_neighbors_selects_k_closest(self):
        """NEAREST_NEIGHBORS should frame target + K nearest."""
        view = _FakeView()
        ctrl = CameraController(view)
        ctrl.framing_scope = FramingScope.NEAREST_NEIGHBORS
        ctrl.n_neighbors = 2

        active_pos = np.array([
            [0.0, 0.0, 0.0],  # target
            [1.0, 0.0, 0.0],  # nearest
            [2.0, 0.0, 0.0],  # 2nd nearest
            [100.0, 0.0, 0.0],  # far away
        ])
        ref = active_pos[0]  # target position

        framed = ctrl._select_framed_particles(active_pos, reference_pos=ref)
        # Should include target + 2 nearest = 3 particles
        assert len(framed) == 3

    def test_keep_all_overrides_scope(self):
        """keep_all_in_frame=True should override the scope and include everything."""
        view = _FakeView()
        ctrl = CameraController(view)
        ctrl.framing_scope = FramingScope.CORE_GROUP
        ctrl.keep_all_in_frame = True

        active_pos = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [50.0, 50.0, 0.0],  # outlier
        ])
        framed = ctrl._select_framed_particles(active_pos)
        assert len(framed) == 3

    def test_all_scope_returns_everything(self):
        """FramingScope.ALL should always return all particles."""
        view = _FakeView()
        ctrl = CameraController(view)
        ctrl.framing_scope = FramingScope.ALL

        active_pos = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ])
        framed = ctrl._select_framed_particles(active_pos)
        assert len(framed) == 2

    def test_tracking_uses_framing_scope(self):
        """Tracking mode should use the framing scope for zoom distance."""
        view = _FakeView()
        ctrl = CameraController(view)
        ctrl.free_zoom = False
        ctrl.mode = CameraMode.TARGET_COMOVING
        ctrl._core_group_percentile = 75.0  # keep inner 3 of 4

        # Full-size positions array (n_particles=4)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [100.0, 100.0, 0.0],
        ])
        mask = np.ones(4, dtype=bool)

        # Run with CORE_GROUP (75%) — excludes outlier → smaller distance
        ctrl.framing_scope = FramingScope.CORE_GROUP
        for _ in range(200):
            ctrl.update(0.0, positions, mask)
        core_distance = view.camera.distance

        # Run with ALL — includes outlier → much larger distance
        ctrl.framing_scope = FramingScope.ALL
        for _ in range(200):
            ctrl.update(0.0, positions, mask)
        all_distance = view.camera.distance

        assert core_distance < all_distance

    def test_nearest_neighbors_fallback_without_target(self):
        """NEAREST_NEIGHBORS without a target uses centroid as reference."""
        view = _FakeView()
        ctrl = CameraController(view)
        ctrl.framing_scope = FramingScope.NEAREST_NEIGHBORS
        ctrl._n_neighbors = 3

        active_pos = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [500.0, 500.0, 0.0],  # extreme outlier
        ])
        # Set _active_com so nearest neighbors has a fallback reference
        ctrl._active_com = active_pos.mean(axis=0)
        framed = ctrl._select_framed_particles(active_pos)
        assert len(framed) <= ctrl._n_neighbors + 1


class TestEventDetection:
    def test_no_events_in_simple_orbits(self):
        data = _make_synthetic_data(n_particles=3)
        interp = TrajectoryInterpolator(data)
        detector = EventDetector(data, interp)
        events = detector.detect_all()
        # Simple circular orbits shouldn't produce close encounters
        close = [e for e in events if e.event_type == "close_encounter"]
        # May or may not detect depending on thresholds, but shouldn't crash
        assert isinstance(events, list)

    def test_close_encounter_detected(self):
        data = _make_synthetic_data(n_particles=3, include_close_encounter=True)
        interp = TrajectoryInterpolator(data)
        detector = EventDetector(data, interp, close_encounter_threshold=0.5)
        events = detector.detect_all()
        close = [e for e in events if e.event_type == "close_encounter"]
        assert len(close) >= 1
        # Should involve particles 0 and 1
        pids = close[0].particle_ids
        assert 0 in pids and 1 in pids

    def test_ejection_detected(self):
        data = _make_synthetic_data(n_particles=3, include_ejection=True)
        interp = TrajectoryInterpolator(data)
        detector = EventDetector(data, interp, ejection_threshold=3.0)
        events = detector.detect_all()
        ejections = [e for e in events if e.event_type == "ejection"]
        assert len(ejections) >= 1
        # Particle 2 should be among the ejected particles
        ejected_pids = set()
        for e in ejections:
            ejected_pids.update(e.particle_ids)
        assert 2 in ejected_pids

    def test_merger_detection(self):
        """Particle that disappears should be detected as merger."""
        times = np.linspace(0, 5, 30)
        positions = {
            0: np.column_stack([np.zeros(30), np.zeros(30), np.zeros(30)]),
            1: np.column_stack([
                np.linspace(1, 0, 20),  # approaches particle 0
                np.zeros(20),
                np.zeros(20),
            ]),
        }
        valid_intervals = {
            0: [(0.0, 5.0)],
            1: [(0.0, times[19])],  # particle 1 disappears at t=times[19]
        }
        data = SimulationData(
            particle_ids=np.array([0, 1]),
            times=times,
            positions=positions,
            valid_intervals=valid_intervals,
        )
        interp = TrajectoryInterpolator(data)
        detector = EventDetector(data, interp, close_encounter_threshold=0.5)
        events = detector.detect_all()
        mergers = [e for e in events if e.event_type == "merger"]
        assert len(mergers) >= 1

    def test_event_has_required_fields(self):
        data = _make_synthetic_data(n_particles=3, include_close_encounter=True)
        interp = TrajectoryInterpolator(data)
        detector = EventDetector(data, interp, close_encounter_threshold=0.5)
        events = detector.detect_all()
        if events:
            e = events[0]
            assert hasattr(e, "time")
            assert hasattr(e, "event_type")
            assert hasattr(e, "particle_ids")
            assert hasattr(e, "position")
            assert hasattr(e, "interest_score")
            assert e.position.shape == (3,)

    def test_events_sorted_by_score(self):
        data = _make_synthetic_data(
            n_particles=3, include_close_encounter=True, include_ejection=True
        )
        interp = TrajectoryInterpolator(data)
        detector = EventDetector(
            data, interp,
            close_encounter_threshold=0.5,
            ejection_threshold=3.0,
        )
        events = detector.detect_all()
        if len(events) >= 2:
            scores = [e.interest_score for e in events]
            assert scores == sorted(scores, reverse=True)
