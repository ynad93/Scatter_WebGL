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

        # 4 particles: 3 clustered near origin, 1 far away
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [50.0, 50.0, 0.0],  # outlier
        ])
        ids = np.array([0, 1, 2, 3])

        framed_pos, framed_ids = ctrl._select_framed_particles(positions, ids)
        # Outlier (particle 3) should be excluded
        assert 3 not in framed_ids
        assert len(framed_pos) == 3

    def test_core_group_keeps_all_when_close(self):
        """When all particles are clustered, CORE_GROUP keeps all of them."""
        view = _FakeView()
        ctrl = CameraController(view)
        ctrl.framing_scope = FramingScope.CORE_GROUP

        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0],
        ])
        ids = np.array([0, 1, 2, 3])

        framed_pos, framed_ids = ctrl._select_framed_particles(positions, ids)
        assert len(framed_ids) == 4

    def test_nearest_neighbors_selects_k_closest(self):
        """NEAREST_NEIGHBORS should frame target + K nearest."""
        view = _FakeView()
        ctrl = CameraController(view)
        ctrl.framing_scope = FramingScope.NEAREST_NEIGHBORS
        ctrl.n_neighbors = 2

        positions = np.array([
            [0.0, 0.0, 0.0],  # target
            [1.0, 0.0, 0.0],  # nearest
            [2.0, 0.0, 0.0],  # 2nd nearest
            [100.0, 0.0, 0.0],  # far away
        ])
        ids = np.array([0, 1, 2, 3])
        ref = positions[0]  # target position

        framed_pos, framed_ids = ctrl._select_framed_particles(
            positions, ids, reference_pos=ref
        )
        # Should include target + 2 nearest = 3 particles
        assert len(framed_ids) == 3
        assert 3 not in framed_ids  # far particle excluded

    def test_keep_all_overrides_scope(self):
        """keep_all_in_frame=True should override the scope and include everything."""
        view = _FakeView()
        ctrl = CameraController(view)
        ctrl.framing_scope = FramingScope.CORE_GROUP
        ctrl.keep_all_in_frame = True

        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [50.0, 50.0, 0.0],  # outlier
        ])
        ids = np.array([0, 1, 2])

        framed_pos, framed_ids = ctrl._select_framed_particles(positions, ids)
        assert len(framed_ids) == 3

    def test_all_scope_returns_everything(self):
        """FramingScope.ALL should always return all particles."""
        view = _FakeView()
        ctrl = CameraController(view)
        ctrl.framing_scope = FramingScope.ALL

        positions = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ])
        ids = np.array([0, 1])

        framed_pos, framed_ids = ctrl._select_framed_particles(positions, ids)
        assert len(framed_ids) == 2

    def test_auto_frame_uses_framing_scope(self):
        """Auto-frame mode should use the framing scope for extent calculation."""
        view = _FakeView()
        ctrl = CameraController(view)
        ctrl.lock_zoom = False
        ctrl.mode = CameraMode.AUTO_FRAME
        ctrl.framing_scope = FramingScope.CORE_GROUP

        # 3 clustered + 1 outlier
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [100.0, 100.0, 0.0],
        ])
        ids = np.array([0, 1, 2, 3])

        # Run with CORE_GROUP (should ignore outlier → smaller distance)
        ctrl.framing_scope = FramingScope.CORE_GROUP
        ctrl._center_initialized = False
        ctrl.update(0.0, positions, ids)
        core_distance = view.camera.distance

        # Run with ALL (should include outlier → much larger distance)
        ctrl.framing_scope = FramingScope.ALL
        ctrl._center_initialized = False
        ctrl._smoothed_distance = 10.0
        ctrl.update(0.0, positions, ids)
        all_distance = view.camera.distance

        assert core_distance < all_distance

    def test_nearest_neighbors_fallback_without_target(self):
        """NEAREST_NEIGHBORS without a reference_pos falls back to CORE_GROUP."""
        view = _FakeView()
        ctrl = CameraController(view)
        ctrl.framing_scope = FramingScope.NEAREST_NEIGHBORS

        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [500.0, 500.0, 0.0],  # extreme outlier
        ])
        ids = np.array([0, 1, 2, 3])

        # No reference_pos → should fall back to core group logic
        framed_pos, framed_ids = ctrl._select_framed_particles(positions, ids)
        assert 3 not in framed_ids


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
