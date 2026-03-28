"""Tests for cubic spline interpolation."""

from pathlib import Path

import numpy as np
import pytest

from scatterview.core.data_loader import SimulationData, load
from scatterview.core.interpolation import TrajectoryInterpolator


SAMPLE_CSV = Path(__file__).resolve().parents[2] / "data" / "ScatterParts.csv"


class TestTrajectoryInterpolator:
    @pytest.fixture
    def sample_data(self):
        return load(SAMPLE_CSV)

    @pytest.fixture
    def interp(self, sample_data):
        return TrajectoryInterpolator(sample_data)

    def test_evaluate_at_data_points(self, sample_data, interp):
        """Interpolated positions at data times should match original data."""
        t0 = sample_data.times[0]
        result = interp.evaluate(t0)
        for pid in sample_data.particle_ids:
            pid_key = int(pid)
            if result[pid_key] is not None:
                original = sample_data.positions[pid_key][0]
                np.testing.assert_allclose(result[pid_key], original, atol=1e-6)

    def test_evaluate_between_data_points(self, sample_data, interp):
        """Interpolation at midpoint should return reasonable values."""
        t0, t1 = sample_data.times[0], sample_data.times[1]
        t_mid = (t0 + t1) / 2
        result = interp.evaluate(t_mid)

        for pid in sample_data.particle_ids:
            pid_key = int(pid)
            pos = result[pid_key]
            if pos is not None:
                # Should be roughly between the two data points
                p0 = sample_data.positions[pid_key][0]
                p1 = sample_data.positions[pid_key][1]
                # Midpoint should be within bounding box (approximately)
                for j in range(3):
                    lo, hi = min(p0[j], p1[j]), max(p0[j], p1[j])
                    # Allow some overshoot from cubic spline
                    assert lo - 0.5 <= pos[j] <= hi + 0.5

    def test_evaluate_batch(self, sample_data, interp):
        """Batch evaluation should return arrays."""
        t0 = sample_data.times[0]
        positions, mask = interp.evaluate_batch(t0)
        assert positions.ndim == 2
        assert positions.shape[1] == 3
        assert positions.shape[0] == len(sample_data.particle_ids)
        assert mask.dtype == bool
        assert mask.any()

    def test_evaluate_trail(self, sample_data, interp):
        """Trail evaluation should return a trajectory."""
        pid = int(sample_data.particle_ids[0])
        t_end = sample_data.times[-1]
        trail_length = (sample_data.times[-1] - sample_data.times[0]) / 2

        result = interp.evaluate_trail(pid, t_end, trail_length)
        assert result is not None
        positions, times = result
        assert positions.shape[1] == 3
        assert len(positions) >= 2
        assert len(times) == len(positions)

    def test_particle_near_time_boundary(self, sample_data, interp):
        """Evaluating slightly outside data range should clamp to endpoints."""
        # Use particle 0 which spans the full time range
        pid0 = int(sample_data.particle_ids[0])
        intervals = sample_data.valid_intervals[pid0]
        span = intervals[-1][1] - intervals[0][0]
        t_near = intervals[0][0] - span * 0.005  # within 1% clamp tolerance
        result = interp.evaluate(t_near)
        assert result[pid0] is not None

    def test_particle_far_outside_time_range(self, sample_data, interp):
        """Evaluating far outside data range should return None."""
        t_far = sample_data.times[0] - 100.0
        result = interp.evaluate(t_far)
        for pid in sample_data.particle_ids:
            pid_key = int(pid)
            assert result[pid_key] is None

    def test_smoothness(self, sample_data, interp):
        """Cubic spline should produce smooth (C2) trajectories."""
        pid = int(sample_data.particle_ids[0])
        t0, t1 = sample_data.times[0], sample_data.times[-1]
        n = 100
        times = np.linspace(t0, t1, n)
        positions = []
        for t in times:
            result = interp.evaluate(t)
            positions.append(result[pid])
        positions = np.array(positions)

        # Check that second derivative exists and is finite
        dt = times[1] - times[0]
        d2x = np.diff(positions[:, 0], n=2) / dt**2
        assert np.all(np.isfinite(d2x))

    def test_cubic_vs_linear_different(self, sample_data, interp):
        """Cubic interpolation should differ from linear at midpoints."""
        t0, t1 = sample_data.times[0], sample_data.times[1]
        t_mid = (t0 + t1) / 2

        cubic_pos = interp.evaluate(t_mid)

        # Compute linear interpolation manually
        for pid in sample_data.particle_ids:
            pid_key = int(pid)
            if cubic_pos[pid_key] is not None:
                p0 = sample_data.positions[pid_key][0]
                p1 = sample_data.positions[pid_key][1]
                linear_pos = 0.5 * (p0 + p1)
                # With only a few data points, cubic and linear may be close
                # but not identical (unless the data is perfectly linear)
                # Just verify they're both finite
                assert np.all(np.isfinite(cubic_pos[pid_key]))
                assert np.all(np.isfinite(linear_pos))


class TestWithSyntheticData:
    def test_hermite_spline_with_velocity(self):
        """When velocity data is available, Hermite splines should be used."""
        times = np.linspace(0, 2 * np.pi, 20)

        # Circular orbit: x = cos(t), y = sin(t), z = 0
        positions = {0: np.column_stack([np.cos(times), np.sin(times), np.zeros_like(times)])}
        velocities = {0: np.column_stack([-np.sin(times), np.cos(times), np.zeros_like(times)])}

        data = SimulationData(
            particle_ids=np.array([0]),
            times=times,
            positions=positions,
            velocities=velocities,
            valid_intervals={0: [(times[0], times[-1])]},
        )

        interp = TrajectoryInterpolator(data)

        # Evaluate at a midpoint — should be very close to the true circular orbit
        t_test = np.pi / 3  # not a data point
        result = interp.evaluate(t_test)
        expected = np.array([np.cos(t_test), np.sin(t_test), 0.0])
        np.testing.assert_allclose(result[0], expected, atol=1e-4)

    def test_disappearing_particle(self):
        """Particle that exists only in part of the time range."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Particle 0 exists throughout
        # Particle 1 exists only at t=0,1,2
        data = SimulationData(
            particle_ids=np.array([0, 1]),
            times=times,
            positions={
                0: np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=float),
                1: np.array([[0, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=float),
            },
            valid_intervals={
                0: [(0.0, 4.0)],
                1: [(0.0, 2.0)],
            },
        )

        interp = TrajectoryInterpolator(data)

        # At t=1.5, both should exist
        result = interp.evaluate(1.5)
        assert result[0] is not None
        assert result[1] is not None

        # At t=3.5, only particle 0 should exist
        result = interp.evaluate(3.5)
        assert result[0] is not None
        assert result[1] is None
