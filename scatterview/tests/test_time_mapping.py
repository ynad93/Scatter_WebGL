"""Tests for variable framerate time mapping."""

import numpy as np
import pytest

from scatterview.core.time_mapping import TimeMapping


class TestTimeMapping:
    def test_uniform_timesteps_linear(self):
        """Uniform timesteps should produce a linear mapping regardless of gamma."""
        times = np.linspace(0, 10, 100)
        tm = TimeMapping(times, gamma=1.0)

        # Forward: sim -> anim should be linear
        s = tm.sim_to_anim(5.0)
        np.testing.assert_allclose(s, 0.5, atol=0.02)

    def test_roundtrip(self):
        """sim_to_anim followed by anim_to_sim should be identity."""
        times = np.array([0, 0.1, 0.15, 0.16, 0.17, 0.5, 1.0, 2.0, 5.0, 10.0])
        tm = TimeMapping(times, gamma=1.0)

        for t in [0.0, 0.15, 1.0, 5.0, 10.0]:
            s = tm.sim_to_anim(t)
            t_back = tm.anim_to_sim(s)
            np.testing.assert_allclose(t_back, t, atol=1e-6)

    def test_dense_region_gets_more_frames(self):
        """Dense timestep regions should occupy more animation time."""
        # Times: 10 steps in [0,1], then 2 steps in [1,10]
        times = np.concatenate([np.linspace(0, 1, 10), np.array([5.0, 10.0])])
        tm = TimeMapping(times, gamma=1.0, smoothing_width=1.0)

        # Animation time at t=1 should be much more than 1/10
        s_at_1 = tm.sim_to_anim(1.0)
        assert s_at_1 > 0.3  # Should be significantly more than 0.1

    def test_gamma_zero_is_linear(self):
        """gamma=0 should produce uniform framerate."""
        times = np.array([0, 0.1, 0.15, 0.16, 1.0, 5.0, 10.0])
        tm = TimeMapping(times, gamma=0.0)

        s = tm.sim_to_anim(5.0)
        np.testing.assert_allclose(s, 0.5, atol=0.01)

    def test_endpoints(self):
        """s=0 maps to t_min, s=1 maps to t_max."""
        times = np.array([2.0, 3.0, 5.0, 8.0, 20.0])
        tm = TimeMapping(times, gamma=0.5)

        np.testing.assert_allclose(tm.anim_to_sim(0.0), 2.0, atol=1e-6)
        np.testing.assert_allclose(tm.anim_to_sim(1.0), 20.0, atol=1e-6)

    def test_monotonic(self):
        """The mapping should be monotonically increasing."""
        times = np.array([0, 0.01, 0.02, 0.5, 1.0, 5.0, 10.0])
        tm = TimeMapping(times, gamma=1.0, smoothing_width=1.0)

        s_values = np.linspace(0, 1, 100)
        t_values = tm.anim_to_sim(s_values)
        assert np.all(np.diff(t_values) > 0)

    def test_get_frame_times(self):
        """get_frame_times should return correct number of times."""
        times = np.linspace(0, 10, 50)
        tm = TimeMapping(times, gamma=0.5)

        frame_times = tm.get_frame_times(100)
        assert len(frame_times) == 100
        assert frame_times[0] == pytest.approx(times[0], abs=1e-6)
        assert frame_times[-1] == pytest.approx(times[-1], abs=1e-6)

    def test_gamma_setter(self):
        """Changing gamma should rebuild the mapping."""
        # Use highly non-uniform data with minimal smoothing for clear effect
        times = np.array([0, 0.01, 0.02, 0.03, 0.04, 5.0, 10.0])
        tm = TimeMapping(times, gamma=0.0, smoothing_width=0.5)

        s_linear = tm.sim_to_anim(5.0)

        tm.gamma = 1.0
        s_adaptive = tm.sim_to_anim(5.0)

        # With highly non-uniform data, gamma=1 should allocate much more
        # animation time to the dense [0, 0.04] region, so s(5.0) should
        # be significantly different from 0.5
        assert abs(s_linear - s_adaptive) > 0.05

    def test_too_few_timesteps(self):
        """Should raise with fewer than 2 timesteps."""
        with pytest.raises(ValueError, match="at least 2"):
            TimeMapping(np.array([1.0]))
