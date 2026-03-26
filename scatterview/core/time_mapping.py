"""Variable framerate mapping for non-uniform simulation timesteps.

Maps simulation time to animation time so that dense timestep regions
(where dynamics are fast-changing) get more frames, and sparse regions
(quiescent phases) get fewer frames.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d


class TimeMapping:
    """Bijective mapping between simulation time and animation time.

    Animation time s in [0, 1] progresses at constant rate (tied to
    wall-clock time). Simulation time t is computed via the inverse
    mapping: t = S_inv(s).

    The mapping adapts to the local timestep density of the input data,
    controlled by the gamma parameter.

    Args:
        times: Sorted unique simulation times (1D array).
        gamma: Adaptation exponent. 0 = uniform framerate, 1 = fully adaptive.
            Values in between provide a blend.
        smoothing_width: Gaussian smoothing width in number of timesteps.
            Prevents frame-rate jitter from individual noisy timesteps.
    """

    def __init__(
        self,
        times: np.ndarray,
        gamma: float = 0.0,
        smoothing_width: float = 5.0,
    ):
        self._times = np.asarray(times, dtype=float)
        self._gamma = gamma
        self._smoothing_width = smoothing_width

        if len(self._times) < 2:
            raise ValueError("Need at least 2 timesteps for time mapping.")

        self._forward_spline: CubicSpline | None = None
        self._inverse_spline: CubicSpline | None = None
        self._build()

    @property
    def t_min(self) -> float:
        return float(self._times[0])

    @property
    def t_max(self) -> float:
        return float(self._times[-1])

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        self._gamma = np.clip(value, 0.0, 1.0)
        self._build()

    def _build(self) -> None:
        """Compute the forward and inverse mappings."""
        times = self._times
        dt = np.diff(times)

        if np.allclose(dt, dt[0]) or self._gamma == 0.0:
            # Uniform timesteps or gamma=0: linear mapping
            self._forward_spline = None
            self._inverse_spline = None
            return

        # Local timestep density (at midpoints, then interpolate to knots)
        rho_mid = 1.0 / np.maximum(dt, 1e-30)

        # Interpolate to knot positions (average of neighbors)
        rho = np.empty(len(times))
        rho[0] = rho_mid[0]
        rho[-1] = rho_mid[-1]
        rho[1:-1] = 0.5 * (rho_mid[:-1] + rho_mid[1:])

        # Apply gamma blending: rho_eff = rho^gamma
        # At gamma=0 this is uniform (all ones), at gamma=1 fully adaptive
        rho_eff = rho ** self._gamma

        # Smooth to avoid frame-rate jitter
        if self._smoothing_width > 0 and len(rho_eff) > 3:
            rho_eff = gaussian_filter1d(rho_eff, sigma=self._smoothing_width)

        # Cumulative integral: S(t) = integral of rho_eff from t_0 to t
        S_values = np.zeros(len(times))
        S_values[1:] = cumulative_trapezoid(rho_eff, times)

        # Normalize to [0, 1]
        S_total = S_values[-1]
        if S_total > 0:
            S_values /= S_total

        # Forward mapping: t -> s
        self._forward_spline = CubicSpline(times, S_values)

        # Inverse mapping: s -> t (swap axes; S is monotonic)
        # Use PchipInterpolator to guarantee monotonicity (CubicSpline can overshoot)
        self._inverse_spline = PchipInterpolator(S_values, times)

    def sim_to_anim(self, t_sim: float | np.ndarray) -> float | np.ndarray:
        """Map simulation time to animation time [0, 1].

        Args:
            t_sim: Simulation time(s).

        Returns:
            Animation time(s) in [0, 1].
        """
        if self._forward_spline is None:
            # Linear mapping
            return (np.asarray(t_sim) - self.t_min) / (self.t_max - self.t_min)
        return self._forward_spline(t_sim)

    def anim_to_sim(self, s: float | np.ndarray) -> float | np.ndarray:
        """Map animation time [0, 1] to simulation time.

        Args:
            s: Animation time(s) in [0, 1].

        Returns:
            Simulation time(s).
        """
        if self._inverse_spline is None:
            # Linear mapping
            return self.t_min + np.asarray(s) * (self.t_max - self.t_min)
        return self._inverse_spline(s)

    def get_frame_times(self, n_frames: int) -> np.ndarray:
        """Get simulation times for a fixed number of uniformly-spaced animation frames.

        Args:
            n_frames: Number of frames.

        Returns:
            (n_frames,) array of simulation times.
        """
        s_values = np.linspace(0, 1, n_frames)
        return self.anim_to_sim(s_values)
