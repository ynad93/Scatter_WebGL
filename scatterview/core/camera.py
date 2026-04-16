"""Automated camera controller for N-body visualization.

Provides composable camera modes: tracking, auto-center,
auto-rotate, event tracking, target tracking, and manual control.

Framing controls which particles drive the camera zoom via a single
count parameter (n_framed): the N closest particles to the reference
point are kept.  When N >= the active count, all particles are framed.

The camera is driven through a turntable-style wrapper exposing
``fov``, ``azimuth``, ``elevation``, ``center``, ``distance``, and a
``view_changed()`` method.  The wrapper is either VisPy's stock
``TurntableCamera`` (historic) or the pygfx-backed wrapper defined in
``scatterview.rendering.engine``; either one works because this module
only touches that public surface.
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

from .. import defaults as D


class CameraMode(Enum):
    MANUAL = auto()
    TARGET_REST_FRAME = auto()
    TARGET_COMOVING = auto()  # unified tracking: deadzone on target or COM


class CameraController:
    """Manages automated camera behavior.

    Args:
        view: VisPy ViewBox whose camera to control.
        masses: Dict of particle ID -> mass array for mass-weighted COM, or None.
        events: List of detected Event objects for event-tracking mode, or None.
        particle_ids: Sorted array of integer particle IDs for pid-to-index lookup.
    """

    def __init__(
        self,
        view,
        masses: dict[int, np.ndarray] | None = None,
        events=None,
        particle_ids: np.ndarray | None = None,
    ):
        self._camera = view.camera
        self._events = events or []

        # pid→index lookup array (same pattern as engine._pid_lookup)
        if particle_ids is not None:
            max_pid = int(np.max(particle_ids)) + 1
            self._pid_lookup = np.full(max_pid, -1, dtype=np.int32)
            for i, pid in enumerate(particle_ids):
                self._pid_lookup[int(pid)] = i
        else:
            self._pid_lookup = np.empty(0, dtype=np.int32)

        # Flat mass array: mass_array[i] = latest mass of particle i
        if masses is not None and particle_ids is not None:
            self._mass_array = np.ones(len(particle_ids), dtype=np.float64)
            for pid_key, mass_arr in masses.items():
                idx = self._pid_lookup[pid_key]
                if idx >= 0 and len(mass_arr) > 0:
                    self._mass_array[idx] = mass_arr[-1]
        else:
            self._mass_array = None

        # Camera mode and rotation
        self._mode = CameraMode.TARGET_COMOVING
        self._auto_rotate_enabled = False
        self._rotation_speed = D.ROTATION_SPEED
        self._azimuth_offset = 0.0

        # Framing: which particles drive the camera.
        # _n_framed = number of closest particles to keep; when >= active
        # count, all particles are framed (the default).
        n = len(particle_ids) if particle_ids is not None else 0
        self._n_framed = n
        self._keep_all_in_frame = False

        # Framing fraction: what fraction of the screen's vertical
        # half-extent the framed group is allowed to occupy.  0.75 means
        # the farthest framed particle appears at 75% of the way from
        # screen center to the top/bottom edge.
        self._framing_fraction = D.FRAMING_FRACTION
        self._cache_fov_trig()

        # Zoom memory: rolling-average window for framing radius.
        # The camera distance tracks the mean radius over the last N
        # frames, filtering out orbital oscillations while responding
        # smoothly to genuine scale changes.  A running sum avoids
        # recomputing the full mean each frame.
        self._zoom_memory_frames = D.ZOOM_MEMORY_FRAMES
        self._radius_ring = np.ones(D.ZOOM_MEMORY_FRAMES, dtype=np.float64)
        self._radius_ring_sum = float(D.ZOOM_MEMORY_FRAMES)
        self._radius_ring_idx = 0

        # Center mode: when tracking a target with neighbors, center on
        # the target particle directly, or on the centroid/CoM of the group.
        self._use_group_center = False   # False = pin on target particle
        self._mass_weighted_center = False  # True = CoM, False = centroid
        self._framed_active_idx = np.empty(0, dtype=np.intp)

        # Event tracking overlay: blends toward upcoming events on top of
        # the active base mode.  Ramp-in and ramp-out durations control
        # how long before/after an event the blend is active.
        self._event_tracking_enabled = False
        self._event_ramp_in = 2.0   # seconds before event to start blending
        self._event_ramp_out = 1.0  # seconds after event to finish blending
        self._event_zoom_tighten = 0.7  # max zoom tightening factor at peak

        # Target tracking
        self._target_pid: int | None = None
        self._target_needs_acquisition = False

        # Panning deadzone: camera center holds still until the tracked
        # point drifts outside this fraction of the visible radius.
        # Only affects center panning, not zoom.
        self._pan_deadzone_fraction = D.PAN_DEADZONE_FRACTION

        # Free zoom: when True, user controls distance via scroll wheel;
        # center tracking still works for target modes
        self._free_zoom = False
        self._free_zoom_callbacks: list = []

        # Camera state — initialized from VisPy camera's current position
        cam_center = self._camera.center
        self._smoothed_center = np.array(cam_center, dtype=np.float64) if cam_center else np.zeros(3)
        self._smoothed_framing_radius = 1.0  # overwritten by set_camera_controller
        self._active_com = self._smoothed_center.copy()

    # --- Properties ---

    @property
    def mode(self) -> CameraMode:
        return self._mode

    @mode.setter
    def mode(self, value: CameraMode) -> None:
        self._mode = value

    @property
    def auto_rotate(self) -> bool:
        return self._auto_rotate_enabled

    @auto_rotate.setter
    def auto_rotate(self, value: bool) -> None:
        self._auto_rotate_enabled = value

    @property
    def rotation_speed(self) -> float:
        return self._rotation_speed

    @rotation_speed.setter
    def rotation_speed(self, value: float) -> None:
        self._rotation_speed = value

    @property
    def target_particle(self) -> int | None:
        return self._target_pid

    @target_particle.setter
    def target_particle(self, pid: int | None) -> None:
        self._target_pid = pid
        # Flag for the next update to jump to the target group
        # instead of slowly chasing from the current camera position
        self._target_needs_acquisition = pid is not None

    @property
    def n_framed(self) -> int:
        return self._n_framed

    @n_framed.setter
    def n_framed(self, value: int) -> None:
        self._n_framed = max(1, value)
        self._target_needs_acquisition = True

    @property
    def keep_all_in_frame(self) -> bool:
        return self._keep_all_in_frame

    @keep_all_in_frame.setter
    def keep_all_in_frame(self, value: bool) -> None:
        self._keep_all_in_frame = value
        self._target_needs_acquisition = True

    @property
    def free_zoom(self) -> bool:
        return self._free_zoom

    @free_zoom.setter
    def free_zoom(self, value: bool) -> None:
        if value == self._free_zoom:
            return
        self._free_zoom = value
        for cb in self._free_zoom_callbacks:
            cb(value)

    @property
    def use_group_center(self) -> bool:
        return self._use_group_center

    @use_group_center.setter
    def use_group_center(self, value: bool) -> None:
        self._use_group_center = value
        self._target_needs_acquisition = True

    @property
    def mass_weighted_center(self) -> bool:
        return self._mass_weighted_center

    @mass_weighted_center.setter
    def mass_weighted_center(self, value: bool) -> None:
        self._mass_weighted_center = value
        self._target_needs_acquisition = True

    @property
    def event_tracking(self) -> bool:
        return self._event_tracking_enabled

    @event_tracking.setter
    def event_tracking(self, value: bool) -> None:
        self._event_tracking_enabled = value

    # --- Core update ---

    def initialize_framing(
        self, positions: np.ndarray, mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Compute initial framing and set internal state.

        Args:
            positions: (n_particles, 3) full-size position array (NaN for inactive).
            mask: (n_particles,) bool — True where particle is active.

        Returns:
            (center, camera_distance) tuple for the caller to apply
            to the VisPy camera.
        """
        active_pos = positions[mask]
        target_pos = self._find_target(positions)
        framed = self._select_framed_particles(active_pos, reference_pos=target_pos)
        center = framed.mean(axis=0)
        radius = self._compute_framing_radius(framed, center)
        self._smoothed_center = center.copy()
        self._smoothed_framing_radius = radius
        # Seed the ring buffer so the camera starts at the correct zoom.
        self._radius_ring[:] = radius
        self._radius_ring_sum = radius * len(self._radius_ring)
        return center, self._ideal_distance(radius)

    def update(
        self,
        sim_time: float,
        positions: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        """Update camera based on current mode and particle positions.

        Called each frame by the RenderEngine.

        Args:
            sim_time: Current simulation time.
            positions: (n_particles, 3) full-size array (NaN for inactive).
            mask: (n_particles,) bool — True where particle is active.
        """
        if self._mode == CameraMode.MANUAL and not self._auto_rotate_enabled:
            return

        if not mask.any():
            return

        active_pos = positions[mask]
        self._active_com = self._compute_center_of_mass(active_pos, mask)

        # Snap the camera when a setting changes (new target, center mode
        # toggle, etc.) so the deadzone doesn't silently absorb the shift.
        if self._target_needs_acquisition:
            self._target_needs_acquisition = False
            center, framing_radius = self._compute_tracking(
                positions, active_pos, mask, use_deadzone=False,
            )
            self._smoothed_framing_radius = framing_radius
            self._radius_ring[:] = framing_radius
            self._radius_ring_sum = framing_radius * len(self._radius_ring)
            self._camera.center = (float(center[0]), float(center[1]), float(center[2]))
            self._camera.distance = self._ideal_distance(framing_radius)

        # Base mode computes center and framing radius
        if self._mode == CameraMode.TARGET_COMOVING:
            center, framing_radius = self._compute_tracking(positions, active_pos, mask, use_deadzone=True)
        elif self._mode == CameraMode.TARGET_REST_FRAME:
            center, framing_radius = self._compute_tracking(positions, active_pos, mask, use_deadzone=False)
        else:
            center, framing_radius = self._smoothed_center, self._smoothed_framing_radius

        # Event tracking overlay: blend toward upcoming events
        if self._event_tracking_enabled and self._events:
            center, framing_radius = self._apply_event_blend(
                sim_time, center, framing_radius, active_pos, mask,
            )

        self._apply_camera(center, framing_radius)

        if self._auto_rotate_enabled:
            self._apply_rotation()

    # --- Framing logic ---

    def _select_framed_particles(
        self,
        active_pos: np.ndarray,
        reference_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Select the closest ``n_framed`` particles to the reference point.

        When ``n_framed >= n_active`` (or ``keep_all_in_frame``), returns
        all particles unchanged.  Also stores ``_framed_active_idx``.

        Uses a hysteresis band to prevent the framed set from flickering
        when two particles are nearly equidistant from the reference
        point.  The current set is kept unless the candidate set's
        farthest member is more than 5% closer than the current set's
        farthest member.

        Args:
            active_pos: (n_active, 3) positions of currently active particles.
            reference_pos: (3,) point to measure distances from. Falls back
                to the current center of mass if None.

        Returns:
            (n_keep, 3) subset of active_pos for the framed particles.
        """
        n = len(active_pos)
        n_keep = min(self._n_framed, n)

        if self._keep_all_in_frame or n <= 2 or n_keep >= n:
            self._framed_active_idx = np.arange(n)
            return active_pos

        center = reference_pos if reference_pos is not None else self._active_com
        distances = np.linalg.norm(active_pos - center, axis=1)

        # Hysteresis: keep the current set unless the candidate set is
        # meaningfully tighter (farthest member >5% closer).
        prev = self._framed_active_idx
        if len(prev) == n_keep and prev.max() < n:
            current_max_dist = distances[prev].max()
            candidate = np.argpartition(distances, n_keep)[:n_keep]
            candidate_max_dist = distances[candidate].max()
            if candidate_max_dist >= current_max_dist * 0.95:
                return active_pos[prev]
            self._framed_active_idx = candidate
            return active_pos[candidate]

        closest = np.argpartition(distances, n_keep)[:n_keep]
        self._framed_active_idx = closest
        return active_pos[closest]

    # --- Helpers ---

    def _compute_center_of_mass(self, active_pos: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compute center of mass of active particles, weighted by mass if available.

        Args:
            active_pos: (n_active, 3) positions of currently active particles.
            mask: (n_particles,) bool — used to index into the full mass array.

        Returns:
            (3,) center of mass position.
        """
        if self._mass_array is not None:
            weights = self._mass_array[mask]
            total = weights.sum()
            if total > 0:
                return (active_pos * weights[:, np.newaxis]).sum(axis=0) / total
        return active_pos.mean(axis=0)

    def _tracking_point(
        self, target_pos: np.ndarray | None, framed_pos: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Determine the camera center point.

        If ``use_group_center`` is False, pins on the target (or COM if
        no target).  Otherwise computes the centroid or mass-weighted
        center of the framed group.

        Args:
            target_pos: (3,) position of the target particle, or None if no
                target is active.
            framed_pos: (n_framed, 3) positions of the framed particle subset.
            mask: (n_particles,) bool — used to map framed indices to full-array
                indices for mass lookup.

        Returns:
            (3,) camera center position.
        """
        if not self._use_group_center:
            return target_pos if target_pos is not None else self._active_com

        if self._mass_weighted_center and self._mass_array is not None:
            # Map framed active-space indices to full-array indices for mass lookup
            full_idx = np.where(mask)[0][self._framed_active_idx]
            weights = self._mass_array[full_idx]
            total = weights.sum()
            if total > 0:
                return (framed_pos * weights[:, np.newaxis]).sum(axis=0) / total

        return framed_pos.mean(axis=0)

    def _cache_fov_trig(self) -> None:
        """Recompute cached FOV-derived trig values.

        Called once at init and whenever framing_fraction changes.
        """
        half_fov = np.radians(self._camera.fov / 2)
        self._tan_half_fov = np.tan(half_fov)
        effective_angle = half_fov * self._framing_fraction
        self._inv_tan_effective_fov = 1.0 / np.tan(effective_angle)

    def _ideal_distance(self, framing_radius: float) -> float:
        """Camera distance that places the farthest framed particle at
        the framing fraction of the screen's vertical half-extent.

        Uses ``tan`` (not ``sin``) because the framing radius is a
        projected (on-screen-plane) distance, so the relationship is
        ``tan(θ) = r / d``.

        Args:
            framing_radius: Maximum effective projected distance from
                center to any framed particle.

        Returns:
            Camera distance in world units.
        """
        return framing_radius * self._inv_tan_effective_fov

    def _compute_framing_radius(self, positions: np.ndarray, center: np.ndarray) -> float:
        """Maximum 3D distance from center to any framed particle.

        Uses the rotation-invariant 3D distance so the framing radius
        does not change when the user orbits the camera.  Depth-axis
        orbital oscillations are handled by the rolling average in
        ``_apply_camera`` rather than by view-dependent projection.

        Args:
            positions: (n_framed, 3) positions of the framed particles.
            center: (3,) camera center point.

        Returns:
            Scalar radius encompassing all framed particles (minimum 1.0).
        """
        if len(positions) == 0:
            return 1.0
        distances = np.linalg.norm(positions - center, axis=1)
        return float(np.max(distances)) or 1.0

    def _apply_pan_deadzone(self, target_position: np.ndarray) -> np.ndarray:
        """Apply panning deadzone: hold camera center until the tracked
        point drifts outside the deadzone radius.

        The camera stays still while the target is within
        `pan_deadzone_fraction` of the visible radius from screen center.
        Once it drifts outside, the camera moves exactly enough to place
        the target on the deadzone edge.

        Args:
            target_position: (3,) world-space position of the tracked point.

        Returns:
            (3,) updated smoothed camera center.
        """
        visible_radius = float(self._camera.distance) * self._tan_half_fov
        deadzone_radius = visible_radius * self._pan_deadzone_fraction

        offset = target_position - self._smoothed_center
        offset_distance = np.linalg.norm(offset)

        if offset_distance > deadzone_radius:
            overshoot = offset_distance - deadzone_radius
            chase_direction = offset / offset_distance
            self._smoothed_center = self._smoothed_center + chase_direction * overshoot

        return self._smoothed_center

    def _apply_camera(self, center: np.ndarray, framing_radius: float) -> None:
        """Push center and distance to the VisPy camera.

        Uses a rolling average over the framing radius so the camera
        tracks the natural scale of the framed group without reacting
        to frame-to-frame orbital oscillations.  Each frame one sample
        enters and one leaves the window, so the target distance drifts
        continuously with no discontinuities.

        Writes to the camera's internal state directly and calls
        view_changed() at most once per frame to avoid redundant
        VisPy transform recomputation.

        Args:
            center: (3,) world-space camera center position.
            framing_radius: Raw (instantaneous) radius encompassing framed particles.
        """
        cam = self._camera
        new_center = (float(center[0]), float(center[1]), float(center[2]))
        cur_center = tuple(cam.center)
        center_changed = new_center != cur_center

        if self._free_zoom:
            if center_changed:
                cam.center = new_center
            return

        # Update the running sum: subtract the sample being overwritten,
        # write the new sample, add it.
        ring = self._radius_ring
        idx = self._radius_ring_idx
        self._radius_ring_sum -= ring[idx]
        ring[idx] = framing_radius
        self._radius_ring_sum += framing_radius
        self._radius_ring_idx = (idx + 1) % len(ring)

        # Camera distance tracks the rolling average directly
        self._smoothed_framing_radius = self._radius_ring_sum / len(ring)
        new_distance = self._ideal_distance(self._smoothed_framing_radius)
        cur_distance = cam.distance
        distance_changed = (cur_distance is None
                            or abs(new_distance - cur_distance) > 1e-4 * cur_distance)

        if center_changed:
            cam.center = new_center
        if distance_changed:
            cam.distance = new_distance

    def _find_target(self, positions: np.ndarray) -> np.ndarray | None:
        """Return the target particle's position, or None if not active.

        Args:
            positions: (n_particles, 3) full-size position array (NaN for inactive).

        Returns:
            (3,) position of the target particle, or None if the target is
            unset, out of range, or inactive this frame.
        """
        if self._target_pid is None or len(self._pid_lookup) == 0:
            return None
        pid = int(self._target_pid)
        if pid >= len(self._pid_lookup):
            return None
        idx = self._pid_lookup[pid]
        if idx < 0:
            return None
        pos = positions[idx]
        if np.isnan(pos[0]):
            return None
        return pos

    # --- Mode implementations ---

    def _compute_tracking(
        self, positions: np.ndarray, active_pos: np.ndarray,
        mask: np.ndarray, use_deadzone: bool,
    ) -> tuple[np.ndarray, float]:
        """Unified tracking: find target, frame particles, compute center.

        When *use_deadzone* is False (rest-frame), locks directly on the
        tracking point.  Falls back to deadzone if no target is active.

        Args:
            positions: (n_particles, 3) full-size position array (NaN for inactive).
            active_pos: (n_active, 3) positions of currently active particles.
            mask: (n_particles,) bool — True where particle is active.
            use_deadzone: If True, apply panning deadzone; if False, lock
                directly on the tracking point.

        Returns:
            (center, framing_radius) tuple.
        """
        target_pos = self._find_target(positions)

        # Rest frame requires a target; fall back to deadzone
        if not use_deadzone and target_pos is None:
            use_deadzone = True

        framed_pos = self._select_framed_particles(
            active_pos, reference_pos=target_pos,
        )
        tracking_point = self._tracking_point(target_pos, framed_pos, mask)

        if use_deadzone:
            center = self._apply_pan_deadzone(tracking_point)
        else:
            self._smoothed_center = tracking_point.copy()
            center = tracking_point

        framing_radius = self._compute_framing_radius(framed_pos, center)
        return center, framing_radius

    def _apply_event_blend(
        self,
        sim_time: float,
        base_center: np.ndarray,
        base_radius: float,
        active_pos: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Blend the base camera state toward the nearest upcoming event.

        Ramps in over ``_event_ramp_in`` seconds before the event and
        ramps out over ``_event_ramp_out`` seconds after.  At peak blend
        the zoom is tightened by ``_event_zoom_tighten``.

        Args:
            sim_time: Current simulation time.
            base_center: (3,) camera center from the base tracking mode.
            base_radius: Framing radius from the base tracking mode.
            active_pos: (n_active, 3) positions of currently active particles.
            mask: (n_particles,) bool — True where particle is active.

        Returns:
            (blended_center, blended_radius) tuple.
        """
        # Find the nearest event (before or after current time)
        best_event = None
        best_dt = float("inf")
        for e in self._events:
            dt = e.time - sim_time
            if abs(dt) < abs(best_dt):
                best_dt = dt
                best_event = e

        if best_event is None:
            return base_center, base_radius

        dt = best_dt
        if dt > 0:
            # Approaching: blend ramps from 0 to 1 over ramp_in seconds
            if dt > self._event_ramp_in:
                return base_center, base_radius
            blend = 1.0 - dt / self._event_ramp_in
        else:
            # Receding: blend ramps from 1 to 0 over ramp_out seconds
            if -dt > self._event_ramp_out:
                return base_center, base_radius
            blend = 1.0 + dt / self._event_ramp_out

        event_pos = best_event.position
        center = (1.0 - blend) * base_center + blend * event_pos
        radius = base_radius * (1.0 - blend * self._event_zoom_tighten)
        return center, radius

    def _apply_rotation(self) -> None:
        """Apply slow auto-rotation around vertical axis."""
        self._azimuth_offset += self._rotation_speed
        self._camera.azimuth = self._azimuth_offset
