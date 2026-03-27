"""Automated camera controller for N-body visualization.

Provides composable camera modes: auto-frame, auto-center,
auto-rotate, event tracking, target tracking, and manual control.

Framing scope controls which particles drive the camera zoom:
- ALL: frame every active particle (can chase outliers)
- CORE_GROUP: ignore outlier particles far from the cluster center
- NEAREST_NEIGHBORS: frame the target particle + its K nearest neighbors
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

from .. import defaults as D


class CameraMode(Enum):
    MANUAL = auto()
    AUTO_FRAME = auto()
    AUTO_ROTATE = auto()
    EVENT_TRACK = auto()
    TARGET_REST_FRAME = auto()
    TARGET_COMOVING = auto()


class FramingScope(Enum):
    ALL = auto()
    CORE_GROUP = auto()
    NEAREST_NEIGHBORS = auto()


class CameraController:
    """Manages automated camera behavior.

    Can be attached to a RenderEngine to control the VisPy camera.

    Args:
        view: VisPy ViewBox whose camera to control.
        masses: Optional dict of particle ID -> mass array for mass-weighted COM.
        events: Optional list of detected events for event-tracking mode.
    """

    def __init__(self, view, masses: dict[int, np.ndarray] | None = None, events=None):
        self._view = view
        self._camera = view.camera
        self._masses = masses
        self._events = events or []

        # Mode
        self._mode = CameraMode.MANUAL
        self._auto_rotate_enabled = False

        # Framing scope
        self._framing_scope = FramingScope.CORE_GROUP
        self._keep_all_in_frame = False
        self._free_zoom = False
        self._free_zoom_callbacks: list = []
        self._n_neighbors = D.CAMERA_N_NEIGHBORS
        self._outlier_sigma = D.CAMERA_OUTLIER_SIGMA

        # Smoothing parameters
        self._ema_alpha = D.CAMERA_EMA_ALPHA
        self._zoom_ema_alpha = D.CAMERA_ZOOM_EMA_ALPHA
        self._deadzone_fraction = 0.4

        # Smoothing state — starts from camera's current position
        cam_center = self._camera.center
        self._smoothed_center = np.array(cam_center, dtype=np.float64) if cam_center else np.zeros(3)
        self._smoothed_distance = float(self._camera.distance or 10.0)

        # Auto-rotate
        self._rotation_speed = D.ROTATION_SPEED
        self._azimuth_offset = 0.0

        # Target tracking
        self._target_pid: int | None = None
        self._target_smoothed_vel = np.zeros(3)
        self._target_needs_acquisition = False

        # Radius percentile for framing (within framed particles)
        self._radius_percentile = D.CAMERA_RADIUS_PERCENTILE

    # --- Properties ---

    @property
    def mode(self) -> CameraMode:
        return self._mode

    @mode.setter
    def mode(self, value: CameraMode) -> None:
        self._mode = value
        if value == CameraMode.AUTO_ROTATE:
            self._auto_rotate_enabled = True

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
    def framing_scope(self) -> FramingScope:
        return self._framing_scope

    @framing_scope.setter
    def framing_scope(self, value: FramingScope) -> None:
        self._framing_scope = value

    @property
    def keep_all_in_frame(self) -> bool:
        return self._keep_all_in_frame

    @keep_all_in_frame.setter
    def keep_all_in_frame(self, value: bool) -> None:
        self._keep_all_in_frame = value

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
    def n_neighbors(self) -> int:
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, value: int) -> None:
        self._n_neighbors = max(1, value)

    # --- Core update ---

    def update(
        self,
        sim_time: float,
        positions: np.ndarray,
        active_ids: np.ndarray,
    ) -> None:
        """Update camera based on current mode and particle positions.

        Called each frame by the RenderEngine.
        """
        if self._mode == CameraMode.MANUAL and not self._auto_rotate_enabled:
            return

        if len(positions) == 0:
            return

        # When a target is first selected, jump the camera to the target
        # group so the deadzone can maintain tracking from a good position
        # (instead of slowly chasing from wherever the camera was).
        if self._target_needs_acquisition:
            self._target_needs_acquisition = False
            target_pos = self._find_target(positions, active_ids)
            if target_pos is not None:
                framed_pos, framed_ids = self._select_framed_particles(
                    positions, active_ids, reference_pos=target_pos,
                )
                group_center = self._compute_center_of_mass(framed_pos, framed_ids)
                framing_radius = self._compute_framing_radius(framed_pos, group_center)
                fov_rad = np.radians(self._camera.fov / 2)
                ideal_distance = framing_radius / np.sin(fov_rad) * 1.25

                self._smoothed_center = group_center.copy()
                self._smoothed_distance = ideal_distance
                self._camera.center = tuple(group_center)
                self._camera.distance = ideal_distance

        if self._mode in (CameraMode.AUTO_FRAME, CameraMode.AUTO_ROTATE):
            self._update_auto_frame(positions, active_ids)
        elif self._mode == CameraMode.EVENT_TRACK:
            self._update_event_track(sim_time, positions, active_ids)
        elif self._mode == CameraMode.TARGET_REST_FRAME:
            self._update_target_rest(positions, active_ids)
        elif self._mode == CameraMode.TARGET_COMOVING:
            self._update_target_comoving(positions, active_ids)

        if self._auto_rotate_enabled:
            self._apply_rotation()

    # --- Framing logic ---

    def _select_framed_particles(
        self,
        positions: np.ndarray,
        active_ids: np.ndarray,
        reference_pos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select which particles should drive the camera framing.

        Args:
            positions: (N, 3) array of all active particle positions.
            active_ids: (N,) array of corresponding particle IDs.
            reference_pos: Optional reference position (e.g. target particle).
                Used as the center for NEAREST_NEIGHBORS distance calculation.

        Returns:
            (framed_positions, framed_ids) — the subset to frame.
        """
        if self._keep_all_in_frame or len(positions) <= 2:
            return positions, active_ids

        scope = self._framing_scope

        if scope == FramingScope.ALL:
            return positions, active_ids

        if scope == FramingScope.CORE_GROUP:
            return self._select_core_group(positions, active_ids)

        if scope == FramingScope.NEAREST_NEIGHBORS:
            return self._select_nearest_neighbors(positions, active_ids, reference_pos)

        return positions, active_ids

    def _select_core_group(
        self, positions: np.ndarray, active_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reject outlier particles far from the cluster center.

        Uses the median distance from the unweighted centroid as the
        scale. Particles beyond outlier_sigma × median_distance are
        excluded from framing (they still render, just don't drive the zoom).
        """
        centroid = positions.mean(axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        median_dist = np.median(distances)

        if median_dist < 1e-12:
            return positions, active_ids

        threshold = self._outlier_sigma * median_dist
        mask = distances <= threshold
        # Always keep at least 2 particles
        if mask.sum() < 2:
            # Keep the closest 2
            closest = np.argsort(distances)[:2]
            mask = np.zeros(len(positions), dtype=bool)
            mask[closest] = True

        return positions[mask], active_ids[mask]

    def _select_nearest_neighbors(
        self,
        positions: np.ndarray,
        active_ids: np.ndarray,
        reference_pos: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select the target particle + its K nearest neighbors.

        If no target is set or reference_pos is None, falls back to CORE_GROUP.
        """
        if reference_pos is None:
            return self._select_core_group(positions, active_ids)

        distances = np.linalg.norm(positions - reference_pos, axis=1)
        # K nearest (including the target itself, which has distance ~0)
        k = min(self._n_neighbors + 1, len(positions))
        nearest_idx = np.argsort(distances)[:k]
        return positions[nearest_idx], active_ids[nearest_idx]

    # --- Helpers ---

    def _compute_center_of_mass(self, positions: np.ndarray, active_ids: np.ndarray) -> np.ndarray:
        """Compute center of mass of framed particles, weighted by mass if available."""
        if self._masses is not None:
            weights = []
            for pid in active_ids:
                pid_key = int(pid)
                if pid_key in self._masses and len(self._masses[pid_key]) > 0:
                    weights.append(self._masses[pid_key][-1])  # use latest mass
                else:
                    weights.append(1.0)
            weights = np.array(weights)
            total = weights.sum()
            if total > 0:
                return (positions * weights[:, np.newaxis]).sum(axis=0) / total
        return positions.mean(axis=0)

    def _compute_framing_radius(self, positions: np.ndarray, center: np.ndarray) -> float:
        """Compute the framing radius (95th percentile of distances from center).

        This determines how far the camera should be to fit the cluster.
        Using a percentile rather than the max avoids chasing distant outliers.
        """
        distances = np.linalg.norm(positions - center, axis=1)
        if len(distances) == 0:
            return 1.0
        return float(np.percentile(distances, self._radius_percentile)) or 1.0

    def _smooth_center(self, target_center: np.ndarray) -> np.ndarray:
        """Move camera center toward target at constant velocity.

        Each frame, the center moves by `ema_alpha` fraction of the gap
        to the target, clamped so it never exceeds `ema_alpha * distance`
        (constant visual speed on screen).
        """
        delta = target_center - self._smoothed_center
        dist = np.linalg.norm(delta)
        if dist < 1e-30:
            return self._smoothed_center

        # Blend toward target: covers ema_alpha of the remaining gap,
        # but capped at a max visual speed (fraction of camera distance)
        step = min(self._ema_alpha, 1.0) * dist
        max_step = self._smoothed_distance * self._ema_alpha
        step = min(step, max_step)
        self._smoothed_center = self._smoothed_center + delta * (step / dist)

        return self._smoothed_center

    def _smooth_distance(self, target_distance: float) -> float:
        """Move camera distance toward target at constant velocity.

        Each frame, the distance changes by `zoom_ema_alpha` fraction
        of the gap, giving smooth zoom without snapping.
        """
        delta = target_distance - self._smoothed_distance
        step = abs(delta) * min(self._zoom_ema_alpha, 1.0)
        max_step = abs(self._smoothed_distance) * self._zoom_ema_alpha
        step = min(step, max_step)
        self._smoothed_distance += step if delta > 0 else -step

        return self._smoothed_distance

    def _apply_deadzone(self, target_position: np.ndarray) -> np.ndarray:
        """Apply deadzone panning: hold camera until target drifts past threshold.

        The camera stays still while the target is within `deadzone_fraction`
        of the visible radius from screen center.  Once it drifts outside,
        the camera moves exactly enough to place the target on the deadzone
        edge.  No additional velocity cap — the deadzone itself IS the
        smoothing.  This ensures the camera always keeps up, even during
        fast slingshots.
        """
        visible_radius = self._smoothed_distance * np.tan(
            np.radians(self._camera.fov / 2)
        )
        deadzone_radius = visible_radius * self._deadzone_fraction

        offset = target_position - self._smoothed_center
        offset_distance = np.linalg.norm(offset)

        if offset_distance > deadzone_radius:
            # Move camera exactly enough to put target on the deadzone edge
            overshoot = offset_distance - deadzone_radius
            chase_direction = offset / offset_distance
            self._smoothed_center = self._smoothed_center + chase_direction * overshoot

        return self._smoothed_center

    def _apply_camera(self, center: np.ndarray, framing_radius: float) -> None:
        """Push smoothed center and distance to the VisPy camera.

        Center is always updated (target tracking should work even if
        the user has scrolled to zoom manually).  Distance is only
        updated when free zoom is off — free zoom gives the user manual
        control of the zoom level while the camera still tracks.

        Zoom uses the deadzone philosophy: the camera distance holds
        steady until the ideal framing distance differs from the current
        by more than the deadzone fraction, then chases to the edge.
        """
        self._camera.center = tuple(center)

        # Free zoom: user controls distance manually via scroll wheel.
        # Center tracking still works so target modes remain functional.
        if self._free_zoom:
            return

        fov_rad = np.radians(self._camera.fov / 2)
        ideal_distance = framing_radius / np.sin(fov_rad) * 1.25

        # Zoom deadzone: hold if the ideal distance is within ±50% of
        # the current smoothed distance.  Chase to the edge if outside.
        # No velocity cap — the deadzone is the only smoothing.
        current = self._smoothed_distance
        delta = ideal_distance - current
        threshold = abs(current) * self._deadzone_fraction

        if abs(delta) > threshold:
            edge_distance = current + (delta - np.sign(delta) * threshold)
            self._smoothed_distance = edge_distance
            self._camera.distance = edge_distance
        else:
            self._camera.distance = current

    # --- Helpers ---

    def _find_target(
        self, positions: np.ndarray, active_ids: np.ndarray,
    ) -> np.ndarray | None:
        """Return the target particle's position, or None if not active."""
        if self._target_pid is None:
            return None
        for i, pid in enumerate(active_ids):
            if int(pid) == self._target_pid:
                return positions[i]
        return None

    # --- Mode implementations ---

    def _update_auto_frame(self, positions: np.ndarray, active_ids: np.ndarray) -> None:
        """Auto-frame: frame particles based on current framing scope.

        When using NEAREST_NEIGHBORS scope, applies a deadzone: the
        camera holds still while the group's center of mass stays within
        30% of the visible radius, and smoothly chases only when the
        group drifts outside.
        """
        reference_pos = self._find_target(positions, active_ids)

        framed_pos, framed_ids = self._select_framed_particles(
            positions, active_ids, reference_pos=reference_pos
        )
        target_com = self._compute_center_of_mass(framed_pos, framed_ids)

        if self._framing_scope == FramingScope.NEAREST_NEIGHBORS:
            center = self._apply_deadzone(target_com)
        else:
            center = self._smooth_center(target_com)

        framing_radius = self._compute_framing_radius(framed_pos, center)
        self._apply_camera(center, framing_radius)

    def _update_event_track(
        self, sim_time: float, positions: np.ndarray, active_ids: np.ndarray
    ) -> None:
        """Event tracking: zoom into upcoming events."""
        # Find the closest upcoming event
        upcoming = [e for e in self._events if e.time >= sim_time]
        if upcoming:
            event = min(upcoming, key=lambda e: e.time)
            # Transition: blend between current framing and event location
            dt = event.time - sim_time
            transition_time = 2.0  # seconds of lead time
            if dt < transition_time and dt > 0:
                blend = 1.0 - dt / transition_time
                event_pos = event.position
                framed_pos, framed_ids = self._select_framed_particles(
                    positions, active_ids, reference_pos=event_pos
                )
                com = self._compute_center_of_mass(framed_pos, framed_ids)
                target = (1 - blend) * com + blend * event_pos
                center = self._smooth_center(target)
                # Zoom in for the event
                framing_radius = self._compute_framing_radius(framed_pos, center)
                event_zoom = framing_radius * (1 - blend * 0.7)  # zoom in up to 70%
                fov_rad = np.radians(self._camera.fov / 2)
                distance = self._smooth_distance(event_zoom / np.sin(fov_rad))
                self._camera.center = tuple(center)
                self._camera.distance = distance
                return

        # Fallback to auto-frame
        self._update_auto_frame(positions, active_ids)

    def _update_target_rest(self, positions: np.ndarray, active_ids: np.ndarray) -> None:
        """Target rest frame: camera locked exactly on target particle.

        Center is locked directly to the target (no smoothing, no deadzone).
        Zoom uses _apply_camera which respects the zoom deadzone.
        """
        target_pos = self._find_target(positions, active_ids)
        if target_pos is None:
            self._update_auto_frame(positions, active_ids)
            return

        # Lock center directly — no smoothing, target stays at screen center
        self._smoothed_center = target_pos.copy()

        framed_pos, _ = self._select_framed_particles(
            positions, active_ids, reference_pos=target_pos,
        )
        framing_radius = self._compute_framing_radius(framed_pos, target_pos)
        self._apply_camera(target_pos, framing_radius)

    def _update_target_comoving(
        self, positions: np.ndarray, active_ids: np.ndarray,
    ) -> None:
        """Target comoving frame with deadzone.

        The camera holds still while the target particle stays within
        the deadzone.  Once it drifts past the edge, the camera smoothly
        chases just enough to pull the target back to the boundary.
        """
        target_pos = self._find_target(positions, active_ids)
        if target_pos is None:
            self._update_auto_frame(positions, active_ids)
            return

        center = self._apply_deadzone(target_pos)

        framed_pos, _ = self._select_framed_particles(
            positions, active_ids, reference_pos=target_pos
        )
        framing_radius = self._compute_framing_radius(framed_pos, center)
        self._apply_camera(center, framing_radius)

    def _apply_rotation(self) -> None:
        """Apply slow auto-rotation around vertical axis."""
        self._azimuth_offset += self._rotation_speed
        if hasattr(self._camera, "azimuth"):
            self._camera.azimuth = self._azimuth_offset
