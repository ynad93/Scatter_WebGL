"""Automated camera controller for N-body visualization.

Provides composable camera modes: tracking, auto-center,
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
    EVENT_TRACK = auto()
    TARGET_REST_FRAME = auto()
    TARGET_COMOVING = auto()  # unified tracking: deadzone on target or COM


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

        # Framing: which particles drive the camera
        self._framing_scope = FramingScope.CORE_GROUP
        self._keep_all_in_frame = False
        self._core_group_percentile = 100.0  # 100% = no filtering; GUI slider adjusts
        self._n_neighbors = D.CAMERA_N_NEIGHBORS

        # Target tracking
        self._target_pid: int | None = None
        self._target_needs_acquisition = False

        # Deadzone: camera holds still until the tracked point drifts
        # outside this fraction of the visible radius
        self._deadzone_fraction = 0.5

        # Free zoom: when True, user controls distance via scroll wheel;
        # center tracking still works for target modes
        self._free_zoom = False
        self._free_zoom_callbacks: list = []


        # Camera state — initialized from VisPy camera's current position
        cam_center = self._camera.center
        self._smoothed_center = np.array(cam_center, dtype=np.float64) if cam_center else np.zeros(3)
        self._smoothed_distance = float(self._camera.distance or 10.0)
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

        # Extract active-only positions for framing calculations
        active_pos = positions[mask]

        # Mass-weighted center of mass of ALL active particles (computed once)
        self._active_com = self._compute_center_of_mass(active_pos, mask)

        # When a target is first selected, jump the camera to the target
        # group so the deadzone can maintain tracking from a good position
        # (instead of slowly chasing from wherever the camera was).
        if self._target_needs_acquisition:
            self._target_needs_acquisition = False
            target_pos = self._find_target(positions)
            if target_pos is not None:
                framed_pos = self._select_framed_particles(
                    active_pos, reference_pos=target_pos,
                )
                group_center = framed_pos.mean(axis=0)
                framing_radius = self._compute_framing_radius(framed_pos, group_center)
                fov_rad = np.radians(self._camera.fov / 2)
                # Distance that fits framing_radius inside the FOV cone,
                # with 25% padding so particles aren't clipped at screen edges
                ideal_distance = framing_radius / np.sin(fov_rad) * 1.25

                self._smoothed_center = group_center.copy()
                self._smoothed_distance = ideal_distance
                self._camera.center = tuple(group_center)
                self._camera.distance = ideal_distance

        if self._mode == CameraMode.TARGET_COMOVING:
            self._update_target_comoving(positions, active_pos)
        elif self._mode == CameraMode.EVENT_TRACK:
            self._update_event_track(sim_time, positions, active_pos)
        elif self._mode == CameraMode.TARGET_REST_FRAME:
            self._update_target_rest(positions, active_pos)

        if self._auto_rotate_enabled:
            self._apply_rotation()

    # --- Framing logic ---

    def _select_framed_particles(
        self,
        active_pos: np.ndarray,
        reference_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Select which particles should drive the camera framing.

        Args:
            active_pos: (N_active, 3) positions of active particles.
            reference_pos: Optional reference position (e.g. target particle).

        Returns:
            Subset of active_pos to frame.
        """
        if self._keep_all_in_frame or len(active_pos) <= 2:
            return active_pos

        scope = self._framing_scope

        if scope == FramingScope.ALL:
            return active_pos

        if scope == FramingScope.CORE_GROUP:
            return self._select_core_group(active_pos)

        if scope == FramingScope.NEAREST_NEIGHBORS:
            return self._select_nearest_neighbors(active_pos, reference_pos)

        return active_pos

    def _select_core_group(self, active_pos: np.ndarray) -> np.ndarray:
        """Keep the closest X% of particles by distance from centroid.

        Particles are ranked by distance from the unweighted centroid.
        The closest `core_group_percentile`% by count are kept for
        framing.  Outliers still render but don't drive the camera.
        """
        centroid = active_pos.mean(axis=0)
        distances = np.linalg.norm(active_pos - centroid, axis=1)

        n_keep = max(2, int(len(active_pos) * self._core_group_percentile / 100.0))
        closest = np.argsort(distances)[:n_keep]
        return active_pos[closest]

    def _select_nearest_neighbors(
        self,
        active_pos: np.ndarray,
        reference_pos: np.ndarray | None,
    ) -> np.ndarray:
        """Select the K nearest neighbors around a reference point.

        If a target particle is set, uses its position as the reference.
        Otherwise uses the mass-weighted center of mass (computed once
        per frame in update()).
        """
        if reference_pos is None:
            reference_pos = self._active_com

        distances = np.linalg.norm(active_pos - reference_pos, axis=1)
        k = min(self._n_neighbors + 1, len(active_pos))
        nearest_idx = np.argsort(distances)[:k]
        return active_pos[nearest_idx]

    # --- Helpers ---

    def _compute_center_of_mass(self, active_pos: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compute center of mass of active particles, weighted by mass if available."""
        if self._mass_array is not None:
            weights = self._mass_array[mask]
            total = weights.sum()
            if total > 0:
                return (active_pos * weights[:, np.newaxis]).sum(axis=0) / total
        return active_pos.mean(axis=0)

    def _compute_framing_radius(self, positions: np.ndarray, center: np.ndarray) -> float:
        """Compute the framing radius (maximum distance from center).

        Returns the distance of the farthest particle from center.
        Since the core group percentile already filters outliers,
        using max here frames all remaining particles.
        """
        distances = np.linalg.norm(positions - center, axis=1)
        if len(distances) == 0:
            return 1.0
        return float(np.max(distances)) or 1.0

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

    def _find_target(self, positions: np.ndarray) -> np.ndarray | None:
        """Return the target particle's position, or None if not active.

        Args:
            positions: (n_particles, 3) full-size array (NaN for inactive).
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

    def _update_event_track(
        self, sim_time: float, positions: np.ndarray, active_pos: np.ndarray,
    ) -> None:
        """Event tracking: zoom into upcoming events."""
        upcoming = [e for e in self._events if e.time >= sim_time]
        if upcoming:
            event = min(upcoming, key=lambda e: e.time)
            dt = event.time - sim_time
            transition_time = 2.0
            if dt < transition_time and dt > 0:
                blend = 1.0 - dt / transition_time
                event_pos = event.position
                framed_pos = self._select_framed_particles(
                    active_pos, reference_pos=event_pos,
                )
                com = framed_pos.mean(axis=0)
                target = (1 - blend) * com + blend * event_pos
                center = self._apply_deadzone(target)
                framing_radius = self._compute_framing_radius(framed_pos, center)
                event_zoom = framing_radius * (1 - blend * 0.7)
                fov_rad = np.radians(self._camera.fov / 2)
                distance = event_zoom / np.sin(fov_rad)
                self._smoothed_distance = distance
                self._camera.center = tuple(center)
                self._camera.distance = distance
                return

        self._update_target_comoving(positions, active_pos)

    def _update_target_rest(
        self, positions: np.ndarray, active_pos: np.ndarray,
    ) -> None:
        """Target rest frame: camera locked exactly on target particle.

        Center is locked directly to the target (no smoothing, no deadzone).
        Zoom uses _apply_camera which respects the zoom deadzone.
        """
        target_pos = self._find_target(positions)
        if target_pos is None:
            self._update_target_comoving(positions, active_pos)
            return

        self._smoothed_center = target_pos.copy()

        framed_pos = self._select_framed_particles(
            active_pos, reference_pos=target_pos,
        )
        framing_radius = self._compute_framing_radius(framed_pos, target_pos)
        self._apply_camera(target_pos, framing_radius)

    def _update_target_comoving(
        self, positions: np.ndarray, active_pos: np.ndarray,
    ) -> None:
        """Unified deadzone tracking mode.

        If a target particle is set, the deadzone tracks that particle.
        Otherwise, it tracks the center of mass of the framing scope
        (Core Group, All, or Nearest Neighbors).

        The camera holds still while the tracked point stays within the
        deadzone.  Once it drifts past the edge, the camera moves exactly
        enough to bring it back to the boundary.
        """
        target_pos = self._find_target(positions)

        framed_pos = self._select_framed_particles(
            active_pos, reference_pos=target_pos,
        )

        if target_pos is not None:
            tracking_point = target_pos
        else:
            tracking_point = self._active_com

        center = self._apply_deadzone(tracking_point)
        framing_radius = self._compute_framing_radius(framed_pos, center)
        self._apply_camera(center, framing_radius)

    def _apply_rotation(self) -> None:
        """Apply slow auto-rotation around vertical axis."""
        self._azimuth_offset += self._rotation_speed
        self._camera.azimuth = self._azimuth_offset
