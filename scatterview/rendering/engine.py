"""VisPy-based OpenGL rendering engine for N-body visualization."""

from __future__ import annotations

from pathlib import Path

import numba as nb
import numpy as np

from .. import defaults as D
from ..core.data_loader import SimulationData
from ..core.interpolation import TrajectoryInterpolator


@nb.njit(cache=True)
def _advance_trail_pointers(
    packed_times,        # (total_pts,) float64 — all precomputed trail times
    segment_start,       # (n_particles,) int64 — start offset in packed_times
    segment_end,         # (n_particles,) int64 — end offset in packed_times
    prev_window_start,   # (n_particles,) int64 — previous tail pointer (-1 = unset)
    prev_window_end,     # (n_particles,) int64 — previous head pointer (-1 = unset)
    t_trail_start,       # float64 — oldest time in the visible trail window
    t_current,           # float64 — current simulation time (newest point)
    is_forward,          # bool — True if time is advancing (normal playback)
    out_window_start,    # (n_particles,) int64 — output tail indices
    out_window_end,      # (n_particles,) int64 — output head indices
):
    """Find the visible trail window [start, end) for each particle.

    During forward playback, advances the previous frame's pointers by
    1-2 positions (O(1) per particle).  On scrub, loop, or first frame,
    falls back to binary search (O(log N) per particle).
    """
    for particle in range(len(segment_start)):
        seg_lo = segment_start[particle]
        n_points = segment_end[particle] - seg_lo

        if is_forward and prev_window_start[particle] >= 0:
            # Forward playback: advance pointers from last frame
            tail = prev_window_start[particle]
            while tail < n_points and packed_times[seg_lo + tail] <= t_trail_start:
                tail += 1
            head = prev_window_end[particle]
            while head < n_points and packed_times[seg_lo + head] < t_current:
                head += 1
        else:
            # Binary search (searchsorted side='right' for tail)
            lo, hi = nb.int64(0), n_points
            while lo < hi:
                mid = (lo + hi) >> 1
                if packed_times[seg_lo + mid] <= t_trail_start:
                    lo = mid + 1
                else:
                    hi = mid
            tail = lo
            # Binary search (searchsorted side='left' for head)
            lo, hi = nb.int64(0), n_points
            while lo < hi:
                mid = (lo + hi) >> 1
                if packed_times[seg_lo + mid] < t_current:
                    lo = mid + 1
                else:
                    hi = mid
            head = lo

        out_window_start[particle] = tail
        out_window_end[particle] = head


@nb.njit(cache=True)
def _assemble_trails(
    out_positions,            # (total_pts, 3) float32 — output positions
    out_colors,               # (total_pts, 4) float32 — output RGBA colors
    out_times,                # (total_pts,) float64 — output timestamps
    write_offsets,            # (n_trails,) int64 — start index in output arrays
    trail_point_counts,       # (n_trails,) int64 — total points per trail
    trail_body_starts,        # (n_trails,) int64 — start index of body in precomp
    trail_body_counts,        # (n_trails,) int64 — number of precomputed body points
    trail_has_tail,           # (n_trails,) bool — whether tail lerp point exists
    trail_tail_positions,     # (n_trails, 3) float32 — lerp'd tail positions
    active_particle_indices,  # (n_trails,) int64 — index into live positions array
    particle_color_indices,   # (n_trails,) int64 — index into color table
    precomp_positions,        # (total_precomp, 3) float32 — packed precomputed positions
    precomp_times,            # (total_precomp,) float64 — packed precomputed times
    live_positions,           # (n_active, 3) float64 — current particle positions
    color_table,              # (n_particles, 4) float32 — per-particle RGBA
    t_trail_start,            # float64 — oldest time in the trail window
    t_current,                # float64 — current simulation time
):
    """Write trail geometry into output buffers for GPU upload.

    Each trail is laid out as: [NaN separator] [tail lerp] [body] [head]
    - Tail: interpolated position at exactly t_trail_start (smooth fade-in)
    - Body: precomputed positions between t_trail_start and t_current
    - Head: live particle position at t_current (from spline evaluation)
    - NaN separators tell VisPy's line renderer to break between trails
    """
    n_trails = len(write_offsets)
    for trail in range(n_trails):
        offset = write_offsets[trail]
        n_points = trail_point_counts[trail]
        n_body = trail_body_counts[trail]
        write_pos = nb.int64(0)

        # NaN separator before this trail (except the first)
        if trail > 0:
            sep = offset - 1
            for dim in range(3):
                out_positions[sep, dim] = np.nan
            for dim in range(4):
                out_colors[sep, dim] = np.nan
            out_times[sep] = np.nan

        # Tail: lerp'd position at the trail window boundary
        if trail_has_tail[trail]:
            for dim in range(3):
                out_positions[offset, dim] = trail_tail_positions[trail, dim]
            out_times[offset] = t_trail_start
            write_pos = 1

        # Body: copy precomputed trajectory points
        body_start = trail_body_starts[trail]
        for i in range(n_body):
            for dim in range(3):
                out_positions[offset + write_pos + i, dim] = precomp_positions[body_start + i, dim]
            out_times[offset + write_pos + i] = precomp_times[body_start + i]
        write_pos += n_body

        # Head: current particle position from spline evaluation
        particle_idx = active_particle_indices[trail]
        for dim in range(3):
            out_positions[offset + write_pos, dim] = np.float32(live_positions[particle_idx, dim])
        out_times[offset + write_pos] = t_current

        # Base RGB color for all points in this trail
        color_idx = particle_color_indices[trail]
        for i in range(n_points):
            for dim in range(3):
                out_colors[offset + i, dim] = color_table[color_idx, dim]


class RenderEngine:
    """Manages the VisPy canvas, particle and trail rendering, camera, and animation."""

    def __init__(
        self, data: SimulationData, interpolator: TrajectoryInterpolator,
        size: tuple[int, int] = (1280, 720),
        title: str = "ScatterView",
    ):
        from vispy import app, scene

        self._data = data
        self._interp = interpolator
        self._t_min = float(data.times[0])
        self._t_max = float(data.times[-1])
        self._time_range = self._t_max - self._t_min

        # Animation state
        self._playing = False
        self._anim_time = 0.0
        self._anim_speed = D.ANIM_SPEED
        self._current_sim_time = data.times[0]

        # Appearance
        self._point_alpha = D.POINT_ALPHA
        self._trail_alpha = D.TRAIL_ALPHA
        self._trail_length_frac = D.TRAIL_LENGTH_FRAC
        self._trail_width = D.TRAIL_WIDTH

        # Per-particle settings
        n = len(data.particle_ids)
        self._colors = self._default_colors(n)
        self._sizing_absolute = False
        self._radius_scale = 1.0
        self._per_particle_scale = np.ones(n, dtype=np.float32)
        self._depth_scaling = D.DEPTH_SCALING

        if data.radii is not None:
            self._raw_radii = np.array(
                [data.radii.get(int(pid), 1.0) for pid in data.particle_ids],
                dtype=np.float32,
            )
        else:
            self._raw_radii = None

        self._base_sizes_relative = self._compute_relative_base_sizes()
        self._base_sizes_absolute = self._compute_absolute_base_sizes()
        self._sizes = self._base_sizes_relative.copy()

        # Black hole rendering
        self._bh_set: set[int] = set()
        if data.startypes is not None:
            for pid_key, k in data.startypes.items():
                if k == D.BH_STARTYPE:
                    self._bh_set.add(pid_key)

        self._id_to_idx = {int(pid): i for i, pid in enumerate(data.particle_ids)}

        # VisPy canvas and view
        self._canvas = scene.SceneCanvas(
            keys="interactive", size=size, title=title, show=False,
        )
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = scene.cameras.TurntableCamera(
            fov=D.CAMERA_FOV, distance=self._compute_initial_distance(),
        )
        self._view.camera.set_range()

        # Sub-view (picture-in-picture)
        self._subview = None
        self._subview_canvas = None
        self._subview_enabled = False
        self._subview_markers = None
        self._subview_camera_controller = None

        # Visuals
        self._particle_visual = None
        self._trail_line = None       # single Line visual for all trails
        self._subview_trail_line = None

        # Vectorized pid→index lookup (used for colors and trail window)
        max_pid = int(max(data.particle_ids)) + 1
        self._pid_lookup = np.full(max_pid, -1, dtype=np.int32)
        for i, pid in enumerate(data.particle_ids):
            self._pid_lookup[int(pid)] = i

        # Pre-computed trails: evaluated once at startup for the entire
        # simulation, then sliced per-frame via a sliding window.
        self._n_particles = n
        self._precomp = interpolator.precompute_all_trails()

        # Pre-allocated trail GPU arrays — reused each frame to avoid
        # per-frame allocation and GC pressure at 60+ FPS.
        self._trail_capacity = 0
        self._combined_pos = np.empty((0, 3), dtype=np.float32)
        self._combined_colors = np.empty((0, 4), dtype=np.float32)
        self._combined_times = np.empty(0, dtype=np.float64)

        # Two-pointer sliding window indices (si, ei) per particle.
        # During forward playback these advance by 0-2 positions per frame
        # (O(1)) instead of binary search (O(log N)).  Reset to -1 to
        # signal that searchsorted is needed (first frame, scrub, loop).
        self._trail_si = np.full(n, -1, dtype=np.int64)
        self._trail_ei = np.full(n, -1, dtype=np.int64)
        self._trail_prev_time = -np.inf

        # Trail alpha gradient: maps time fraction (0=oldest, 1=newest) to
        # opacity via a t^1.5 power curve.  Trails fade from transparent at
        # the tail to opaque at the head.
        self._alpha_lut = (np.linspace(0, 1, 1024) ** 1.5).astype(np.float32)

        self._build_visuals()

        # Timer and camera
        self._timer = app.Timer(interval=1.0 / 60.0, connect=self._on_timer, start=False)
        self._camera_controller = None

        # Auto-enable free zoom on scroll wheel / trackpad zoom
        self._canvas.events.mouse_wheel.connect(self._on_mouse_wheel)
        self._canvas.events.key_press.connect(self._on_key_press)
        self._canvas.events.key_release.connect(self._on_key_release)

        # Keyboard pan: track which keys are held for smooth continuous movement
        self._pan_keys_held: set[str] = set()
        self._ctrl_held = False

    @staticmethod
    def _has_ctrl(event) -> bool:
        """Check if Ctrl is held during an event."""
        modifiers = getattr(event, 'modifiers', None) or ()
        return any(
            (m.name if hasattr(m, 'name') else str(m)).lower() == 'control'
            for m in modifiers
        )

    def _on_mouse_wheel(self, event) -> None:
        """Zoom toward the mouse cursor position.

        Shifts the camera center toward the world-space point under the
        cursor as it zooms in (like Google Maps).  The cursor offset
        from screen center is converted to world units using the camera
        distance and field of view, then the center is nudged by that
        offset proportional to the zoom amount.

        Holding Ctrl multiplies zoom speed by 5x.
        """
        if self._camera_controller is not None and not self._camera_controller.free_zoom:
            self._camera_controller.free_zoom = True

        camera = self._view.camera
        if not hasattr(camera, 'distance') or camera.distance is None:
            return

        delta = getattr(event, 'delta', None)
        if delta is None:
            return
        scroll = delta[1] if hasattr(delta, '__len__') else float(delta)
        if scroll == 0:
            return

        speed = 5.0 if self._has_ctrl(event) else 1.0
        zoom_factor = 1.1 ** (-scroll * speed)

        # Shift center toward cursor: convert the cursor's pixel offset
        # from screen center into world-space displacement at the focal
        # plane (the plane through camera.center).
        import math
        mouse_pos = getattr(event, 'pos', None)
        canvas_size = self._canvas.size
        if mouse_pos is not None and len(mouse_pos) >= 2 and canvas_size[1] > 0:
            # Cursor offset from screen center, in pixels
            cursor_offset_x = mouse_pos[0] - canvas_size[0] / 2
            cursor_offset_y = mouse_pos[1] - canvas_size[1] / 2

            # Convert pixels to world units: at the focal plane, the
            # visible half-height is distance * tan(fov/2).  Pixel scale
            # is (visible height) / (screen height).
            fov_rad = math.radians(camera.fov / 2)
            world_per_pixel = 2.0 * camera.distance * math.tan(fov_rad) / canvas_size[1]

            # World-space offset in camera-relative coordinates
            dx_screen = cursor_offset_x * world_per_pixel
            dy_screen = -cursor_offset_y * world_per_pixel  # screen Y is flipped

            # Rotate screen offset into world coordinates using camera azimuth
            az_rad = math.radians(camera.azimuth)
            right = np.array([math.cos(az_rad), math.sin(az_rad), 0.0])
            up = np.array([0.0, 0.0, 1.0])

            center = np.array(camera.center, dtype=np.float64)
            world_offset = right * dx_screen + up * dy_screen

            # Shift center toward cursor proportional to zoom amount:
            # zoom_factor < 1 (zooming in) → positive shift toward cursor
            # zoom_factor > 1 (zooming out) → negative shift away
            center += world_offset * (1.0 - zoom_factor)
            camera.center = tuple(center)

        camera.distance *= zoom_factor
        camera.view_changed()
        event.handled = True

    _PAN_KEYS = {'W', 'Up', 'S', 'Down', 'A', 'Left', 'D', 'Right'}

    def _on_key_press(self, event) -> None:
        """Track held keys for smooth continuous panning."""
        key = getattr(event, 'key', None)
        if key is None:
            return
        key_name = key.name if hasattr(key, 'name') else str(key)
        if key_name in self._PAN_KEYS:
            self._pan_keys_held.add(key_name)
            # Enable free zoom so auto-framing doesn't fight the pan
            if self._camera_controller is not None and not self._camera_controller.free_zoom:
                self._camera_controller.free_zoom = True
        if key_name == 'Control':
            self._ctrl_held = True

    def _on_key_release(self, event) -> None:
        """Stop panning when key is released."""
        key = getattr(event, 'key', None)
        if key is None:
            return
        key_name = key.name if hasattr(key, 'name') else str(key)
        self._pan_keys_held.discard(key_name)
        if key_name == 'Control':
            self._ctrl_held = False

    def _apply_keyboard_pan(self) -> None:
        """Apply smooth continuous panning from held keys.

        Called each frame from _on_timer.  Pan speed is proportional to
        camera distance so it feels consistent at any zoom level.
        Holding Ctrl multiplies speed by 5x.
        """
        if not self._pan_keys_held:
            return

        import math
        camera = self._view.camera
        distance = camera.distance or 1.0

        # Pan speed per frame (fraction of camera distance)
        base_speed = 0.02
        speed = base_speed * (5.0 if self._ctrl_held else 1.0)
        step = distance * speed

        # Camera-relative directions based on azimuth
        az_rad = math.radians(camera.azimuth)
        right = np.array([math.cos(az_rad), math.sin(az_rad), 0.0])
        forward = np.array([-math.sin(az_rad), math.cos(az_rad), 0.0])

        center = np.array(camera.center, dtype=np.float64)
        for key_name in self._pan_keys_held:
            if key_name in ('W', 'Up'):
                center += forward * step
            elif key_name in ('S', 'Down'):
                center -= forward * step
            elif key_name in ('A', 'Left'):
                center -= right * step
            elif key_name in ('D', 'Right'):
                center += right * step

        camera.center = tuple(center)

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def _compute_relative_base_sizes(self) -> np.ndarray:
        n = len(self._data.particle_ids)
        if self._raw_radii is not None:
            compressed = np.cbrt(self._raw_radii)
            max_c = compressed.max()
            if max_c > 0:
                normalized = compressed / max_c
                rng = D.RELATIVE_SIZE_MAX_PX - D.RELATIVE_SIZE_MIN_PX
                return (D.RELATIVE_SIZE_MIN_PX + normalized * rng).astype(np.float32)
        return np.full(n, D.DEFAULT_SIZE_PX, dtype=np.float32)

    def _compute_absolute_base_sizes(self) -> np.ndarray:
        n = len(self._data.particle_ids)
        if self._raw_radii is not None:
            return 2.0 * self._raw_radii
        positions = list(self._data.positions.values())
        if positions and len(positions[0]) > 0:
            all_pos = np.vstack([p[0:1] for p in positions if len(p) > 0])
            extent = np.ptp(all_pos, axis=0).max()
            default_r = max(extent * 0.01, 1.0)
        else:
            default_r = 1.0
        return np.full(n, 2.0 * default_r, dtype=np.float32)

    def _recompute_sizes(self) -> None:
        base = self._base_sizes_absolute if self._sizing_absolute else self._base_sizes_relative
        self._sizes = base * self._radius_scale * self._per_particle_scale

    # ------------------------------------------------------------------
    # Colors / helpers
    # ------------------------------------------------------------------

    def _default_colors(self, n: int) -> np.ndarray:
        base = np.array([
            [1.0, 0.3, 0.3, 1.0], [0.3, 0.6, 1.0, 1.0],
            [0.3, 1.0, 0.3, 1.0], [1.0, 0.8, 0.2, 1.0],
            [0.8, 0.3, 1.0, 1.0], [1.0, 0.5, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0], [1.0, 0.4, 0.7, 1.0],
        ], dtype=np.float32)
        idx = np.arange(n) % len(base)
        return base[idx]

    def _compute_initial_distance(self) -> float:
        positions, _, _ = self._interp.evaluate_batch(self._data.times[0])
        if len(positions) == 0:
            return 10.0
        return max(np.max(np.linalg.norm(positions, axis=1)) * 5.0, 1.0)

    def _get_particle_attrs(self, active_ids: np.ndarray) -> dict:
        """Compute face_color, edge_color, edge_width, size in one pass."""
        idx = self._pid_lookup[active_ids.astype(np.intp)]
        n = len(active_ids)
        face_color = self._colors[idx].copy()
        edge_color = np.zeros((n, 4), dtype=np.float32)
        edge_width = np.zeros(n, dtype=np.float32)
        sizes = self._sizes[idx].copy()

        if self._bh_set:
            bh_face = np.array(D.BH_FACE_COLOR, dtype=np.float32)
            for i, pid in enumerate(active_ids):
                if int(pid) in self._bh_set:
                    face_color[i] = bh_face
                    edge_color[i] = self._colors[idx[i]]
                    edge_color[i, 3] = 1.0
                    edge_width[i] = D.BH_EDGE_WIDTH

        return {"face_color": face_color, "edge_color": edge_color,
                "edge_width": edge_width, "size": sizes}

    def _resolve_scaling_mode(self) -> str:
        if self._sizing_absolute:
            return "scene"
        return "visual" if self._depth_scaling else "fixed"

    # ------------------------------------------------------------------
    # Visual construction
    # ------------------------------------------------------------------

    def _build_visuals(self) -> None:
        from vispy import scene

        positions, active_ids, _ = self._interp.evaluate_batch(self._data.times[0])
        if len(positions) == 0:
            positions = np.zeros((1, 3), dtype=np.float32)

        attrs = self._get_particle_attrs(active_ids)
        self._particle_visual = scene.Markers(
            spherical=True, scaling=self._resolve_scaling_mode(),
            light_color=D.LIGHT_COLOR, light_position=D.LIGHT_POSITION,
            light_ambient=D.LIGHT_AMBIENT, parent=self._view.scene,
        )
        self._particle_visual.set_data(pos=positions, **attrs)
        self._update_trails(self._data.times[0], positions, active_ids)

    def _rebuild_particle_visual(self) -> None:
        from vispy import scene

        if self._particle_visual is not None:
            self._particle_visual.parent = None

        self._particle_visual = scene.Markers(
            spherical=True, scaling=self._resolve_scaling_mode(),
            light_color=D.LIGHT_COLOR, light_position=D.LIGHT_POSITION,
            light_ambient=D.LIGHT_AMBIENT, parent=self._view.scene,
        )
        self._particle_visual.alpha = self._point_alpha

        positions, active_ids, _ = self._interp.evaluate_batch(self._current_sim_time)
        if len(positions) > 0:
            attrs = self._get_particle_attrs(active_ids)
            self._particle_visual.set_data(pos=positions, **attrs)

    # ------------------------------------------------------------------
    # Trail rendering
    #
    # Trails are rendered as polylines showing each particle's recent
    # trajectory.  The positions are precomputed at startup (from the
    # simulation's adaptive timesteps + angle-based refinement) and
    # stored in packed arrays.  Each frame, we extract the visible
    # time window [t_trail_start, t_current] via a sliding-window
    # lookup, interpolate smooth boundary points at the window edges,
    # and upload the result to VisPy's Line visual.
    # ------------------------------------------------------------------

    def _update_trails(self, time: float, positions: np.ndarray,
                       active_ids: np.ndarray) -> None:
        from vispy import scene

        # Trail window: [t_trail_start, time] covers trail_length_frac
        # of the total simulation duration
        time_range = self._data.times[-1] - self._data.times[0]
        trail_length = time_range * self._trail_length_frac
        t_trail_start = max(time - trail_length, self._data.times[0])
        if t_trail_start >= time:
            if self._trail_line is not None:
                self._trail_line.visible = False
            return

        precomp = self._precomp
        alpha_table = self._alpha_lut
        max_alpha_index = len(alpha_table) - 1
        n_active = len(active_ids)
        inv_trail_window = 1.0 / (time - t_trail_start)

        # --- Map active particle IDs to precomputed trail segments ---
        particle_indices = self._pid_lookup[active_ids.astype(np.intp)]
        has_precomp = particle_indices >= 0
        segment_starts_all = precomp.offsets[particle_indices[has_precomp]]
        segment_ends_all = precomp.offsets[particle_indices[has_precomp] + 1]
        has_trail_data = segment_starts_all < segment_ends_all

        # Filter to particles that have precomputed trail points
        valid_particle_mask = np.where(has_precomp)[0][has_trail_data]
        n_valid = len(valid_particle_mask)
        if n_valid == 0:
            if self._trail_line is not None:
                self._trail_line.visible = False
            return

        segment_starts = precomp.offsets[particle_indices[valid_particle_mask]]
        segment_ends = precomp.offsets[particle_indices[valid_particle_mask] + 1]
        precomp_counts = segment_ends - segment_starts

        # --- Sliding window: find visible trail range per particle ---
        # During forward playback, the numba function advances pointers
        # by 1-2 positions (O(1)).  On scrub/loop it falls back to
        # binary search (O(log N)).
        trail_particle_idx = particle_indices[valid_particle_mask]
        is_forward = time >= self._trail_prev_time
        self._trail_prev_time = time

        window_starts = np.empty(n_valid, dtype=np.int64)
        window_ends = np.empty(n_valid, dtype=np.int64)

        _advance_trail_pointers(
            precomp.times, segment_starts, segment_ends,
            self._trail_si[trail_particle_idx],
            self._trail_ei[trail_particle_idx],
            t_trail_start, time, is_forward,
            window_starts, window_ends,
        )

        self._trail_si[trail_particle_idx] = window_starts
        self._trail_ei[trail_particle_idx] = window_ends

        body_counts = np.maximum(window_ends - window_starts, 0)
        body_starts = segment_starts + window_starts

        # --- Tail interpolation ---
        # The trail's oldest visible point should be at exactly
        # t_trail_start, not at the nearest precomputed timestamp.
        # We linearly interpolate between the precomputed points
        # straddling t_trail_start so the tail fades in smoothly.
        can_interpolate_tail = (window_starts > 0) & (window_starts <= precomp_counts)
        idx_before_tail = np.maximum(segment_starts + window_starts - 1, segment_starts)
        idx_after_tail = np.minimum(segment_starts + window_starts, segment_ends - 1)
        t_before_tail = precomp.times[idx_before_tail]
        t_after_tail = precomp.times[idx_after_tail]
        tail_time_gap = t_after_tail - t_before_tail
        has_tail = can_interpolate_tail & (tail_time_gap > 0)

        # Lerp factor: 0 = at before point, 1 = at after point
        tail_lerp_factor = np.where(
            has_tail,
            (t_trail_start - t_before_tail) / np.maximum(tail_time_gap, 1e-30),
            0.0,
        )
        tail_positions = (
            precomp.positions[idx_before_tail] * (1 - tail_lerp_factor[:, np.newaxis])
            + precomp.positions[idx_after_tail] * tail_lerp_factor[:, np.newaxis]
        )

        # --- Filter to particles with enough points to draw ---
        # Need at least 2 points (tail/body/head) to render a line segment
        trail_point_counts = body_counts + 1 + has_tail.astype(np.int64)
        drawable = trail_point_counts >= 2
        if not drawable.any():
            if self._trail_line is not None:
                self._trail_line.visible = False
            return

        drawable_idx = np.where(drawable)[0]
        n_trails = len(drawable_idx)
        draw_counts = trail_point_counts[drawable_idx]
        draw_body_starts = body_starts[drawable_idx]
        draw_body_counts = body_counts[drawable_idx]
        draw_has_tail = has_tail[drawable_idx]
        draw_tail_positions = tail_positions[drawable_idx]
        draw_particle_idx = valid_particle_mask[drawable_idx]

        total_pts = int(draw_counts.sum()) + n_trails - 1  # NaN separators

        # --- Reuse pre-allocated GPU arrays (grow if needed) ---
        if total_pts > self._trail_capacity:
            self._trail_capacity = int(total_pts * 1.5)
            self._combined_pos = np.empty((self._trail_capacity, 3), dtype=np.float32)
            self._combined_colors = np.empty((self._trail_capacity, 4), dtype=np.float32)
            self._combined_times = np.empty(self._trail_capacity, dtype=np.float64)
        combined_pos = self._combined_pos[:total_pts]
        combined_colors = self._combined_colors[:total_pts]

        # Output offset per trail: accounts for NaN separators between trails
        write_offsets = np.empty(n_trails, dtype=np.int64)
        write_offsets[0] = 0
        if n_trails > 1:
            cumulative_counts = np.cumsum(draw_counts)
            write_offsets[1:] = cumulative_counts[:-1] + np.arange(1, n_trails)

        # Per-particle color lookup
        particle_color_indices = self._pid_lookup[
            active_ids[draw_particle_idx].astype(np.intp)
        ]

        # Assemble all trail geometry in compiled code (numba)
        combined_times = self._combined_times[:total_pts]
        _assemble_trails(
            combined_pos, combined_colors, combined_times,
            write_offsets, draw_counts, draw_body_starts, draw_body_counts,
            draw_has_tail, draw_tail_positions, draw_particle_idx,
            particle_color_indices,
            precomp.positions, precomp.times, positions, self._colors,
            t_trail_start, time,
        )

        # --- Alpha gradient: older points fade out, newest are opaque ---
        # Each point's alpha is determined by its time position within
        # the trail window, mapped through a power-law curve (t^1.5)
        # stored in the alpha lookup table.
        valid_time_mask = ~np.isnan(combined_times)
        time_fraction = np.zeros(total_pts)
        time_fraction[valid_time_mask] = (
            (combined_times[valid_time_mask] - t_trail_start) * inv_trail_window
        )
        alpha_index = (time_fraction * max_alpha_index).astype(np.intp)
        combined_colors[:, 3] = alpha_table[alpha_index] * self._trail_alpha

        if self._trail_line is not None:
            self._trail_line.set_data(
                pos=combined_pos, color=combined_colors,
                width=self._trail_width,
            )
            self._trail_line.visible = True
        else:
            self._trail_line = scene.Line(
                pos=combined_pos, color=combined_colors,
                parent=self._view.scene, width=self._trail_width,
                antialias=True, connect="strip",
            )

        if self._subview_enabled and self._subview is not None:
            sub_width = max(1.5, self._trail_width * 0.7)
            if self._subview_trail_line is not None:
                self._subview_trail_line.set_data(
                    pos=combined_pos, color=combined_colors,
                    width=sub_width,
                )
                self._subview_trail_line.visible = True
            else:
                self._subview_trail_line = scene.Line(
                    pos=combined_pos, color=combined_colors,
                    parent=self._subview.scene,
                    width=max(1.5, self._trail_width * 0.7),
                    antialias=True, connect="strip",
                )

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def _on_timer(self, event) -> None:
        if self._playing:
            dt = event.dt if hasattr(event, "dt") else 1.0 / 60.0
            self._anim_time += self._anim_speed * dt
            if self._anim_time > 1.0:
                self._anim_time = 0.0
                self._trail_si[:] = -1
                self._trail_ei[:] = -1
            self._current_sim_time = self._t_min + self._anim_time * self._time_range
        self._apply_keyboard_pan()
        self._update_frame()

    def _update_light_direction(self) -> None:
        """Transform the world-space light direction into eye space.

        VisPy's Markers shader expects the light direction in eye space
        (camera-relative coordinates).  By transforming a fixed world-space
        light direction through the camera's view rotation each frame,
        particles get consistent directional lighting: the bright side
        faces the light and the dark side faces away, regardless of
        camera angle.  This gives strong depth cues when orbiting.
        """
        import math
        camera = self._view.camera
        world_light_dir = np.array(D.LIGHT_POSITION, dtype=np.float64)
        world_light_dir /= np.linalg.norm(world_light_dir)

        # Rotate world light direction by the inverse of the camera's
        # azimuth and elevation to get eye-space direction
        az = math.radians(-camera.azimuth)
        el = math.radians(-camera.elevation)

        # Azimuth rotation (around Z axis)
        cos_az, sin_az = math.cos(az), math.sin(az)
        x = world_light_dir[0] * cos_az - world_light_dir[1] * sin_az
        y = world_light_dir[0] * sin_az + world_light_dir[1] * cos_az
        z = world_light_dir[2]

        # Elevation rotation (around X axis)
        cos_el, sin_el = math.cos(el), math.sin(el)
        eye_y = y * cos_el - z * sin_el
        eye_z = y * sin_el + z * cos_el

        eye_light = np.array([x, eye_y, eye_z])
        eye_light /= np.linalg.norm(eye_light)

        if self._particle_visual is not None:
            self._particle_visual.light_position = tuple(eye_light)
        if self._subview_markers is not None:
            self._subview_markers.light_position = tuple(eye_light)

    def _update_frame(self) -> None:
        positions, active_ids, _ = self._interp.evaluate_batch(self._current_sim_time)
        if len(positions) == 0:
            self._canvas.update()
            return

        self._update_light_direction()
        attrs = self._get_particle_attrs(active_ids)

        if self._particle_visual is not None:
            self._particle_visual.set_data(pos=positions, **attrs)

        self._update_trails(self._current_sim_time, positions, active_ids)

        if self._subview_enabled and self._subview_markers is not None:
            self._subview_markers.set_data(
                pos=positions, face_color=attrs["face_color"],
                size=attrs["size"] * 0.7, edge_color=attrs["edge_color"],
                edge_width=attrs["edge_width"],
            )
            if self._subview_canvas is not None:
                self._subview_canvas.update()

        if self._camera_controller is not None:
            self._camera_controller.update(self._current_sim_time, positions, active_ids)
        if self._subview_camera_controller is not None and self._subview_enabled:
            self._subview_camera_controller.update(self._current_sim_time, positions, active_ids)

        self._canvas.update()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def canvas(self):
        return self._canvas

    @property
    def view(self):
        return self._view

    @property
    def sim_time(self) -> float:
        return self._current_sim_time

    @sim_time.setter
    def sim_time(self, value: float) -> None:
        self._current_sim_time = np.clip(value, self._t_min, self._t_max)
        self._anim_time = (self._current_sim_time - self._t_min) / self._time_range
        self._update_frame()

    @property
    def playing(self) -> bool:
        return self._playing

    def play(self) -> None:
        self._playing = True
        self._timer.start()

    def pause(self) -> None:
        self._playing = False

    def toggle_play(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def set_speed(self, speed: float) -> None:
        self._anim_speed = max(0.001, speed)

    def set_particle_color(self, pid: int, rgba: tuple[float, ...]) -> None:
        idx = self._id_to_idx.get(pid)
        if idx is not None:
            self._colors[idx] = np.array(rgba, dtype=np.float32)

    def set_particle_size(self, pid: int, size: float) -> None:
        idx = self._id_to_idx.get(pid)
        if idx is not None:
            self._per_particle_scale[idx] = size
            self._recompute_sizes()

    def set_radius_scale(self, scale: float) -> None:
        self._radius_scale = max(0.01, scale)
        self._recompute_sizes()

    def set_sizing_mode(self, absolute: bool) -> None:
        if absolute == self._sizing_absolute:
            return
        self._sizing_absolute = absolute
        self._recompute_sizes()
        self._rebuild_particle_visual()

    def set_depth_scaling(self, enabled: bool) -> None:
        if enabled == self._depth_scaling:
            return
        self._depth_scaling = enabled
        self._rebuild_particle_visual()

    def set_black_hole(self, pid: int, is_bh: bool = True) -> None:
        if is_bh:
            self._bh_set.add(pid)
        else:
            self._bh_set.discard(pid)

    def set_trail_length(self, frac: float) -> None:
        self._trail_length_frac = np.clip(frac, 0.0, 1.0)
        # Reset two-pointer cache — trail window bounds changed
        self._trail_si[:] = -1
        self._trail_ei[:] = -1

    def set_trail_alpha(self, alpha: float) -> None:
        self._trail_alpha = np.clip(alpha, 0.0, 1.0)

    def set_trail_width(self, width: float) -> None:
        self._trail_width = max(1.0, width)
        if self._trail_line is not None:
            self._trail_line.set_data(width=self._trail_width)
        if self._subview_trail_line is not None:
            self._subview_trail_line.set_data(width=max(1.5, self._trail_width * 0.7))

    def set_point_alpha(self, alpha: float) -> None:
        self._point_alpha = np.clip(alpha, 0.0, 1.0)
        if self._particle_visual is not None:
            self._particle_visual.alpha = self._point_alpha
        if self._subview_markers is not None:
            self._subview_markers.alpha = self._point_alpha

    def set_camera_controller(self, controller) -> None:
        self._camera_controller = controller

    # ------------------------------------------------------------------
    # Sub-view
    # ------------------------------------------------------------------

    def enable_subview(self, corner: str = "bottom-right", size_frac: float = 0.3) -> None:
        from vispy import scene

        if self._subview_canvas is not None:
            return

        main_widget = self._canvas.native
        w, h = main_widget.width(), main_widget.height()
        sw, sh = int(w * size_frac), int(h * size_frac)
        margin = 10
        corners = {
            "bottom-right": (w - sw - margin, h - sh - margin),
            "bottom-left": (margin, h - sh - margin),
            "top-right": (w - sw - margin, margin),
            "top-left": (margin, margin),
        }
        x, y = corners.get(corner, corners["bottom-right"])

        self._subview_canvas = scene.SceneCanvas(keys=None, show=False, parent=main_widget)
        self._subview_canvas.native.setFixedSize(sw, sh)
        self._subview_canvas.native.move(x, y)
        self._subview_canvas.native.setStyleSheet("border: 2px solid white; background: black;")

        self._subview = self._subview_canvas.central_widget.add_view()
        self._subview.camera = scene.cameras.TurntableCamera(
            fov=D.SUBVIEW_FOV, distance=self._compute_initial_distance() * 1.5,
        )

        from ..core.camera import CameraController, CameraMode
        self._subview_camera_controller = CameraController(self._subview, masses=self._data.masses)
        self._subview_camera_controller.mode = CameraMode.AUTO_FRAME

        positions, active_ids, _ = self._interp.evaluate_batch(self._current_sim_time)
        if len(positions) == 0:
            positions = np.zeros((1, 3), dtype=np.float32)
            active_ids = np.array([0])

        attrs = self._get_particle_attrs(active_ids)
        self._subview_markers = scene.Markers(
            spherical=True, light_color=D.LIGHT_COLOR,
            light_position=D.LIGHT_POSITION, light_ambient=D.LIGHT_AMBIENT,
            parent=self._subview.scene,
        )
        self._subview_markers.set_data(
            pos=positions, face_color=attrs["face_color"],
            size=attrs["size"] * 0.7, edge_color=attrs["edge_color"],
            edge_width=attrs["edge_width"],
        )

        self._subview_canvas.native.show()
        self._subview_enabled = True

    def disable_subview(self) -> None:
        if self._subview_canvas is not None:
            self._subview_canvas.native.hide()
            self._subview_canvas.native.setParent(None)
            self._subview_canvas = None
            self._subview = None
            self._subview_markers = None
            self._subview_camera_controller = None
            self._subview_trail_line = None
            self._subview_enabled = False

    # ------------------------------------------------------------------
    # Display / export
    # ------------------------------------------------------------------

    def show(self) -> None:
        from vispy import app
        self._canvas.show()
        self._timer.start()
        app.run()

    def screenshot(self, filepath: str | Path, size: tuple[int, int] | None = None) -> None:
        from vispy import io
        img = self._canvas.render(size=size)
        io.write_png(str(filepath), img)

    def render_video(
        self, filepath: str | Path, duration: float = D.VIDEO_DURATION,
        fps: int = D.VIDEO_FPS, size: tuple[int, int] | None = None,
    ) -> None:
        import imageio.v3 as iio

        n_frames = int(duration * fps)
        frame_sim_times = np.linspace(self._t_min, self._t_max, n_frames)
        filepath = Path(filepath)
        render_size = size or self._canvas.size

        with iio.imopen(str(filepath), "w") as writer:
            for i, t in enumerate(frame_sim_times):
                self._current_sim_time = t
                self._anim_time = (t - self._t_min) / self._time_range
                self._update_frame()

                img = self._canvas.render(size=render_size)
                writer.write(img, plugin="pyav", codec="libx264", fps=fps)

                if (i + 1) % (n_frames // 10 or 1) == 0:
                    print(f"Rendering: {(i + 1) / n_frames * 100:.0f}%")

        print(f"Video saved to {filepath}")
