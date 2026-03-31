"""VisPy-based OpenGL rendering engine for N-body visualization."""

from __future__ import annotations

import math
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
    particle_indices,         # (n_trails,) int64 — index into full positions array
    trail_is_active,          # (n_trails,) bool — whether particle is currently active
    particle_color_indices,   # (n_trails,) int64 — index into color table
    precomp_positions,        # (total_precomp, 3) float32 — packed precomputed positions
    precomp_times,            # (total_precomp,) float64 — packed precomputed times
    live_positions,           # (n_particles, 3) float64 — full-size positions (NaN if inactive)
    color_table,              # (n_particles, 4) float32 — per-particle RGBA
    t_trail_start,            # float64 — oldest time in the trail window
    t_current,                # float64 — current simulation time
):
    """Write trail geometry into output buffers for GPU upload.

    Each trail is laid out as: [NaN separator] [tail lerp] [body] [head]
    - Tail: interpolated position at exactly t_trail_start (smooth fade-in)
    - Body: precomputed positions between t_trail_start and t_current
    - Head: live particle position at t_current (only for active particles)
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

        # Head: current particle position (only for active particles)
        if trail_is_active[trail]:
            p_idx = particle_indices[trail]
            for dim in range(3):
                out_positions[offset + write_pos, dim] = np.float32(live_positions[p_idx, dim])
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
        """Create the rendering engine.

        Args:
            data: Loaded simulation data (positions, times, masses, etc.).
            interpolator: Pre-built TrajectoryInterpolator for spline evaluation.
            size: Initial canvas size as (width, height) in pixels.
            title: Window title string.
        """
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

        # Manual control speeds (WASD pan and scroll zoom)
        self._pan_speed = 0.02       # fraction of camera distance per frame
        self._zoom_speed = 1.0       # scroll zoom multiplier (1.0 = default)

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

        self._id_to_idx = {int(pid): i for i, pid in enumerate(data.particle_ids)}

        self._time_unit = D.UNIT_TIME

        # Time overlay
        self._time_display_enabled = True
        self._time_font_size = D.TIME_FONT_SIZE
        self._time_color = D.TIME_COLOR
        self._time_anchor = D.TIME_ANCHOR
        self._time_text = None  # VisPy Text visual, created in _build_visuals

        # Star field background
        self._stars_enabled = True
        self._star_visual = None
        self._star_directions = None   # (N, 3) float32 unit vectors, fixed
        self._star_positions = None    # (N, 3) float32 working buffer, scaled each frame
        self._star_base_sizes = None   # (N,) float32
        self._star_base_colors = None  # (N, 4) float32
        self._star_max_particle_dist = 1.0  # max distance any particle reaches from origin
        self._star_shell_factor = D.STAR_SHELL_FACTOR
        self._star_min_shell_radius = 1.0  # computed as max_dist * factor

        # Black hole rendering — pre-computed boolean array for vectorized lookup
        self._is_bh = np.zeros(n, dtype=bool)
        if data.startypes is not None:
            for pid_key, k in data.startypes.items():
                if k == D.BH_STARTYPE:
                    idx = self._id_to_idx.get(pid_key)
                    if idx is not None:
                        self._is_bh[idx] = True

        # VisPy canvas and view
        self._canvas = scene.SceneCanvas(
            keys="interactive", size=size, title=title, show=False,
        )
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = scene.cameras.TurntableCamera(
            fov=D.CAMERA_FOV, distance=10.0,
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
        self._trail_line = None
        self._subview_trail_line = None

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
        self._time_fraction = np.empty(0, dtype=np.float32)

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

        # Keyboard pan/rotate: track which keys are held for smooth continuous movement
        self._pan_keys_held: set[str] = set()
        self._ctrl_held = False
        self._alt_held = False
        self._shift_held = False
        self._manual_mode_callbacks: list = []  # called when arrow-key pan forces manual mode

        # Cached camera trig — computed once per frame in _update_frame,
        # reused by lighting and panning.  Physics spherical convention:
        # θ = π/2 − elevation (polar from +z),  φ = azimuth.
        # VisPy axis mapping:  x = r sinθ sinφ,  y = r sinθ cosφ,  z = r cosθ.
        self._cos_az = 1.0
        self._sin_az = 0.0
        self._cos_el = 1.0
        self._sin_el = 0.0

        # Normalized world-space light direction (constant, precomputed once)
        _lw = np.array(D.LIGHT_POSITION, dtype=np.float64)
        self._light_world = _lw / np.linalg.norm(_lw)

    def _cache_camera_trig(self) -> None:
        """Precompute sin/cos of camera azimuth and elevation."""
        az = math.radians(self._view.camera.azimuth)
        el = math.radians(self._view.camera.elevation)
        self._cos_az = math.cos(az)
        self._sin_az = math.sin(az)
        self._cos_el = math.cos(el)
        self._sin_el = math.sin(el)

    def _camera_axes(self):
        """Camera-relative right and forward vectors from cached trig."""
        return (
            np.array([self._cos_az, self._sin_az, 0.0]),
            np.array([-self._sin_az, self._cos_az, 0.0]),
        )

    @staticmethod
    def _has_ctrl(event) -> bool:
        """Check if Ctrl is held during a VisPy input event.

        Args:
            event: VisPy input event with a `modifiers` attribute.
        """
        return any(m.name.lower() == 'control' for m in event.modifiers)

    def _on_mouse_wheel(self, event) -> None:
        """Zoom toward the mouse cursor position.

        Shifts the camera center toward the world-space point under the
        cursor as it zooms in (like Google Maps).  The cursor offset
        from screen center is converted to world units using the camera
        distance and field of view, then the center is nudged by that
        offset proportional to the zoom amount.

        Holding Ctrl multiplies zoom speed by 5x.

        Args:
            event: VisPy mouse wheel event with `delta` and `pos` attributes.
        """
        if self._camera_controller is not None and not self._camera_controller.free_zoom:
            self._camera_controller.free_zoom = True

        camera = self._view.camera

        scroll = event.delta[1]
        if scroll == 0:
            return

        speed = self._zoom_speed * (5.0 if self._has_ctrl(event) else 1.0)
        # 1.1^(-scroll*speed): scroll up → zoom_factor < 1 → closer
        zoom_factor = 1.1 ** (-scroll * speed)

        # Shift center toward cursor: convert the cursor's pixel offset
        # from screen center into world-space displacement at the focal
        # plane (the plane through camera.center).
        mouse_pos = event.pos
        canvas_size = self._canvas.size
        if mouse_pos is not None:
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

            # Rotate screen offset into world coordinates
            self._cache_camera_trig()
            right, _ = self._camera_axes()
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
        """Track held keys for smooth continuous panning/rotation.

        Args:
            event: VisPy key press event with `key.name` attribute.
        """
        key_name = event.key.name
        if key_name == 'Space':
            self.toggle_play()
            event.handled = True
            return
        if key_name in self._PAN_KEYS:
            self._pan_keys_held.add(key_name)
            # Switch to manual mode so auto-tracking doesn't fight the pan
            if not self._alt_held and self._camera_controller is not None:
                from ..core.camera import CameraMode
                if self._camera_controller.mode != CameraMode.MANUAL:
                    self._camera_controller.mode = CameraMode.MANUAL
                    for cb in self._manual_mode_callbacks:
                        cb()
                if not self._camera_controller.free_zoom:
                    self._camera_controller.free_zoom = True
        if key_name == 'Control':
            self._ctrl_held = True
        if key_name == 'Alt':
            self._alt_held = True
        if key_name == 'Shift':
            self._shift_held = True

    def _on_key_release(self, event) -> None:
        """Stop panning/rotation when key is released.

        Args:
            event: VisPy key release event with `key.name` attribute.
        """
        key_name = event.key.name
        self._pan_keys_held.discard(key_name)
        if key_name == 'Control':
            self._ctrl_held = False
        if key_name == 'Alt':
            self._alt_held = False
        if key_name == 'Shift':
            self._shift_held = False

    def _apply_keyboard_pan(self) -> None:
        """Apply smooth continuous panning and time scrubbing from held keys.

        Called each frame from _on_timer.  Pan speed is proportional to
        camera distance so it feels consistent at any zoom level.
        Holding Ctrl multiplies speed by 5x.  Skipped when Alt is held
        (arrow keys rotate instead).

        Default:
            Up/Down — move up/down (world Z)
            Left/Right, A/D — move left/right
            W/S — move forward/backward

        Shift held:
            Up/Down — move forward/backward
            Left/Right — scrub time forward/backward
        """
        if not self._pan_keys_held or self._alt_held:
            return

        camera = self._view.camera
        speed = self._pan_speed * (5.0 if self._ctrl_held else 1.0)
        step = camera.distance * speed

        right, forward = self._camera_axes()
        up = np.array([0.0, 0.0, 1.0])

        center = np.array(camera.center, dtype=np.float64)
        time_step = 0.0

        for key_name in self._pan_keys_held:
            if self._shift_held:
                # Shift: Up/Down = forward/backward, Left/Right = time scrub
                if key_name in ('Up',):
                    center += forward * step
                elif key_name in ('Down',):
                    center -= forward * step
                elif key_name in ('Left',):
                    time_step -= 1.0
                elif key_name in ('Right',):
                    time_step += 1.0
            else:
                # Default: Up/Down = vertical, Left/Right = horizontal
                if key_name in ('Up',):
                    center += up * step
                elif key_name in ('Down',):
                    center -= up * step
                elif key_name in ('Left',):
                    center -= right * step
                elif key_name in ('Right',):
                    center += right * step

            # WASD always pan (unaffected by Shift)
            if key_name == 'W':
                center += forward * step
            elif key_name == 'S':
                center -= forward * step
            elif key_name == 'A':
                center -= right * step
            elif key_name == 'D':
                center += right * step

        camera.center = tuple(center)

        if time_step != 0.0:
            scrub_speed = self._anim_speed * (5.0 if self._ctrl_held else 1.0)
            self._anim_time = max(0.0, min(1.0, self._anim_time + time_step * scrub_speed / 60.0))
            self._current_sim_time = self._t_min + self._anim_time * self._time_range

    _ROTATE_SPEED_DEG = 1.5  # degrees per frame for keyboard rotation

    def _apply_keyboard_rotate(self) -> None:
        """Apply smooth continuous orbit rotation from held arrow keys + Alt.

        Called each frame from _on_timer.  Left/Right change azimuth,
        Up/Down change elevation.  Holding Ctrl multiplies speed by 5x.
        """
        if not self._pan_keys_held or not self._alt_held:
            return

        camera = self._view.camera
        speed = self._ROTATE_SPEED_DEG * (5.0 if self._ctrl_held else 1.0)

        for key_name in self._pan_keys_held:
            if key_name in ('A', 'Left'):
                camera.azimuth -= speed
            elif key_name in ('D', 'Right'):
                camera.azimuth += speed
            elif key_name in ('W', 'Up'):
                camera.elevation = min(90.0, camera.elevation + speed)
            elif key_name in ('S', 'Down'):
                camera.elevation = max(-90.0, camera.elevation - speed)

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def _compute_relative_base_sizes(self) -> np.ndarray:
        """Map particle radii to screen-pixel sizes.

        Cube root compresses the dynamic range (proportional to volume^(1/3)),
        then linearly maps [0, max] → [MIN_PX, MAX_PX].
        """
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
        """Map particle radii to world-unit diameters.

        If radii are provided, diameter = 2 * radius.
        Otherwise, default to 1% of the simulation spatial extent.
        """
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
        """Generate a default RGBA color palette cycling through 8 distinct colors.

        Args:
            n: Number of particles to assign colors to.

        Returns:
            (n, 4) float32 RGBA array.
        """
        base = np.array([
            [1.0, 0.3, 0.3, 1.0], [0.3, 0.6, 1.0, 1.0],
            [0.3, 1.0, 0.3, 1.0], [1.0, 0.8, 0.2, 1.0],
            [0.8, 0.3, 1.0, 1.0], [1.0, 0.5, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0], [1.0, 0.4, 0.7, 1.0],
        ], dtype=np.float32)
        idx = np.arange(n) % len(base)
        return base[idx]

    def _generate_star_field(self) -> None:
        """Generate random star directions, sizes, and colors.

        Stores unit vectors on the sphere.  Each frame, these are scaled
        to a dynamic shell radius that tracks the camera distance so the
        stars always remain within the viewing frustum.
        """
        rng = np.random.default_rng(D.STAR_SEED)
        n = D.STAR_COUNT

        # Uniform unit vectors on the sphere
        z = rng.uniform(-1, 1, n)
        phi = rng.uniform(0, 2 * np.pi, n)
        r_xy = np.sqrt(1 - z ** 2)
        self._star_directions = np.column_stack([
            r_xy * np.cos(phi), r_xy * np.sin(phi), z,
        ]).astype(np.float32)

        # Shell radius: maximum finite distance any particle reaches from the
        # origin across the entire simulation, times a safety factor.
        max_dist = 0.0
        for pos in self._data.positions.values():
            finite = np.isfinite(pos).all(axis=1)
            if finite.any():
                d = np.linalg.norm(pos[finite], axis=1).max()
                if d > max_dist:
                    max_dist = d
        self._star_max_particle_dist = max_dist
        self._star_min_shell_radius = max(max_dist * self._star_shell_factor, 1.0)


        # Sizes: power-law — many dim, few bright
        u = rng.uniform(0, 1, n)
        self._star_base_sizes = (D.STAR_BASE_SIZE * (1.0 + 3.0 * u ** 3)).astype(np.float32)

        # Colors: rough main-sequence distribution
        colors = np.ones((n, 4), dtype=np.float32)
        idx = rng.choice(n, int(0.3 * n), replace=False)
        colors[idx, 0] *= rng.uniform(0.7, 0.9, len(idx)).astype(np.float32)
        colors[idx, 1] *= rng.uniform(0.8, 0.95, len(idx)).astype(np.float32)
        idx = rng.choice(n, int(0.1 * n), replace=False)
        colors[idx, 2] *= rng.uniform(0.4, 0.7, len(idx)).astype(np.float32)
        idx = rng.choice(n, int(0.05 * n), replace=False)
        colors[idx, 1] *= rng.uniform(0.3, 0.6, len(idx)).astype(np.float32)
        colors[idx, 2] *= rng.uniform(0.1, 0.3, len(idx)).astype(np.float32)
        brightness = rng.uniform(0.3, 1.0, n).astype(np.float32)
        colors[:, :3] *= brightness[:, np.newaxis]
        colors[:, 3] = brightness
        self._star_base_colors = colors

        # Pre-allocated working buffers (reused each frame)
        self._star_positions = np.empty_like(self._star_directions)

    def _get_particle_attrs(self, mask: np.ndarray) -> dict:
        """Compute face_color, edge_color, edge_width, size for active particles.

        Args:
            mask: (n_particles,) bool — True for active particles.
        """
        n = int(mask.sum())
        sizes = self._sizes[mask]
        face_color = self._colors[mask]
        edge_color = np.zeros((n, 4), dtype=np.float32)
        edge_width = np.zeros(n, dtype=np.float32)

        bh_mask = self._is_bh[mask]
        if bh_mask.any():
            face_color = face_color.copy()
            face_color[bh_mask] = np.array(D.BH_FACE_COLOR, dtype=np.float32)
            edge_color[bh_mask] = self._colors[mask][bh_mask]
            edge_color[bh_mask, 3] = 1.0
            edge_width[bh_mask] = D.BH_EDGE_WIDTH

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

        positions, mask = self._interp.evaluate_batch(self._data.times[0])
        active_pos = positions[mask] if mask.any() else np.zeros((1, 3), dtype=np.float32)

        attrs = self._get_particle_attrs(mask)
        self._particle_visual = scene.Markers(
            spherical=True, scaling=self._resolve_scaling_mode(),
            light_color=D.LIGHT_COLOR, light_position=D.LIGHT_POSITION,
            light_ambient=D.LIGHT_AMBIENT, parent=self._view.scene,
        )
        self._particle_visual.set_data(pos=active_pos, **attrs)
        self._update_trails(self._data.times[0], positions, mask)

        # Generate star field and create visual if enabled by default
        self._generate_star_field()
        if self._stars_enabled:
            self.enable_stars(True)

        # Time overlay — attached to the canvas root so it stays fixed on screen
        self._time_text = scene.visuals.Text(
            text=D.format_sim_time(self._t_min, self._time_unit),
            color=self._time_color,
            font_size=self._time_font_size,
            pos=D.TIME_OFFSET,
            anchor_x="left",
            anchor_y="top",
            parent=self._canvas.scene,
        )
        self._time_text.order = 10
        self._time_text.visible = self._time_display_enabled

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

        positions, mask = self._interp.evaluate_batch(self._current_sim_time)
        if mask.any():
            attrs = self._get_particle_attrs(mask)
            self._particle_visual.set_data(pos=positions[mask], **attrs)

    def _hide_all_trails(self) -> None:
        """Hide all trail visuals."""
        if self._trail_line is not None:
            self._trail_line.visible = False
        if self._subview_trail_line is not None:
            self._subview_trail_line.visible = False

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
                       mask: np.ndarray) -> None:
        """Extract visible trail windows and upload geometry to VisPy.

        Args:
            time: Current simulation time.
            positions: (n_particles, 3) full-size position array (NaN for inactive).
            mask: (n_particles,) bool — True where particle is active.
        """
        from vispy import scene

        # Trail window: [t_trail_start, time] covers trail_length_frac
        # of the total simulation duration
        trail_length = self._time_range * self._trail_length_frac
        t_trail_start = max(time - trail_length, self._t_min)
        if t_trail_start >= time:
            self._hide_all_trails()
            return

        precomp = self._precomp
        alpha_table = self._alpha_lut
        max_alpha_index = len(alpha_table) - 1
        n_particles = self._n_particles
        inv_trail_window = 1.0 / (time - t_trail_start)

        # --- Check ALL particles for visible trail data ---
        # (not just active ones — disappeared particles keep their trails
        # until the trail window slides past their last precomputed point)
        all_indices = np.arange(n_particles)
        segment_starts = precomp.offsets[:-1]
        segment_ends = precomp.offsets[1:]
        has_trail_data = segment_starts < segment_ends

        valid_indices = all_indices[has_trail_data]
        n_valid = len(valid_indices)
        if n_valid == 0:
            self._hide_all_trails()
            return

        seg_starts = segment_starts[valid_indices]
        seg_ends = segment_ends[valid_indices]
        precomp_counts = seg_ends - seg_starts

        # --- Sliding window: find visible trail range per particle ---
        is_forward = time >= self._trail_prev_time
        self._trail_prev_time = time

        window_starts = np.empty(n_valid, dtype=np.int64)
        window_ends = np.empty(n_valid, dtype=np.int64)

        _advance_trail_pointers(
            precomp.times, seg_starts, seg_ends,
            self._trail_si[valid_indices],
            self._trail_ei[valid_indices],
            t_trail_start, time, is_forward,
            window_starts, window_ends,
        )

        self._trail_si[valid_indices] = window_starts
        self._trail_ei[valid_indices] = window_ends

        body_counts = np.maximum(window_ends - window_starts, 0)
        body_starts = seg_starts + window_starts

        # --- Tail interpolation ---
        can_interpolate_tail = (window_starts > 0) & (window_starts <= precomp_counts)
        idx_before_tail = np.maximum(seg_starts + window_starts - 1, seg_starts)
        idx_after_tail = np.minimum(seg_starts + window_starts, seg_ends - 1)
        t_before_tail = precomp.times[idx_before_tail]
        t_after_tail = precomp.times[idx_after_tail]
        tail_time_gap = t_after_tail - t_before_tail
        has_tail = can_interpolate_tail & (tail_time_gap > 0)

        # Lerp factor: 0.0 = at before point, 1.0 = at after point.
        # 1e-30 prevents division by zero when times are identical.
        tail_lerp_factor = np.where(
            has_tail,
            (t_trail_start - t_before_tail) / np.maximum(tail_time_gap, 1e-30),
            0.0,
        )
        tail_positions = (
            precomp.positions[idx_before_tail] * (1 - tail_lerp_factor[:, np.newaxis])
            + precomp.positions[idx_after_tail] * tail_lerp_factor[:, np.newaxis]
        )

        # --- Determine which trails have an active head ---
        # Active particles get a live head position from the spline;
        # inactive particles' trails end at the last precomputed point.
        is_active = mask[valid_indices]

        # Trail point count: body + head (if active) + tail (if interpolated)
        trail_point_counts = (
            body_counts
            + is_active.astype(np.int64)
            + has_tail.astype(np.int64)
        )
        drawable = trail_point_counts >= 2
        if not drawable.any():
            self._hide_all_trails()
            return

        drawable_idx = np.where(drawable)[0]
        n_trails = len(drawable_idx)
        draw_counts = trail_point_counts[drawable_idx]
        draw_body_starts = body_starts[drawable_idx]
        draw_body_counts = body_counts[drawable_idx]
        draw_has_tail = has_tail[drawable_idx]
        draw_tail_positions = tail_positions[drawable_idx]
        draw_particle_idx = valid_indices[drawable_idx]
        draw_is_active = is_active[drawable_idx]

        total_pts = int(draw_counts.sum()) + n_trails - 1  # NaN separators

        # --- Reuse pre-allocated GPU arrays (grow if needed) ---
        if total_pts > self._trail_capacity:
            self._trail_capacity = int(total_pts * 1.5)
            self._combined_pos = np.empty((self._trail_capacity, 3), dtype=np.float32)
            self._combined_colors = np.empty((self._trail_capacity, 4), dtype=np.float32)
            self._combined_times = np.empty(self._trail_capacity, dtype=np.float64)
            self._time_fraction = np.empty(self._trail_capacity, dtype=np.float32)
        combined_pos = self._combined_pos[:total_pts]
        combined_colors = self._combined_colors[:total_pts]

        # Output offset per trail: each trail starts after the previous
        # trail's points + 1 NaN separator (used by VisPy to break the line strip)
        write_offsets = np.empty(n_trails, dtype=np.int64)
        write_offsets[0] = 0
        if n_trails > 1:
            cumulative_counts = np.cumsum(draw_counts)
            # +arange(1,n) adds one NaN separator per preceding trail
            write_offsets[1:] = cumulative_counts[:-1] + np.arange(1, n_trails)

        # Assemble all trail geometry in compiled code (numba)
        combined_times = self._combined_times[:total_pts]
        _assemble_trails(
            combined_pos, combined_colors, combined_times,
            write_offsets, draw_counts, draw_body_starts, draw_body_counts,
            draw_has_tail, draw_tail_positions, draw_particle_idx,
            draw_is_active, draw_particle_idx,  # color_indices = particle indices
            precomp.positions, precomp.times, positions, self._colors,
            t_trail_start, time,
        )

        # --- Alpha gradient: older points fade out, newest are opaque ---
        valid_time_mask = ~np.isnan(combined_times)
        time_fraction = self._time_fraction[:total_pts]
        time_fraction[:] = 0.0
        time_fraction[valid_time_mask] = (
            (combined_times[valid_time_mask] - t_trail_start) * inv_trail_window
        )
        alpha_index = np.clip(
            (time_fraction * max_alpha_index).astype(np.intp), 0, max_alpha_index,
        )
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
                    parent=self._subview.scene, width=sub_width,
                    antialias=True, connect="strip",
                )

    # ------------------------------------------------------------------
    # Star field
    # ------------------------------------------------------------------

    def _update_stars(self) -> None:
        """Update star field positions on the fixed shell.

        Called only when stars are enabled and visual exists.
        """
        np.multiply(self._star_directions, self._star_min_shell_radius,
                    out=self._star_positions)
        self._star_visual.set_data(
            pos=self._star_positions, face_color=self._star_base_colors,
            size=self._star_base_sizes, edge_width=0,
        )

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def _on_timer(self, event) -> None:
        """Advance animation and render a frame.

        Args:
            event: VisPy timer event with `dt` attribute (seconds since last tick).
        """
        if self._playing:
            dt = event.dt if hasattr(event, "dt") else 1.0 / 60.0
            self._anim_time += self._anim_speed * dt
            if self._anim_time > 1.0:
                self._anim_time = 0.0
                self._trail_si[:] = -1
                self._trail_ei[:] = -1
            self._current_sim_time = self._t_min + self._anim_time * self._time_range
        self._apply_keyboard_pan()
        self._apply_keyboard_rotate()
        self._update_frame()

    def _update_light_direction(self) -> None:
        """Transform the precomputed world-space light direction into eye space
        using cached camera trig (inverse rotation: cos(-φ)=cosφ, sin(-φ)=-sinφ).
        """
        w = self._light_world
        x = w[0] * self._cos_az + w[1] * self._sin_az
        y = -w[0] * self._sin_az + w[1] * self._cos_az
        z = w[2]
        eye_y = y * self._cos_el + z * self._sin_el
        eye_z = -y * self._sin_el + z * self._cos_el
        inv_norm = 1.0 / math.sqrt(x * x + eye_y * eye_y + eye_z * eye_z)
        light = (x * inv_norm, eye_y * inv_norm, eye_z * inv_norm)

        if self._particle_visual is not None:
            self._particle_visual.light_position = light
        if self._subview_markers is not None:
            self._subview_markers.light_position = light

    def _update_frame(self) -> None:
        self._cache_camera_trig()

        positions, mask = self._interp.evaluate_batch(self._current_sim_time)
        if not mask.any():
            self._canvas.update()
            return

        self._update_light_direction()
        active_pos = positions[mask]
        attrs = self._get_particle_attrs(mask)

        # Camera update before GPU upload so tracking drives this frame's view
        if self._camera_controller is not None:
            self._camera_controller.update(self._current_sim_time, positions, mask)
        if self._subview_camera_controller is not None and self._subview_enabled:
            self._subview_camera_controller.update(self._current_sim_time, positions, mask)

        if self._particle_visual is not None:
            self._particle_visual.set_data(pos=active_pos, **attrs)

        self._update_trails(self._current_sim_time, positions, mask)

        # Star field
        if self._stars_enabled and self._star_visual is not None:
            self._update_stars()

        # Time overlay — only update the VisPy text when the display string
        # actually changes (avoids per-frame texture re-render)
        if self._time_text is not None and self._time_display_enabled:
            text = D.format_sim_time(self._current_sim_time, self._time_unit)
            if text != self._time_text.text:
                self._time_text.text = text

        if self._subview_enabled and self._subview_markers is not None:
            self._subview_markers.set_data(
                pos=active_pos, face_color=attrs["face_color"],
                size=attrs["size"] * 0.7, edge_color=attrs["edge_color"],
                edge_width=attrs["edge_width"],
            )
            if self._subview_canvas is not None:
                self._subview_canvas.update()

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
        """Set playback speed.

        Args:
            speed: Fraction of total simulation duration advanced per real second.
        """
        self._anim_speed = max(0.001, speed)

    def set_particle_color(self, pid: int, rgba: tuple[float, ...]) -> None:
        """Set the color of a single particle.

        Args:
            pid: Integer particle ID.
            rgba: (R, G, B, A) color tuple with values in [0, 1].
        """
        idx = self._id_to_idx.get(pid)
        if idx is not None:
            self._colors[idx] = np.array(rgba, dtype=np.float32)

    def set_particle_size(self, pid: int, size: float) -> None:
        """Set the per-particle size multiplier.

        Args:
            pid: Integer particle ID.
            size: Scale factor relative to the base size (1.0 = default).
        """
        idx = self._id_to_idx.get(pid)
        if idx is not None:
            self._per_particle_scale[idx] = size
            self._recompute_sizes()

    def set_radius_scale(self, scale: float) -> None:
        """Set the global radius multiplier applied to all particles.

        Args:
            scale: Multiplier for particle sizes (clamped to >= 0.01).
        """
        self._radius_scale = max(0.01, scale)
        self._recompute_sizes()

    def set_sizing_mode(self, absolute: bool) -> None:
        """Switch between absolute (world-unit) and relative (pixel) sizing.

        Args:
            absolute: If True, particle sizes are in world units (same as x,y,z).
                If False, sizes are in screen pixels.
        """
        if absolute == self._sizing_absolute:
            return
        self._sizing_absolute = absolute
        self._recompute_sizes()
        self._rebuild_particle_visual()

    def set_depth_scaling(self, enabled: bool) -> None:
        """Toggle perspective depth scaling on particle markers.

        Args:
            enabled: If True, closer particles appear larger (perspective projection).
        """
        if enabled == self._depth_scaling:
            return
        self._depth_scaling = enabled
        self._rebuild_particle_visual()

    def set_black_hole(self, pid: int, is_bh: bool = True) -> None:
        """Mark or unmark a particle as a black hole for special rendering.

        Args:
            pid: Integer particle ID.
            is_bh: If True, render with black-hole styling (dark face, colored edge ring).
        """
        idx = self._id_to_idx.get(pid)
        if idx is not None:
            self._is_bh[idx] = is_bh

    def set_trail_length(self, frac: float) -> None:
        """Set trail length as a fraction of total simulation time.

        Args:
            frac: Fraction in [0, 1]. E.g. 0.01 = 1% of the simulation shown as trail.
        """
        self._trail_length_frac = np.clip(frac, 0.0, 1.0)
        # Reset two-pointer cache — trail window bounds changed
        self._trail_si[:] = -1
        self._trail_ei[:] = -1

    def set_trail_alpha(self, alpha: float) -> None:
        """Set peak trail opacity at the head (newest point).

        Args:
            alpha: Opacity in [0, 1]. Trails fade from 0 at the tail to this value.
        """
        self._trail_alpha = np.clip(alpha, 0.0, 1.0)

    def set_trail_width(self, width: float) -> None:
        """Set trail line width.

        Args:
            width: Line width in pixels (clamped to >= 0.5).
        """
        self._trail_width = max(0.5, width)


    def set_point_alpha(self, alpha: float) -> None:
        """Set particle marker opacity.

        Args:
            alpha: Opacity in [0, 1]. 0 = invisible, 1 = fully opaque.
        """
        self._point_alpha = np.clip(alpha, 0.0, 1.0)
        if self._particle_visual is not None:
            self._particle_visual.alpha = self._point_alpha
        if self._subview_markers is not None:
            self._subview_markers.alpha = self._point_alpha

    # ------------------------------------------------------------------
    # Units
    # ------------------------------------------------------------------

    def set_units(self, time_unit: str | None = None) -> None:
        """Update the time unit label for the overlay.

        Args:
            time_unit: Time unit string (e.g. "yr", "Myr"). No-op if None.
        """
        if time_unit:
            self._time_unit = time_unit

    # ------------------------------------------------------------------
    # Time overlay
    # ------------------------------------------------------------------

    def set_time_display(self, enabled: bool) -> None:
        """Show or hide the on-screen time overlay.

        Args:
            enabled: If True, the simulation time is drawn on the canvas.
        """
        self._time_display_enabled = enabled
        if self._time_text is not None:
            self._time_text.visible = enabled

    def set_time_font_size(self, size: float) -> None:
        """Set the font size of the time overlay.

        Args:
            size: Font size in points.
        """
        self._time_font_size = size
        if self._time_text is not None:
            self._time_text.font_size = size

    def set_time_color(self, color) -> None:
        """Set the color of the time overlay text.

        Args:
            color: Any VisPy-compatible color (RGBA tuple, color name, etc.).
        """
        self._time_color = color
        if self._time_text is not None:
            self._time_text.color = color

    def set_time_anchor(self, anchor: str) -> None:
        """Set time overlay position.

        Args:
            anchor: Corner name — "top-left", "top-right", "bottom-left",
                or "bottom-right".
        """
        self._time_anchor = anchor
        self._reposition_time_text()

    def _reposition_time_text(self) -> None:
        if self._time_text is None:
            return
        w, h = self._canvas.size
        ox, oy = D.TIME_OFFSET
        anchors = {
            "top-left": ((ox, oy), "left", "top"),
            "top-right": ((w - ox, oy), "right", "top"),
            "bottom-left": ((ox, h - oy), "left", "bottom"),
            "bottom-right": ((w - ox, h - oy), "right", "bottom"),
        }
        pos, ax, ay = anchors.get(self._time_anchor, anchors["top-left"])
        self._time_text.pos = pos
        self._time_text.anchors = (ax, ay)

    # ------------------------------------------------------------------
    # Star field
    # ------------------------------------------------------------------

    def enable_stars(self, enabled: bool = True) -> None:
        """Toggle the background star field.

        Args:
            enabled: If True, show a spherical shell of background stars.
        """
        from vispy import scene

        self._stars_enabled = enabled
        if enabled and self._star_visual is None:
            if self._star_directions is None:
                self._generate_star_field()
            np.multiply(self._star_directions, self._star_min_shell_radius,
                        out=self._star_positions)
            self._star_visual = scene.Markers(
                parent=self._view.scene,
            )
            self._star_visual.set_data(
                pos=self._star_positions,
                face_color=self._star_base_colors,
                size=self._star_base_sizes,
                edge_width=0,
            )
            self._star_visual.order = -10
        if self._star_visual is not None:
            self._star_visual.visible = enabled

    def set_star_shell_factor(self, factor: float) -> None:
        """Set the star field shell radius as a multiple of the max particle distance.

        Args:
            factor: Multiplier applied to the maximum particle distance from the
                origin. Larger values push stars further out.
        """
        self._star_shell_factor = max(0.1, factor)
        self._star_min_shell_radius = max(
            self._star_max_particle_dist * self._star_shell_factor, 1.0,
        )

    def set_camera_controller(self, controller) -> None:
        """Attach a CameraController and initialize its framing.

        Args:
            controller: CameraController instance to drive the VisPy camera.
        """
        self._camera_controller = controller
        positions, mask = self._interp.evaluate_batch(self._current_sim_time)
        if not mask.any():
            return
        center, distance = controller.initialize_framing(positions, mask)
        self._view.camera.center = tuple(center)
        self._view.camera.distance = distance

    # ------------------------------------------------------------------
    # Sub-view
    # ------------------------------------------------------------------

    def enable_subview(self, corner: str = "bottom-right", size_frac: float = 0.3) -> None:
        """Create and show a picture-in-picture sub-view.

        Args:
            corner: Screen corner for placement — "bottom-right", "bottom-left",
                "top-right", or "top-left".
            size_frac: Fraction of main canvas size for the sub-view (0 to 1).
        """
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
            fov=D.SUBVIEW_FOV, distance=10.0,
        )

        from ..core.camera import CameraController, CameraMode
        self._subview_camera_controller = CameraController(
            self._subview, masses=self._data.masses, particle_ids=self._data.particle_ids,
        )
        self._subview_camera_controller.mode = CameraMode.TARGET_COMOVING

        positions, mask = self._interp.evaluate_batch(self._current_sim_time)
        active_pos = positions[mask] if mask.any() else np.zeros((1, 3), dtype=np.float32)

        attrs = self._get_particle_attrs(mask)
        self._subview_markers = scene.Markers(
            spherical=True, light_color=D.LIGHT_COLOR,
            light_position=D.LIGHT_POSITION, light_ambient=D.LIGHT_AMBIENT,
            parent=self._subview.scene,
        )
        self._subview_markers.set_data(
            pos=active_pos, face_color=attrs["face_color"],
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
        """Save the current frame as a PNG image.

        Args:
            filepath: Output file path.
            size: Render resolution as (width, height). Uses canvas size if None.
        """
        from vispy import io
        img = self._canvas.render(size=size)
        io.write_png(str(filepath), img)

    def render_video(
        self, filepath: str | Path, duration: float = D.VIDEO_DURATION,
        fps: int = D.VIDEO_FPS, size: tuple[int, int] | None = None,
        t_start: float | None = None, t_end: float | None = None,
        progress_callback: callable | None = None,
    ) -> None:
        """Render the simulation to a video file.

        Args:
            filepath: Output file path (.mp4 or .gif).
            duration: Video duration in real-time seconds.
            fps: Frames per second.
            size: Render resolution as (width, height). Uses canvas size if None.
            t_start: Simulation start time. Uses data start if None.
            t_end: Simulation end time. Uses data end if None.
            progress_callback: Called as callback(current_frame, total_frames).
                If it raises InterruptedError, rendering is cancelled.
        """
        import av

        t0 = t_start if t_start is not None else self._t_min
        t1 = t_end if t_end is not None else self._t_max
        n_frames = int(duration * fps)
        frame_sim_times = np.linspace(t0, t1, n_frames)
        filepath = Path(filepath)
        render_size = size or self._canvas.size

        container = av.open(str(filepath), mode="w")
        stream = None

        try:
            for i, t in enumerate(frame_sim_times):
                self._current_sim_time = t
                self._anim_time = (t - self._t_min) / self._time_range
                self._update_frame()

                img = self._canvas.render(size=render_size)

                # Create stream from actual rendered dimensions (first frame)
                if stream is None:
                    h, w = img.shape[:2]
                    stream = container.add_stream("libx264", rate=fps)
                    stream.width = w
                    stream.height = h
                    stream.pix_fmt = "yuv420p"

                frame = av.VideoFrame.from_ndarray(img, format="rgba")
                for packet in stream.encode(frame):
                    container.mux(packet)

                if progress_callback is not None:
                    progress_callback(i + 1, n_frames)
                elif (i + 1) % (n_frames // 10 or 1) == 0:
                    print(f"Rendering: {(i + 1) / n_frames * 100:.0f}%")

            # Flush encoder
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()

        if progress_callback is None:
            print(f"Video saved to {filepath}")
