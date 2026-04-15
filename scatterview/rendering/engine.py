"""VisPy-based OpenGL rendering engine for N-body visualization."""

from __future__ import annotations

import contextlib
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

        # Manual control speeds (WASD pan and scroll zoom)
        self._pan_speed = 0.02       # fraction of camera distance per frame
        self._zoom_speed = 1.0       # scroll zoom multiplier (1.0 = default)

        # Per-particle settings
        n = len(data.particle_ids)
        self._colors = self._default_colors(n)
        self._sizing_absolute = False
        self._equal_sizes = False
        self._radius_scale = 1.0
        self._per_particle_scale = np.ones(n, dtype=np.float32)
        self._depth_scaling = D.DEPTH_SCALING
        self._subview_depth_scaling = False
        self._subview_lock_orientation = True
        self._subview_lock_last_az: float | None = None
        self._subview_lock_last_el: float | None = None

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
        self._star_count = D.STAR_COUNT
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

        # VisPy canvas and view — use a grid so split-screen can add
        # the sub-view as a sibling cell rather than an overlay.
        self._canvas = scene.SceneCanvas(
            keys="interactive", size=size, title=title, show=False,
        )
        self._grid = self._canvas.central_widget.add_grid()
        self._view = self._grid.add_view(row=0, col=0)
        self._view.camera = scene.cameras.TurntableCamera(
            fov=D.CAMERA_FOV, distance=10.0,
        )
        self._view.camera.set_range()

        # Sub-view — own ViewBox on the same canvas
        self._subview = None
        self._subview_enabled = False
        self._subview_camera_controller = None
        self._subview_markers = None
        self._subview_trail_line = None
        self._subview_star_visual = None
        self._subview_layout = None
        self._subview_size_frac = 0.3
        self._subview_pip_margin = 10
        self._subview_radius_scale = 1.0

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
        # opacity via a quadratic curve (t**2).  The tail fades off
        # aggressively so only the most recent motion reads strongly.
        self._alpha_lut = np.linspace(0, 1, 1024, dtype=np.float32) ** 2

        self._build_visuals()

        # Timer and camera
        self._timer = app.Timer(interval=1.0 / 60.0, connect=self._on_timer, start=False)
        self._camera_controller = None

        # Auto-enable free zoom on scroll wheel / trackpad zoom
        self._canvas.events.mouse_wheel.connect(self._on_mouse_wheel)
        self._canvas.events.key_press.connect(self._on_key_press)
        self._canvas.events.key_release.connect(self._on_key_release)
        self._canvas.events.resize.connect(self._on_canvas_resize)

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
        """Zoom the view under the cursor (main or sub-view).

        Dispatches to the sub-view when the cursor is inside its rect,
        otherwise to the main view.  Each view toggles its own
        controller's ``free_zoom`` so the two settings are independent.

        Holding Ctrl multiplies zoom speed by 5x.

        Args:
            event: VisPy mouse wheel event with `delta` and `pos` attributes.
        """
        scroll = event.delta[1]
        if scroll == 0:
            return

        if self._cursor_in_subview(event.pos):
            view = self._subview
            controller = self._subview_camera_controller
        else:
            view = self._view
            controller = self._camera_controller

        if controller is not None and not controller.free_zoom:
            controller.free_zoom = True

        self._zoom_view(view, event, scroll)
        event.handled = True

    def _cursor_in_subview(self, pos) -> bool:
        """Return True if ``pos`` is inside the sub-view's on-canvas rect."""
        if not self._subview_enabled or self._subview is None or pos is None:
            return False
        sx, sy = self._subview.pos
        sw, sh = self._subview.size
        x, y = pos[0], pos[1]
        return sx <= x < sx + sw and sy <= y < sy + sh

    def _zoom_view(self, view, event, scroll) -> None:
        camera = view.camera
        speed = self._zoom_speed * (5.0 if self._has_ctrl(event) else 1.0)
        # 1.1^(-scroll*speed): scroll up → zoom_factor < 1 → closer
        zoom_factor = 1.1 ** (-scroll * speed)

        # Shift center toward cursor (view-local coords, not canvas coords).
        mouse_pos = event.pos
        view_size = view.size
        if mouse_pos is not None and view_size[0] and view_size[1]:
            vx, vy = view.pos
            local_x = mouse_pos[0] - vx
            local_y = mouse_pos[1] - vy
            cursor_offset_x = local_x - view_size[0] / 2
            cursor_offset_y = local_y - view_size[1] / 2

            fov_rad = math.radians(camera.fov / 2)
            world_per_pixel = 2.0 * camera.distance * math.tan(fov_rad) / view_size[1]

            dx_screen = cursor_offset_x * world_per_pixel
            dy_screen = -cursor_offset_y * world_per_pixel  # screen Y is flipped

            # Rotate screen offset into world coordinates using *this*
            # camera's azimuth/elevation (not the main view's cached trig).
            az = math.radians(camera.azimuth)
            el = math.radians(camera.elevation)
            sin_az, cos_az = math.sin(az), math.cos(az)
            sin_el, cos_el = math.sin(el), math.cos(el)
            right = np.array([cos_az, sin_az, 0.0])
            up = np.array([-sin_az * sin_el, cos_az * sin_el, cos_el])

            center = np.array(camera.center, dtype=np.float64)
            center += (right * dx_screen + up * dy_screen) * (1.0 - zoom_factor)
            camera.center = tuple(center)

        camera.distance *= zoom_factor
        camera.view_changed()

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
        up = np.array([
            -self._sin_el * self._sin_az,
            self._sin_el * self._cos_az,
            self._cos_el,
        ])

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

        Cube root compresses the dynamic range (proportional to volume^(1/3)).
        The smallest particle is pinned to MIN_PX; larger particles scale up
        proportionally with no upper cap, so true mass ratios show through.
        """
        n = len(self._data.particle_ids)
        if self._raw_radii is not None:
            compressed = np.cbrt(self._raw_radii)
            min_c = compressed[compressed > 0].min() if np.any(compressed > 0) else 0.0
            if min_c > 0:
                return (compressed / min_c * D.RELATIVE_SIZE_MIN_PX).astype(np.float32)
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
        if self._equal_sizes:
            n = len(self._base_sizes_relative)
            uniform = self._base_sizes_absolute.mean() if self._sizing_absolute else D.DEFAULT_SIZE_PX
            base = np.full(n, uniform, dtype=np.float32)
        else:
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
        n = self._star_count

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

    def _resolve_subview_scaling_mode(self):
        if self._sizing_absolute:
            return "scene"
        return "visual" if self._subview_depth_scaling else False

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
            self._trail_line.set_data(pos=combined_pos, color=combined_colors)
            self._trail_line.visible = True
        else:
            self._trail_line = scene.Line(
                pos=combined_pos, color=combined_colors,
                parent=self._view.scene, width=1,
                antialias=True, connect="strip",
            )

        # Sub-view: share the main view's trail colors directly.
        if self._subview_enabled and self._subview is not None:
            if self._subview_trail_line is not None:
                self._subview_trail_line.set_data(
                    pos=combined_pos, color=combined_colors,
                )
                self._subview_trail_line.visible = True
            else:
                self._subview_trail_line = scene.Line(
                    pos=combined_pos, color=combined_colors,
                    parent=self._subview.scene, width=1,
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
        if self._subview_star_visual is not None:
            self._subview_star_visual.set_data(
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
            if self._subview_lock_orientation and self._subview is not None:
                self._sync_subview_orientation()

        if self._particle_visual is not None:
            self._particle_visual.set_data(pos=active_pos, **attrs)

        self._update_trails(self._current_sim_time, positions, mask)

        # Sub-view: feed the same computed data to its own visuals
        if self._subview_enabled and self._subview_markers is not None:
            sub_size = attrs["size"] * self._subview_radius_scale
            self._subview_markers.set_data(
                pos=active_pos, face_color=attrs["face_color"],
                size=sub_size, edge_color=attrs["edge_color"],
                edge_width=attrs["edge_width"],
            )

        # Star field
        if self._stars_enabled and self._star_visual is not None:
            self._update_stars()

        # Time overlay — only update the VisPy text when the display string
        # actually changes (avoids per-frame texture re-render)
        if self._time_text is not None and self._time_display_enabled:
            text = D.format_sim_time(self._current_sim_time, self._time_unit)
            if text != self._time_text.text:
                self._time_text.text = text

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
        if self._subview_markers is not None:
            self._subview_markers.scaling = self._resolve_subview_scaling_mode()

    def set_equal_sizes(self, enabled: bool) -> None:
        """Toggle uniform sizing: when enabled, every particle renders at the same size."""
        if enabled == self._equal_sizes:
            return
        self._equal_sizes = enabled
        self._recompute_sizes()

    def set_depth_scaling(self, enabled: bool) -> None:
        """Toggle perspective depth scaling on particle markers.

        Args:
            enabled: If True, closer particles appear larger (perspective projection).
        """
        if enabled == self._depth_scaling:
            return
        self._depth_scaling = enabled
        self._rebuild_particle_visual()

    def set_subview_depth_scaling(self, enabled: bool) -> None:
        """Toggle perspective depth scaling on sub-view markers.

        Absolute sizing always uses scene-space scaling in the sub-view,
        so the depth toggle is a no-op while Absolute Sizes is on.
        """
        self._subview_depth_scaling = enabled
        if self._subview_markers is not None:
            self._subview_markers.scaling = self._resolve_subview_scaling_mode()

    def set_subview_lock_orientation(self, enabled: bool) -> None:
        """Bi-directionally link main and sub-view azimuth/elevation."""
        self._subview_lock_orientation = enabled
        # Re-seed the cache so the first post-toggle frame treats both
        # cameras as already in sync (avoids a spurious snap).
        self._subview_lock_last_az = None
        self._subview_lock_last_el = None

    def _sync_subview_orientation(self) -> None:
        """Mirror rotations between the main and sub-view cameras.

        Whichever camera moved since the last sync wins; the other
        is updated to match.  Handles user drags on either view as
        well as auto-rotate on either controller.
        """
        main_cam = self._view.camera
        sub_cam = self._subview.camera
        main_az, main_el = main_cam.azimuth, main_cam.elevation
        sub_az, sub_el = sub_cam.azimuth, sub_cam.elevation

        last_az = self._subview_lock_last_az
        last_el = self._subview_lock_last_el
        if last_az is None or last_el is None:
            # First sync: adopt the main view's orientation as the truth.
            target_az, target_el = main_az, main_el
        else:
            main_moved = (main_az != last_az) or (main_el != last_el)
            sub_moved = (sub_az != last_az) or (sub_el != last_el)
            if main_moved:
                target_az, target_el = main_az, main_el
            elif sub_moved:
                target_az, target_el = sub_az, sub_el
            else:
                target_az, target_el = main_az, main_el

        if (main_az, main_el) != (target_az, target_el):
            main_cam.azimuth = target_az
            main_cam.elevation = target_el
        if (sub_az, sub_el) != (target_az, target_el):
            sub_cam.azimuth = target_az
            sub_cam.elevation = target_el

        self._subview_lock_last_az = target_az
        self._subview_lock_last_el = target_el

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

    def set_star_count(self, count: int) -> None:
        """Change the number of background stars and regenerate the field.

        Args:
            count: Number of stars on the spherical shell.
        """
        self._star_count = max(100, int(count))
        self._generate_star_field()
        if self._star_visual is not None:
            self._star_visual.set_data(
                pos=self._star_positions,
                face_color=self._star_base_colors,
                size=self._star_base_sizes,
                edge_width=0,
            )
        if self._subview_star_visual is not None:
            self._subview_star_visual.set_data(
                pos=self._star_positions,
                face_color=self._star_base_colors,
                size=self._star_base_sizes,
                edge_width=0,
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

    def _recompute_pip_geometry(self) -> None:
        """Recompute PiP sub-view pos/size from current canvas size.

        No-op when the sub-view is disabled or in Split mode.  Called on
        canvas resize and before offscreen renders at a different size,
        so the PiP corner stays anchored and proportional.
        """
        if (self._subview is None
                or self._subview_layout is None
                or self._subview_layout.startswith("Split")):
            return
        w, h = self._canvas.size
        sw = int(w * self._subview_size_frac)
        sh = int(h * self._subview_size_frac)
        margin = self._subview_pip_margin
        pip_corners = {
            "PiP Bottom-Right": (w - sw - margin, h - sh - margin),
            "PiP Bottom-Left": (margin, h - sh - margin),
            "PiP Top-Right": (w - sw - margin, margin),
            "PiP Top-Left": (margin, margin),
        }
        sx, sy = pip_corners.get(self._subview_layout, pip_corners["PiP Bottom-Right"])
        self._subview.pos = (sx, sy)
        self._subview.size = (sw, sh)

    def _on_canvas_resize(self, event) -> None:
        self._recompute_pip_geometry()

    def enable_subview(self, layout: str = "PiP Bottom-Right", size_frac: float = 0.3) -> None:
        """Create a sub-view on the same canvas.

        Supports PiP (overlay in a corner) and split-screen layouts.
        The sub-view has its own scene and visuals but shares the same
        OpenGL context.  Per-frame data is computed once and passed to
        both sets of visuals.

        Args:
            layout: Layout name — "PiP Bottom-Right", "PiP Bottom-Left",
                "PiP Top-Right", "PiP Top-Left",
                "Split Left | Right", "Split Top | Bottom".
            size_frac: Fraction of main canvas size for PiP mode (0 to 1).
        """
        from vispy import scene

        if self._subview is not None:
            return

        self._subview_layout = layout
        self._subview_size_frac = size_frac

        # Create sub-view: grid cell for split, overlay for PiP
        if layout.startswith("Split"):
            if "Left" in layout:
                self._subview = self._grid.add_view(row=0, col=1)
            else:
                self._subview = self._grid.add_view(row=1, col=0)
        else:
            self._subview = self._canvas.central_widget.add_view()
            self._recompute_pip_geometry()

        self._subview.camera = scene.cameras.TurntableCamera(
            fov=D.CAMERA_FOV, distance=10.0,
        )
        self._subview.border_color = (1.0, 1.0, 1.0, 0.6)

        from ..core.camera import CameraController, CameraMode
        self._subview_camera_controller = CameraController(
            self._subview, masses=self._data.masses, particle_ids=self._data.particle_ids,
        )
        self._subview_camera_controller.mode = CameraMode.TARGET_COMOVING

        # Create visuals in the sub-view's own scene
        positions, mask = self._interp.evaluate_batch(self._current_sim_time)
        active_pos = positions[mask] if mask.any() else np.zeros((1, 3), dtype=np.float32)
        attrs = self._get_particle_attrs(mask)

        self._subview_markers = scene.Markers(
            spherical=True, scaling=self._resolve_subview_scaling_mode(),
            light_color=D.LIGHT_COLOR,
            light_position=D.LIGHT_POSITION, light_ambient=D.LIGHT_AMBIENT,
            parent=self._subview.scene,
        )
        self._subview_markers.set_data(
            pos=active_pos, face_color=attrs["face_color"],
            size=attrs["size"], edge_color=attrs["edge_color"],
            edge_width=attrs["edge_width"],
        )

        # Star field for the sub-view (same data, own visual)
        if self._stars_enabled and self._star_positions is not None:
            self._subview_star_visual = scene.Markers(
                parent=self._subview.scene,
            )
            self._subview_star_visual.set_data(
                pos=self._star_positions,
                face_color=self._star_base_colors,
                size=self._star_base_sizes,
                edge_width=0,
            )
            self._subview_star_visual.order = -10

        # Defer framing to the first _update_frame call so the GUI
        # has a chance to apply target/mode settings first.
        # The controller's _target_needs_acquisition flag (set when
        # target_particle is assigned) triggers an immediate snap.

        self._subview_enabled = True

    def disable_subview(self) -> None:
        if self._subview is not None:
            # Remove from grid's internal layout tracking if it was
            # added as a grid cell (split mode).
            if self._subview_layout and self._subview_layout.startswith("Split"):
                grid = self._grid
                to_remove = [
                    k for k, v in grid._grid_widgets.items()
                    if v[4] is self._subview
                ]
                for k in to_remove:
                    row, col = grid._grid_widgets[k][:2]
                    del grid._grid_widgets[k]
                    if row in grid._cells and col in grid._cells[row]:
                        del grid._cells[row][col]
                grid._need_solver_recreate = True

            self._subview.parent = None
            self._subview = None
            self._subview_markers = None
            self._subview_trail_line = None
            self._subview_star_visual = None
            self._subview_camera_controller = None
            self._subview_enabled = False
            self._subview_layout = None

    # ------------------------------------------------------------------
    # Display / export
    # ------------------------------------------------------------------

    def show(self) -> None:
        from vispy import app
        self._canvas.show()
        self._timer.start()
        app.run()

    @contextlib.contextmanager
    def _canvas_resized_to(self, size: tuple[int, int] | None):
        """Temporarily spoof the canvas's reported size for offscreen renders.

        Works around vispy's ``canvas.render(size=...)`` only mapping
        ``canvas.size`` onto a fraction of the larger FBO (content lands in the
        lower-left quadrant).  Inside the context, the backend reports the
        requested size as both logical and physical size and the central widget
        is resized so child views (including the PiP sub-view) lay out against
        the render resolution.  The real Qt widget is untouched.
        """
        if size is None or tuple(size) == tuple(self._canvas.size):
            yield
            return

        backend = self._canvas._backend
        orig_physical = backend._physical_size
        orig_cw_size = self._canvas._central_widget.size

        backend._vispy_get_size = lambda: tuple(size)
        backend._physical_size = tuple(size)
        self._canvas._central_widget.size = tuple(size)
        self._recompute_pip_geometry()
        # Force each camera to recompute its projection for the render aspect
        # ratio.  The viewbox resize event does this automatically, but we
        # trigger it explicitly so the first rendered frame is already framed
        # for the render size rather than the GUI aspect.
        self._view.camera.view_changed()
        if self._subview is not None and self._subview.camera is not None:
            self._subview.camera.view_changed()
        try:
            yield
        finally:
            del backend._vispy_get_size
            backend._physical_size = orig_physical
            self._canvas._central_widget.size = orig_cw_size
            self._recompute_pip_geometry()
            self._view.camera.view_changed()
            if self._subview is not None and self._subview.camera is not None:
                self._subview.camera.view_changed()

    def _render_at(self, size: tuple[int, int] | None):
        """Render the canvas at an arbitrary size (see ``_canvas_resized_to``)."""
        with self._canvas_resized_to(size):
            return self._canvas.render(size=size)

    def screenshot(self, filepath: str | Path, size: tuple[int, int] | None = None) -> None:
        """Save the current frame as a PNG image.

        Args:
            filepath: Output file path.
            size: Render resolution as (width, height). Uses canvas size if None.
        """
        from vispy import io
        img = self._render_at(size)
        io.write_png(str(filepath), img)

    def render_video(
        self, filepath: str | Path, duration: float = D.VIDEO_DURATION,
        fps: int = D.VIDEO_FPS, size: tuple[int, int] | None = None,
        t_start: float | None = None, t_end: float | None = None,
        progress_callback: callable | None = None,
        codec: str = "libx264",
        codec_options: dict | None = None,
    ) -> None:
        """Render the simulation to a video file.

        Rendering runs on the calling thread (it owns the GL context);
        encoding and muxing run on a worker thread fed by a 1-frame queue,
        so GPU readback overlaps with CPU (or GPU) encoding.

        Args:
            filepath: Output file path (.mp4 or .gif).
            duration: Video duration in real-time seconds.
            fps: Frames per second.
            size: Render resolution as (width, height). Uses canvas size if None.
            t_start: Simulation start time. Uses data start if None.
            t_end: Simulation end time. Uses data end if None.
            progress_callback: Called as callback(current_frame, total_frames).
                If it raises InterruptedError, rendering is cancelled.
            codec: ffmpeg codec name, e.g. "libx264", "h264_nvenc", "hevc_nvenc".
            codec_options: ffmpeg stream options dict (preset, crf/cq, rc, ...).
        """
        import av
        import queue
        import threading

        t0 = t_start if t_start is not None else self._t_min
        t1 = t_end if t_end is not None else self._t_max
        n_frames = int(duration * fps)
        frame_sim_times = np.linspace(t0, t1, n_frames)
        filepath = Path(filepath)
        render_size = tuple(size) if size is not None else tuple(self._canvas.size)

        container = av.open(str(filepath), mode="w")

        frame_q: queue.Queue = queue.Queue(maxsize=1)
        worker_error: list[BaseException] = []

        def _make_stream(first_img):
            h, w = first_img.shape[:2]
            requested = codec
            try:
                s = container.add_stream(requested, rate=fps)
            except (av.FFmpegError, ValueError) as exc:
                if requested != "libx264":
                    print(f"[render_video] codec {requested!r} init failed "
                          f"({exc}); falling back to libx264")
                    s = container.add_stream("libx264", rate=fps)
                    s.options = {"preset": "veryfast", "crf": "20"}
                else:
                    raise
            else:
                if codec_options:
                    s.options = dict(codec_options)
            s.width, s.height = w, h
            # Encode directly in RGB (planar GBR) instead of going through
            # YUV.  libx264 supports ``gbrp`` in the High 4:4:4 Predictive
            # profile and skips the lossy RGB→YUV colour-matrix step entirely,
            # so output colours match what the GUI draws pixel-for-pixel.
            # NVENC codecs don't support gbrp — fall back to full-range YUV 4:4:4.
            if s.codec.name == "libx264":
                s.pix_fmt = "gbrp"
            else:
                s.pix_fmt = "yuv444p"
                cc = s.codec_context
                cc.color_range = av.video.reformatter.ColorRange.JPEG
                cc.colorspace = av.video.reformatter.Colorspace.ITU709
            return s

        def _encoder_worker():
            stream = None
            try:
                while True:
                    img = frame_q.get()
                    if img is None:
                        break
                    if stream is None:
                        stream = _make_stream(img)
                    vf = av.VideoFrame.from_ndarray(img[..., :3], format="rgb24")
                    for packet in stream.encode(vf):
                        container.mux(packet)
                if stream is not None:
                    for packet in stream.encode():
                        container.mux(packet)
            except BaseException as exc:  # propagate to main thread
                worker_error.append(exc)
                # Drain any further frames so producer doesn't block.
                while True:
                    try:
                        if frame_q.get_nowait() is None:
                            break
                    except queue.Empty:
                        break

        worker = threading.Thread(target=_encoder_worker, daemon=True)
        worker.start()

        native = self._canvas.native
        native.setUpdatesEnabled(False)
        try:
            with self._canvas_resized_to(render_size):
                for i, t in enumerate(frame_sim_times):
                    if worker_error:
                        break
                    self._current_sim_time = t
                    self._anim_time = (t - self._t_min) / self._time_range
                    self._update_frame()

                    img = self._canvas.render(size=render_size)
                    frame_q.put(img)

                    if progress_callback is not None:
                        progress_callback(i + 1, n_frames)
                    elif (i + 1) % (n_frames // 10 or 1) == 0:
                        print(f"Rendering: {(i + 1) / n_frames * 100:.0f}%")
        finally:
            # Signal the worker to flush and exit, then wait for it.
            frame_q.put(None)
            worker.join()
            container.close()
            native.setUpdatesEnabled(True)

        if worker_error:
            raise worker_error[0]

        if progress_callback is None:
            print(f"Video saved to {filepath}")
