"""wgpu / pygfx rendering engine for N-body visualization.

Replaces the prior VisPy/OpenGL implementation.  The public API is
preserved verbatim; only the GPU upload and window plumbing changed.
The Numba JIT trail assembly, sliding-window trail pointers, spline
evaluation, and camera math are unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path

import numba as nb
import numpy as np
import pygfx as gfx
from rendercanvas.offscreen import OffscreenRenderCanvas
from rendercanvas.pyqt6 import QRenderCanvas

from .. import defaults as D
from ..core.data_loader import SimulationData
from ..core.interpolation import TrajectoryInterpolator
from .sphere_material import SphereImpostorMaterial


# ---------------------------------------------------------------------------
# Numba JIT hot paths (unchanged from VisPy implementation)
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _advance_trail_pointers(
    packed_times,        # (total_pts,) float64
    segment_start,       # (n_particles,) int64
    segment_end,         # (n_particles,) int64
    prev_window_start,   # (n_particles,) int64
    prev_window_end,     # (n_particles,) int64
    t_trail_start,       # float64
    t_current,           # float64
    is_forward,          # bool
    out_window_start,    # (n_particles,) int64
    out_window_end,      # (n_particles,) int64
):
    """Find the visible trail window [start, end) for each particle.

    O(1) during forward playback (pointer advance), O(log N) on scrub
    or first frame (binary search).
    """
    for particle in range(len(segment_start)):
        seg_lo = segment_start[particle]
        n_points = segment_end[particle] - seg_lo

        if is_forward and prev_window_start[particle] >= 0:
            tail = prev_window_start[particle]
            while tail < n_points and packed_times[seg_lo + tail] <= t_trail_start:
                tail += 1
            head = prev_window_end[particle]
            while head < n_points and packed_times[seg_lo + head] < t_current:
                head += 1
        else:
            lo, hi = nb.int64(0), n_points
            while lo < hi:
                mid = (lo + hi) >> 1
                if packed_times[seg_lo + mid] <= t_trail_start:
                    lo = mid + 1
                else:
                    hi = mid
            tail = lo
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
    out_positions,
    out_colors,
    out_times,
    write_offsets,
    trail_point_counts,
    trail_body_starts,
    trail_body_counts,
    trail_has_tail,
    trail_tail_positions,
    particle_indices,
    trail_is_active,
    particle_color_indices,
    precomp_positions,
    precomp_times,
    live_positions,
    color_table,
    t_trail_start,
    t_current,
):
    """Write trail geometry into output buffers for GPU upload.

    Layout per trail: [NaN separator] [tail lerp] [body] [head]
    NaN separators tell the line renderer to break between trails.
    """
    n_trails = len(write_offsets)
    for trail in range(n_trails):
        offset = write_offsets[trail]
        n_points = trail_point_counts[trail]
        n_body = trail_body_counts[trail]
        write_pos = nb.int64(0)

        if trail > 0:
            sep = offset - 1
            for dim in range(3):
                out_positions[sep, dim] = np.nan
            for dim in range(4):
                out_colors[sep, dim] = np.nan
            out_times[sep] = np.nan

        if trail_has_tail[trail]:
            for dim in range(3):
                out_positions[offset, dim] = trail_tail_positions[trail, dim]
            out_times[offset] = t_trail_start
            write_pos = 1

        body_start = trail_body_starts[trail]
        for i in range(n_body):
            for dim in range(3):
                out_positions[offset + write_pos + i, dim] = precomp_positions[body_start + i, dim]
            out_times[offset + write_pos + i] = precomp_times[body_start + i]
        write_pos += n_body

        if trail_is_active[trail]:
            p_idx = particle_indices[trail]
            for dim in range(3):
                out_positions[offset + write_pos, dim] = np.float32(live_positions[p_idx, dim])
            out_times[offset + write_pos] = t_current

        color_idx = particle_color_indices[trail]
        for i in range(n_points):
            for dim in range(3):
                out_colors[offset + i, dim] = color_table[color_idx, dim]


# ---------------------------------------------------------------------------
# Adapters so controls.py keeps working unchanged
# ---------------------------------------------------------------------------


class _CanvasAdapter:
    """Wraps a QRenderCanvas so `engine.canvas.native` returns the QWidget.

    The old engine exposed `canvas.native` (VisPy's term for the Qt widget).
    pygfx's QRenderCanvas *is* a QWidget, so `.native` just points back to it.
    """

    def __init__(self, canvas):
        self._canvas = canvas

    @property
    def native(self):
        return self._canvas

    @property
    def size(self):
        return self._canvas.get_logical_size()

    def __getattr__(self, name):
        return getattr(self._canvas, name)


# ---------------------------------------------------------------------------
# Camera state: a lightweight struct that looks like VisPy's TurntableCamera
# enough for core/camera.py and gui/controls.py to keep working.
# ---------------------------------------------------------------------------


class PygfxTurntableCamera:
    """VisPy-TurntableCamera-compatible wrapper around a pygfx PerspectiveCamera.

    Stores (fov, azimuth, elevation, distance, center) as canonical state.
    On every ``view_changed()`` call it recomputes the pygfx camera's
    world-space position and orientation.  This lets the legacy
    CameraController keep writing ``_center`` / ``_distance`` directly
    without knowing anything about pygfx.
    """

    # Axis convention: VisPy's TurntableCamera default uses a +Z-up camera
    # that orbits around the center.  Here we replicate that exactly so
    # the light-direction transform and pan axes keep the same math.
    def __init__(self, pygfx_camera: gfx.PerspectiveCamera, fov: float = 45.0,
                 azimuth: float = 0.0, elevation: float = 30.0,
                 distance: float = 10.0, center=(0.0, 0.0, 0.0)):
        self._pygfx = pygfx_camera
        self._fov = float(fov)
        self._azimuth = float(azimuth)
        self._elevation = float(elevation)
        self._distance = float(distance)
        self._center = (float(center[0]), float(center[1]), float(center[2]))
        self._pygfx.fov = self._fov
        self._push_to_pygfx()

    # Public attributes (mirror VisPy's TurntableCamera API)
    @property
    def fov(self) -> float:
        return self._fov

    @fov.setter
    def fov(self, value: float) -> None:
        self._fov = float(value)
        self._pygfx.fov = self._fov

    @property
    def azimuth(self) -> float:
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value: float) -> None:
        # Wrap to match VisPy (-180, 180] behavior
        value = float(value)
        while value > 180.0:
            value -= 360.0
        while value <= -180.0:
            value += 360.0
        self._azimuth = value
        self._push_to_pygfx()

    @property
    def elevation(self) -> float:
        return self._elevation

    @elevation.setter
    def elevation(self, value: float) -> None:
        self._elevation = max(-90.0, min(90.0, float(value)))
        self._push_to_pygfx()

    @property
    def distance(self) -> float:
        return self._distance

    @distance.setter
    def distance(self, value: float) -> None:
        self._distance = max(1e-9, float(value))
        self._push_to_pygfx()

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value) -> None:
        self._center = (float(value[0]), float(value[1]), float(value[2]))
        self._push_to_pygfx()

    # view_changed() is called by CameraController after poking ``_center``
    # / ``_distance`` privates — push that state to pygfx.
    def view_changed(self) -> None:
        self._push_to_pygfx()

    def set_range(self) -> None:
        """No-op for compatibility; pygfx doesn't need an explicit recompute."""
        self._push_to_pygfx()

    def _spherical_to_direction(self) -> np.ndarray:
        """Unit vector from center toward camera using (az, el).

        VisPy's TurntableCamera puts the camera at:
            x = d * sin(az) * cos(el)
            y = -d * cos(az) * cos(el)   (start looking +Y)
            z = d * sin(el)
        with +Z up.  We reproduce that mapping exactly so the existing
        light and pan math (which reads camera.azimuth/elevation) stays
        correct.
        """
        az = math.radians(self._azimuth)
        el = math.radians(self._elevation)
        cos_el = math.cos(el)
        return np.array([
            math.sin(az) * cos_el,
            -math.cos(az) * cos_el,
            math.sin(el),
        ], dtype=np.float64)

    def _push_to_pygfx(self) -> None:
        direction = self._spherical_to_direction()
        center = np.array(self._center, dtype=np.float64)
        position = center + direction * self._distance
        self._pygfx.local.position = tuple(position.tolist())
        # +Z up, look at center
        self._pygfx.show_pos(self._center, up=(0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RenderEngine:
    """wgpu / pygfx renderer managing particles, trails, camera, animation."""

    def __init__(
        self, data: SimulationData, interpolator: TrajectoryInterpolator,
        size: tuple[int, int] = (1280, 720),
        title: str = "ScatterView",
    ):
        """Create the rendering engine.

        Args:
            data: Loaded simulation data.
            interpolator: Pre-built TrajectoryInterpolator.
            size: Initial canvas size (width, height) in logical pixels.
            title: Window title string.
        """
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

        # Manual control speeds
        self._pan_speed = 0.02
        self._zoom_speed = 1.0

        # Per-particle settings
        n = len(data.particle_ids)
        self._colors = self._default_colors(n)
        self._sizing_absolute = False
        self._radius_scale = 1.0
        self._per_particle_scale = np.ones(n, dtype=np.float32)
        self._depth_scaling = D.DEPTH_SCALING
        self._equal_sizes = False   # toggle: render every particle at a uniform size

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
        self._time_text = None  # pygfx.Text
        self._time_text_str = ""  # cached to avoid rebuilding glyphs each frame

        # Star field
        self._stars_enabled = True
        self._star_visual = None   # pygfx.Points
        self._star_count = D.STAR_COUNT
        self._star_directions = None
        self._star_positions = None
        self._star_base_sizes = None
        self._star_base_colors = None
        self._star_max_particle_dist = 1.0
        self._star_shell_factor = D.STAR_SHELL_FACTOR
        self._star_min_shell_radius = 1.0
        self._star_positions_dirty = True  # re-upload positions only on change

        # Black hole markers
        self._is_bh = np.zeros(n, dtype=bool)
        if data.startypes is not None:
            for pid_key, k in data.startypes.items():
                if k == D.BH_STARTYPE:
                    idx = self._id_to_idx.get(pid_key)
                    if idx is not None:
                        self._is_bh[idx] = True

        # --- Window & pygfx stack ---
        # update_mode="continuous" drives draws up to max_fps via the
        # canvas's internal scheduler; wall-clock dt is derived inside
        # the draw callback so playback stays independent of render rate.
        #
        # vsync=False: the WSLg/Wayland/X compositor does its own vsync,
        # so enabling wgpu's vsync *on top* stacks a second present queue
        # and visibly delays input-to-photon response.  Disabling wgpu's
        # vsync drops that extra buffer; the compositor still prevents
        # tearing.  max_fps caps CPU/GPU work at the display refresh rate.
        self._canvas_raw = QRenderCanvas(
            size=size, title=title, update_mode="continuous", max_fps=120, vsync=False,
        )
        self._canvas = _CanvasAdapter(self._canvas_raw)
        self._renderer = gfx.WgpuRenderer(self._canvas_raw)
        self._scene = gfx.Scene()

        # Overlay scene kept for future non-text overlays; time text now
        # uses a QLabel (see _build_time_overlay_label) to avoid pygfx's
        # float32-filterable requirement on some GPU adapters.

        # Main camera — wrapped so legacy code keeps working
        self._pygfx_camera = gfx.PerspectiveCamera(
            fov=D.CAMERA_FOV, aspect=size[0] / max(size[1], 1),
        )
        self._view_camera = PygfxTurntableCamera(
            self._pygfx_camera, fov=D.CAMERA_FOV,
            azimuth=0.0, elevation=30.0, distance=10.0,
        )
        # Orbit controller for drag-to-rotate / right-drag-pan
        self._orbit_controller = _PygfxOrbitAdapter(self._view_camera, self._canvas_raw)

        # `engine.view` mimics VisPy's ViewBox just enough for the legacy
        # CameraController and the CLI to reach the camera wrapper.
        self._view = _ViewAdapter(self._view_camera, self._scene)

        # Sub-view
        self._subview = None
        self._subview_enabled = False
        self._subview_camera = None              # PygfxTurntableCamera
        self._subview_pygfx_camera = None        # gfx.PerspectiveCamera
        self._subview_camera_controller = None
        self._subview_markers = None             # _MarkerAdapter wrapping subview Points
        self._subview_stars = None               # pygfx.Points (stars in subview)
        self._subview_trail_line = None          # pygfx.Line
        self._subview_trail_geom = None
        self._subview_layout = None
        self._subview_size_frac = 0.3
        self._subview_radius_scale = 1.0
        self._subview_trail_alpha = D.TRAIL_ALPHA
        self._subview_point_alpha = D.POINT_ALPHA
        self._subview_depth_scaling = D.DEPTH_SCALING
        # Bi-directional azimuth/elevation link between main and sub-view.
        self._subview_lock_orientation = True
        self._subview_lock_last_az = None
        self._subview_lock_last_el = None

        # Main particle / trail / star pygfx objects (built later)
        self._particle_visual = None       # pygfx.Points (regular particles, Gaussian blobs)
        self._bh_visual = None             # pygfx.Points (black holes, ring marker)
        self._trail_line = None            # pygfx.Line
        self._trail_geom = None
        self._particle_geom = None
        self._bh_geom = None
        self._n_active_cached = -1

        # Pre-computed trails
        self._n_particles = n
        self._precomp = interpolator.precompute_all_trails()

        # Pre-allocated trail arrays — grown on demand.
        self._trail_capacity = 0
        self._combined_pos = np.empty((0, 3), dtype=np.float32)
        self._combined_colors = np.empty((0, 4), dtype=np.float32)
        self._combined_times = np.empty(0, dtype=np.float64)
        self._time_fraction = np.empty(0, dtype=np.float32)
        # Pre-allocated GPU-side buffers (same capacity, different lifetime).
        self._trail_gpu_pos = None
        self._trail_gpu_colors = None
        self._trail_gpu_capacity = 0

        # Two-pointer sliding window
        self._trail_si = np.full(n, -1, dtype=np.int64)
        self._trail_ei = np.full(n, -1, dtype=np.int64)
        self._trail_prev_time = -np.inf

        # Alpha gradient lookup table
        self._alpha_lut = np.linspace(0, 1, 1024, dtype=np.float32)

        # Timer & camera controller
        self._camera_controller = None
        self._manual_mode_callbacks: list = []

        # Keyboard state
        self._pan_keys_held: set[str] = set()
        self._ctrl_held = False
        self._alt_held = False
        self._shift_held = False

        # Cached camera trig
        self._cos_az = 1.0
        self._sin_az = 0.0
        self._cos_el = 1.0
        self._sin_el = 0.0

        # World-space light direction (unit vector)
        _lw = np.array(D.LIGHT_POSITION, dtype=np.float64)
        self._light_world = _lw / np.linalg.norm(_lw)
        self._light_eye = (0.0, 0.0, 1.0)
        self._light_eye_last = None  # cache to skip redundant uniform writes

        # Build initial visuals
        self._build_visuals()

        # Hook draw + input events
        self._canvas_raw.request_draw(self._animate)
        self._canvas_raw.add_event_handler(
            self._on_event,
            "wheel", "key_down", "key_up", "resize",
        )

        # Wall-clock reference for per-frame dt advancement inside _animate.
        self._last_tick = None

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _on_event(self, event) -> None:
        et = event.get("event_type")
        if et == "wheel":
            self._on_wheel(event)
        elif et == "key_down":
            self._on_key_down(event)
        elif et == "key_up":
            self._on_key_up(event)
        elif et == "resize":
            self._on_resize(event)

    @staticmethod
    def _has_ctrl(event) -> bool:
        mods = event.get("modifiers", ()) or ()
        return any(m and m.lower() in ("control", "ctrl") for m in mods)

    def _on_wheel(self, event) -> None:
        """Zoom toward the mouse cursor position.

        Holding Ctrl multiplies zoom speed by 5x.
        """
        if self._camera_controller is not None and not self._camera_controller.free_zoom:
            self._camera_controller.free_zoom = True

        # pygfx/jupyter-rfb wheel event: dy positive = scroll up = zoom in
        scroll = -event.get("dy", 0.0) / 120.0  # normalize: 120 units = one notch
        if scroll == 0:
            return

        camera = self._view_camera
        speed = self._zoom_speed * (5.0 if self._has_ctrl(event) else 1.0)
        zoom_factor = 1.1 ** (-scroll * speed)

        mouse_x = event.get("x")
        mouse_y = event.get("y")
        canvas_w, canvas_h = self._canvas.size
        if mouse_x is not None and mouse_y is not None:
            cursor_offset_x = mouse_x - canvas_w / 2
            cursor_offset_y = mouse_y - canvas_h / 2

            fov_rad = math.radians(camera.fov / 2)
            world_per_pixel = 2.0 * camera.distance * math.tan(fov_rad) / max(canvas_h, 1)

            dx_screen = cursor_offset_x * world_per_pixel
            dy_screen = -cursor_offset_y * world_per_pixel

            self._cache_camera_trig()
            right, _ = self._camera_axes()
            up = np.array([0.0, 0.0, 1.0])

            center = np.array(camera.center, dtype=np.float64)
            world_offset = right * dx_screen + up * dy_screen
            center += world_offset * (1.0 - zoom_factor)
            camera.center = tuple(center)

        camera.distance *= zoom_factor
        camera.view_changed()
        self._canvas_raw.request_draw()

    _PAN_KEYS = {'W', 'UP', 'S', 'DOWN', 'A', 'LEFT', 'D', 'RIGHT',
                 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'}

    @staticmethod
    def _normalize_key(name: str) -> str:
        """Map rendercanvas key names to the short names the engine uses."""
        mapping = {
            'ArrowUp': 'Up', 'ArrowDown': 'Down',
            'ArrowLeft': 'Left', 'ArrowRight': 'Right',
            ' ': 'Space',
        }
        return mapping.get(name, name)

    def _on_key_down(self, event) -> None:
        raw = event.get("key", "")
        if raw is None:
            return
        name = self._normalize_key(raw)
        if name == 'Space':
            self.toggle_play()
            return
        # Normalize single letters to uppercase so the pan map hits
        key_name = name.upper() if len(name) == 1 else name
        if key_name in ('W', 'S', 'A', 'D', 'Up', 'Down', 'Left', 'Right'):
            self._pan_keys_held.add(key_name)
            if not self._alt_held and self._camera_controller is not None:
                from ..core.camera import CameraMode
                if self._camera_controller.mode != CameraMode.MANUAL:
                    self._camera_controller.mode = CameraMode.MANUAL
                    for cb in self._manual_mode_callbacks:
                        cb()
                if not self._camera_controller.free_zoom:
                    self._camera_controller.free_zoom = True
        if name in ('Control', 'Ctrl'):
            self._ctrl_held = True
        if name == 'Alt':
            self._alt_held = True
        if name == 'Shift':
            self._shift_held = True

    def _on_key_up(self, event) -> None:
        raw = event.get("key", "")
        if raw is None:
            return
        name = self._normalize_key(raw)
        key_name = name.upper() if len(name) == 1 else name
        self._pan_keys_held.discard(key_name)
        if name in ('Control', 'Ctrl'):
            self._ctrl_held = False
        if name == 'Alt':
            self._alt_held = False
        if name == 'Shift':
            self._shift_held = False

    def _on_resize(self, event) -> None:
        # Update camera aspect + overlay layout
        w = event.get("width", None) or self._canvas.size[0]
        h = event.get("height", None) or self._canvas.size[1]
        self._pygfx_camera.aspect = w / max(h, 1)
        if self._subview_pygfx_camera is not None:
            self._subview_pygfx_camera.aspect = w / max(h, 1)
        self._reposition_time_text()  # keep QLabel anchored

    # ------------------------------------------------------------------
    # Camera helpers (cached trig)
    # ------------------------------------------------------------------

    def _cache_camera_trig(self) -> None:
        az = math.radians(self._view_camera.azimuth)
        el = math.radians(self._view_camera.elevation)
        self._cos_az = math.cos(az)
        self._sin_az = math.sin(az)
        self._cos_el = math.cos(el)
        self._sin_el = math.sin(el)

    def _camera_axes(self):
        # Right / forward vectors in world space (consistent with the +Z-up
        # VisPy-compatible convention used in the light and pan math).
        return (
            np.array([self._cos_az, self._sin_az, 0.0]),
            np.array([-self._sin_az, self._cos_az, 0.0]),
        )

    def _apply_keyboard_pan(self) -> None:
        if not self._pan_keys_held or self._alt_held:
            return

        camera = self._view_camera
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
                if key_name == 'Up':
                    center += forward * step
                elif key_name == 'Down':
                    center -= forward * step
                elif key_name == 'Left':
                    time_step -= 1.0
                elif key_name == 'Right':
                    time_step += 1.0
            else:
                if key_name == 'Up':
                    center += up * step
                elif key_name == 'Down':
                    center -= up * step
                elif key_name == 'Left':
                    center -= right * step
                elif key_name == 'Right':
                    center += right * step

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

    _ROTATE_SPEED_DEG = 1.5

    def _apply_keyboard_rotate(self) -> None:
        if not self._pan_keys_held or not self._alt_held:
            return
        camera = self._view_camera
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

        Cube root compresses the dynamic range (proportional to
        volume^(1/3)).  The smallest particle is pinned to ``MIN_PX``;
        larger particles scale up proportionally with no upper cap, so
        true mass ratios remain visible.
        """
        n = len(self._data.particle_ids)
        if self._raw_radii is not None:
            compressed = np.cbrt(self._raw_radii)
            positive = compressed[compressed > 0]
            if positive.size > 0:
                min_c = positive.min()
                return (compressed / min_c * D.RELATIVE_SIZE_MIN_PX).astype(np.float32)
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
        if self._equal_sizes:
            n = len(self._base_sizes_relative)
            uniform = (self._base_sizes_absolute.mean()
                       if self._sizing_absolute else D.DEFAULT_SIZE_PX)
            base = np.full(n, uniform, dtype=np.float32)
        else:
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

    def _generate_star_field(self) -> None:
        rng = np.random.default_rng(D.STAR_SEED)
        n = self._star_count

        z = rng.uniform(-1, 1, n)
        phi = rng.uniform(0, 2 * np.pi, n)
        r_xy = np.sqrt(1 - z ** 2)
        self._star_directions = np.column_stack([
            r_xy * np.cos(phi), r_xy * np.sin(phi), z,
        ]).astype(np.float32)

        max_dist = 0.0
        for pos in self._data.positions.values():
            finite = np.isfinite(pos).all(axis=1)
            if finite.any():
                d = np.linalg.norm(pos[finite], axis=1).max()
                if d > max_dist:
                    max_dist = d
        self._star_max_particle_dist = max_dist
        self._star_min_shell_radius = max(max_dist * self._star_shell_factor, 1.0)

        u = rng.uniform(0, 1, n)
        self._star_base_sizes = (D.STAR_BASE_SIZE * (1.0 + 3.0 * u ** 3)).astype(np.float32)

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

        self._star_positions = np.empty_like(self._star_directions)

    def _get_particle_attrs(self, mask: np.ndarray) -> dict:
        """Compute face_color, edge_color, edge_width, size for active particles."""
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
                "edge_width": edge_width, "size": sizes,
                "bh_mask": bh_mask}

    def _resolve_size_space(self) -> str:
        """Map legacy sizing flags onto pygfx ``size_space``.

        - absolute          → "world" (radius in world units)
        - relative + depth  → "world" (pixels via camera projection)
        - relative, no depth → "screen" (fixed pixel size)
        """
        if self._sizing_absolute:
            return "world"
        return "world" if self._depth_scaling else "screen"

    # ------------------------------------------------------------------
    # Visual construction
    # ------------------------------------------------------------------

    def _build_visuals(self) -> None:
        # --- Background ---
        # Dark blue → black gradient matches the prior VisPy look.
        bg_mat = gfx.BackgroundMaterial((0.0, 0.0, 0.0), (0.02, 0.02, 0.08))
        self._scene.add(gfx.Background(None, bg_mat))

        positions, mask = self._interp.evaluate_batch(self._data.times[0])

        # --- Particle buffers (pre-allocated to full capacity so we can
        # reuse the same Geometry object across frames) ---
        n = self._n_particles
        self._particle_positions = np.zeros((n, 3), dtype=np.float32)
        self._particle_colors = np.zeros((n, 4), dtype=np.float32)
        self._particle_sizes = np.zeros(n, dtype=np.float32)
        self._bh_positions = np.zeros((max(self._is_bh.sum(), 1), 3), dtype=np.float32)
        self._bh_colors = np.ones((max(self._is_bh.sum(), 1), 4), dtype=np.float32)
        self._bh_sizes = np.full(max(self._is_bh.sum(), 1), 10.0, dtype=np.float32)

        # Geometry + material for regular particles
        self._particle_geom = gfx.Geometry(
            positions=self._particle_positions.copy(),
            colors=self._particle_colors.copy(),
            sizes=self._particle_sizes.copy(),
        )
        self._particle_material = SphereImpostorMaterial(
            size_mode="vertex", size_space=self._resolve_size_space(),
            color_mode="vertex",
            ambient=D.LIGHT_AMBIENT,
        )
        self._particle_visual = gfx.Points(self._particle_geom, self._particle_material)
        self._scene.add(self._particle_visual)

        # Black hole particles — ring marker on top
        if self._is_bh.any():
            self._bh_geom = gfx.Geometry(
                positions=self._bh_positions.copy(),
                colors=self._bh_colors.copy(),
                sizes=self._bh_sizes.copy(),
            )
            self._bh_material = gfx.PointsMarkerMaterial(
                marker="ring",
                size_mode="vertex",
                size_space="screen",
                color_mode="vertex",
                edge_width=D.BH_EDGE_WIDTH,
                edge_color="white",
            )
            self._bh_visual = gfx.Points(self._bh_geom, self._bh_material)
            self._bh_visual.render_order = 1  # draw on top of regular particles
            self._scene.add(self._bh_visual)

        # Initial particle data
        self._upload_particle_frame(positions, mask)

        # Trails + star field
        self._update_trails(self._data.times[0], positions, mask)
        self._generate_star_field()
        if self._stars_enabled:
            self.enable_stars(True)

        # Time overlay.  pygfx's Text material samples float32 textures
        # (requires the float32-filterable WGPU feature which our GPU
        # adapter may not expose).  To keep text rendering fast and
        # portable, we use a Qt QLabel overlaid on the canvas for
        # onscreen display, and PIL text composite for offscreen export.
        # `self._time_text` is a QLabel; `self._time_text_str` caches the
        # current string to avoid per-frame rewrites.
        self._time_text_str = D.format_sim_time(self._t_min, self._time_unit)
        self._time_text = self._build_time_overlay_label(self._time_text_str)
        self._reposition_time_text()

        # Seed camera framing so the default view is reasonable before the
        # CLI attaches a CameraController.
        valid = np.isfinite(positions).all(axis=1)
        if valid.any():
            center = positions[valid].mean(axis=0)
            radius = float(np.max(np.linalg.norm(positions[valid] - center, axis=1))) or 1.0
            self._view_camera.center = tuple(center)
            self._view_camera.distance = max(radius * 2.5, 1.0)

    def _upload_particle_frame(self, positions: np.ndarray, mask: np.ndarray) -> None:
        """Push one frame's particle state into the Geometry buffers.

        Inactive particles get size=0 (invisible) but keep their buffer
        slot; this keeps the draw range constant and avoids rebuilding
        the pipeline on every frame.
        """
        n = self._n_particles
        pos_buf = self._particle_geom.positions
        col_buf = self._particle_geom.colors
        size_buf = self._particle_geom.sizes

        # Regular particles: compute full attributes then zero out BH slots
        # and inactive slots.
        face_color = np.zeros((n, 4), dtype=np.float32)
        sizes = np.zeros(n, dtype=np.float32)

        if mask.any():
            # Only active, non-BH particles render through the Gaussian-blob
            # Points; black holes render via the separate ring Points so they
            # get the correct look.
            main_mask = mask & ~self._is_bh
            if main_mask.any():
                face_color[main_mask] = self._colors[main_mask]
                # apply global point_alpha
                face_color[main_mask, 3] *= self._point_alpha
                sizes[main_mask] = self._sizes[main_mask]

        pos_buf.data[:] = np.where(
            np.isfinite(positions), positions.astype(np.float32), 0.0,
        )
        col_buf.data[:] = face_color
        size_buf.data[:] = sizes

        pos_buf.update_range(0, n)
        col_buf.update_range(0, n)
        size_buf.update_range(0, n)

        # Black holes
        if self._bh_visual is not None and self._is_bh.any():
            bh_idx = np.where(self._is_bh)[0]
            bh_active = mask[bh_idx]
            n_bh = len(bh_idx)
            bh_pos = np.zeros((n_bh, 3), dtype=np.float32)
            bh_col = np.zeros((n_bh, 4), dtype=np.float32)
            bh_size = np.zeros(n_bh, dtype=np.float32)
            if bh_active.any():
                pos_active = positions[bh_idx[bh_active]].astype(np.float32)
                bh_pos[bh_active] = np.where(np.isfinite(pos_active), pos_active, 0.0)
                # Ring edge color uses the particle's own color, fully opaque.
                bh_col[bh_active, :3] = self._colors[bh_idx[bh_active], :3]
                bh_col[bh_active, 3] = 1.0 * self._point_alpha
                bh_size[bh_active] = self._sizes[bh_idx[bh_active]]
            self._bh_geom.positions.data[:] = bh_pos
            self._bh_geom.colors.data[:] = bh_col
            self._bh_geom.sizes.data[:] = bh_size
            self._bh_geom.positions.update_range(0, n_bh)
            self._bh_geom.colors.update_range(0, n_bh)
            self._bh_geom.sizes.update_range(0, n_bh)

    def _rebuild_particle_visual(self) -> None:
        """Rebuild particle material after size-space or depth-scaling change.

        pygfx's PointsMaterial is immutable w.r.t. size_space after first
        draw, so we recreate it.  Geometry buffers stay put.
        """
        if self._particle_visual is not None:
            self._scene.remove(self._particle_visual)
        self._particle_material = SphereImpostorMaterial(
            size_mode="vertex", size_space=self._resolve_size_space(),
            color_mode="vertex",
            ambient=D.LIGHT_AMBIENT,
        )
        self._particle_visual = gfx.Points(self._particle_geom, self._particle_material)
        self._scene.add(self._particle_visual)

        positions, mask = self._interp.evaluate_batch(self._current_sim_time)
        self._upload_particle_frame(positions, mask)

    # ------------------------------------------------------------------
    # Trails
    # ------------------------------------------------------------------

    def _hide_all_trails(self) -> None:
        if self._trail_line is not None:
            self._trail_line.visible = False
        if self._subview_trail_line is not None:
            self._subview_trail_line.visible = False

    def _ensure_trail_geom(self, capacity: int) -> None:
        """Make sure the GPU trail buffers can hold ``capacity`` vertices."""
        if self._trail_line is not None and self._trail_gpu_capacity >= capacity:
            return
        # (Re)allocate pygfx buffers.
        positions = np.zeros((capacity, 3), dtype=np.float32)
        colors = np.zeros((capacity, 4), dtype=np.float32)
        self._trail_geom = gfx.Geometry(positions=positions, colors=colors)
        if self._trail_line is not None:
            self._scene.remove(self._trail_line)
        trail_mat = gfx.LineMaterial(
            thickness=1.5, thickness_space="screen",
            color_mode="vertex", aa=True,
        )
        self._trail_line = gfx.Line(self._trail_geom, trail_mat)
        self._scene.add(self._trail_line)
        self._trail_gpu_capacity = capacity

        # Sub-view mirror if active
        if self._subview_enabled and self._subview_trail_line is not None:
            sub_positions = np.zeros((capacity, 3), dtype=np.float32)
            sub_colors = np.zeros((capacity, 4), dtype=np.float32)
            self._subview_trail_geom = gfx.Geometry(positions=sub_positions, colors=sub_colors)
            self._scene.remove(self._subview_trail_line)
            sub_mat = gfx.LineMaterial(
                thickness=1.5, thickness_space="screen",
                color_mode="vertex", aa=True,
            )
            self._subview_trail_line = gfx.Line(self._subview_trail_geom, sub_mat)
            self._scene.add(self._subview_trail_line)

    def _update_trails(self, time: float, positions: np.ndarray,
                       mask: np.ndarray) -> None:
        """Extract visible trail windows and upload to GPU."""
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

        can_interpolate_tail = (window_starts > 0) & (window_starts <= precomp_counts)
        idx_before_tail = np.maximum(seg_starts + window_starts - 1, seg_starts)
        idx_after_tail = np.minimum(seg_starts + window_starts, seg_ends - 1)
        t_before_tail = precomp.times[idx_before_tail]
        t_after_tail = precomp.times[idx_after_tail]
        tail_time_gap = t_after_tail - t_before_tail
        has_tail = can_interpolate_tail & (tail_time_gap > 0)

        tail_lerp_factor = np.where(
            has_tail,
            (t_trail_start - t_before_tail) / np.maximum(tail_time_gap, 1e-30),
            0.0,
        )
        tail_positions = (
            precomp.positions[idx_before_tail] * (1 - tail_lerp_factor[:, np.newaxis])
            + precomp.positions[idx_after_tail] * tail_lerp_factor[:, np.newaxis]
        )

        is_active = mask[valid_indices]

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

        total_pts = int(draw_counts.sum()) + n_trails - 1

        if total_pts > self._trail_capacity:
            self._trail_capacity = int(total_pts * 1.5)
            self._combined_pos = np.empty((self._trail_capacity, 3), dtype=np.float32)
            self._combined_colors = np.empty((self._trail_capacity, 4), dtype=np.float32)
            self._combined_times = np.empty(self._trail_capacity, dtype=np.float64)
            self._time_fraction = np.empty(self._trail_capacity, dtype=np.float32)
        combined_pos = self._combined_pos[:total_pts]
        combined_colors = self._combined_colors[:total_pts]

        write_offsets = np.empty(n_trails, dtype=np.int64)
        write_offsets[0] = 0
        if n_trails > 1:
            cumulative_counts = np.cumsum(draw_counts)
            write_offsets[1:] = cumulative_counts[:-1] + np.arange(1, n_trails)

        combined_times = self._combined_times[:total_pts]
        _assemble_trails(
            combined_pos, combined_colors, combined_times,
            write_offsets, draw_counts, draw_body_starts, draw_body_counts,
            draw_has_tail, draw_tail_positions, draw_particle_idx,
            draw_is_active, draw_particle_idx,
            precomp.positions, precomp.times, positions, self._colors,
            t_trail_start, time,
        )

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

        # Upload to pygfx — resize GPU buffer if needed.  draw_range
        # clips rendering to [0, total_pts) so we only need to upload the
        # valid range; slots beyond total_pts won't be drawn.
        self._ensure_trail_geom(max(total_pts, 1))
        pos_buf = self._trail_geom.positions
        col_buf = self._trail_geom.colors
        pos_buf.data[:total_pts] = combined_pos
        col_buf.data[:total_pts] = combined_colors
        pos_buf.update_range(0, total_pts)
        col_buf.update_range(0, total_pts)
        self._trail_geom.positions.draw_range = (0, total_pts)
        self._trail_line.visible = True

        # Sub-view
        if self._subview_enabled and self._subview_trail_line is not None:
            if self._subview_trail_alpha != self._trail_alpha:
                sub_colors = combined_colors.copy()
                alpha_ratio = self._subview_trail_alpha / max(self._trail_alpha, 1e-6)
                sub_colors[:, 3] *= alpha_ratio
            else:
                sub_colors = combined_colors
            sp = self._subview_trail_geom.positions
            sc = self._subview_trail_geom.colors
            sp.data[:total_pts] = combined_pos
            sc.data[:total_pts] = sub_colors
            sp.update_range(0, total_pts)
            sc.update_range(0, total_pts)
            self._subview_trail_geom.positions.draw_range = (0, total_pts)
            self._subview_trail_line.visible = True

    # ------------------------------------------------------------------
    # Star field
    # ------------------------------------------------------------------

    def _update_stars(self) -> None:
        # Star positions only change when the shell radius changes
        # (driven by the particle extent).  Skip GPU upload otherwise.
        if not self._star_positions_dirty:
            return
        np.multiply(self._star_directions, self._star_min_shell_radius,
                    out=self._star_positions)
        n = len(self._star_positions)
        geom = self._star_visual.geometry
        geom.positions.data[:] = self._star_positions
        geom.positions.update_range(0, n)
        if self._subview_stars is not None:
            sg = self._subview_stars.geometry
            sg.positions.data[:] = self._star_positions
            sg.positions.update_range(0, n)
        self._star_positions_dirty = False

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def _advance_sim_time(self) -> None:
        """Advance ``_current_sim_time`` by one wall-clock step.

        Called at the top of each draw callback.  Playback speed is
        wall-clock-driven (fraction of sim duration per real second),
        so the render rate only affects smoothness, not pace.  Holds
        at the same time while paused.
        """
        import time as _time
        now = _time.monotonic()
        if self._last_tick is None:
            dt = 0.0
        else:
            dt = max(0.0, now - self._last_tick)
        self._last_tick = now

        if self._playing:
            self._anim_time += self._anim_speed * dt
            if self._anim_time > 1.0:
                self._anim_time = 0.0
                self._trail_si[:] = -1
                self._trail_ei[:] = -1
            self._current_sim_time = self._t_min + self._anim_time * self._time_range

        self._apply_keyboard_pan()
        self._apply_keyboard_rotate()

    def _update_light_direction(self) -> None:
        """Transform world-space light direction to eye space and push it
        to the sphere-impostor material uniforms.

        The light stays fixed in world space (e.g. upper-left of the
        scene), so as the camera orbits we transform the world direction
        into view space each frame so the highlight appears to shift
        with the geometry instead of following the camera.
        """
        w = self._light_world
        x = w[0] * self._cos_az + w[1] * self._sin_az
        y = -w[0] * self._sin_az + w[1] * self._cos_az
        z = w[2]
        eye_y = y * self._cos_el + z * self._sin_el
        eye_z = -y * self._sin_el + z * self._cos_el
        inv_norm = 1.0 / math.sqrt(x * x + eye_y * eye_y + eye_z * eye_z)
        new_light = (x * inv_norm, eye_y * inv_norm, eye_z * inv_norm)

        # Skip GPU uniform write when direction hasn't changed (camera
        # static + paused → steady light direction).
        if new_light == self._light_eye_last:
            return
        self._light_eye = new_light
        self._light_eye_last = new_light

        if self._particle_material is not None:
            self._particle_material.light_dir = new_light
        if self._subview_markers is not None and self._subview_markers._material is not None:
            self._subview_markers._material.light_dir = new_light

    def _update_frame(self) -> None:
        """Compute one frame's state (no GPU present yet)."""
        self._cache_camera_trig()

        positions, mask = self._interp.evaluate_batch(self._current_sim_time)
        if not mask.any():
            return

        self._update_light_direction()

        if self._camera_controller is not None:
            self._camera_controller.update(self._current_sim_time, positions, mask)
        if self._subview_camera_controller is not None and self._subview_enabled:
            self._subview_camera_controller.update(self._current_sim_time, positions, mask)
            if self._subview_lock_orientation:
                self._sync_subview_orientation()

        self._upload_particle_frame(positions, mask)
        self._update_trails(self._current_sim_time, positions, mask)

        if self._subview_enabled and self._subview_markers is not None:
            self._subview_markers.update_from(
                positions, mask, self._is_bh, self._colors, self._sizes,
                self._subview_radius_scale, self._subview_point_alpha,
            )

        if self._stars_enabled and self._star_visual is not None:
            self._update_stars()

        if self._time_text is not None and self._time_display_enabled:
            text = D.format_sim_time(self._current_sim_time, self._time_unit)
            if text != self._time_text_str:
                self._time_text_str = text
                self._time_text.setText(text)
                # Width may change when the formatted string length changes
                # (e.g. scientific notation), so keep the corner anchored.
                self._reposition_time_text()

    def _animate(self) -> None:
        """Draw callback registered with the canvas.

        Invoked by the canvas's scheduler on every present tick.  Handles
        wall-clock-driven simulation-time advancement, keyboard-held
        pan/rotate, per-frame data upload, and the pygfx render passes.
        """
        self._advance_sim_time()
        self._update_frame()
        self._render_passes()

    def _render_passes(self) -> None:
        """Issue main-view and optional sub-view render passes.

        The time overlay is a QLabel child of the canvas widget, so Qt
        draws it for us on top of the rendered surface — no extra pygfx
        pass needed.
        """
        w, h = self._canvas.size
        self._pygfx_camera.aspect = w / max(h, 1)

        if self._subview_enabled and self._subview is not None:
            self._render_with_subview(w, h)
        else:
            self._renderer.render(self._scene, self._pygfx_camera, flush=True)

    def _render_with_subview(self, w: int, h: int) -> None:
        if self._subview["layout"].startswith("Split"):
            if "Left" in self._subview["layout"]:
                # main fills left half, sub fills right
                rect_main = (0, 0, w // 2, h)
                rect_sub = (w // 2, 0, w - w // 2, h)
            else:
                # horizontal split: top main / bottom sub
                rect_main = (0, 0, w, h // 2)
                rect_sub = (0, h // 2, w, h - h // 2)
            self._renderer.render(self._scene, self._pygfx_camera,
                                  rect=rect_main, flush=False, clear=True)
            self._renderer.render(self._scene, self._subview_pygfx_camera,
                                  rect=rect_sub, flush=True, clear=False)
        else:
            # PiP: full main + inset corner
            self._renderer.render(self._scene, self._pygfx_camera, flush=False, clear=True)
            sw, sh = self._subview["size"]
            sx, sy = self._subview["pos"]
            self._renderer.render(self._scene, self._subview_pygfx_camera,
                                  rect=(sx, sy, sw, sh), flush=True, clear=True)

        self._subview_pygfx_camera.aspect = max(self._subview["size"][0], 1) / max(self._subview["size"][1], 1)

    # ------------------------------------------------------------------
    # Public API (preserved verbatim from the VisPy implementation)
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
        self._current_sim_time = float(np.clip(value, self._t_min, self._t_max))
        self._anim_time = (self._current_sim_time - self._t_min) / self._time_range
        self._canvas_raw.request_draw()

    @property
    def playing(self) -> bool:
        return self._playing

    def play(self) -> None:
        self._playing = True
        # Reset the wall-clock reference so the first post-play frame
        # doesn't inherit a stale dt that would jump the sim forward.
        self._last_tick = None

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

    def set_equal_sizes(self, enabled: bool) -> None:
        """Render every particle at a uniform size (ignores per-particle radii)."""
        if enabled == self._equal_sizes:
            return
        self._equal_sizes = enabled
        self._recompute_sizes()

    def set_subview_depth_scaling(self, enabled: bool) -> None:
        """Toggle perspective depth scaling on the sub-view markers."""
        self._subview_depth_scaling = enabled
        if self._subview_markers is not None:
            self._subview_markers.scaling = "visual" if enabled else False

    def set_subview_lock_orientation(self, enabled: bool) -> None:
        """Bi-directionally link main and sub-view azimuth/elevation."""
        self._subview_lock_orientation = enabled
        # Re-seed the cache so the first post-toggle frame treats both
        # cameras as already in sync (avoids a spurious snap).
        self._subview_lock_last_az = None
        self._subview_lock_last_el = None

    def _sync_subview_orientation(self) -> None:
        """Mirror rotations between main and sub-view cameras.

        Whichever camera moved since the last sync wins; the other is
        updated to match.  Handles drags on either view and auto-rotate
        on either controller.
        """
        if self._subview_camera is None:
            return
        main_cam = self._view_camera
        sub_cam = self._subview_camera
        main_az, main_el = main_cam.azimuth, main_cam.elevation
        sub_az, sub_el = sub_cam.azimuth, sub_cam.elevation

        last_az = self._subview_lock_last_az
        last_el = self._subview_lock_last_el
        if last_az is None or last_el is None:
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
        idx = self._id_to_idx.get(pid)
        if idx is not None:
            self._is_bh[idx] = is_bh

    def set_trail_length(self, frac: float) -> None:
        self._trail_length_frac = float(np.clip(frac, 0.0, 1.0))
        self._trail_si[:] = -1
        self._trail_ei[:] = -1

    def set_trail_alpha(self, alpha: float) -> None:
        self._trail_alpha = float(np.clip(alpha, 0.0, 1.0))

    def set_point_alpha(self, alpha: float) -> None:
        self._point_alpha = float(np.clip(alpha, 0.0, 1.0))
        # Re-upload current frame so the alpha change takes effect even when paused.
        if self._particle_geom is not None:
            positions, mask = self._interp.evaluate_batch(self._current_sim_time)
            self._upload_particle_frame(positions, mask)

    # --- Units ---

    def set_units(self, time_unit: str | None = None) -> None:
        if time_unit:
            self._time_unit = time_unit

    # --- Time overlay ---

    def set_time_display(self, enabled: bool) -> None:
        self._time_display_enabled = enabled
        if self._time_text is not None:
            self._time_text.setVisible(enabled)

    def set_time_font_size(self, size: float) -> None:
        self._time_font_size = float(size)
        if self._time_text is not None:
            self._time_text.setStyleSheet(self._time_label_stylesheet())
            self._reposition_time_text()

    def set_time_color(self, color) -> None:
        self._time_color = color
        if self._time_text is not None:
            self._time_text.setStyleSheet(self._time_label_stylesheet())

    def set_time_anchor(self, anchor: str) -> None:
        self._time_anchor = anchor
        self._reposition_time_text()

    def _build_time_overlay_label(self, text: str):
        """Create a QLabel child of the canvas for the time overlay."""
        from PyQt6 import QtWidgets, QtCore, QtGui
        label = QtWidgets.QLabel(text, parent=self._canvas_raw)
        label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        label.setStyleSheet(self._time_label_stylesheet())
        label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        label.adjustSize()
        label.raise_()
        return label

    def _time_label_stylesheet(self) -> str:
        r, g, b, a = self._time_color
        return (
            f"color: rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, "
            f"{int(a * 255)}); background: transparent; "
            f"font-size: {int(self._time_font_size)}pt; font-weight: 600;"
        )

    def _reposition_time_text(self) -> None:
        """Place the time overlay QLabel in the requested corner."""
        if self._time_text is None:
            return
        self._time_text.adjustSize()
        w, h = self._canvas.size
        lw = self._time_text.width()
        lh = self._time_text.height()
        ox, oy = D.TIME_OFFSET
        corners = {
            "top-left":     (ox,           oy),
            "top-right":    (w - lw - ox,  oy),
            "bottom-left":  (ox,           h - lh - oy),
            "bottom-right": (w - lw - ox,  h - lh - oy),
        }
        x, y = corners.get(self._time_anchor, corners["top-left"])
        self._time_text.move(x, y)
        self._time_text.setVisible(self._time_display_enabled)

    # --- Star field ---

    def enable_stars(self, enabled: bool = True) -> None:
        self._stars_enabled = enabled
        if enabled and self._star_visual is None:
            if self._star_directions is None:
                self._generate_star_field()
            np.multiply(self._star_directions, self._star_min_shell_radius,
                        out=self._star_positions)
            star_geom = gfx.Geometry(
                positions=self._star_positions.astype(np.float32).copy(),
                colors=self._star_base_colors.copy(),
                sizes=self._star_base_sizes.copy(),
            )
            star_mat = gfx.PointsMaterial(
                size_mode="vertex", size_space="screen",
                color_mode="vertex",
            )
            self._star_visual = gfx.Points(star_geom, star_mat)
            self._star_visual.render_order = -10
            self._scene.add(self._star_visual)
        if self._star_visual is not None:
            self._star_visual.visible = enabled

    def set_star_shell_factor(self, factor: float) -> None:
        self._star_shell_factor = max(0.1, float(factor))
        self._star_min_shell_radius = max(
            self._star_max_particle_dist * self._star_shell_factor, 1.0,
        )
        self._star_positions_dirty = True

    def set_star_count(self, count: int) -> None:
        self._star_count = max(100, int(count))
        self._generate_star_field()
        # Recreate star visual with the new count (geometry buffer size changed).
        if self._star_visual is not None:
            self._scene.remove(self._star_visual)
            self._star_visual = None
        if self._subview_stars is not None:
            self._scene.remove(self._subview_stars)
            self._subview_stars = None
        if self._stars_enabled:
            self.enable_stars(True)
            if self._subview_enabled:
                self._ensure_subview_stars()

    # --- Camera controller attach ---

    def set_camera_controller(self, controller) -> None:
        self._camera_controller = controller
        positions, mask = self._interp.evaluate_batch(self._current_sim_time)
        if not mask.any():
            return
        center, distance = controller.initialize_framing(positions, mask)
        self._view_camera.center = tuple(center)
        self._view_camera.distance = distance

    # ------------------------------------------------------------------
    # Sub-view
    # ------------------------------------------------------------------

    def enable_subview(self, layout: str = "PiP Bottom-Right", size_frac: float = 0.3) -> None:
        if self._subview is not None:
            return

        w, h = self._canvas.size
        self._subview_size_frac = size_frac
        self._subview_layout = layout

        # Build sub-view rect
        if layout.startswith("Split"):
            sw = w // 2
            sh = h // 2 if "Top" in layout else h
            sx, sy = (w - sw, 0) if "Left" in layout else (0, 0)
        else:
            sw, sh = int(w * size_frac), int(h * size_frac)
            margin = 10
            pip_corners = {
                "PiP Bottom-Right": (w - sw - margin, margin),
                "PiP Bottom-Left":  (margin,          margin),
                "PiP Top-Right":    (w - sw - margin, h - sh - margin),
                "PiP Top-Left":     (margin,          h - sh - margin),
            }
            sx, sy = pip_corners.get(layout, pip_corners["PiP Bottom-Right"])

        self._subview = {
            "layout": layout, "pos": (sx, sy), "size": (sw, sh),
        }

        # Sub-view camera — shares the main view's FOV
        self._subview_pygfx_camera = gfx.PerspectiveCamera(
            fov=D.CAMERA_FOV, aspect=max(sw, 1) / max(sh, 1),
        )
        self._subview_camera = PygfxTurntableCamera(
            self._subview_pygfx_camera, fov=D.CAMERA_FOV,
            azimuth=0.0, elevation=30.0, distance=10.0,
        )

        from ..core.camera import CameraController, CameraMode
        self._subview_camera_controller = CameraController(
            _ViewAdapter(self._subview_camera, self._scene),
            masses=self._data.masses, particle_ids=self._data.particle_ids,
        )
        self._subview_camera_controller.mode = CameraMode.TARGET_COMOVING

        # Sub-view markers — adapter renders into its own geometry so
        # size/color overrides for the sub-view don't clobber the main view.
        self._subview_markers = _MarkerAdapter(self)
        self._subview_markers.create_in_scene(self._scene)

        # Sub-view trail line — created lazily inside _ensure_trail_geom.
        sub_pos = np.zeros((max(self._trail_gpu_capacity, 64), 3), dtype=np.float32)
        sub_col = np.zeros_like(sub_pos[:, :1].repeat(4, axis=1))
        self._subview_trail_geom = gfx.Geometry(positions=sub_pos, colors=sub_col)
        sub_trail_mat = gfx.LineMaterial(
            thickness=1.5, thickness_space="screen",
            color_mode="vertex", aa=True,
        )
        self._subview_trail_line = gfx.Line(self._subview_trail_geom, sub_trail_mat)
        self._scene.add(self._subview_trail_line)

        self._ensure_subview_stars()
        self._subview_enabled = True

        # Seed camera framing
        positions, mask = self._interp.evaluate_batch(self._current_sim_time)
        if mask.any():
            center, distance = self._subview_camera_controller.initialize_framing(positions, mask)
            self._subview_camera.center = tuple(center)
            self._subview_camera.distance = distance

    def _ensure_subview_stars(self) -> None:
        if self._subview_stars is not None or not self._stars_enabled:
            return
        if self._star_positions is None:
            return
        geom = gfx.Geometry(
            positions=self._star_positions.astype(np.float32).copy(),
            colors=self._star_base_colors.copy(),
            sizes=self._star_base_sizes.copy(),
        )
        mat = gfx.PointsMaterial(
            size_mode="vertex", size_space="screen", color_mode="vertex",
        )
        self._subview_stars = gfx.Points(geom, mat)
        self._subview_stars.render_order = -10
        self._scene.add(self._subview_stars)

    def disable_subview(self) -> None:
        if self._subview is None:
            return
        if self._subview_markers is not None:
            self._subview_markers.dispose(self._scene)
            self._subview_markers = None
        if self._subview_trail_line is not None:
            self._scene.remove(self._subview_trail_line)
            self._subview_trail_line = None
            self._subview_trail_geom = None
        if self._subview_stars is not None:
            self._scene.remove(self._subview_stars)
            self._subview_stars = None
        self._subview_camera_controller = None
        self._subview_camera = None
        self._subview_pygfx_camera = None
        self._subview = None
        self._subview_enabled = False
        self._subview_layout = None

    # ------------------------------------------------------------------
    # Display / export
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Show the canvas and start the event loop (standalone path)."""
        try:
            from PyQt6 import QtWidgets
        except ImportError:  # pragma: no cover
            raise RuntimeError("PyQt6 is required to show the canvas")
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self._canvas_raw.show()
        self.play()
        app.exec()

    # Offscreen machinery (lazily built per size)
    _off_canvas = None
    _off_renderer = None
    _off_size = None

    def _ensure_offscreen(self, size: tuple[int, int]):
        w, h = int(size[0]), int(size[1])
        if self._off_canvas is None or self._off_size != (w, h):
            self._off_canvas = OffscreenRenderCanvas(size=(w, h), pixel_ratio=1)
            self._off_renderer = gfx.WgpuRenderer(self._off_canvas)
            self._off_size = (w, h)
        return self._off_canvas, self._off_renderer

    def _render_offscreen(self, size: tuple[int, int]) -> np.ndarray:
        w, h = int(size[0]), int(size[1])
        canvas, renderer = self._ensure_offscreen((w, h))

        # Run per-frame data update (in case time/camera changed).
        self._update_frame()

        saved_aspect = self._pygfx_camera.aspect

        def draw():
            self._pygfx_camera.aspect = w / max(h, 1)
            if self._subview_enabled and self._subview is not None:
                self._render_with_subview_into(renderer, w, h)
            else:
                renderer.render(self._scene, self._pygfx_camera, flush=True)

        canvas.request_draw(draw)
        img = np.asarray(canvas.draw()).copy()
        self._pygfx_camera.aspect = saved_aspect

        # Composite the Qt-rendered time overlay as PIL text.  QLabel
        # only paints over the live canvas; for offscreen export we
        # draw the same string onto the numpy buffer ourselves.
        if self._time_display_enabled and self._time_text_str:
            img = self._composite_time_overlay(img, w, h)
        return img

    def _composite_time_overlay(self, img: np.ndarray, w: int, h: int) -> np.ndarray:
        """Draw the current time text onto an RGBA offscreen frame."""
        from PIL import Image, ImageDraw, ImageFont

        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)

        # Resolve a font at the current engine font size.  Fall back to
        # PIL's default (tiny bitmap) if DejaVu isn't available.
        font_px = int(round(self._time_font_size * 1.33))  # pt -> px
        font = None
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ):
            try:
                font = ImageFont.truetype(path, font_px)
                break
            except (OSError, IOError):
                continue
        if font is None:
            font = ImageFont.load_default()

        # Measure for corner anchoring
        bbox = draw.textbbox((0, 0), self._time_text_str, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        ox, oy = D.TIME_OFFSET
        corners = {
            "top-left":     (ox,          oy),
            "top-right":    (w - tw - ox, oy),
            "bottom-left":  (ox,          h - th - oy * 2),
            "bottom-right": (w - tw - ox, h - th - oy * 2),
        }
        x, y = corners.get(self._time_anchor, corners["top-left"])

        r, g, b, a = self._time_color
        rgba = (int(r * 255), int(g * 255), int(b * 255), int(a * 255))
        draw.text((x - bbox[0], y - bbox[1]), self._time_text_str, font=font, fill=rgba)
        return np.asarray(pil)

    def _render_with_subview_into(self, renderer: gfx.WgpuRenderer,
                                  w: int, h: int) -> None:
        if self._subview["layout"].startswith("Split"):
            if "Left" in self._subview["layout"]:
                rect_main = (0, 0, w // 2, h)
                rect_sub = (w // 2, 0, w - w // 2, h)
            else:
                rect_main = (0, 0, w, h // 2)
                rect_sub = (0, h // 2, w, h - h // 2)
            renderer.render(self._scene, self._pygfx_camera,
                            rect=rect_main, flush=False, clear=True)
            renderer.render(self._scene, self._subview_pygfx_camera,
                            rect=rect_sub, flush=False, clear=False)
        else:
            renderer.render(self._scene, self._pygfx_camera, flush=False, clear=True)
            sw, sh = self._subview["size"]
            sx, sy = self._subview["pos"]
            # PiP rect must be scaled to the export resolution.
            cw, ch = self._canvas.size
            scale_x = w / max(cw, 1)
            scale_y = h / max(ch, 1)
            rect = (int(sx * scale_x), int(sy * scale_y),
                    int(sw * scale_x), int(sh * scale_y))
            renderer.render(self._scene, self._subview_pygfx_camera,
                            rect=rect, flush=False, clear=True)

    def screenshot(self, filepath: str | Path, size: tuple[int, int] | None = None) -> None:
        """Save the current frame as a PNG image."""
        from PIL import Image
        render_size = size or self._canvas.size
        img = self._render_offscreen(render_size)
        # (H, W, 4) RGBA uint8
        Image.fromarray(img).save(str(filepath))

    def render_video(
        self, filepath: str | Path, duration: float = D.VIDEO_DURATION,
        fps: int = D.VIDEO_FPS, size: tuple[int, int] | None = None,
        t_start: float | None = None, t_end: float | None = None,
        progress_callback=None,
        codec: str = "libx264",
        codec_options: dict | None = None,
    ) -> None:
        """Render the simulation to a video file.

        Args:
            codec: ffmpeg encoder name (``libx264``, ``h264_nvenc``,
                ``hevc_nvenc``, ...).  Falls back to libx264 on stream
                init failure.
            codec_options: ffmpeg stream options dict (preset, crf/cq,
                rc, ...).  Per-codec defaults apply when omitted.
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

        def _open_stream(chosen_codec: str, w: int, h: int):
            s = container.add_stream(chosen_codec, rate=fps)
            s.width = w
            s.height = h
            s.pix_fmt = "yuv420p"
            if codec_options:
                s.options = dict(codec_options)
            return s

        try:
            for i, t in enumerate(frame_sim_times):
                self._current_sim_time = float(t)
                self._anim_time = (self._current_sim_time - self._t_min) / self._time_range

                img = self._render_offscreen(render_size)  # (H, W, 4) uint8 RGBA

                if stream is None:
                    h, w = img.shape[:2]
                    try:
                        stream = _open_stream(codec, w, h)
                    except Exception:
                        # Fall back to libx264 if the requested codec
                        # (e.g. NVENC) fails to initialize.
                        if codec != "libx264":
                            stream = _open_stream("libx264", w, h)
                        else:
                            raise

                frame = av.VideoFrame.from_ndarray(img, format="rgba")
                for packet in stream.encode(frame):
                    container.mux(packet)

                if progress_callback is not None:
                    progress_callback(i + 1, n_frames)
                elif (i + 1) % (n_frames // 10 or 1) == 0:
                    print(f"Rendering: {(i + 1) / n_frames * 100:.0f}%")

            if stream is not None:
                for packet in stream.encode():
                    container.mux(packet)
        finally:
            container.close()

        if progress_callback is None:
            print(f"Video saved to {filepath}")


# ---------------------------------------------------------------------------
# Sub-view marker adapter
#
# controls.py accesses engine._subview_markers.alpha and .scaling directly.
# This tiny adapter translates those VisPy-style attribute writes into pygfx
# size_space / material updates, and owns its own Points object that
# renders into the main scene (visibility is gated by the render pass rect).
# ---------------------------------------------------------------------------


class _MarkerAdapter:
    def __init__(self, engine: RenderEngine):
        self._engine = engine
        self._alpha = D.POINT_ALPHA
        self._scaling = "visual" if engine._depth_scaling else False
        self._visual = None
        self._geom = None
        self._material = None

    def create_in_scene(self, scene: gfx.Scene) -> None:
        n = self._engine._n_particles
        pos = np.zeros((n, 3), dtype=np.float32)
        col = np.zeros((n, 4), dtype=np.float32)
        siz = np.zeros(n, dtype=np.float32)
        self._geom = gfx.Geometry(positions=pos, colors=col, sizes=siz)
        self._material = SphereImpostorMaterial(
            size_mode="vertex",
            size_space="world" if self._scaling == "visual" else "screen",
            color_mode="vertex",
            ambient=D.LIGHT_AMBIENT,
        )
        self._visual = gfx.Points(self._geom, self._material)
        scene.add(self._visual)

    def dispose(self, scene: gfx.Scene) -> None:
        if self._visual is not None:
            scene.remove(self._visual)
            self._visual = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = float(value)
        self._engine._subview_point_alpha = self._alpha

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, value) -> None:
        self._scaling = value
        # Rebuild material because size_space is immutable
        n = self._engine._n_particles
        scene = self._visual.parent if self._visual is not None else None
        if scene is None:
            return
        scene.remove(self._visual)
        self._material = SphereImpostorMaterial(
            size_mode="vertex",
            size_space="world" if value == "visual" else "screen",
            color_mode="vertex",
            ambient=D.LIGHT_AMBIENT,
        )
        self._visual = gfx.Points(self._geom, self._material)
        scene.add(self._visual)

    def update_from(self, positions: np.ndarray, mask: np.ndarray,
                    is_bh: np.ndarray, colors: np.ndarray, sizes: np.ndarray,
                    radius_scale: float, point_alpha: float) -> None:
        """Upload this frame's data, keyed by the engine's main-view state."""
        if self._visual is None:
            return
        n = len(mask)
        face = np.zeros((n, 4), dtype=np.float32)
        sz = np.zeros(n, dtype=np.float32)
        if mask.any():
            active_nonbh = mask & ~is_bh
            if active_nonbh.any():
                face[active_nonbh] = colors[active_nonbh]
                face[active_nonbh, 3] *= point_alpha
                sz[active_nonbh] = sizes[active_nonbh] * radius_scale
            active_bh = mask & is_bh
            if active_bh.any():
                # Sub-view: draw BHs as normal blobs (no separate ring pass
                # — keeps the sub-view visual simple; they still show up).
                face[active_bh, :3] = colors[active_bh, :3]
                face[active_bh, 3] = point_alpha
                sz[active_bh] = sizes[active_bh] * radius_scale
        pos = np.where(np.isfinite(positions), positions.astype(np.float32), 0.0)
        self._geom.positions.data[:] = pos
        self._geom.colors.data[:] = face
        self._geom.sizes.data[:] = sz
        self._geom.positions.update_range(0, n)
        self._geom.colors.update_range(0, n)
        self._geom.sizes.update_range(0, n)


# ---------------------------------------------------------------------------
# Orbit / view adapters
# ---------------------------------------------------------------------------


class _ViewAdapter:
    """Minimal VisPy-ViewBox substitute so CameraController stays unchanged.

    CameraController reads ``view.camera`` and then pokes ``center``,
    ``distance``, ``azimuth``, ``fov`` on it; those all exist on
    PygfxTurntableCamera.
    """

    def __init__(self, camera: PygfxTurntableCamera, scene: gfx.Scene):
        self.camera = camera
        self.scene = scene


class _PygfxOrbitAdapter:
    """Mouse drag rotation + right-drag pan, pushed into PygfxTurntableCamera.

    The built-in ``gfx.OrbitController`` manipulates the pygfx camera
    directly (position/orientation), which would bypass our turntable
    wrapper's azimuth/elevation state.  This adapter implements the
    same gestures but routes them through the wrapper so all consumers
    (light direction, framing, CameraController) stay in sync.
    """

    _LEFT = 1
    _RIGHT = 2
    _MIDDLE = 4

    def __init__(self, camera_wrap: PygfxTurntableCamera, canvas) -> None:
        self._camera = camera_wrap
        self._canvas = canvas
        self._drag_button = 0
        self._last_pos = (0, 0)
        canvas.add_event_handler(
            self._on_event,
            "pointer_down", "pointer_up", "pointer_move",
        )

    def _on_event(self, event) -> None:
        et = event["event_type"]
        if et == "pointer_down":
            self._drag_button = event.get("button", 0)
            self._last_pos = (event.get("x", 0), event.get("y", 0))
        elif et == "pointer_up":
            self._drag_button = 0
        elif et == "pointer_move" and self._drag_button:
            x, y = event.get("x", 0), event.get("y", 0)
            dx = x - self._last_pos[0]
            dy = y - self._last_pos[1]
            self._last_pos = (x, y)
            if self._drag_button == self._LEFT:
                # Rotate: horizontal → azimuth, vertical → elevation
                self._camera.azimuth = self._camera.azimuth - dx * 0.4
                self._camera.elevation = self._camera.elevation + dy * 0.4
                self._canvas.request_draw()
            elif self._drag_button == self._RIGHT:
                # Pan: right-drag moves the camera center in screen space
                az = math.radians(self._camera.azimuth)
                el = math.radians(self._camera.elevation)
                right = np.array([math.cos(az), math.sin(az), 0.0])
                up = np.array([
                    -math.sin(el) * math.sin(az),
                    math.sin(el) * math.cos(az),
                    math.cos(el),
                ])
                # Scale pan rate to camera distance so it feels consistent.
                step = self._camera.distance * 0.002
                center = np.array(self._camera.center, dtype=np.float64)
                center -= right * (dx * step)
                center += up * (dy * step)
                self._camera.center = tuple(center)
                self._canvas.request_draw()
