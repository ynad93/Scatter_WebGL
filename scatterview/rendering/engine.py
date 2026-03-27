"""VisPy-based OpenGL rendering engine for N-body visualization."""

from __future__ import annotations

from pathlib import Path

import numba as nb
import numpy as np

from .. import defaults as D
from ..core.data_loader import SimulationData
from ..core.interpolation import TrajectoryInterpolator
from ..core.time_mapping import TimeMapping


@nb.njit(cache=True)
def _advance_trail_pointers(
    packed_times: np.ndarray,   # (total_pts,) float64 — all precomputed times
    lo: np.ndarray,             # (n,) int64 — start offset per particle
    hi: np.ndarray,             # (n,) int64 — end offset per particle
    si_prev: np.ndarray,        # (n,) int64 — previous si (-1 = needs full search)
    ei_prev: np.ndarray,        # (n,) int64 — previous ei (-1 = needs full search)
    t_start: float,
    t_end: float,
    forward: bool,
    si_out: np.ndarray,         # (n,) int64 — output
    ei_out: np.ndarray,         # (n,) int64 — output
) -> None:
    """Advance or binary-search trail window pointers for all particles."""
    for k in range(len(lo)):
        lo_k = lo[k]
        n_k = hi[k] - lo_k

        if forward and si_prev[k] >= 0:
            # Fast path: advance from previous position
            s = si_prev[k]
            while s < n_k and packed_times[lo_k + s] <= t_start:
                s += 1
            e = ei_prev[k]
            while e < n_k and packed_times[lo_k + e] < t_end:
                e += 1
        else:
            # Slow path: binary search
            # searchsorted(side='right') for si
            s_lo, s_hi = nb.int64(0), n_k
            while s_lo < s_hi:
                mid = (s_lo + s_hi) >> 1
                if packed_times[lo_k + mid] <= t_start:
                    s_lo = mid + 1
                else:
                    s_hi = mid
            s = s_lo
            # searchsorted(side='left') for ei
            e_lo, e_hi = nb.int64(0), n_k
            while e_lo < e_hi:
                mid = (e_lo + e_hi) >> 1
                if packed_times[lo_k + mid] < t_end:
                    e_lo = mid + 1
                else:
                    e_hi = mid
            e = e_lo

        si_out[k] = s
        ei_out[k] = e


class RenderEngine:
    """Manages the VisPy canvas, particle and trail rendering, camera, and animation."""

    def __init__(
        self, data: SimulationData, interpolator: TrajectoryInterpolator,
        time_mapping: TimeMapping, size: tuple[int, int] = (1280, 720),
        title: str = "ScatterView",
    ):
        from vispy import app, scene

        self._data = data
        self._interp = interpolator
        self._time_mapping = time_mapping

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

        # Pre-computed alpha lookup table (eliminates per-particle linspace + pow)
        self._alpha_lut = (np.linspace(0, 1, 1024) ** 1.5).astype(np.float32)

        self._build_visuals()

        # Timer and camera
        self._timer = app.Timer(interval=1.0 / 60.0, connect=self._on_timer, start=False)
        self._camera_controller = None

        # Auto-enable free zoom on scroll wheel / trackpad zoom
        self._canvas.events.mouse_wheel.connect(self._on_mouse_wheel)

    def _on_mouse_wheel(self, event) -> None:
        """Enable free zoom when the user scrolls."""
        if self._camera_controller is not None and not self._camera_controller.free_zoom:
            self._camera_controller.free_zoom = True

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
    # Trail rendering — precomputed trails with sliding-window extraction
    # ------------------------------------------------------------------

    def _update_trails(self, time: float, positions: np.ndarray,
                       active_ids: np.ndarray) -> None:
        from vispy import scene

        time_range = self._data.times[-1] - self._data.times[0]
        trail_length = time_range * self._trail_length_frac
        t_start = max(time - trail_length, self._data.times[0])
        if t_start >= time:
            if self._trail_line is not None:
                self._trail_line.visible = False
            return

        precomp = self._precomp
        lut = self._alpha_lut
        lut_max = len(lut) - 1
        n_active = len(active_ids)
        inv_window = 1.0 / (time - t_start)

        # --- Vectorized window lookup for all particles at once ---
        # Map active IDs to precomp indices via the lookup table
        buf_idx = self._pid_lookup[active_ids.astype(np.intp)]
        valid = buf_idx >= 0
        lo_all = precomp.offsets[buf_idx[valid]]
        hi_all = precomp.offsets[buf_idx[valid] + 1]
        has_data = lo_all < hi_all

        # Build arrays only for particles that have precomputed data
        valid_idx = np.where(valid)[0][has_data]
        n_valid = len(valid_idx)
        if n_valid == 0:
            if self._trail_line is not None:
                self._trail_line.visible = False
            return

        lo = precomp.offsets[buf_idx[valid_idx]]
        hi = precomp.offsets[buf_idx[valid_idx] + 1]
        n_precomp = hi - lo  # (n_valid,) — points per particle

        # Two-pointer sliding window via numba: during forward playback,
        # advance si/ei from the previous frame (O(1) per particle).
        # Falls back to binary search on first frame, backward scrub, or loop.
        particle_idx = buf_idx[valid_idx]  # index into _trail_si/_trail_ei
        forward = time >= self._trail_prev_time
        self._trail_prev_time = time

        si_arr = np.empty(n_valid, dtype=np.int64)
        ei_arr = np.empty(n_valid, dtype=np.int64)

        _advance_trail_pointers(
            precomp.times, lo, hi,
            self._trail_si[particle_idx],
            self._trail_ei[particle_idx],
            t_start, time, forward,
            si_arr, ei_arr,
        )

        self._trail_si[particle_idx] = si_arr
        self._trail_ei[particle_idx] = ei_arr

        body_counts = np.maximum(ei_arr - si_arr, 0)
        body_starts = lo + si_arr  # absolute index into packed arrays

        # --- Vectorized tail lerp ---
        # Tail exists when si > 0 (there's a precomputed point before t_start)
        can_tail = (si_arr > 0) & (si_arr <= n_precomp)
        # Clamp to valid range — particles with si=0 get dummy indices
        # (their results are masked out by can_tail / has_tail).
        i_before = np.maximum(lo + si_arr - 1, lo)
        i_after = np.minimum(lo + si_arr, hi - 1)
        t_before = precomp.times[i_before]
        t_after = precomp.times[i_after]
        dt_tail = t_after - t_before
        has_tail = can_tail & (dt_tail > 0)
        alpha_tail = np.where(has_tail, (t_start - t_before) / np.maximum(dt_tail, 1e-30), 0.0)
        # Compute all lerp'd tail positions (unused ones will be skipped)
        tail_pos_all = (
            precomp.positions[i_before] * (1 - alpha_tail[:, np.newaxis])
            + precomp.positions[i_after] * alpha_tail[:, np.newaxis]
        )

        # --- Compute per-particle point counts and filter ---
        counts = body_counts + 1 + has_tail.astype(np.int64)  # +1 for head
        active_mask = counts >= 2
        if not active_mask.any():
            if self._trail_line is not None:
                self._trail_line.visible = False
            return

        # Filter to only particles with >= 2 trail points
        a_idx = np.where(active_mask)[0]
        n_trails = len(a_idx)
        a_counts = counts[a_idx]
        a_body_starts = body_starts[a_idx]
        a_body_counts = body_counts[a_idx]
        a_has_tail = has_tail[a_idx]
        a_tail_pos = tail_pos_all[a_idx]
        a_valid_idx = valid_idx[a_idx]  # index into active_ids / positions

        total_pts = int(a_counts.sum()) + n_trails - 1  # +NaN separators

        # --- Reuse pre-allocated arrays (grow if needed) ---
        if total_pts > self._trail_capacity:
            self._trail_capacity = int(total_pts * 1.5)
            self._combined_pos = np.empty((self._trail_capacity, 3), dtype=np.float32)
            self._combined_colors = np.empty((self._trail_capacity, 4), dtype=np.float32)
            self._combined_times = np.empty(self._trail_capacity, dtype=np.float64)
        combined_pos = self._combined_pos[:total_pts]
        combined_colors = self._combined_colors[:total_pts]

        # Compute write offsets for each particle's block
        # write_offsets[k] = start index in combined arrays for trail k.
        # Layout: [trail_0] [NaN] [trail_1] [NaN] ... [trail_N-1]
        write_offsets = np.empty(n_trails, dtype=np.int64)
        write_offsets[0] = 0
        if n_trails > 1:
            cum = np.cumsum(a_counts)
            for k in range(1, n_trails):
                write_offsets[k] = cum[k - 1] + k  # k NaN separators before trail k

        # Pre-fill NaN separators
        for k in range(1, n_trails):
            sep_idx = write_offsets[k] - 1
            combined_pos[sep_idx] = np.nan
            combined_colors[sep_idx] = np.nan

        # Look up color indices for all active trail particles
        color_idx = self._pid_lookup[active_ids[a_valid_idx].astype(np.intp)]

        # Write each particle's trail data
        for k in range(n_trails):
            off = write_offsets[k]
            count = int(a_counts[k])
            bc = int(a_body_counts[k])
            w = 0

            # Tail
            if a_has_tail[k]:
                combined_pos[off] = a_tail_pos[k]
                w = 1

            # Body (bulk copy from precomputed)
            if bc > 0:
                bs = int(a_body_starts[k])
                combined_pos[off + w:off + w + bc] = precomp.positions[bs:bs + bc]
                w += bc

            # Head
            combined_pos[off + w] = positions[a_valid_idx[k]]

            # Colors: base RGB + time-based alpha
            combined_colors[off:off + count, :3] = self._colors[color_idx[k], :3]

        # --- Vectorized time-based alpha for ALL trail points at once ---
        combined_times = self._combined_times[:total_pts]
        combined_times[:] = np.nan  # NaN separators get NaN time (ignored)

        for k in range(n_trails):
            off = write_offsets[k]
            count = int(a_counts[k])
            bc = int(a_body_counts[k])
            w = 0
            if a_has_tail[k]:
                combined_times[off] = t_start
                w = 1
            if bc > 0:
                bs = int(a_body_starts[k])
                combined_times[off + w:off + w + bc] = precomp.times[bs:bs + bc]
                w += bc
            combined_times[off + w] = time

        # Single vectorized alpha computation across all points
        valid_mask = ~np.isnan(combined_times)
        t_frac = np.zeros(total_pts)
        t_frac[valid_mask] = (combined_times[valid_mask] - t_start) * inv_window
        lut_idx = (t_frac * lut_max).astype(np.intp)
        combined_colors[:, 3] = lut[lut_idx] * self._trail_alpha

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
            self._current_sim_time = float(
                self._time_mapping.anim_to_sim(self._anim_time),
            )
        self._update_frame()

    def _update_frame(self) -> None:
        positions, active_ids, _ = self._interp.evaluate_batch(self._current_sim_time)
        if len(positions) == 0:
            self._canvas.update()
            return

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
        self._current_sim_time = np.clip(value, self._data.times[0], self._data.times[-1])
        self._anim_time = float(self._time_mapping.sim_to_anim(self._current_sim_time))
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
        frame_sim_times = self._time_mapping.get_frame_times(n_frames)
        filepath = Path(filepath)
        render_size = size or self._canvas.size

        with iio.imopen(str(filepath), "w") as writer:
            for i, t in enumerate(frame_sim_times):
                self._current_sim_time = t
                self._anim_time = float(self._time_mapping.sim_to_anim(t))
                self._update_frame()

                img = self._canvas.render(size=render_size)
                writer.write(img, plugin="pyav", codec="libx264", fps=fps)

                if (i + 1) % (n_frames // 10 or 1) == 0:
                    print(f"Rendering: {(i + 1) / n_frames * 100:.0f}%")

        print(f"Video saved to {filepath}")
