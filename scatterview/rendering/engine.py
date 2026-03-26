"""VisPy-based OpenGL rendering engine for N-body visualization."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .. import defaults as D
from ..core.data_loader import SimulationData
from ..core.interpolation import TrajectoryInterpolator
from ..core.time_mapping import TimeMapping


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

        # Trail cache: parallel lists per particle
        self._trail_times: dict[int, list[float]] = {}
        self._trail_pos: dict[int, list[np.ndarray]] = {}  # list of (3,) arrays
        self._trail_pos_array: dict[int, np.ndarray] = {}  # cached (N,3) for GPU
        self._trail_dirty: set[int] = set()

        # Pre-computed alpha lookup table (eliminates per-particle linspace + pow)
        self._alpha_lut = (np.linspace(0, 1, 1024) ** 1.5).astype(np.float32)

        self._build_visuals()

        # Timer and camera
        self._timer = app.Timer(interval=1.0 / 120.0, connect=self._on_timer, start=False)
        self._camera_controller = None

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
        idx = np.array(
            [self._id_to_idx.get(int(pid), 0) for pid in active_ids], dtype=np.intp,
        )
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
    # Trail rendering (no cache — evaluate fresh each frame)
    # ------------------------------------------------------------------

    def _trail_append(self, pid_key: int, head: np.ndarray,
                      time: float, trail_length: float):
        """Append head position and trim expired tail points.

        Args:
            pid_key: Particle ID.
            head: (3,) position array (already evaluated by caller).
            time: Current simulation time.
            trail_length: Trail window in time units.
        """
        t_start = max(time - trail_length, self._data.times[0])
        if t_start >= time:
            return

        times = self._trail_times.get(pid_key)

        # Cache miss: seed from full evaluation
        if times is None:
            result = self._interp.evaluate_trail(pid_key, time, trail_length)
            if result is None or len(result[0]) < 2:
                return
            pos_arr, t_arr = result
            self._trail_times[pid_key] = t_arr.tolist()
            self._trail_pos[pid_key] = [pos_arr[i] for i in range(len(pos_arr))]
            self._trail_dirty.add(pid_key)
            return

        pos = self._trail_pos[pid_key]

        # Trim expired tail points
        trim_idx = 0
        while trim_idx < len(times) and times[trim_idx] < t_start:
            trim_idx += 1
        if trim_idx > 0:
            del times[:trim_idx]
            del pos[:trim_idx]

        if not times or time <= times[-1]:
            if trim_idx > 0:
                self._trail_dirty.add(pid_key)
            return

        # Subdivide if the new segment bends sharply
        last_t = times[-1]
        if len(pos) >= 2:
            d1 = pos[-1] - pos[-2]
            d2 = head - pos[-1]
            dot = (d1 * d2).sum()
            cos_a = dot / np.sqrt((d1 * d1).sum() * (d2 * d2).sum())
            n_fill = int(math.ceil(math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))))
            if n_fill > 1:
                fill_times = np.linspace(last_t, time, n_fill + 1)[1:]
                fill_pos = self._interp.evaluate_spline(pid_key, fill_times)
                if fill_pos is not None:
                    times.extend(fill_times.tolist())
                    pos.extend(fill_pos)
                    self._trail_dirty.add(pid_key)
                    return

        times.append(time)
        pos.append(head)
        self._trail_dirty.add(pid_key)

    def _update_trails(self, time: float, positions: np.ndarray,
                       active_ids: np.ndarray) -> None:
        from vispy import scene

        time_range = self._data.times[-1] - self._data.times[0]
        trail_length = time_range * self._trail_length_frac

        # First frame: batch evaluate to seed all caches at once
        if not self._trail_times:
            active_pids = [int(pid) for pid in active_ids]
            batch = self._interp.evaluate_trails_batch(active_pids, time, trail_length)
            for pid_key, (pos_arr, t_arr) in batch.items():
                if len(pos_arr) >= 2:
                    self._trail_times[pid_key] = t_arr.tolist()
                    self._trail_pos[pid_key] = [pos_arr[i] for i in range(len(pos_arr))]

        # Update each particle's trail — positions already evaluated by caller
        for i, pid in enumerate(active_ids):
            self._trail_append(int(pid), positions[i], time, trail_length)

        # Build combined GPU array
        active_pids = [int(pid) for pid in active_ids]
        total_pts = 0
        trail_lengths = {}
        for pid_key in active_pids:
            t = self._trail_times.get(pid_key)
            if t and len(t) >= 2:
                trail_lengths[pid_key] = len(t)
                total_pts += len(t)
        n_trails = len(trail_lengths)
        if n_trails == 0:
            if self._trail_line is not None:
                self._trail_line.visible = False
            return
        total_pts += n_trails - 1  # NaN separators

        combined_pos = np.empty((total_pts, 3))
        combined_colors = np.empty((total_pts, 4))
        offset = 0

        lut = self._alpha_lut
        lut_max = len(lut) - 1

        for pid_key, n in trail_lengths.items():
            if offset > 0:
                combined_pos[offset] = np.nan
                combined_colors[offset] = np.nan
                offset += 1

            # Rebuild cached array only when dirty
            if pid_key in self._trail_dirty or pid_key not in self._trail_pos_array:
                self._trail_pos_array[pid_key] = np.array(self._trail_pos[pid_key])
                self._trail_dirty.discard(pid_key)
            combined_pos[offset:offset + n] = self._trail_pos_array[pid_key]

            base_color = self._colors[self._id_to_idx.get(pid_key, 0)]
            combined_colors[offset:offset + n, :3] = base_color[:3]
            indices = (np.arange(n) * (lut_max / max(n - 1, 1))).astype(np.intp)
            combined_colors[offset:offset + n, 3] = lut[indices] * self._trail_alpha
            offset += n

        if self._trail_line is not None:
            self._trail_line.set_data(pos=combined_pos, color=combined_colors)
            self._trail_line.visible = True
        else:
            self._trail_line = scene.Line(
                pos=combined_pos, color=combined_colors,
                parent=self._view.scene, width=self._trail_width,
                antialias=True, connect="strip",
            )

        if self._subview_enabled and self._subview is not None:
            if self._subview_trail_line is not None:
                self._subview_trail_line.set_data(pos=combined_pos, color=combined_colors)
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
                self._trail_times.clear()
                self._trail_pos.clear()
                self._trail_pos_array.clear()
                self._trail_dirty.clear()
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
        old = self._current_sim_time
        self._current_sim_time = np.clip(value, self._data.times[0], self._data.times[-1])
        self._anim_time = float(self._time_mapping.sim_to_anim(self._current_sim_time))
        if self._current_sim_time < old:
            self._trail_times.clear()
            self._trail_pos.clear()
            self._trail_pos_array.clear()
            self._trail_dirty.clear()
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
        self._trail_times.clear()
        self._trail_pos.clear()
        self._trail_pos_array.clear()
        self._trail_dirty.clear()

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
