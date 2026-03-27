"""Qt-based GUI control panel for ScatterView."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.camera import CameraController
    from ..rendering.engine import RenderEngine


class ControlPanel:
    """Qt control panel alongside the VisPy 3D viewport.

    Provides controls for time, appearance, camera modes,
    sub-view, and export.
    """

    def __init__(self, engine: RenderEngine, camera_controller: CameraController):
        from PyQt6 import QtCore, QtGui, QtWidgets

        self._engine = engine
        self._camera = camera_controller
        self._qt = QtWidgets
        self._QtCore = QtCore

        self._app = QtWidgets.QApplication.instance()
        if self._app is None:
            self._app = QtWidgets.QApplication([])

        # Main window
        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle("ScatterView 2.0")
        self._window.resize(1400, 800)

        # Central widget with horizontal layout
        central = QtWidgets.QWidget()
        self._window.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # 3D viewport on the left (stretch factor 3)
        canvas_widget = engine.canvas.native
        layout.addWidget(canvas_widget, stretch=3)

        # Control panel on the right (stretch factor 1)
        self._panel = QtWidgets.QScrollArea()
        self._panel.setWidgetResizable(True)
        self._panel.setMinimumWidth(300)
        self._panel.setMaximumWidth(400)

        panel_widget = QtWidgets.QWidget()
        self._panel_layout = QtWidgets.QVBoxLayout(panel_widget)
        self._panel.setWidget(panel_widget)
        layout.addWidget(self._panel, stretch=1)

        # Build control sections
        self._build_time_controls()
        self._build_appearance_controls()
        self._build_camera_controls()
        self._build_subview_controls()
        self._build_export_controls()

        # Stretch at bottom
        self._panel_layout.addStretch()

    def _add_section(self, title: str) -> "QtWidgets.QVBoxLayout":
        from PyQt6 import QtWidgets

        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout()
        group.setLayout(layout)
        self._panel_layout.addWidget(group)
        return layout

    def _add_slider(
        self, layout, label: str, min_val: float, max_val: float,
        value: float, callback, steps: int = 100
    ):
        from PyQt6 import QtCore, QtWidgets

        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(steps)
        slider.setValue(int((value - min_val) / (max_val - min_val) * steps))
        val_label = QtWidgets.QLabel(f"{value:.3g}")

        def on_change(v):
            real_val = min_val + (v / steps) * (max_val - min_val)
            val_label.setText(f"{real_val:.3g}")
            callback(real_val)

        slider.valueChanged.connect(on_change)
        row.addWidget(lbl)
        row.addWidget(slider)
        row.addWidget(val_label)
        layout.addLayout(row)
        # Attach value label for programmatic updates
        slider._val_label = val_label
        return slider

    def _add_log_slider(
        self, layout, label: str, min_val: float, max_val: float,
        value: float, callback, steps: int = 100
    ):
        """Slider that maps linearly in log-space (logarithmic feel)."""
        from PyQt6 import QtCore, QtWidgets

        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        log_val = np.log10(max(value, min_val))

        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(steps)
        slider.setValue(int((log_val - log_min) / (log_max - log_min) * steps))
        val_label = QtWidgets.QLabel(f"{value:.3g}")

        def on_change(v):
            log_v = log_min + (v / steps) * (log_max - log_min)
            real_val = 10.0 ** log_v
            val_label.setText(f"{real_val:.3g}")
            callback(real_val)

        slider.valueChanged.connect(on_change)
        row.addWidget(lbl)
        row.addWidget(slider)
        row.addWidget(val_label)
        layout.addLayout(row)
        # Store log params and label for programmatic updates
        slider._val_label = val_label
        slider._log_min = log_min
        slider._log_max = log_max
        return slider

    def _build_time_controls(self) -> None:
        from PyQt6 import QtCore, QtWidgets

        section = self._add_section("Time")

        # Time slider
        self._t_min = float(self._engine._data.times[0])
        self._t_max = float(self._engine._data.times[-1])
        self._time_steps = 10000
        self._time_slider = self._add_slider(
            section, "Time", self._t_min, self._t_max, self._t_min,
            self._on_time_slider_changed,
            steps=self._time_steps,
        )
        self._time_val_label = self._time_slider._val_label

        # Sync timer: update slider from engine during playback
        self._sync_timer = QtCore.QTimer()
        self._sync_timer.setInterval(50)  # 20 Hz update
        self._sync_timer.timeout.connect(self._sync_time_slider)
        self._slider_updating = False  # guard against feedback loops

        # Play/Pause button
        row = QtWidgets.QHBoxLayout()
        self._play_btn = QtWidgets.QPushButton("Play")
        self._play_btn.clicked.connect(self._on_play_toggle)
        row.addWidget(self._play_btn)
        section.addLayout(row)

        # Speed slider (logarithmic, 1/60 at center)
        self._speed_slider = self._add_log_slider(
            section, "Speed", 0.001, 0.1, self._engine._anim_speed,
            self._on_speed_change,
        )


    def _on_time_slider_changed(self, value: float) -> None:
        """User dragged the time slider — update engine (but skip if syncing)."""
        if not self._slider_updating:
            self._engine.sim_time = value

    def _sync_time_slider(self) -> None:
        """Push current engine time back to the slider during playback."""
        if not self._engine.playing:
            return
        t = self._engine.sim_time
        frac = (t - self._t_min) / (self._t_max - self._t_min)
        slider_val = int(frac * self._time_steps)
        slider_val = max(0, min(self._time_steps, slider_val))

        self._time_slider.blockSignals(True)
        self._time_slider.setValue(slider_val)
        self._time_slider.blockSignals(False)
        self._time_val_label.setText(f"{t:.3g}")

    def _on_play_toggle(self) -> None:
        self._engine.toggle_play()
        playing = self._engine.playing
        self._play_btn.setText("Pause" if playing else "Play")
        if playing:
            self._sync_timer.start()
        else:
            self._sync_timer.stop()


    def _on_speed_change(self, speed: float) -> None:
        """Update playback speed."""
        self._engine.set_speed(speed)

    def _set_log_slider(self, slider, value: float) -> None:
        """Programmatically update a log-slider's position and label."""
        log_min = slider._log_min
        log_max = slider._log_max
        log_val = np.log10(max(value, 10.0 ** log_min))
        steps = slider.maximum()
        pos = int((log_val - log_min) / (log_max - log_min) * steps)
        pos = max(0, min(steps, pos))
        slider.blockSignals(True)
        slider.setValue(pos)
        slider.blockSignals(False)
        slider._val_label.setText(f"{value:.3g}")

    def _build_appearance_controls(self) -> None:
        from PyQt6 import QtWidgets

        section = self._add_section("Appearance")

        # Point alpha
        self._add_slider(
            section, "Point Alpha", 0.0, 1.0, self._engine._point_alpha,
            self._engine.set_point_alpha,
        )

        # Sizing mode toggle
        self._absolute_cb = QtWidgets.QCheckBox("Absolute Sizes (world units)")
        self._absolute_cb.setToolTip(
            "Absolute: radius in same units as x,y,z coordinates.\n"
            "Relative: radii normalized to screen pixels."
        )
        self._absolute_cb.toggled.connect(self._engine.set_sizing_mode)
        section.addWidget(self._absolute_cb)

        # Depth scaling toggle (uses VisPy's native perspective scaling)
        self._depth_cb = QtWidgets.QCheckBox("Depth Scaling")
        self._depth_cb.setToolTip(
            "Closer particles appear larger, farther ones smaller.\n"
            "Uses GPU-native perspective projection on marker sizes."
        )
        self._depth_cb.setChecked(self._engine._depth_scaling)
        self._depth_cb.toggled.connect(self._engine.set_depth_scaling)
        section.addWidget(self._depth_cb)

        # Global radius scale (logarithmic, 1.0 at center)
        self._add_log_slider(
            section, "Radius Scale", 0.1, 10.0, 1.0,
            self._engine.set_radius_scale,
        )

        # Per-particle scale sliders (only if manageable number)
        n_particles = len(self._engine._data.particle_ids)
        if n_particles <= 20:
            labels = self._engine._data.id_labels
            for pid in self._engine._data.particle_ids:
                pid_key = int(pid)
                name = labels[pid_key] if labels else str(pid_key)
                self._add_slider(
                    section, f"Scale {name}", 0.1, 5.0, 1.0,
                    lambda v, p=pid_key: self._engine.set_particle_size(p, v),
                )

        # Trail length (logarithmic, fraction of total simulation time)
        self._trail_slider = self._add_log_slider(
            section, "Trail (frac)", 0.001, 0.5, self._engine._trail_length_frac,
            self._engine.set_trail_length,
        )

        # Trail width
        self._add_slider(
            section, "Trail Width", 1.0, 10.0, self._engine._trail_width,
            self._engine.set_trail_width,
        )

        # Trail alpha
        self._add_slider(
            section, "Trail Alpha", 0.0, 1.0, self._engine._trail_alpha,
            self._engine.set_trail_alpha,
        )

    def _build_camera_controls(self) -> None:
        from PyQt6 import QtWidgets

        from ..core.camera import CameraMode, FramingScope

        section = self._add_section("Camera")

        # Mode selector
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Mode"))
        self._mode_combo = self._make_mode_combo()
        self._mode_names = self._camera_mode_names
        self._mode_combo.currentTextChanged.connect(self._on_camera_mode_change)
        row.addWidget(self._mode_combo)
        section.addLayout(row)

        # Framing scope selector
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Framing"))
        self._framing_combo = self._make_framing_combo()
        self._framing_names = self._framing_scope_names
        self._framing_combo.currentTextChanged.connect(self._on_framing_change)
        row.addWidget(self._framing_combo)
        section.addLayout(row)

        # Keep all in frame toggle
        self._keep_all_cb = QtWidgets.QCheckBox("Keep All in Frame")
        self._keep_all_cb.toggled.connect(
            lambda v: setattr(self._camera, "keep_all_in_frame", v)
        )
        section.addWidget(self._keep_all_cb)

        # Free zoom toggle
        self._free_zoom_cb = QtWidgets.QCheckBox("Free Zoom")
        self._free_zoom_cb.setChecked(self._camera.free_zoom)
        self._free_zoom_cb.setToolTip(
            "Manual zoom control (scroll wheel / trackpad).\n"
            "Auto-enabled when you scroll. Uncheck to restore auto-zoom."
        )
        self._free_zoom_cb.toggled.connect(
            lambda v: setattr(self._camera, "free_zoom", v)
        )
        self._camera._free_zoom_callbacks.append(self._on_free_zoom_changed)
        section.addWidget(self._free_zoom_cb)

        # Neighbor count (for NEAREST_NEIGHBORS scope)
        self._neighbor_slider = self._add_slider(
            section, "Neighbors", 1, 10, self._camera.n_neighbors,
            lambda v: setattr(self._camera, "n_neighbors", int(v)),
            steps=9,
        )

        # Auto-rotate toggle
        self._rotate_cb = QtWidgets.QCheckBox("Auto-Rotate")
        self._rotate_cb.toggled.connect(
            lambda v: setattr(self._camera, "auto_rotate", v)
        )
        section.addWidget(self._rotate_cb)

        # Rotation speed
        self._add_slider(
            section, "Rot. Speed", 0.0, 5.0, self._camera.rotation_speed,
            lambda v: setattr(self._camera, "rotation_speed", v),
        )

        # Camera pan/zoom speed (hidden for modes that don't use smoothing)
        self._speed_container = QtWidgets.QWidget()
        speed_layout = QtWidgets.QVBoxLayout(self._speed_container)
        speed_layout.setContentsMargins(0, 0, 0, 0)
        self._add_slider(
            speed_layout, "Pan Speed", 0.01, 0.5, self._camera._ema_alpha,
            lambda v: setattr(self._camera, "_ema_alpha", v),
        )
        self._add_slider(
            speed_layout, "Zoom Speed", 0.005, 0.2, self._camera._zoom_ema_alpha,
            lambda v: setattr(self._camera, "_zoom_ema_alpha", v),
        )
        section.addWidget(self._speed_container)

        # Target particle selector (searchable)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Target"))
        self._target_combo = self._make_particle_combo()
        self._target_combo.activated.connect(
            lambda _: self._on_target_change(self._target_combo.currentText())
        )
        row.addWidget(self._target_combo)
        section.addLayout(row)

    def _on_free_zoom_changed(self, value: bool) -> None:
        """Sync checkbox when free zoom is toggled externally (e.g. scroll wheel)."""
        if self._free_zoom_cb.isChecked() != value:
            self._free_zoom_cb.blockSignals(True)
            self._free_zoom_cb.setChecked(value)
            self._free_zoom_cb.blockSignals(False)

    def _make_particle_combo(self) -> "QtWidgets.QComboBox":
        """Create a searchable combo box with all particle IDs."""
        from PyQt6 import QtCore, QtWidgets

        combo = QtWidgets.QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        combo.addItem("None")
        labels = self._engine._data.id_labels
        for pid in self._engine._data.particle_ids:
            pid_key = int(pid)
            name = labels[pid_key] if labels else str(pid_key)
            combo.addItem(name)
        # Enable filtering by typing
        completer = QtWidgets.QCompleter(
            [combo.itemText(i) for i in range(combo.count())]
        )
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        combo.setCompleter(completer)
        return combo

    def _make_mode_combo(self) -> "QtWidgets.QComboBox":
        """Create a camera mode combo box with per-item tooltips."""
        from PyQt6 import QtWidgets

        from ..core.camera import CameraMode

        self._camera_mode_names = {
            "Manual": CameraMode.MANUAL,
            "Auto-Frame": CameraMode.AUTO_FRAME,
            "Auto-Rotate": CameraMode.AUTO_ROTATE,
            "Event Track": CameraMode.EVENT_TRACK,
            "Target (Rest)": CameraMode.TARGET_REST_FRAME,
            "Target (Comoving)": CameraMode.TARGET_COMOVING,
        }
        tips = {
            "Manual": (
                "Free trackball camera.\n"
                "Drag to rotate, scroll to zoom, middle-drag to pan."
            ),
            "Auto-Frame": (
                "Automatically center and zoom to keep the framed\n"
                "particles in view. Which particles are framed is\n"
                "controlled by the Framing dropdown below."
            ),
            "Auto-Rotate": (
                "Like Auto-Frame, but also slowly orbits the camera\n"
                "around the vertical axis. Good for presentations\n"
                "and video export."
            ),
            "Event Track": (
                "Smoothly zoom into detected events (close encounters,\n"
                "mergers) as they approach in simulation time. Falls\n"
                "back to Auto-Frame between events."
            ),
            "Target (Rest)": (
                "Lock the camera center on the selected Target particle.\n"
                "Everything else moves relative to the target. Useful\n"
                "for studying one star's interactions."
            ),
            "Target (Comoving)": (
                "Follow the Target particle with a smoothed velocity\n"
                "estimate, reducing jitter from orbital oscillations.\n"
                "The target stays roughly centered."
            ),
        }
        combo = QtWidgets.QComboBox()
        for i, name in enumerate(self._camera_mode_names):
            combo.addItem(name)
            combo.setItemData(i, tips.get(name, ""), self._QtCore.Qt.ItemDataRole.ToolTipRole)
        combo.setToolTip("Camera behavior mode. Hover items for details.")
        return combo

    def _make_framing_combo(self) -> "QtWidgets.QComboBox":
        """Create a framing scope combo box with per-item tooltips."""
        from PyQt6 import QtWidgets

        from ..core.camera import FramingScope

        self._framing_scope_names = {
            "Core Group": FramingScope.CORE_GROUP,
            "Nearest Neighbors": FramingScope.NEAREST_NEIGHBORS,
            "All Particles": FramingScope.ALL,
        }
        tips = {
            "Core Group": (
                "Ignore outlier particles (beyond 2x the median distance\n"
                "from the centroid). The camera frames the dense core\n"
                "without chasing ejected bodies."
            ),
            "Nearest Neighbors": (
                "Frame only the Target particle and its K nearest\n"
                "neighbors (set K with the Neighbors slider).\n"
                "Requires a Target to be selected; otherwise falls\n"
                "back to Core Group."
            ),
            "All Particles": (
                "Frame every active particle. The camera always\n"
                "zooms out far enough to keep everything visible,\n"
                "including ejected particles."
            ),
        }
        combo = QtWidgets.QComboBox()
        for i, name in enumerate(self._framing_scope_names):
            combo.addItem(name)
            combo.setItemData(i, tips.get(name, ""), self._QtCore.Qt.ItemDataRole.ToolTipRole)
        combo.setToolTip("Which particles drive camera center and zoom. Hover items for details.")
        return combo

    def _on_camera_mode_change(self, text: str) -> None:
        from ..core.camera import CameraMode

        mode = self._mode_names.get(text, CameraMode.MANUAL)
        self._camera.mode = mode
        if mode == CameraMode.AUTO_ROTATE:
            self._rotate_cb.setChecked(True)

        # Show pan/zoom speed sliders only for modes that use smoothing
        uses_smoothing = mode in (
            CameraMode.AUTO_FRAME, CameraMode.AUTO_ROTATE,
            CameraMode.EVENT_TRACK, CameraMode.TARGET_COMOVING,
        )
        self._speed_container.setVisible(uses_smoothing)

    def _on_framing_change(self, text: str) -> None:
        scope = self._framing_names.get(text)
        if scope is not None:
            self._camera.framing_scope = scope

    def _resolve_pid(self, text: str) -> int | None:
        """Resolve a label string from a combo box to an integer particle ID."""
        if text == "None":
            return None
        labels = self._engine._data.id_labels
        if labels:
            label_to_id = {v: k for k, v in labels.items()}
            return label_to_id.get(text)
        try:
            return int(text)
        except ValueError:
            return None

    def _on_target_change(self, text: str) -> None:
        self._camera.target_particle = self._resolve_pid(text)

    def _build_subview_controls(self) -> None:
        from PyQt6 import QtWidgets

        from ..core.camera import CameraMode, FramingScope

        section = self._add_section("Sub-View (PiP)")

        self._subview_cb = QtWidgets.QCheckBox("Enable Sub-View")
        self._subview_cb.toggled.connect(self._on_subview_toggle)
        section.addWidget(self._subview_cb)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Corner"))
        self._corner_combo = QtWidgets.QComboBox()
        for corner in ["bottom-right", "bottom-left", "top-right", "top-left"]:
            self._corner_combo.addItem(corner)
        row.addWidget(self._corner_combo)
        section.addLayout(row)

        # Sub-view camera mode
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Mode"))
        self._subview_mode_combo = self._make_mode_combo()
        self._subview_mode_names = self._camera_mode_names
        self._subview_mode_combo.setCurrentText("Auto-Frame")
        self._subview_mode_combo.currentTextChanged.connect(self._on_subview_mode_change)
        row.addWidget(self._subview_mode_combo)
        section.addLayout(row)

        # Sub-view framing scope
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Framing"))
        self._subview_framing_combo = self._make_framing_combo()
        self._subview_framing_names = self._framing_scope_names
        self._subview_framing_combo.currentTextChanged.connect(self._on_subview_framing_change)
        row.addWidget(self._subview_framing_combo)
        section.addLayout(row)

        # Sub-view target (searchable)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Target"))
        self._subview_target_combo = self._make_particle_combo()
        self._subview_target_combo.activated.connect(
            lambda _: self._on_subview_target_change(self._subview_target_combo.currentText())
        )
        row.addWidget(self._subview_target_combo)
        section.addLayout(row)

        # Sub-view neighbor count
        from .. import defaults as _D
        self._add_slider(
            section, "Neighbors", 1, 10, _D.CAMERA_N_NEIGHBORS,
            lambda v: self._set_subview_neighbors(int(v)),
            steps=9,
        )

        # Sub-view auto-rotate
        self._subview_rotate_cb = QtWidgets.QCheckBox("Auto-Rotate")
        self._subview_rotate_cb.toggled.connect(self._on_subview_rotate_toggle)
        section.addWidget(self._subview_rotate_cb)

    def _on_subview_toggle(self, enabled: bool) -> None:
        if enabled:
            corner = self._corner_combo.currentText()
            self._engine.enable_subview(corner=corner)
            # Apply current sub-view settings to the new controller
            self._on_subview_mode_change(self._subview_mode_combo.currentText())
            self._on_subview_framing_change(self._subview_framing_combo.currentText())
            self._on_subview_target_change(self._subview_target_combo.currentText())
            self._on_subview_rotate_toggle(self._subview_rotate_cb.isChecked())
        else:
            self._engine.disable_subview()

    def _on_subview_mode_change(self, text: str) -> None:
        from ..core.camera import CameraMode

        ctrl = self._engine._subview_camera_controller
        if ctrl is None:
            return
        mode = self._subview_mode_names.get(text)
        if mode is not None:
            ctrl.mode = mode
            if mode == CameraMode.AUTO_ROTATE:
                self._subview_rotate_cb.setChecked(True)

    def _on_subview_framing_change(self, text: str) -> None:
        ctrl = self._engine._subview_camera_controller
        if ctrl is None:
            return
        scope = self._subview_framing_names.get(text)
        if scope is not None:
            ctrl.framing_scope = scope

    def _on_subview_target_change(self, text: str) -> None:
        ctrl = self._engine._subview_camera_controller
        if ctrl is not None:
            ctrl.target_particle = self._resolve_pid(text)

    def _set_subview_neighbors(self, n: int) -> None:
        ctrl = self._engine._subview_camera_controller
        if ctrl is not None:
            ctrl.n_neighbors = n

    def _on_subview_rotate_toggle(self, enabled: bool) -> None:
        ctrl = self._engine._subview_camera_controller
        if ctrl is not None:
            ctrl.auto_rotate = enabled

    def _build_export_controls(self) -> None:
        from PyQt6 import QtWidgets

        section = self._add_section("Export")

        # Screenshot
        self._screenshot_btn = QtWidgets.QPushButton("Screenshot (PNG)")
        self._screenshot_btn.clicked.connect(self._on_screenshot)
        section.addWidget(self._screenshot_btn)

        # Video record
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Duration (s)"))
        self._duration_spin = QtWidgets.QDoubleSpinBox()
        self._duration_spin.setRange(1.0, 300.0)
        self._duration_spin.setValue(10.0)
        row.addWidget(self._duration_spin)
        section.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("FPS"))
        self._fps_spin = QtWidgets.QSpinBox()
        self._fps_spin.setRange(1, 120)
        self._fps_spin.setValue(30)
        row.addWidget(self._fps_spin)
        section.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Resolution"))
        self._res_w = QtWidgets.QSpinBox()
        self._res_w.setRange(320, 7680)
        self._res_w.setValue(1920)
        self._res_h = QtWidgets.QSpinBox()
        self._res_h.setRange(240, 4320)
        self._res_h.setValue(1080)
        row.addWidget(self._res_w)
        row.addWidget(QtWidgets.QLabel("x"))
        row.addWidget(self._res_h)
        section.addLayout(row)

        self._record_btn = QtWidgets.QPushButton("Record Video (MP4)")
        self._record_btn.clicked.connect(self._on_record)
        section.addWidget(self._record_btn)

    def _on_screenshot(self) -> None:
        from PyQt6 import QtWidgets

        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._window, "Save Screenshot", "screenshot.png", "PNG (*.png)"
        )
        if filepath:
            self._engine.screenshot(filepath)

    def _on_record(self) -> None:
        from PyQt6 import QtWidgets

        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._window, "Save Video", "video.mp4", "MP4 (*.mp4);;GIF (*.gif)"
        )
        if filepath:
            duration = self._duration_spin.value()
            fps = self._fps_spin.value()
            size = (self._res_w.value(), self._res_h.value())
            self._engine.render_video(filepath, duration=duration, fps=fps, size=size)

    def show(self) -> None:
        """Show the main window and start Qt event loop."""
        self._window.show()
        if self._engine._timer is not None:
            self._engine._timer.start()
        self._app.exec()

    @property
    def window(self):
        return self._window
