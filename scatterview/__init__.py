"""ScatterView: N-body simulation visualization tool."""

import os
import sys

# Qt's Wayland backend breaks QComboBox popups when a QOpenGLWidget is
# embedded in the same window: the popup opens but can't create a
# grabbing window ("Failed to create grabbing popup ... transientParent")
# so clicks never reach it and it lingers as a ghost image until the
# compositor times it out.  Force the xcb (X11) platform on Linux when
# Wayland is detected; Xwayland handles this correctly on WSLg and on
# standard Linux desktops.  Users can opt out by setting QT_QPA_PLATFORM
# themselves before launching.
if sys.platform.startswith("linux") and "QT_QPA_PLATFORM" not in os.environ:
    if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland":
        os.environ["QT_QPA_PLATFORM"] = "xcb"


# Route VisPy's DPI query through Qt.  VisPy's Linux DPI probe shells
# out to `xdpyinfo` / `xrandr`; on WSLg (and any headless/virtual X
# display) xrandr reports a physical screen size of 0mm, so VisPy
# warns and falls back to 96.  Qt, meanwhile, already has a correct
# logical DPI from its platform integration.  Overriding VisPy's
# `get_dpi` to read `QScreen.logicalDotsPerInch()` gives VisPy the real
# value and eliminates the xrandr shell-out entirely.
def _qt_get_dpi(raise_error=True):
    from PyQt6 import QtGui

    app = QtGui.QGuiApplication.instance()
    if app is None:
        return 96.0
    screen = app.primaryScreen()
    if screen is None:
        return 96.0
    return float(screen.logicalDotsPerInch())


def _install_vispy_dpi_shim():
    import vispy.util.dpi as _dpi

    _dpi.get_dpi = _qt_get_dpi
    # vispy.app.canvas binds get_dpi at module load; patch it too in
    # case canvas was imported before this shim ran.
    try:
        import vispy.app.canvas as _canvas
        _canvas.get_dpi = _qt_get_dpi
    except ImportError:
        pass


_install_vispy_dpi_shim()


__version__ = "2.0.0"
