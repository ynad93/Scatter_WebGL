"""ScatterView: N-body simulation visualization tool."""

# Waive pygfx's default ``float32-filterable`` feature requirement so the
# WGPU device can bind to the real GPU on systems where that feature
# isn't exposed (WSL2 D3D12/OpenGL passthrough, some drivers).  Without
# this waiver pygfx falls back to the CPU (lavapipe) adapter and
# rendering becomes 10-100x slower.  None of ScatterView's render paths
# (points, lines, text, background) use float32-filterable sampling,
# so dropping it is safe here.
#
# Must run BEFORE pygfx's first device creation, i.e. before
# scatterview.rendering.engine (or any module that imports pygfx) is
# imported.  Import order: scatterview -> __init__.py -> user code ->
# rendering.engine.
from pygfx.renderers.wgpu import enable_wgpu_features as _enable
_enable("!float32-filterable")
del _enable

__version__ = "2.0.0"
