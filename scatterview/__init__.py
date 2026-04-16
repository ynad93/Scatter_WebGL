"""ScatterView: N-body simulation visualization tool."""

import os as _os
import sys as _sys

# On Linux (including WSL2), prefer the Vulkan backend over the default
# OpenGL-via-D3D12 fallback that lacks features like float32-filterable
# on WSL2 GPU passthrough.  `setdefault` leaves a user-supplied value
# untouched.
if _sys.platform.startswith("linux"):
    _os.environ.setdefault("WGPU_BACKEND_TYPE", "Vulkan")

__version__ = "2.0.0"
