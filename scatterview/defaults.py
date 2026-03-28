"""Single source of truth for all ScatterView default settings.

Every default value lives here. The engine, CLI, and GUI all read from
this module rather than hardcoding their own values.
"""

# --- Animation ---
ANIM_SPEED = 1.0 / 60.0       # fraction of total sim duration advanced per second
                               # (1/60 ≈ 60-second full playback at 1x speed)

# --- Appearance ---
POINT_ALPHA = 1.0              # particle opacity (0 = invisible, 1 = opaque)
TRAIL_ALPHA = 0.6              # peak trail opacity (at the head / newest point)
TRAIL_LENGTH_FRAC = 0.005      # trail window as fraction of total simulation time
TRAIL_WIDTH = 3.0              # trail line width in pixels

# --- Particle sizing ---
RELATIVE_SIZE_MIN_PX = 3.0     # smallest particle in relative mode (screen pixels)
RELATIVE_SIZE_MAX_PX = 20.0    # largest particle in relative mode (screen pixels)
DEFAULT_SIZE_PX = 10.0         # uniform size when no radii are provided
DEPTH_SCALING = False          # if True, closer particles appear larger (perspective)

# --- Camera ---
CAMERA_FOV = 45                # field of view in degrees
CAMERA_EMA_ALPHA = 0.15        # event-track camera smoothing: max displacement per
                               # frame as a fraction of the current camera distance
CAMERA_N_NEIGHBORS = 3         # nearest-neighbors framing: track the target
                               # particle plus this many closest neighbors
ROTATION_SPEED = 0.5           # auto-rotate: degrees of azimuth per frame

# --- Black hole rendering (BSE stellar evolution codes) ---
BH_STARTYPE = 14               # BSE stellar type code identifying black holes
BH_FACE_COLOR = (0.02, 0.02, 0.05, 0.15)  # near-black with low opacity
BH_EDGE_WIDTH = 2.0            # edge ring width in pixels

# --- Lighting (world-space directional light for spherical markers) ---
# The light direction is transformed into eye space each frame so that
# shading changes as you orbit.  Offset from the default camera view
# direction so particles show visible shadow contrast on startup.
LIGHT_AMBIENT = 0.15
LIGHT_COLOR = "white"
LIGHT_POSITION = (-0.5, -0.3, 1.0)  # upper-left, offset from default camera

# --- Window ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
SUBVIEW_FOV = 60

# --- Video export ---
VIDEO_DURATION = 10.0          # default export duration in seconds
VIDEO_FPS = 30                 # default export frame rate

# --- Trail refinement ---
REFINE_ANGLE_DEG = 3.0         # maximum chord angle (degrees) between consecutive
                               # trail points; segments exceeding this get subdivided
                               # during precomputation
