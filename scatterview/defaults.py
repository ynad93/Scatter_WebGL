"""Single source of truth for all ScatterView default settings.

Every default value lives here. The engine, CLI, and GUI all read from
this module rather than hardcoding their own values.
"""

# --- Animation ---
ANIM_SPEED = 1.0 / 60.0       # fraction of total duration per second (~60s playback)
GAMMA = 0.0                    # time mapping gamma (0 = uniform sim-time advance)

# --- Appearance ---
POINT_ALPHA = 1.0
TRAIL_ALPHA = 0.6
TRAIL_LENGTH_FRAC = 0.005      # fraction of total simulation time (0.5%)
TRAIL_WIDTH = 3.0              # pixels
TRAIL_INITIAL_POINTS = 100     # starting uniform sample count before refinement

# --- Sizing ---
RELATIVE_SIZE_MIN_PX = 3.0     # smallest particle in relative mode
RELATIVE_SIZE_MAX_PX = 20.0    # largest particle in relative mode
DEFAULT_SIZE_PX = 10.0         # when no radii are provided
DEPTH_SCALING = True           # closer particles appear larger (perspective-scaled sizes)

# --- Camera ---
CAMERA_FOV = 45
CAMERA_EMA_ALPHA = 0.15        # center smoothing (exponential moving average)
CAMERA_ZOOM_EMA_ALPHA = 0.03   # zoom smoothing (slower than center)
CAMERA_COM_JUMP_THRESHOLD = 0.5
CAMERA_OUTLIER_SIGMA = 2.0     # core-group rejection: N × median distance
CAMERA_RADIUS_PERCENTILE = 95.0
CAMERA_N_NEIGHBORS = 3
ROTATION_SPEED = 0.5           # degrees per frame

# --- Black hole rendering ---
BH_STARTYPE = 14               # BSE stellar type code for black holes
BH_FACE_COLOR = (0.02, 0.02, 0.05, 0.15)
BH_EDGE_WIDTH = 2.0

# --- Lighting ---
LIGHT_AMBIENT = 0.2
LIGHT_COLOR = "white"
LIGHT_POSITION = (1, -1, 1)

# --- Window ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
SUBVIEW_SIZE_FRAC = 0.3
SUBVIEW_FOV = 60

# --- Rendering / export ---
VIDEO_DURATION = 10.0
VIDEO_FPS = 30

# --- Adaptive trail refinement ---
REFINE_ANGLE_THRESHOLD = 0.0524  # 3 degrees in radians
