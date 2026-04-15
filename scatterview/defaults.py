"""Single source of truth for all ScatterView default settings.

Every default value lives here. The engine, CLI, and GUI all read from
this module rather than hardcoding their own values.
"""

import math

# --- Animation ---
ANIM_SPEED = 0.01              # fraction of total sim duration advanced per second
                               # (0.01 ≈ 100-second full playback at 1x speed)

# --- Appearance ---
POINT_ALPHA = 1.0              # particle opacity (0 = invisible, 1 = opaque)
TRAIL_ALPHA = 0.85             # peak trail opacity (at the head / newest point)
TRAIL_LENGTH_FRAC = 0.005      # trail window as fraction of total simulation time

# --- Particle sizing ---
RELATIVE_SIZE_MIN_PX = 3.0     # smallest particle in relative mode (screen pixels);
                               # other particles scale up proportionally to cbrt(radius)
DEFAULT_SIZE_PX = 10.0         # uniform size when no radii are provided, and the
                               # per-particle pixel size used by the "Equal Sizes" toggle
DEPTH_SCALING = False          # if True, closer particles appear larger (perspective)

# --- Camera ---
CAMERA_FOV = 45                # field of view in degrees
ROTATION_SPEED = 0.5           # auto-rotate: degrees of azimuth per frame
FRAMING_FRACTION = 0.95        # framed particles stay within this fraction
                               # of the screen's vertical half-extent
ZOOM_MEMORY_FRAMES = 60        # rolling-average window for framing radius (frames)
                               # camera distance tracks the mean radius over this
                               # many frames, filtering orbital oscillations while
                               # responding smoothly to scale changes.
                               # At 60 Hz, 120 frames = 2-second memory.
PAN_DEADZONE_FRACTION = 0.5    # panning deadzone: fraction of visible radius

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
LIGHT_POSITION = (-0.5, -0.3, 1.0)  # world-space direction, upper-left offset from camera

# --- Window ---
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080

# --- Video export ---
VIDEO_DURATION = 60.0          # default export duration in seconds
VIDEO_FPS = 60                 # default export frame rate
VIDEO_WIDTH = 2560             # default export width in pixels
VIDEO_HEIGHT = 1440            # default export height in pixels

# --- Trail refinement ---
REFINE_ANGLE_DEG = 3.6         # maximum chord angle (degrees) between consecutive
                               # trail points; segments exceeding this get subdivided
                               # during precomputation


# --- Physical units ---
# Labels for the simulation's unit system.  These describe the units of
# the *input data* (positions, masses, times) and are used for the time
# overlay display.
UNIT_MASS = "Msun"
UNIT_DISTANCE = "AU"
UNIT_TIME = "yr"

# Conversion to CGS for each supported unit label.
_MASS_TO_CGS = {
    "Msun": 1.98892e33,
    "kg": 1e3,
    "g": 1.0,
}
_DISTANCE_TO_CGS = {
    "AU": 1.496e13,
    "pc": 3.0857e18,
    "kpc": 3.0857e21,
    "Mpc": 3.0857e24,
    "Rsun": 6.957e10,
    "km": 1e5,
    "m": 1e2,
    "cm": 1.0,
}
_TIME_TO_CGS = {
    "yr": 3.15576e7,
    "Myr": 3.15576e13,
    "Gyr": 3.15576e16,
    "kyr": 3.15576e10,
    "s": 1.0,
}

MASS_UNITS = list(_MASS_TO_CGS.keys())
DISTANCE_UNITS = list(_DISTANCE_TO_CGS.keys())
TIME_UNITS = list(_TIME_TO_CGS.keys())

# Fundamental constants in CGS
_G_CGS = 6.67430e-8           # cm³ g⁻¹ s⁻²
_C_CGS = 2.99792458e10        # cm s⁻¹


def G_in_units(mass_unit: str = UNIT_MASS, dist_unit: str = UNIT_DISTANCE,
               time_unit: str = UNIT_TIME) -> float:
    """Gravitational constant G in the given unit system.

    Args:
        mass_unit: Mass unit label (key into _MASS_TO_CGS).
        dist_unit: Distance unit label (key into _DISTANCE_TO_CGS).
        time_unit: Time unit label (key into _TIME_TO_CGS).

    Returns:
        G in the specified unit system.
    """
    m = _MASS_TO_CGS[mass_unit]
    d = _DISTANCE_TO_CGS[dist_unit]
    t = _TIME_TO_CGS[time_unit]
    return _G_CGS * m * t ** 2 / d ** 3


def c_in_units(dist_unit: str = UNIT_DISTANCE,
               time_unit: str = UNIT_TIME) -> float:
    """Speed of light c in the given unit system.

    Args:
        dist_unit: Distance unit label (key into _DISTANCE_TO_CGS).
        time_unit: Time unit label (key into _TIME_TO_CGS).

    Returns:
        c in the specified unit system.
    """
    d = _DISTANCE_TO_CGS[dist_unit]
    t = _TIME_TO_CGS[time_unit]
    return _C_CGS * t / d


# --- Time overlay ---
TIME_FONT_SIZE = 14            # points
TIME_COLOR = (1.0, 1.0, 1.0, 0.9)
TIME_ANCHOR = "top-left"       # top-left | top-right | bottom-left | bottom-right
TIME_OFFSET = (15, 25)         # pixel offset from the anchor corner

_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def format_sim_time(t: float, unit: str | None = None, decimals: int = 2) -> str:
    """Format a simulation time for on-screen display.

    Uses scientific notation for values outside [0.01, 9999].
    Example: 3140000.0 with unit="yr" → "3.14 × 10⁶ yr"

    Args:
        t: Simulation time value to format.
        unit: Unit label appended to the formatted string. Defaults to UNIT_TIME.
        decimals: Number of decimal places in the mantissa.

    Returns:
        Human-readable time string with unit suffix.
    """
    unit = unit or UNIT_TIME
    if t == 0.0:
        return f"0.{'0' * decimals} {unit}"
    abs_t = abs(t)
    if 0.01 <= abs_t < 10000:
        return f"{t:.{decimals}f} {unit}"
    exponent = int(math.floor(math.log10(abs_t)))
    mantissa = t / 10 ** exponent
    exp_str = str(exponent).translate(_SUPERSCRIPT)
    return f"{mantissa:.{decimals}f} × 10{exp_str} {unit}"


# --- Star field ---
STAR_COUNT = 5000              # background stars on the spherical shell
STAR_SHELL_FACTOR = 2.0        # shell radius = factor × max particle distance
STAR_SEED = 42                 # RNG seed for reproducibility
STAR_BASE_SIZE = 1.0           # base marker size in pixels
