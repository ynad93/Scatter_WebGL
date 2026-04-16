# ScatterView

A Python + VisPy visualization tool for N-body scattering simulations. Renders particle trajectories in real-time 3D with cubic spline interpolation, precomputed trail rendering, and an interactive camera system.

## Features

- **Real-time 3D rendering** via VisPy (OpenGL) with spherical markers and world-space directional lighting
- **Cubic spline interpolation** of particle trajectories (CubicHermiteSpline with velocity data, CubicSpline fallback)
- **Precomputed trails** with angle-based refinement at simulation timesteps — zero spline evaluation during playback
- **Sliding-window trail extraction** via numba-compiled two-pointer advance (O(1) per particle per frame)
- **Smooth trail boundaries** with lerp'd tail/head points so trails don't snap to discrete timestamps
- **Deadzone camera system** — camera holds still while targets are near screen center, chases only when they drift past a configurable threshold
- **Multiple camera modes**: Manual, Auto-Frame, Auto-Rotate, Event Track, Target Rest, Target Comoving
- **Framing scopes**: All particles, Core Group (outlier rejection), Nearest Neighbors
- **Black hole rendering** with BSE stellar type detection
- **GUI control panel** (Qt) with sliders for speed, trail length, trail alpha, particle sizing, camera deadzone, and more
- **Video export** to MP4 via imageio/pyav
- **Picture-in-picture** sub-view with independent camera controls

## Installation

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `vispy`, `PyQt6`, `numba`. Optional: `imageio[pyav]` for video export, `h5py` for HDF5 data.

## Usage

### Command line

Loading an HDF5 file (layout auto-detected — see [Data Format](#data-format)):

```bash
python -m scatterview simulation.h5
```

If the file contains multiple pandas datasets or nests the data under a non-root group, pass `--key` to point at it:

```bash
python -m scatterview simulation.h5 --key simulation
```

Loading a CSV file:

```bash
python -m scatterview data/ScatterParts.csv
```

Or with options:

```bash
python -m scatterview simulation.csv --camera auto-frame --trail-length 0.01 --width 1920 --height 1080
```

#### Arguments

Positional:

- `datafile` — Path to the simulation data file (CSV or HDF5). Format is inferred from the extension unless `--format` overrides it.

Data loading:

- `--format`, `-f` `{csv,hdf5}` — Force the data format. Default: auto-detected from the file extension (`.csv` → csv; `.h5`/`.hdf5` → hdf5).
- `--key KEY` — HDF5 key (group or pandas dataset name) holding the phase-space data. Use this when the file contains multiple pandas datasets or nests the data under a non-root group. Without it, the loader assumes the chosen layout lives at the top level and raises a `ValueError` otherwise. CSV files ignore this flag. Default: `None`.

Output:

- `--output`, `-o` `PATH` — If set, ScatterView runs in batch mode and writes to `PATH` instead of launching the GUI. A `.png` suffix saves a single screenshot; any other extension is rendered as a video (MP4 recommended, delegated to `imageio`/`pyav`). Default: `None` (interactive GUI).
- `--duration FLOAT` — Video duration in seconds (ignored for `.png` output and interactive mode). Default: `10.0`.
- `--fps INT` — Video frames per second (ignored for `.png` and interactive mode). Default: `60`.
- `--width INT` — Window/render width in pixels. Applies to both interactive viewer and batch output. Default: `1920`.
- `--height INT` — Window/render height in pixels. Default: `1080`.

Camera:

- `--camera {manual,tracking,event-track}` — Initial camera mode. `manual` = full user control, `tracking` = deadzone comoving tracking of the `--target` particle (or group COM if none), `event-track` = tracking with automatic retargeting onto detected close encounters. Default: `tracking`.
- `--target INT` — Particle ID to track when using a tracking camera mode. Must exist in the data file. Default: `None` (camera frames the whole cluster / core group instead).

Trails:

- `--trail-length FLOAT` — Trail length as a fraction of the total simulation time range (0 = no trail, 1 = trail spans the entire run). Default: `0.005`.

Analysis:

- `--detect-events` — Run the close-encounter event detector on load and print detected events to the terminal. Does not alter rendering. Default: off.

Units (affect axis labels, HUD readouts, and unit conversions — they do **not** rescale or reinterpret the numerical values in your data file, which the loader always takes at face value):

- `--mass-unit {Msun,kg,g}` — Mass unit of the input data. Default: `Msun`.
- `--distance-unit {AU,pc,kpc,Mpc,Rsun,km,m,cm}` — Distance unit of the input data. Default: `AU`.
- `--time-unit {yr,Myr,Gyr,kyr,s}` — Time unit of the input data. Default: `yr`.

### From Python

```python
from scatterview.core.data_loader import load
from scatterview.core.interpolation import TrajectoryInterpolator
from scatterview.rendering.engine import RenderEngine

data = load("simulation.csv")
interpolator = TrajectoryInterpolator(data)
engine = RenderEngine(data, interpolator, size=(1280, 720))
engine.play()
engine.show()
```

## Data Format

Format is auto-detected from the file extension (`.csv`, `.h5`, `.hdf5`), or forced with `--format` / `fmt=`.

### CSV

The CSV must have columns:

```
ID, time, x, y, z
```

Optional columns: `vx, vy, vz` (velocities — enables CubicHermiteSpline for physically accurate orbital interpolation), `mass`, `radius` (also detected as `r` or `rad`), `startype` (also `kstar` or `k` — BSE stellar evolution code).

Column names are case-insensitive. The time column also accepts `t` if `time` is not present.

Each particle gets one row per timestep. Particles can appear/disappear mid-simulation — use empty cells or NaN for position columns at timesteps where a particle doesn't exist (gaps are handled via per-segment splines). An example CSV is in `data/ScatterParts.csv`.

### HDF5

Three layouts are supported, tried in this order:

#### HDF5 structure primer

An HDF5 file works like a filesystem. **Groups** are directories, **datasets** are files (numpy arrays on disk), and **attributes** are small metadata tags attached to any group or dataset. The root of the file is itself a group, `/`.

Here is a 3-body simulation stored as snapshot groups (Layout 3) — each timestep gets its own directory:

```
simulation.h5                         <- the file (root group "/")
├── snap_0/                           <- group (like a directory: mkdir snap_0)
│   ├── positions                     <- dataset (like a file: a (3,3) numpy array)
│   ├── ids                           <- dataset: [0, 1, 2]
│   └── (time = 0.0)                  <- attribute (metadata tag on the group)
├── snap_1/
│   ├── positions                     <- same datasets, different timestep
│   ├── ids
│   └── (time = 0.5)
└── snap_2/
    ├── positions
    ├── ids
    └── (time = 1.0)
```

With `h5py`, groups behave like dicts — you access datasets and subgroups by string key, the same way you would navigate a directory tree:

```python
import h5py
import numpy as np

# --- Writing (creating the file above) ---
with h5py.File("simulation.h5", "w") as f:
    for i, t in enumerate([0.0, 0.5, 1.0]):
        # Create a group (like mkdir snap_0/)
        grp = f.create_group(f"snap_{i}")       # no zero-padding needed

        # Create datasets inside it (like writing files into that directory)
        grp.create_dataset("positions", data=positions_at_t[i])  # (N, 3)
        grp.create_dataset("ids", data=np.array([0, 1, 2]))

        # Attach an attribute (a metadata tag, not a dataset)
        grp.attrs["time"] = t

# --- Reading ---
with h5py.File("simulation.h5", "r") as f:
    # List keys at the root (like ls /)
    print(list(f.keys()))
    # ['snap_0', 'snap_1', 'snap_2']

    # Navigate into a group by key (like cd snap_0/)
    snap = f["snap_0"]                  # group object, acts like a dict

    # Read a dataset inside it (like reading a file in that directory)
    pos = snap["positions"][:]           # numpy array, shape (3, 3)

    # Read an attribute on the group
    t = snap.attrs["time"]               # 0.0

    # List datasets inside a group (like ls snap_0/)
    print(list(snap.keys()))             # ['positions', 'ids']
```

The three ScatterView layouts use this structure differently:
- **Layout 1** (multi-index): pandas handles the HDF5 structure internally via `df.to_hdf()` — you don't create groups or datasets manually.
- **Layout 2** (single-file): everything lives at the root as top-level datasets — no groups, like a flat directory with just files in `/`.
- **Layout 3** (snapshots): each timestep is a group (subdirectory) containing that snapshot's datasets, as shown above.

#### 1. Multi-index DataFrame (pandas HDF5 table)

The recommended format for N-body codes that already use pandas. Build a DataFrame whose rows are indexed by `(time, particle_id)` and whose columns are the per-particle quantities at that timestep:

```
DataFrame contents (what you build in pandas):

                  mass*  radius*    x      y      z      vx     vy     vz
time   id
0.0    0           1.0    0.05    1.00   0.00   0.00    0.00   0.50   0.00
       1           1.0    0.05   -1.00   0.00   0.00    0.00  -0.50   0.00
       2           0.5    0.03    0.00   1.00   0.00    0.50   0.00   0.00
0.5    0           1.0    0.05    1.00   0.25   0.00    0.00   0.50   0.00
       1           1.0    0.05   -1.00  -0.25   0.00    0.00  -0.50   0.00
       2           0.5    0.03    0.25   1.00   0.00    0.50   0.00   0.00
1.0    0           ...

* mass and radius are OPTIONAL — omit either or both columns if you don't have them.
  (vx/vy/vz are also optional; without them, ScatterView falls back to a non-Hermite spline.)
```

Logically, the file looks like a single snapshot group whose contents are indexed first by time and then by particle id (compare with Layout 3, where each timestep is its own top-level group):

```
simulation.h5
└── data/                                       <- one group (the key= argument)
    ├── time = 0.0
    │   ├── id 0:  mass*, radius*, x, y, z, vx, vy, vz
    │   ├── id 1:  mass*, radius*, x, y, z, vx, vy, vz
    │   └── id 2:  mass*, radius*, x, y, z, vx, vy, vz
    ├── time = 0.5
    │   ├── id 0:  mass*, radius*, x, y, z, vx, vy, vz
    │   ├── id 1:  mass*, radius*, x, y, z, vx, vy, vz
    │   └── id 2:  mass*, radius*, x, y, z, vx, vy, vz
    └── time = 1.0
        └── ...
```

Hand the whole DataFrame to `df.to_hdf()` in one call — pandas serializes the multi-index and columns into the `data/` group for you. You do not create subgroups or datasets yourself:

```python
import pandas as pd

# df has columns: x, y, z, and optionally vx, vy, vz, mass, radius, k
# index levels: (time, particle_id)
df.index = pd.MultiIndex.from_arrays([times, ids], names=["time", "id"])
df.to_hdf("simulation.h5", key="data")
```

**Required columns:** `x`, `y`, `z` (also accepts `r0/r1/r2` or `pos_x/pos_y/pos_z`)

**Optional columns:**
- `vx`, `vy`, `vz` (or `v0/v1/v2`, `vel_x/vel_y/vel_z`) — velocities for Hermite spline interpolation
- `mass` (or `m`) — particle masses for mass-weighted center of mass
- `radius` (or `rad`) — particle radii for size scaling
- `k` (or `startype`, `kstar`, `stellar_type`) — BSE stellar type code (14 = black hole)

Column names are detected case-insensitively. Particle IDs can be integers or strings (strings are mapped to integer keys internally; the original labels are preserved for GUI display).

Per-particle data is extracted via `df.xs(pid, level=1)`, which reads directly from HDF5 without expanding the full table into memory.

**Particles appearing or disappearing mid-simulation:** two equivalent ways to mark a particle as absent at a given timestep:
- **Omit the `(time, pid)` row entirely** from the multi-index — cheaper on disk.
- **Keep the row, but write `NaN` into the position columns** — explicit, easier to inspect.

The loader treats both identically: it reindexes each particle against the global time axis (the union of all times appearing in the index) and any row that is missing or has non-finite positions is recorded as a gap in `valid_intervals`. The spline is then split into separate segments around each gap, so trajectories never interpolate through regions where the particle didn't exist.

#### 2. Single-file arrays

Every quantity is stored in its **own top-level dataset** (an HDF5 dataset is the n-dimensional array on disk; groups are the directory-like containers — Layout 2 has no groups, just datasets sitting directly under the root `/`):

```
simulation.h5
├── positions    (N, T, 3)        <- required
├── ids          (N,)              <- optional in principle, but see note below
├── times        (T,)              <- optional, defaults to 0..T-1
├── velocities   (N, T, 3)        <- optional
├── masses       (N,) or (N, T)   <- optional
├── radii        (N,)              <- optional, one scalar per particle
└── startypes    (N,)              <- optional, BSE stellar-type code per particle
```

**Row alignment is the contract that makes trajectories reconstructable.** The leading `N` axis of every per-particle dataset (`positions`, `velocities`, `masses` when 2D, `radii`, `startypes`) is indexed in the **same order** as `ids`. That is:

```
positions[i, t, :]  ==  position of particle ids[i] at time times[t]
velocities[i, t, :] ==  velocity of particle ids[i] at time times[t]
masses[i] (or masses[i, t]) ==  mass of particle ids[i]
radii[i]            ==  radius of particle ids[i]
startypes[i]        ==  BSE stellar-type code of particle ids[i]
```

Without `ids`, the rows are still aligned across datasets, but particles are anonymous (referred to internally as `0..N-1`) — fine if you don't care about labeling, fatal if you need to cross-reference a specific particle in your downstream analysis. Whatever order you write `ids` in is the order the loader will use; rows are not re-sorted.

NaN positions mark timesteps where a particle doesn't exist (Layout 2 supports disappearing particles natively — the loader masks NaN rows per particle and computes valid intervals from them).

```python
import h5py

with h5py.File("simulation.h5", "w") as f:
    f.create_dataset("positions",  data=pos)       # (N, T, 3) float64
    f.create_dataset("ids",        data=ids)       # (N,) int — defines row order
    f.create_dataset("times",      data=times)     # (T,) float64
    f.create_dataset("velocities", data=vel)       # (N, T, 3) float64
    f.create_dataset("masses",     data=masses)    # (N,) or (N, T) float64
    f.create_dataset("radii",      data=radii)     # (N,) float64
    f.create_dataset("startypes",  data=startypes) # (N,) int — BSE stellar type
```

#### 3. Snapshot groups

Each timestep lives in its own group (subdirectory). Any top-level group whose name starts with `snapshot` or `snap_` is treated as a snapshot. Groups are sorted numerically by the digits in their name, so both zero-padded (`snapshot_0000`, `snapshot_0001`) and unpadded (`snap_0`, `snap_1`, ..., `snap_102`) names work correctly.

```
simulation.h5
├── snap_0/
│   ├── positions    (N, 3)
│   ├── ids          (N,)
│   ├── velocities   (N, 3)        <- optional
│   ├── masses       (N,) or scalar <- optional
│   ├── radii        (N,)           <- optional
│   ├── startypes    (N,)           <- optional, BSE stellar-type code
│   └── (time = 0.0)               <- attribute, or a "times" dataset
├── snap_1/
│   ├── positions    (N, 3)
│   ├── ids          (N,)
│   └── ...
└── ...
```

Within each snapshot group, every per-particle dataset (`positions`, `velocities`, `masses`, `radii`, `startypes`) is row-aligned to that snapshot's own `ids`. Different snapshots can carry different particle sets — the loader uses each snapshot's `ids` to look up the canonical particle index, so particles can appear and disappear over time.

Time for each snapshot is read from three sources, tried in order:
1. A `times` dataset inside the group — `f["snap_0"]["times"]`
2. A `time` attribute on the group — `f["snap_0"].attrs["time"]`
3. The snapshot index (0, 1, 2, ...) as a last resort

`radii` and `startypes` are treated as static per-particle properties even though they live inside each snapshot: the loader collapses them to one scalar per particle by recording the **first value** it encounters as it walks snapshots in time order. Subsequent appearances of the same particle's radius/startype are ignored, so it's fine (and natural) to write the same value into every snapshot.

```python
import h5py

with h5py.File("simulation.h5", "w") as f:
    for i, t in enumerate(timesteps):
        grp = f.create_group(f"snap_{i}")              # no zero-padding needed
        grp.create_dataset("positions",  data=pos_at_t[i])   # (N, 3)
        grp.create_dataset("ids",        data=ids)            # (N,)
        grp.create_dataset("velocities", data=vel_at_t[i])   # (N, 3)
        grp.create_dataset("masses",     data=masses)         # (N,) or scalar
        grp.create_dataset("radii",      data=radii)          # (N,) float64
        grp.create_dataset("startypes",  data=startypes)      # (N,) int
        grp.attrs["time"] = t                                  # scalar
```

#### Custom dataset names

For layouts 2 and 3, pass a `field_map` dict to remap dataset names if your file uses different keys:

```python
from scatterview.core.data_loader import load

data = load("simulation.h5", field_map={
    "positions": "coords",
    "ids": "particle_ids",
    "times": "t",
    "velocities": "vel",
    "masses": "m",
    "radii": "rad",
    "startypes": "kstar",
})
```

## Architecture

### Interpolation (`scatterview/core/interpolation.py`)

Particle positions are interpolated from simulation data using scipy cubic splines:

- **CubicHermiteSpline** when velocity data is available (exact derivative matching — physically superior for orbital mechanics)
- **CubicSpline** with natural boundary conditions as fallback
- Splines are pre-built at load time; polynomial coefficients are extracted for fast Horner evaluation
- **`evaluate_batch(time)`**: evaluates all particles at a single time using vectorized Horner polynomial evaluation with shared breakpoints (one `searchsorted` for all particles)

### Trail Precomputation (`precompute_all_trails`)

Trails are precomputed once at startup:

1. Evaluate each particle's spline at its **simulation timesteps** (the integrator's adaptive step sizes already concentrate points at close encounters)
2. Apply single-pass **angle-based refinement**: where consecutive chord vectors deflect by more than 3 degrees, insert additional points to smooth the polyline
3. Pack all particles' trails into contiguous arrays with an offset table for O(1) per-particle slicing
4. Optionally parallelized via `multiprocessing.Pool`

### Per-Frame Trail Rendering (`_update_trails` in `scatterview/rendering/engine.py`)

Each frame extracts the visible trail window with zero spline evaluation:

1. **Two-pointer sliding window** (numba-compiled): advances tail/head indices by 1-2 positions during forward playback (O(1) per particle), falls back to binary search on scrub/loop
2. **Tail interpolation**: linearly interpolates between the two precomputed points straddling the trail start time for smooth fade-in
3. **Head position**: uses the live particle position from `evaluate_batch` (always matches the particle exactly)
4. **Assembly** (numba-compiled): writes positions, times, and base colors into pre-allocated GPU arrays
5. **Alpha gradient**: time-based opacity via a t^1.5 power-law lookup table (oldest points fade out)

### Camera System (`scatterview/core/camera.py`)

The camera uses a **deadzone** approach:

- The camera holds still while the tracked target (particle, group COM, or cluster center) stays within a configurable fraction of the visible radius from screen center
- When the target drifts past the deadzone edge, the camera moves exactly enough to place it back on the edge — no velocity cap, so it keeps up with fast slingshots
- Zoom uses the same deadzone: camera distance holds until the ideal framing distance differs by more than the deadzone fraction
- **Target acquisition**: selecting a target jumps the camera to the target group immediately, then the deadzone maintains tracking

Camera modes:
- **Manual**: full user control (mouse orbit, WASD pan, scroll zoom)
- **Auto-Frame**: frames the selected particle group (Core Group, All, or Nearest Neighbors)
- **Auto-Rotate**: auto-frame + continuous azimuth rotation
- **Event Track**: smoothly transitions to detected close encounters
- **Target Rest**: locks camera center exactly on the target particle
- **Target Comoving**: deadzone tracking of the target with smooth chase

### Lighting

World-space directional lighting: the light direction is fixed in world coordinates and transformed into eye space each frame based on camera azimuth/elevation. This means particle shading changes when you orbit — the bright hemisphere faces the light, the dark hemisphere faces away, giving depth cues.

## Controls

### Mouse
- **Left drag**: orbit camera
- **Scroll wheel**: zoom toward cursor position (Ctrl = 5x speed)
- **Shift + left drag**: pan

### Keyboard

All keys are continuous while held. **Ctrl** multiplies speed by 5x for any action below.

- **Space**: play/pause
- **WASD**: pan forward/backward/left/right (relative to camera facing)
- **Arrow Up/Down**: pan up/down
- **Arrow Left/Right**: pan left/right
- **Shift + Arrow Up/Down**: pan forward/backward (instead of vertical)
- **Shift + Arrow Left/Right**: scrub time backward/forward
- **Alt + WASD / Arrow keys**: orbit camera — Left/Right rotate azimuth, Up/Down rotate elevation

Pressing any pan key automatically switches to Manual camera mode.

### GUI Sliders
- **Time**: scrub through simulation
- **Speed**: playback speed
- **Trail (frac)**: trail length as fraction of total simulation time
- **Trail Width**: trail line width in pixels
- **Trail Alpha**: trail opacity
- **Radius Scale**: global particle size multiplier
- **Pan Speed**: WASD pan sensitivity (log-spaced)
- **Zoom Speed**: scroll zoom sensitivity (log-spaced)
- **Deadzone**: fraction of visible radius where camera holds still (0.1–0.8)
- **Neighbors**: number of nearest neighbors for framing (visible when using Nearest Neighbors scope)

## Profiling

```bash
python profile_engine.py [datafile] [--frames N] [--warmup N]
```

Reports per-component timing breakdown, cProfile stats, and trail cache statistics.

## License

See [LICENSE](LICENSE).
