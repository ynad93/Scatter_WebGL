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

```bash
python -m scatterview data/ScatterParts.csv
```

Or with options:

```bash
python -m scatterview simulation.csv --camera auto-frame --trail-length 0.01 --width 1920 --height 1080
```

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

### CSV

The CSV must have columns:

```
ID, time, x, y, z
```

Optional columns: `vx, vy, vz` (velocities — enables CubicHermiteSpline for physically accurate orbital interpolation), `mass`, `radius`, `startype` (BSE stellar evolution code).

Each particle gets one row per timestep. Particles can appear/disappear mid-simulation (gaps are handled via per-segment splines). An example CSV is in `data/ScatterParts.csv`.

### HDF5

Two layouts are supported:
- **Single-file**: datasets `times (T,)`, `positions (N, T, 3)`, `particle_ids (N,)`
- **Snapshot groups**: groups named by snapshot index, each containing `positions (N, 3)`, `particle_ids (N,)`

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
- **Space**: play/pause
- **WASD / Arrow keys**: pan camera (continuous while held, Ctrl = 5x speed)

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

## Performance

Benchmarked with 101 particles, 24,393 timesteps:

| Component | Time per frame |
|-----------|---------------|
| `evaluate_batch` (spline eval) | 0.20 ms |
| `update_trails` (window + assembly) | 0.40 ms |
| `get_particle_attrs` (colors/sizes) | 0.03 ms |
| **Total CPU** | **0.63 ms** |
| GPU render (VisPy) | ~5–8 ms |

Trail precomputation: ~1s for 101 particles, 5M refined points (60 MB). Parallelizable via `n_workers` parameter.

## License

See [LICENSE](LICENSE).
