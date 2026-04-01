# ScatterView

Real-time 3D visualization tool for N-body scattering simulations.
Renders particle trajectories with cubic spline interpolation, precomputed trails, and interactive camera control using VisPy (OpenGL) and Qt6.

## Environment

All running, testing, and development must be done from the **`moviemaking`** conda environment:

```bash
conda activate moviemaking
```

The environment is defined in `environment.yml`. To create it:

```bash
conda env create -f environment.yml
```

## Install

```bash
pip install -e .
```

## Running

```bash
python -m scatterview <datafile> [options]
# or after install:
scatterview <datafile> [options]
```

Example:

```bash
scatterview data/ScatterParts.csv --camera tracking --trail-length 0.01
```

## Tests

```bash
pytest scatterview/tests/
```

## Project layout

```
scatterview/
  cli.py              # CLI entry point
  defaults.py         # All tunable constants
  core/
    data_loader.py    # CSV and HDF5 loading
    interpolation.py  # Cubic spline trajectory interpolation
    camera.py         # Camera modes and deadzone tracking
    event_detection.py
  rendering/
    engine.py         # VisPy OpenGL renderer
  gui/
    controls.py       # Qt6 control panel
    export.py         # MP4/image export
  tests/
```

## Key conventions

- Python 3.10+, full type annotations
- NumPy vectorized operations and Numba JIT (`@nb.njit(cache=True)`) for hot paths
- No defensive isinstance/type guards in internal code; let it crash if types are wrong
- All default constants live in `defaults.py`
- GUI uses PyQt6 (not PyQt5 despite pyproject.toml listing)

## Data formats

- **CSV**: columns `ID`, `time` (or `t`), `x`, `y`, `z`; optional `vx,vy,vz`, `mass`, `radius`, `startype`
- **HDF5**: three layouts supported (multi-index DataFrame, flat arrays, snapshot groups)
