"""Data loading for N-body simulation output files.

Supports CSV and HDF5 formats with automatic format detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SimulationData:
    """Standardized container for N-body simulation data.

    Attributes:
        particle_ids: Unique integer particle identifiers.
        times: Sorted unique simulation times.
        positions: Mapping of particle ID -> (T_i, 3) position array.
        velocities: Mapping of particle ID -> (T_i, 3) velocity array, or None.
        masses: Mapping of particle ID -> (T_i,) mass array, or None.
        valid_intervals: Mapping of particle ID -> list of (t_start, t_end) tuples
            indicating continuous segments where the particle exists.
        id_labels: Mapping of integer ID -> original label string (e.g. "BH0").
            Only populated when the source data uses non-numeric IDs.
    """

    particle_ids: np.ndarray
    times: np.ndarray
    positions: dict[int, np.ndarray]
    velocities: dict[int, np.ndarray] | None = None
    masses: dict[int, np.ndarray] | None = None
    radii: dict[int, float] | None = None
    startypes: dict[int, int] | None = None
    valid_intervals: dict[int, list[tuple[float, float]]] = field(
        default_factory=dict
    )
    id_labels: dict[int, str] | None = None


def load(filepath: str | Path, fmt: str | None = None, **kwargs) -> SimulationData:
    """Load simulation data from file.

    Args:
        filepath: Path to the data file.
        fmt: Format override ('csv' or 'hdf5'). Auto-detected from extension if None.
        **kwargs: Passed to the format-specific loader.

    Returns:
        SimulationData with all fields populated.
    """
    filepath = Path(filepath)
    if fmt is None:
        fmt = _detect_format(filepath)

    loaders = {
        "csv": load_csv,
        "hdf5": load_hdf5,
        "h5": load_hdf5,
    }
    loader = loaders.get(fmt)
    if loader is None:
        raise ValueError(
            f"Unknown format '{fmt}'. Supported: {list(loaders.keys())}"
        )
    return loader(filepath, **kwargs)


def _detect_format(filepath: Path) -> str:
    suffix = filepath.suffix.lower()
    format_map = {
        ".csv": "csv",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
    }
    fmt = format_map.get(suffix)
    if fmt is None:
        raise ValueError(
            f"Cannot detect format from extension '{suffix}'. "
            f"Use fmt= to specify. Supported: {list(format_map.values())}"
        )
    return fmt


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv(
    filepath: Path,
    id_col: str = "ID",
    time_col: str = "time",
    pos_cols: tuple[str, str, str] = ("x", "y", "z"),
    vel_cols: tuple[str, str, str] | None = ("vx", "vy", "vz"),
    mass_col: str | None = "mass",
    radius_col: str | None = "radius",
) -> SimulationData:
    """Load simulation data from a CSV file.

    Expected CSV columns: at minimum ID, time, x, y, z.
    Optional columns: mass, vx, vy, vz.

    Handles particles that appear/disappear mid-simulation (null/NaN coordinates).
    """
    df = pd.read_csv(filepath)

    # Normalize column names to lowercase for flexible matching
    df.columns = df.columns.str.strip().str.lower()
    id_col = id_col.lower()
    time_col = time_col.lower()
    pos_cols = tuple(c.lower() for c in pos_cols)
    if vel_cols is not None:
        vel_cols = tuple(c.lower() for c in vel_cols)
    if mass_col is not None:
        mass_col = mass_col.lower()
    if radius_col is not None:
        radius_col = radius_col.lower()

    # Auto-detect time column: accept "t" if "time" not present
    if time_col not in df.columns and "t" in df.columns:
        time_col = "t"

    # Validate required columns
    required = [id_col, time_col] + list(pos_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Check which optional columns are present
    has_velocity = vel_cols is not None and all(c in df.columns for c in vel_cols)
    has_mass = mass_col is not None and mass_col in df.columns
    # Auto-detect radius column: try "radius", "r", "rad"
    has_radius = radius_col is not None and radius_col in df.columns
    if not has_radius:
        for candidate in ("r", "rad", "radius"):
            if candidate in df.columns:
                radius_col = candidate
                has_radius = True
                break

    # Auto-detect startype column: "startype", "kstar", "k"
    startype_col: str | None = None
    for candidate in ("startype", "kstar", "k"):
        if candidate in df.columns:
            startype_col = candidate
            break
    has_startype = startype_col is not None

    # Sort by time then ID for consistent ordering
    df = df.sort_values([time_col, id_col]).reset_index(drop=True)

    # Extract global sorted unique times
    times = np.sort(df[time_col].unique())

    # Group by particle ID
    particle_ids_raw = df[id_col].unique()

    # Map IDs to integers if they are strings
    _try_numeric = True
    for pid in particle_ids_raw:
        try:
            int(pid)
        except (ValueError, TypeError):
            _try_numeric = False
            break

    if _try_numeric:
        # Numeric IDs — use as-is
        particle_ids_raw = np.array([int(p) for p in particle_ids_raw])
        particle_ids = np.sort(particle_ids_raw)
        id_label_map = {int(p): int(p) for p in particle_ids}
        reverse_label_map = id_label_map.copy()
    else:
        # String IDs — assign stable integer keys in sorted order
        sorted_labels = sorted(particle_ids_raw, key=str)
        id_label_map = {label: i for i, label in enumerate(sorted_labels)}
        reverse_label_map = {i: label for label, i in id_label_map.items()}
        particle_ids = np.arange(len(sorted_labels))

    positions: dict[int, np.ndarray] = {}
    velocities: dict[int, np.ndarray] | None = {} if has_velocity else None
    masses: dict[int, np.ndarray] | None = {} if has_mass else None
    radii: dict[int, float] | None = {} if has_radius else None
    startypes: dict[int, int] | None = {} if has_startype else None
    valid_intervals: dict[int, list[tuple[float, float]]] = {}

    for label, pid_key in id_label_map.items():
        mask = df[id_col] == label
        pdf = df.loc[mask].sort_values(time_col)

        p_times = pdf[time_col].values.astype(float)
        # Coerce to numeric — handles empty strings and whitespace from CSVs
        p_pos = (
            pd.to_numeric(pdf[pos_cols[0]], errors="coerce").values,
            pd.to_numeric(pdf[pos_cols[1]], errors="coerce").values,
            pd.to_numeric(pdf[pos_cols[2]], errors="coerce").values,
        )

        # Detect valid (non-null) rows
        valid_mask = np.isfinite(p_pos[0]) & np.isfinite(p_pos[1]) & np.isfinite(p_pos[2])

        # Build position array for valid rows only
        pos_array = np.column_stack([p_pos[0][valid_mask], p_pos[1][valid_mask], p_pos[2][valid_mask]])
        valid_times = p_times[valid_mask]

        positions[pid_key] = pos_array

        if has_velocity and velocities is not None:
            v_data = (
                pdf[vel_cols[0]].values[valid_mask],
                pdf[vel_cols[1]].values[valid_mask],
                pdf[vel_cols[2]].values[valid_mask],
            )
            velocities[pid_key] = np.column_stack(v_data)

        if has_mass and masses is not None:
            masses[pid_key] = pdf[mass_col].values[valid_mask]

        if has_radius and radii is not None:
            r_vals = pd.to_numeric(pdf[radius_col], errors="coerce").values[valid_mask]
            finite = r_vals[np.isfinite(r_vals)]
            radii[pid_key] = float(finite[0]) if len(finite) > 0 else 1.0

        if has_startype and startypes is not None:
            k_vals = pd.to_numeric(pdf[startype_col], errors="coerce").values[valid_mask]
            finite_k = k_vals[np.isfinite(k_vals)]
            startypes[pid_key] = int(finite_k[0]) if len(finite_k) > 0 else -1

        # Compute valid intervals (continuous segments without nulls)
        valid_intervals[pid_key] = _compute_valid_intervals(p_times, valid_mask)

    # Build label map for display (None if IDs were already numeric)
    id_labels = None
    if not _try_numeric:
        id_labels = {i: str(label) for label, i in id_label_map.items()}

    return SimulationData(
        particle_ids=particle_ids,
        times=times,
        positions=positions,
        velocities=velocities,
        masses=masses,
        radii=radii if radii else None,
        startypes=startypes if startypes else None,
        valid_intervals=valid_intervals,
        id_labels=id_labels,
    )


def _compute_valid_intervals(
    times: np.ndarray, valid_mask: np.ndarray
) -> list[tuple[float, float]]:
    """Find continuous time intervals where a particle exists.

    Returns a list of (t_start, t_end) tuples for each unbroken
    sequence of valid (non-null) timesteps.
    """
    if not np.any(valid_mask):
        return []

    intervals = []
    in_interval = False
    t_start = 0.0

    for i, (t, valid) in enumerate(zip(times, valid_mask)):
        if valid and not in_interval:
            t_start = t
            in_interval = True
        elif not valid and in_interval:
            intervals.append((t_start, times[i - 1]))
            in_interval = False

    if in_interval:
        intervals.append((t_start, times[len(times) - 1]))

    return intervals


# ---------------------------------------------------------------------------
# HDF5 loader
# ---------------------------------------------------------------------------

def load_hdf5(
    filepath: Path,
    field_map: dict[str, str] | None = None,
    snapshot_group_pattern: str = "snapshot_{:04d}",
) -> SimulationData:
    """Load simulation data from an HDF5 file.

    Supports two common layouts:
    1. Single-file with datasets: /times, /positions (N, T, 3), /ids (N,), etc.
    2. Snapshot groups: /snapshot_0000/positions (N, 3), /snapshot_0000/time, etc.

    Args:
        filepath: Path to the HDF5 file.
        field_map: Optional mapping from standard names to HDF5 dataset paths.
            Standard names: 'ids', 'times', 'positions', 'velocities', 'masses'.
            Example: {'positions': 'PartType1/Coordinates', 'ids': 'PartType1/ParticleIDs'}
        snapshot_group_pattern: Format string for snapshot group names if using
            snapshot layout. Use '{}' for the snapshot index.
    """
    import h5py

    default_field_map = {
        "ids": "ids",
        "times": "times",
        "positions": "positions",
        "velocities": "velocities",
        "masses": "masses",
    }
    if field_map is not None:
        default_field_map.update(field_map)
    fmap = default_field_map

    with h5py.File(filepath, "r") as f:
        # Detect layout: check if top-level has 'positions' dataset or snapshot groups
        if fmap["positions"] in f:
            return _load_hdf5_single(f, fmap)
        else:
            return _load_hdf5_snapshots(f, fmap, snapshot_group_pattern)


def _load_hdf5_single(f, fmap: dict[str, str]) -> SimulationData:
    """Load from a single HDF5 file with arrays: positions (N, T, 3), etc."""
    import h5py

    pos_data = f[fmap["positions"]][:]  # (N, T, 3)
    ids = f[fmap["ids"]][:] if fmap["ids"] in f else np.arange(pos_data.shape[0])
    times = f[fmap["times"]][:] if fmap["times"] in f else np.arange(pos_data.shape[1], dtype=float)

    has_vel = fmap["velocities"] in f
    has_mass = fmap["masses"] in f

    particle_ids = np.sort(np.unique(ids))
    positions = {}
    velocities = {} if has_vel else None
    masses = {} if has_mass else None
    valid_intervals = {}

    vel_data = f[fmap["velocities"]][:] if has_vel else None
    mass_data = f[fmap["masses"]][:] if has_mass else None

    for idx, pid in enumerate(particle_ids):
        pid_key = int(pid)
        p = pos_data[idx]  # (T, 3)

        # Detect valid timesteps
        valid_mask = np.all(np.isfinite(p), axis=1)
        positions[pid_key] = p[valid_mask]

        if has_vel and velocities is not None and vel_data is not None:
            velocities[pid_key] = vel_data[idx][valid_mask]

        if has_mass and masses is not None and mass_data is not None:
            if mass_data.ndim == 2:
                masses[pid_key] = mass_data[idx][valid_mask]
            else:
                # Scalar mass per particle, broadcast across valid times
                masses[pid_key] = np.full(valid_mask.sum(), mass_data[idx])

        valid_intervals[pid_key] = _compute_valid_intervals(times, valid_mask)

    return SimulationData(
        particle_ids=particle_ids,
        times=times,
        positions=positions,
        velocities=velocities,
        masses=masses,
        valid_intervals=valid_intervals,
    )


def _load_hdf5_snapshots(f, fmap: dict[str, str], pattern: str) -> SimulationData:
    """Load from snapshot groups: /snapshot_0000/, /snapshot_0001/, etc."""
    # Find all snapshot groups
    snap_groups = sorted(
        [k for k in f.keys() if k.startswith("snapshot") or k.startswith("snap_")],
        key=lambda k: int("".join(filter(str.isdigit, k)) or "0"),
    )
    if not snap_groups:
        raise ValueError(
            "HDF5 file has no recognized snapshot groups and no top-level 'positions' dataset."
        )

    # Read first snapshot to get particle IDs
    first = f[snap_groups[0]]
    if fmap["ids"] in first:
        particle_ids = np.sort(np.unique(first[fmap["ids"]][:]))
    else:
        n_particles = first[fmap["positions"]].shape[0]
        particle_ids = np.arange(n_particles)

    n_snaps = len(snap_groups)
    n_particles = len(particle_ids)

    # Pre-allocate arrays
    times = np.empty(n_snaps)
    all_pos = np.empty((n_particles, n_snaps, 3))
    all_pos[:] = np.nan

    has_vel = fmap["velocities"] in first
    has_mass = fmap["masses"] in first
    all_vel = np.empty((n_particles, n_snaps, 3)) if has_vel else None
    all_mass = np.empty((n_particles, n_snaps)) if has_mass else None
    if all_vel is not None:
        all_vel[:] = np.nan
    if all_mass is not None:
        all_mass[:] = np.nan

    pid_to_idx = {int(pid): i for i, pid in enumerate(particle_ids)}

    for si, gname in enumerate(snap_groups):
        g = f[gname]

        # Read time
        if fmap["times"] in g:
            times[si] = float(np.array(g[fmap["times"]]).flat[0])
        elif "time" in g.attrs:
            times[si] = float(g.attrs["time"])
        else:
            times[si] = float(si)

        # Read positions
        pos = g[fmap["positions"]][:]
        snap_ids = g[fmap["ids"]][:] if fmap["ids"] in g else np.arange(pos.shape[0])

        for pi, pid in enumerate(snap_ids):
            idx = pid_to_idx.get(int(pid))
            if idx is not None:
                all_pos[idx, si] = pos[pi]
                if has_vel and all_vel is not None and fmap["velocities"] in g:
                    all_vel[idx, si] = g[fmap["velocities"]][pi]
                if has_mass and all_mass is not None and fmap["masses"] in g:
                    m = g[fmap["masses"]]
                    if m.ndim == 0:
                        all_mass[idx, si] = float(m[()])
                    else:
                        all_mass[idx, si] = m[pi]

    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    all_pos = all_pos[:, sort_idx]
    if all_vel is not None:
        all_vel = all_vel[:, sort_idx]
    if all_mass is not None:
        all_mass = all_mass[:, sort_idx]

    # Build per-particle data
    positions = {}
    velocities = {} if has_vel else None
    masses = {} if has_mass else None
    valid_intervals = {}

    for idx, pid in enumerate(particle_ids):
        pid_key = int(pid)
        p = all_pos[idx]  # (T, 3)
        valid_mask = np.all(np.isfinite(p), axis=1)

        positions[pid_key] = p[valid_mask]

        if has_vel and velocities is not None and all_vel is not None:
            velocities[pid_key] = all_vel[idx][valid_mask]

        if has_mass and masses is not None and all_mass is not None:
            masses[pid_key] = all_mass[idx][valid_mask]

        valid_intervals[pid_key] = _compute_valid_intervals(times, valid_mask)

    return SimulationData(
        particle_ids=particle_ids,
        times=times,
        positions=positions,
        velocities=velocities,
        masses=masses,
        valid_intervals=valid_intervals,
    )
