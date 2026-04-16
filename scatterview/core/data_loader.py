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
    """Infer data format from the file extension.

    Args:
        filepath: Path to the data file.

    Returns:
        Format string ('csv' or 'hdf5').

    Raises:
        ValueError: If the extension is not recognized.
    """
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

    Args:
        filepath: Path to the CSV file.
        id_col: Column name for particle identifiers.
        time_col: Column name for simulation time.
        pos_cols: Column names for (x, y, z) position components.
        vel_cols: Column names for (vx, vy, vz) velocity components, or None to skip.
        mass_col: Column name for particle mass, or None to skip.
        radius_col: Column name for particle radius, or None to skip.

    Returns:
        SimulationData with all fields populated.
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
    else:
        # String IDs — assign stable integer keys in sorted order
        sorted_labels = sorted(particle_ids_raw, key=str)
        id_label_map = {label: i for i, label in enumerate(sorted_labels)}
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
        # Positions use pd.to_numeric because empty/whitespace cells are
        # the normal way to represent "particle doesn't exist at this
        # timestep" in ragged CSV grids.  These become NaN → filtered below.
        p_pos = np.column_stack([
            pd.to_numeric(pdf[pos_cols[0]], errors="coerce").values,
            pd.to_numeric(pdf[pos_cols[1]], errors="coerce").values,
            pd.to_numeric(pdf[pos_cols[2]], errors="coerce").values,
        ])

        # Filter to rows where the particle exists (non-NaN positions)
        valid_mask = np.isfinite(p_pos).all(axis=1)
        positions[pid_key] = p_pos[valid_mask]

        # Velocities, masses, radii, startypes only exist at valid rows.
        # Bad values here are genuinely malformed data and should crash.
        if has_velocity and velocities is not None:
            velocities[pid_key] = np.column_stack([
                pdf[vel_cols[0]].values[valid_mask].astype(float),
                pdf[vel_cols[1]].values[valid_mask].astype(float),
                pdf[vel_cols[2]].values[valid_mask].astype(float),
            ])

        if has_mass and masses is not None:
            masses[pid_key] = pdf[mass_col].values[valid_mask].astype(float)

        if has_radius and radii is not None:
            radii[pid_key] = float(pdf[radius_col].values[valid_mask][0])

        if has_startype and startypes is not None:
            startypes[pid_key] = int(pdf[startype_col].values[valid_mask][0])

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

    Args:
        times: (N,) array of simulation times for this particle's rows.
        valid_mask: (N,) bool array — True where position data is finite.

    Returns:
        List of (t_start, t_end) tuples for each contiguous valid segment.
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
    key: str | None = None,
    field_map: dict[str, str] | None = None,
) -> SimulationData:
    """Load simulation data from an HDF5 file.

    Supports three layouts:
    1. Multi-index DataFrame (pandas HDF5 table): multi-index with
       (time, ID), columns include x, y, z and optionally vx, vy, vz,
       mass, k (startype), radius.
    2. Single-file with datasets: /times, /positions (N, T, 3), /ids (N,).
    3. Snapshot groups: any top-level group whose name starts with
       "snapshot" or "snap_" is treated as a snapshot.  Groups are
       sorted numerically by the digits in their name, so both
       zero-padded (snapshot_0000) and unpadded (snap_0, snap_1, ...,
       snap_102) names work correctly.

    By default the loader assumes the chosen layout begins at the top
    level of the file.  If ``key`` is given, that key is treated as the
    effective root: pandas reads use ``pd.read_hdf(..., key=key)``, and
    raw-h5py layouts look for ``positions`` / ``snapshot*`` groups inside
    the ``key`` group.  If no layout matches, the loader raises a
    ``ValueError`` rather than silently returning empty data.

    Args:
        filepath: Path to the HDF5 file.
        key: Optional HDF5 key identifying the group (or pandas dataset)
            that holds the phase-space data.  Required when the file
            contains multiple pandas datasets or nests the data under a
            non-root group.
        field_map: Optional mapping from standard names to HDF5 dataset paths.
    """
    import h5py

    # Try multi-index DataFrame first (pandas HDF5 table format).
    # If the user passed a key, trust it — let pandas errors propagate.
    # If not, a failure here just means this file isn't a pandas table;
    # fall through to the raw-h5py layouts.
    try:
        return _load_hdf5_multiindex(filepath, key=key)
    except (KeyError, ValueError, TypeError, ImportError) as pandas_err:
        if key is not None:
            raise
        pandas_error = pandas_err

    default_field_map = {
        "ids": "ids",
        "times": "times",
        "positions": "positions",
        "velocities": "velocities",
        "masses": "masses",
        "radii": "radii",
        "startypes": "startypes",
    }
    if field_map is not None:
        default_field_map.update(field_map)
    fmap = default_field_map

    with h5py.File(filepath, "r") as f:
        root = f[key] if key is not None else f

        if fmap["positions"] in root:
            return _load_hdf5_single(root, fmap)

        snap_groups = [
            k for k in root.keys()
            if k.startswith("snapshot") or k.startswith("snap_")
        ]
        if snap_groups:
            return _load_hdf5_snapshots(root, fmap)

        where = f"key {key!r}" if key is not None else "the top level"
        raise ValueError(
            f"HDF5 file '{filepath}' has no recognized ScatterView layout at "
            f"{where}. Expected one of:\n"
            f"  1. A pandas multi-index DataFrame (pass key= to select among "
            f"multiple datasets)\n"
            f"  2. A '{fmap['positions']}' dataset\n"
            f"  3. Groups named 'snapshot*' or 'snap_*'\n"
            f"Pandas load error was: {pandas_error}"
        )


def _load_hdf5_multiindex(filepath: Path, key: str | None = None) -> SimulationData:
    """Load from a pandas HDF5 table with multi-index (time, ID).

    Expected format: DataFrame written with ``df.to_hdf(path, key=...)``
    where the index has two levels:
        - Level 0: simulation time (float), assumed already sorted
        - Level 1: particle ID (int or string — strings become user labels)

    All bodies are assumed to share the same block timesteps (no gaps).
    Per-particle data is extracted via pandas multi-index slicing
    (``df.xs(pid, level=1)``) which reads directly from HDF5 without
    expanding the full N_particles × N_timesteps table into memory.

    Columns (detected flexibly):
        Required: x, y, z
        Optional: vx, vy, vz, mass, radius, k (BSE stellar type)

    Args:
        filepath: Path to the HDF5 file containing a pandas table.

    Returns:
        SimulationData with all fields populated.
    """
    df = pd.read_hdf(filepath) if key is None else pd.read_hdf(filepath, key=key)

    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels < 2:
        raise ValueError("HDF5 file does not contain a multi-index DataFrame")

    # Flexible column detection (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}

    def _find_col(candidates):
        for c in candidates:
            if c.lower() in col_map:
                return col_map[c.lower()]
        return None

    x_col = _find_col(["x", "r0", "pos_x"])
    y_col = _find_col(["y", "r1", "pos_y"])
    z_col = _find_col(["z", "r2", "pos_z"])
    if x_col is None or y_col is None or z_col is None:
        raise ValueError(
            f"HDF5 multi-index DataFrame must have x, y, z columns. "
            f"Found: {list(df.columns)}"
        )

    vx_col = _find_col(["vx", "v0", "vel_x"])
    vy_col = _find_col(["vy", "v1", "vel_y"])
    vz_col = _find_col(["vz", "v2", "vel_z"])
    has_vel = vx_col is not None and vy_col is not None and vz_col is not None

    mass_col = _find_col(["mass", "m"])
    radius_col = _find_col(["radius", "rad"])
    startype_col = _find_col(["k", "startype", "kstar", "stellar_type"])

    # Times from the first index level (assumed sorted)
    times = np.array(df.index.get_level_values(0).unique(), dtype=float)

    # Particle IDs from the second index level
    raw_ids = df.index.get_level_values(1).unique()

    # Convert IDs: use originals for targeting, map to integer keys internally
    id_labels = None
    try:
        particle_ids = np.sort(np.array([int(pid) for pid in raw_ids]))
        id_to_key = {pid: int(pid) for pid in raw_ids}
    except (ValueError, TypeError):
        # String IDs — assign integer keys, store original labels for GUI
        sorted_labels = sorted(raw_ids, key=str)
        id_labels = {i: str(label) for i, label in enumerate(sorted_labels)}
        id_to_key = {label: i for i, label in enumerate(sorted_labels)}
        particle_ids = np.arange(len(sorted_labels))

    # Extract per-particle data using multi-index slicing.
    # df.xs(pid, level=1) returns a DataFrame indexed by time for one particle,
    # without copying the data for other particles.
    positions = {}
    velocities = {} if has_vel else None
    masses = {} if mass_col else None
    radii = {} if radius_col else None
    startypes = {} if startype_col else None
    valid_intervals = {}

    for raw_pid in raw_ids:
        pid_key = id_to_key[raw_pid]

        # Slice this particle's data directly from the multi-index, then
        # reindex against the global times so that any (time, pid) rows the
        # user omitted appear as NaN. This makes row omission semantically
        # equivalent to NaN sentinels — the loader treats both as "particle
        # absent at this timestep" and records the gap in valid_intervals.
        particle_df = df.xs(raw_pid, level=1).reindex(times)

        pos_x = particle_df[x_col].values.astype(float)
        pos_y = particle_df[y_col].values.astype(float)
        pos_z = particle_df[z_col].values.astype(float)

        valid_mask = np.isfinite(pos_x) & np.isfinite(pos_y) & np.isfinite(pos_z)
        positions[pid_key] = np.column_stack([
            pos_x[valid_mask], pos_y[valid_mask], pos_z[valid_mask],
        ])

        if has_vel and velocities is not None:
            velocities[pid_key] = np.column_stack([
                particle_df[vx_col].values[valid_mask].astype(float),
                particle_df[vy_col].values[valid_mask].astype(float),
                particle_df[vz_col].values[valid_mask].astype(float),
            ])

        if mass_col and masses is not None:
            mass_vals = particle_df[mass_col].values[valid_mask].astype(float)
            finite = mass_vals[np.isfinite(mass_vals)]
            if len(finite) > 0:
                masses[pid_key] = finite

        if radius_col and radii is not None:
            rad_vals = particle_df[radius_col].values[valid_mask].astype(float)
            finite = rad_vals[np.isfinite(rad_vals)]
            if len(finite) > 0:
                radii[pid_key] = float(np.median(finite))

        if startype_col and startypes is not None:
            k_vals = particle_df[startype_col].values[valid_mask]
            finite_k = k_vals[np.isfinite(k_vals.astype(float))]
            if len(finite_k) > 0:
                startypes[pid_key] = int(finite_k[-1])

        valid_intervals[pid_key] = _compute_valid_intervals(times, valid_mask)

    return SimulationData(
        particle_ids=particle_ids,
        times=times,
        positions=positions,
        velocities=velocities,
        masses=masses,
        radii=radii,
        startypes=startypes,
        valid_intervals=valid_intervals,
        id_labels=id_labels,
    )


def _load_hdf5_single(f, fmap: dict[str, str]) -> SimulationData:
    """Load from a single HDF5 file with arrays: positions (N, T, 3), etc.

    Args:
        f: Open h5py.File handle.
        fmap: Mapping from standard field names ('ids', 'times', 'positions',
            'velocities', 'masses') to HDF5 dataset paths within the file.

    Returns:
        SimulationData with all fields populated.
    """
    import h5py

    pos_data = f[fmap["positions"]][:]  # (N, T, 3)
    ids = f[fmap["ids"]][:] if fmap["ids"] in f else np.arange(pos_data.shape[0])
    times = f[fmap["times"]][:] if fmap["times"] in f else np.arange(pos_data.shape[1], dtype=float)

    has_vel = fmap["velocities"] in f
    has_mass = fmap["masses"] in f
    has_rad = fmap["radii"] in f
    has_kst = fmap["startypes"] in f

    # Row order in pos_data / vel_data / mass_data / rad_data / kst_data is the
    # user's intended particle order: row i is the trajectory of particle ids[i].
    # Preserve it.
    particle_ids = np.asarray(ids)
    positions = {}
    velocities = {} if has_vel else None
    masses = {} if has_mass else None
    radii = {} if has_rad else None
    startypes = {} if has_kst else None
    valid_intervals = {}

    vel_data = f[fmap["velocities"]][:] if has_vel else None
    mass_data = f[fmap["masses"]][:] if has_mass else None
    rad_data = f[fmap["radii"]][:] if has_rad else None
    kst_data = f[fmap["startypes"]][:] if has_kst else None

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

        if has_rad and radii is not None and rad_data is not None:
            radii[pid_key] = float(rad_data[idx])

        if has_kst and startypes is not None and kst_data is not None:
            startypes[pid_key] = int(kst_data[idx])

        valid_intervals[pid_key] = _compute_valid_intervals(times, valid_mask)

    return SimulationData(
        particle_ids=particle_ids,
        times=times,
        positions=positions,
        velocities=velocities,
        masses=masses,
        radii=radii,
        startypes=startypes,
        valid_intervals=valid_intervals,
    )


def _load_hdf5_snapshots(f, fmap: dict[str, str]) -> SimulationData:
    """Load from snapshot groups: /snapshot_0000/, /snap_0/, etc.

    Detects any top-level group whose name starts with "snapshot" or
    "snap_".  Groups are sorted numerically by the digits in their name,
    so both zero-padded (snapshot_0000) and unpadded (snap_0, snap_102)
    names work correctly.

    Args:
        f: Open h5py.File handle.
        fmap: Mapping from standard field names to HDF5 dataset paths.

    Returns:
        SimulationData with all fields populated.
    """
    # Find all snapshot groups (any key starting with "snapshot" or "snap_"),
    # sorted by the numeric value of the digits in the name.
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

    # Per-snapshot static per-particle data (radii, startypes) — values come
    # from inside each snapshot group, aligned to that snapshot's `ids`. Each
    # is collapsed to a single scalar per particle by recording the first value
    # encountered as we iterate snapshots in time order.
    has_rad = fmap["radii"] in first
    has_kst = fmap["startypes"] in first
    radii = {} if has_rad else None
    startypes = {} if has_kst else None

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

        # Read static per-particle datasets if present in this snapshot
        snap_rad = g[fmap["radii"]][:] if (radii is not None and fmap["radii"] in g) else None
        snap_kst = g[fmap["startypes"]][:] if (startypes is not None and fmap["startypes"] in g) else None

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

                pid_int = int(pid)
                if snap_rad is not None and pid_int not in radii:
                    radii[pid_int] = float(snap_rad[pi])
                if snap_kst is not None and pid_int not in startypes:
                    startypes[pid_int] = int(snap_kst[pi])

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
        radii=radii,
        startypes=startypes,
        valid_intervals=valid_intervals,
    )
