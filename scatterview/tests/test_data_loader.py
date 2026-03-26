"""Tests for data loading."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from scatterview.core.data_loader import SimulationData, load, load_csv


SAMPLE_CSV = Path(__file__).resolve().parents[2] / "data" / "ScatterParts.csv"


class TestLoadCSV:
    def test_loads_sample_data(self):
        data = load(SAMPLE_CSV, fmt="csv")
        assert isinstance(data, SimulationData)
        assert len(data.particle_ids) == 5
        assert data.times[0] < data.times[-1]
        # All particles should have position data
        for pid in data.particle_ids:
            assert int(pid) in data.positions
            assert data.positions[int(pid)].shape[1] == 3

    def test_times_sorted(self):
        data = load(SAMPLE_CSV)
        assert np.all(np.diff(data.times) >= 0)

    def test_mass_loaded(self):
        data = load(SAMPLE_CSV)
        assert data.masses is not None
        for pid in data.particle_ids:
            assert int(pid) in data.masses
            assert len(data.masses[int(pid)]) > 0

    def test_valid_intervals_present(self):
        data = load(SAMPLE_CSV)
        for pid in data.particle_ids:
            intervals = data.valid_intervals[int(pid)]
            assert len(intervals) >= 1
            for t_start, t_end in intervals:
                assert t_start <= t_end

    def test_handles_nulls(self):
        """Particles with null coordinates should have shorter position arrays."""
        csv_content = (
            "ID,time,x,y,z\n"
            "0,0.0,1.0,2.0,3.0\n"
            "1,0.0,4.0,5.0,6.0\n"
            "0,1.0,1.1,2.1,3.1\n"
            "1,1.0,,, \n"  # particle 1 disappears
            "0,2.0,1.2,2.2,3.2\n"
            "1,2.0,,, \n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            data = load(f.name)

        # Particle 0 should have 3 timesteps
        assert data.positions[0].shape[0] == 3
        # Particle 1 should have 1 timestep (only t=0 is valid)
        assert data.positions[1].shape[0] == 1
        # Particle 1 should have one valid interval
        assert len(data.valid_intervals[1]) == 1
        assert data.valid_intervals[1][0] == (0.0, 0.0)

    def test_format_autodetection(self):
        data = load(SAMPLE_CSV)
        assert isinstance(data, SimulationData)

    def test_velocity_columns_optional(self):
        """CSV without velocity columns should load with velocities=None."""
        data = load(SAMPLE_CSV)
        # Sample CSV has no vx, vy, vz columns
        assert data.velocities is None

    def test_reappearing_particle(self):
        """Particle that disappears and reappears should have two valid intervals."""
        csv_content = (
            "ID,time,x,y,z\n"
            "0,0.0,1.0,2.0,3.0\n"
            "0,1.0,,, \n"
            "0,2.0,1.2,2.2,3.2\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            data = load(f.name)

        assert data.positions[0].shape[0] == 2  # two valid timesteps
        assert len(data.valid_intervals[0]) == 2  # two separate intervals


class TestLoadHDF5:
    def test_single_layout(self, tmp_path):
        """Test HDF5 with single-file layout: /positions (N, T, 3)."""
        h5py = pytest.importorskip("h5py")

        filepath = tmp_path / "test.h5"
        n_particles, n_times = 3, 10
        times = np.linspace(0, 1, n_times)
        positions = np.random.randn(n_particles, n_times, 3)
        ids = np.array([0, 1, 2])

        with h5py.File(filepath, "w") as f:
            f.create_dataset("positions", data=positions)
            f.create_dataset("times", data=times)
            f.create_dataset("ids", data=ids)

        data = load(filepath)
        assert len(data.particle_ids) == 3
        assert len(data.times) == 10
        for pid in data.particle_ids:
            assert data.positions[int(pid)].shape == (10, 3)

    def test_snapshot_layout(self, tmp_path):
        """Test HDF5 with snapshot groups."""
        h5py = pytest.importorskip("h5py")

        filepath = tmp_path / "test.h5"
        n_particles = 3
        n_snaps = 5
        ids = np.array([0, 1, 2])

        with h5py.File(filepath, "w") as f:
            for si in range(n_snaps):
                g = f.create_group(f"snapshot_{si:04d}")
                g.create_dataset("positions", data=np.random.randn(n_particles, 3))
                g.create_dataset("ids", data=ids)
                g.attrs["time"] = float(si) * 0.5

        data = load(filepath, field_map={"times": "time"})
        assert len(data.particle_ids) == 3
        assert len(data.times) == 5
