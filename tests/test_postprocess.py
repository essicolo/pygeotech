"""Tests for the postprocess module."""

import numpy as np
import pytest

from pygeotech.geometry.primitives import Rectangle, Line
from pygeotech.materials.base import Material, assign
from pygeotech.physics.darcy import Darcy
from pygeotech.boundaries.base import BoundaryConditions, Dirichlet
from pygeotech.boundaries.locators import left, right
from pygeotech.solvers.fipy_backend import FiPyBackend
from pygeotech.postprocess.fields import compute_gradient, compute_velocity
from pygeotech.postprocess.probes import PointProbe, LineProbe
from pygeotech.postprocess.export import export_csv


class TestComputeGradient:
    def test_linear_field(self):
        """Gradient of a linear field should be constant."""
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=1.0)
        # Linear field: f = 2*x + 3*y
        field = 2.0 * mesh.nodes[:, 0] + 3.0 * mesh.nodes[:, 1]
        grad = compute_gradient(mesh, field)
        np.testing.assert_allclose(grad[:, 0], 2.0, atol=1e-10)
        np.testing.assert_allclose(grad[:, 1], 3.0, atol=1e-10)


class TestProbes:
    def test_point_probe(self):
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=1.0)
        field = mesh.nodes[:, 0]  # f = x
        probe = PointProbe(location=(5, 2.5))
        val = probe.sample(mesh, field)
        assert abs(val - 5.0) < 1.0

    def test_line_probe(self):
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=1.0)
        field = mesh.nodes[:, 0]
        probe = LineProbe(start=(0, 2.5), end=(10, 2.5), n_points=20)
        dist, vals = probe.sample(mesh, field)
        assert len(dist) == 20
        assert len(vals) == 20
        # Values should increase along x
        assert vals[-1] > vals[0]


class TestExportCSV:
    def test_export(self, tmp_path):
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=2.0)
        mat = Material(name="test", hydraulic_conductivity=1e-5, porosity=0.3)
        materials = assign(mesh, {"default": mat})
        darcy = Darcy(mesh, materials)
        bc = BoundaryConditions()
        bc.add(Dirichlet(field="H", value=10.0, where=left()))
        bc.add(Dirichlet(field="H", value=5.0, where=right()))
        sol = FiPyBackend().solve(darcy, bc)

        out_file = str(tmp_path / "test.csv")
        export_csv(sol, out_file)

        data = np.genfromtxt(out_file, delimiter=",", names=True)
        assert "x" in data.dtype.names
        assert "y" in data.dtype.names
        assert "H" in data.dtype.names

    def test_export_along_line(self, tmp_path):
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=2.0)
        mat = Material(name="test", hydraulic_conductivity=1e-5, porosity=0.3)
        materials = assign(mesh, {"default": mat})
        darcy = Darcy(mesh, materials)
        bc = BoundaryConditions()
        bc.add(Dirichlet(field="H", value=10.0, where=left()))
        bc.add(Dirichlet(field="H", value=5.0, where=right()))
        sol = FiPyBackend().solve(darcy, bc)

        line = Line(start=(0, 2.5), end=(10, 2.5))
        out_file = str(tmp_path / "line.csv")
        export_csv(sol, out_file, along=line)

        data = np.genfromtxt(out_file, delimiter=",", names=True)
        assert len(data) == 100  # default sample count
