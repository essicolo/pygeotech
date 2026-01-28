"""Tests for the solver module."""

import numpy as np
import pytest

from pygeotech.geometry.primitives import Rectangle
from pygeotech.materials.base import Material, assign
from pygeotech.physics.darcy import Darcy
from pygeotech.boundaries.base import BoundaryConditions, Dirichlet, Neumann
from pygeotech.boundaries.locators import left, right, top, bottom
from pygeotech.solvers.fipy_backend import FiPyBackend
from pygeotech.solvers.base import Solution
from pygeotech.solvers.analytical import AnalyticalSolver


class TestFiPyBackendDarcy:
    """Test the built-in sparse solver for steady-state Darcy."""

    def _setup_1d_flow(self, nx: int = 11):
        """Create a 1-D-like flow problem (thin strip)."""
        domain = Rectangle(Lx=10, Ly=0.5)
        mesh = domain.generate_mesh(resolution=10.0 / (nx - 1))
        mat = Material(name="uniform", hydraulic_conductivity=1e-4, porosity=0.3)
        materials = assign(mesh, {"default": mat})
        darcy = Darcy(mesh, materials)

        bc = BoundaryConditions()
        bc.add(Dirichlet(field="H", value=10.0, where=left()))
        bc.add(Dirichlet(field="H", value=5.0, where=right()))

        return darcy, bc, mesh

    def test_solve_returns_solution(self):
        darcy, bc, mesh = self._setup_1d_flow()
        solver = FiPyBackend()
        sol = solver.solve(darcy, bc)
        assert isinstance(sol, Solution)
        assert "H" in sol.fields
        assert sol.fields["H"].shape == (mesh.n_nodes,)

    def test_head_monotonic(self):
        """Head should decrease from left (10) to right (5)."""
        darcy, bc, mesh = self._setup_1d_flow()
        solver = FiPyBackend()
        sol = solver.solve(darcy, bc)
        H = sol.fields["H"]

        # Check that left nodes have higher head than right nodes
        left_mask = mesh.nodes[:, 0] < 1.0
        right_mask = mesh.nodes[:, 0] > 9.0
        assert H[left_mask].mean() > H[right_mask].mean()

    def test_head_bounds(self):
        """Head should be between boundary values."""
        darcy, bc, mesh = self._setup_1d_flow()
        solver = FiPyBackend()
        sol = solver.solve(darcy, bc)
        H = sol.fields["H"]
        assert H.min() >= 5.0 - 0.1
        assert H.max() <= 10.0 + 0.1


class TestFiPyBackendDamSeepage:
    """Integration test: dam seepage problem."""

    def test_dam_seepage(self):
        domain = Rectangle(Lx=120, Ly=20, origin=(0, 0))
        cutoff = Rectangle(x0=58, y0=8, width=4, height=12)
        domain.add_subdomain("cutoff", cutoff)

        mesh = domain.generate_mesh(resolution=2.0)

        foundation = Material(
            name="foundation",
            hydraulic_conductivity=1e-5,
            porosity=0.35,
        )
        concrete = Material(
            name="concrete",
            hydraulic_conductivity=1e-10,
            porosity=0.05,
        )

        materials = assign(mesh, {
            "default": foundation,
            "cutoff": concrete,
        })

        darcy = Darcy(mesh, materials)

        bc = BoundaryConditions()
        from pygeotech.boundaries.locators import x_less_than, x_greater_than
        bc.add(Dirichlet(field="H", value=27.0,
                         where=top() & x_less_than(40)))
        bc.add(Dirichlet(field="H", value=20.0,
                         where=top() & x_greater_than(80)))

        solver = FiPyBackend()
        sol = solver.solve(darcy, bc)
        H = sol.fields["H"]

        # Head should be within physical bounds
        assert H.min() >= 19.0
        assert H.max() <= 28.0


class TestAnalyticalSolver:
    def test_confined_1d(self):
        x = np.linspace(0, 10, 50)
        H = AnalyticalSolver.confined_1d(x, H_left=20.0, H_right=10.0, L=10.0)
        assert H[0] == pytest.approx(20.0)
        assert H[-1] == pytest.approx(10.0)
        # Linear
        np.testing.assert_allclose(H, 20.0 - x)

    def test_terzaghi(self):
        z = np.linspace(0, 1, 50)
        u = AnalyticalSolver.terzaghi_consolidation(
            z, t=100, H_drain=1.0, cv=1e-3, delta_sigma=100e3,
        )
        assert u.shape == (50,)
        # Excess pore pressure should be non-negative
        assert np.all(u >= -1e-6)
