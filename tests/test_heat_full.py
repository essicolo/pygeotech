"""Comprehensive tests for the heat transfer module."""

import numpy as np
import pytest

from pygeotech.geometry.primitives import Rectangle
from pygeotech.materials.base import Material, assign
from pygeotech.physics.heat import HeatTransfer
from pygeotech.boundaries.base import BoundaryConditions, Dirichlet, Neumann
from pygeotech.boundaries.locators import left, right, top, bottom
from pygeotech.solvers.fipy_backend import FiPyBackend
from pygeotech.solvers.base import Solution
from pygeotech.time.stepper import Stepper


def _make_heat(resolution=2.0, Lx=10.0, Ly=2.0):
    """Create a heat transfer problem."""
    domain = Rectangle(Lx=Lx, Ly=Ly)
    mesh = domain.generate_mesh(resolution=resolution)
    mat = Material(
        name="soil",
        thermal_conductivity=2.0,
        dry_density=1600.0,
        specific_heat=800.0,
        hydraulic_conductivity=1e-5,
        porosity=0.35,
    )
    materials = assign(mesh, {"default": mat})
    return HeatTransfer(mesh, materials), mesh


class TestHeatProperties:
    def test_name_and_field(self):
        h, _ = _make_heat()
        assert h.name == "heat"
        assert h.primary_field == "T"

    def test_is_transient(self):
        h, _ = _make_heat()
        assert h.is_transient is True

    def test_set_velocity(self):
        h, mesh = _make_heat()
        v = np.ones((mesh.n_cells, 2)) * 1e-5
        h.set_velocity(v)
        assert h._velocity is not None

    def test_coefficients(self):
        h, _ = _make_heat()
        coeff = h.coefficients()
        assert "thermal_conductivity" in coeff
        assert "bulk_heat_capacity" in coeff
        # bulk_heat_capacity = rho * cp = 1600 * 800 = 1_280_000
        np.testing.assert_allclose(coeff["bulk_heat_capacity"], 1600.0 * 800.0)


class TestHeatAssembly:
    def test_assemble_system_shape(self):
        h, mesh = _make_heat()
        bc = BoundaryConditions()
        bc.add(Dirichlet(field="T", value=100.0, where=left()))
        bc.add(Dirichlet(field="T", value=20.0, where=right()))

        A, rhs, d_nodes, d_values = h.assemble_system(bc)
        assert A.shape == (mesh.n_nodes, mesh.n_nodes)
        assert rhs.shape == (mesh.n_nodes,)
        assert len(d_nodes) > 0

    def test_assemble_system_transient(self):
        h, mesh = _make_heat()
        bc = BoundaryConditions()
        bc.add(Dirichlet(field="T", value=100.0, where=left()))
        bc.add(Dirichlet(field="T", value=20.0, where=right()))

        T_old = np.full(mesh.n_nodes, 20.0)
        A, rhs, d_nodes, d_values = h.assemble_system(
            bc, T_old=T_old, dt=3600.0,
        )
        assert A.shape == (mesh.n_nodes, mesh.n_nodes)
        # With transient terms, rhs should have M/dt * T_old contribution
        assert np.any(rhs != 0)


class TestHeatSolve:
    def test_steady_state_conduction(self):
        """Steady-state conduction in a thin strip: linear T profile."""
        h, mesh = _make_heat(Lx=10.0, Ly=0.5)
        bc = BoundaryConditions()
        bc.add(Dirichlet(field="T", value=100.0, where=left()))
        bc.add(Dirichlet(field="T", value=20.0, where=right()))

        solver = FiPyBackend()
        sol = solver.solve(h, bc)
        T = sol.fields["T"]
        assert T.shape == (mesh.n_nodes,)
        # Temperature should be bounded
        assert T.min() >= 19.0
        assert T.max() <= 101.0
        # Left should be hotter than right
        left_mask = mesh.nodes[:, 0] < 1.0
        right_mask = mesh.nodes[:, 0] > 9.0
        assert T[left_mask].mean() > T[right_mask].mean()

    def test_transient_heat(self):
        """Transient heat conduction."""
        h, mesh = _make_heat()
        bc = BoundaryConditions()
        bc.add(Dirichlet(field="T", value=100.0, where=left()))
        bc.add(Dirichlet(field="T", value=20.0, where=right()))

        time = Stepper(t_end=7200.0, dt=3600.0)
        solver = FiPyBackend()
        sol = solver.solve(h, bc, time=time, initial_condition={"T": 20.0})
        T = sol.fields["T"]
        assert T.shape == (mesh.n_nodes,)
        assert sol.times is not None
        assert len(sol.times) == 3  # t=0, t=3600, t=7200

    def test_steady_state_with_advection(self):
        """Conduction + advection: hot side should still be hotter."""
        h, mesh = _make_heat(Lx=10.0, Ly=0.5, resolution=0.5)
        v = np.zeros((mesh.n_cells, 2))
        v[:, 0] = 1e-6  # small velocity to keep Peclet number low
        h.set_velocity(v)

        bc = BoundaryConditions()
        bc.add(Dirichlet(field="T", value=100.0, where=left()))
        bc.add(Dirichlet(field="T", value=20.0, where=right()))

        solver = FiPyBackend()
        sol = solver.solve(h, bc)
        T = sol.fields["T"]
        # Left should be hotter than right
        left_mask = mesh.nodes[:, 0] < 1.0
        right_mask = mesh.nodes[:, 0] > 9.0
        assert T[left_mask].mean() > T[right_mask].mean()


class TestHeatResidual:
    def test_residual_shape(self):
        h, mesh = _make_heat()
        T = np.full(mesh.n_nodes, 50.0)
        res = h.residual(T)
        assert res.shape == (mesh.n_nodes,)

    def test_constant_temp_residual(self):
        """Uniform temperature should give near-zero residual."""
        h, mesh = _make_heat()
        T = np.full(mesh.n_nodes, 50.0)
        res = h.residual(T)
        np.testing.assert_allclose(res, 0.0, atol=1e-8)
