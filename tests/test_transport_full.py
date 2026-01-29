"""Comprehensive tests for the solute transport module."""

import numpy as np
import pytest

from pygeotech.geometry.primitives import Rectangle
from pygeotech.materials.base import Material, assign
from pygeotech.physics.transport import Transport
from pygeotech.boundaries.base import BoundaryConditions, Dirichlet, Neumann
from pygeotech.boundaries.locators import left, right, top, bottom
from pygeotech.solvers.fipy_backend import FiPyBackend
from pygeotech.solvers.base import Solution
from pygeotech.time.stepper import Stepper


def _make_transport(resolution=2.0, **kwargs):
    """Create a transport problem."""
    domain = Rectangle(Lx=20, Ly=2)
    mesh = domain.generate_mesh(resolution=resolution)
    mat = Material(
        name="sand",
        hydraulic_conductivity=1e-4,
        porosity=0.35,
    )
    materials = assign(mesh, {"default": mat})
    return Transport(mesh, materials, **kwargs), mesh


class TestTransportProperties:
    def test_name_and_field(self):
        t, _ = _make_transport()
        assert t.name == "transport"
        assert t.primary_field == "C"

    def test_is_transient(self):
        t, _ = _make_transport()
        assert t.is_transient is True

    def test_set_velocity(self):
        t, mesh = _make_transport()
        v = np.ones((mesh.n_cells, 2)) * 1e-5
        t.set_velocity(v)
        assert t._velocity is not None
        assert t._velocity.shape == (mesh.n_cells, 2)

    def test_coefficients(self):
        t, mesh = _make_transport()
        coeff = t.coefficients()
        assert "porosity" in coeff
        assert "retardation" in coeff
        assert "decay_rate" in coeff


class TestRetardation:
    def test_no_sorption(self):
        t, mesh = _make_transport()
        R = t.compute_retardation()
        np.testing.assert_allclose(R, 1.0)

    def test_linear_sorption(self):
        t, mesh = _make_transport(
            sorption="linear",
            sorption_params={"Kd": 0.001, "bulk_density": 1600.0},
        )
        R = t.compute_retardation()
        # R = 1 + rho_b * Kd / theta
        theta = 0.35
        expected = 1.0 + 1600.0 * 0.001 / theta
        np.testing.assert_allclose(R, expected, rtol=0.01)
        assert np.all(R > 1.0)

    def test_freundlich_sorption(self):
        t, mesh = _make_transport(
            sorption="freundlich",
            sorption_params={"Kf": 0.01, "nf": 0.8, "bulk_density": 1600.0},
        )
        # Without concentration (nf != 1), still returns a scalar R
        C = np.full(mesh.n_nodes, 0.5)
        R = t.compute_retardation(C)
        assert R.shape == (mesh.n_nodes,)
        assert np.all(R >= 1.0)

    def test_langmuir_sorption(self):
        t, mesh = _make_transport(
            sorption="langmuir",
            sorption_params={"Kl": 0.1, "S_max": 0.01, "bulk_density": 1600.0},
        )
        C = np.full(mesh.n_nodes, 0.1)
        R = t.compute_retardation(C)
        assert R.shape == (mesh.n_nodes,)
        assert np.all(R >= 1.0)

    def test_custom_retardation_factor(self):
        t, mesh = _make_transport(retardation_factor=2.5)
        R = t.compute_retardation()
        np.testing.assert_allclose(R, 2.5)


class TestDispersionTensor:
    def test_shape(self):
        t, mesh = _make_transport()
        v = np.random.rand(mesh.n_cells, 2) * 1e-5
        D = t.dispersion_tensor(v)
        assert D.shape == (mesh.n_cells, 2, 2)

    def test_isotropic_when_no_velocity(self):
        """Zero velocity: D = D_m * I."""
        t, mesh = _make_transport(molecular_diffusion=1e-9)
        v = np.zeros((mesh.n_cells, 2))
        D = t.dispersion_tensor(v)
        for ic in range(min(5, mesh.n_cells)):
            np.testing.assert_allclose(
                D[ic], 1e-9 * np.eye(2), atol=1e-20
            )

    def test_symmetric(self):
        t, mesh = _make_transport()
        v = np.random.rand(mesh.n_cells, 2) * 1e-5
        D = t.dispersion_tensor(v)
        for ic in range(min(5, mesh.n_cells)):
            np.testing.assert_allclose(D[ic], D[ic].T, atol=1e-15)


class TestTransportAssembly:
    def test_assemble_system_shape(self):
        t, mesh = _make_transport()
        bc = BoundaryConditions()
        bc.add(Dirichlet(field="C", value=1.0, where=left()))
        bc.add(Dirichlet(field="C", value=0.0, where=right()))

        C = np.zeros(mesh.n_nodes)
        A, rhs, d_nodes, d_values = t.assemble_system(
            bc, C_current=C, C_old=C, dt=100.0,
        )
        assert A.shape == (mesh.n_nodes, mesh.n_nodes)
        assert rhs.shape == (mesh.n_nodes,)
        assert len(d_nodes) > 0

    def test_decay_increases_diagonal(self):
        """First-order decay adds λ*M to the system matrix."""
        t_no_decay, mesh = _make_transport(decay_rate=0.0)
        t_decay, _ = _make_transport(decay_rate=0.01)

        bc = BoundaryConditions()
        C = np.zeros(mesh.n_nodes)

        A1, _, _, _ = t_no_decay.assemble_system(bc, C_current=C, C_old=C, dt=100.0)
        A2, _, _, _ = t_decay.assemble_system(bc, C_current=C, C_old=C, dt=100.0)

        # A with decay should have larger diagonal elements
        assert A2.diagonal().sum() >= A1.diagonal().sum()


class TestTransportSolve:
    def test_steady_state_diffusion(self):
        """Steady-state diffusion with no advection: should be linear."""
        t, mesh = _make_transport(
            dispersion_longitudinal=0.0,
            dispersion_transverse=0.0,
            molecular_diffusion=1e-6,
        )
        bc = BoundaryConditions()
        bc.add(Dirichlet(field="C", value=1.0, where=left()))
        bc.add(Dirichlet(field="C", value=0.0, where=right()))

        solver = FiPyBackend()
        sol = solver.solve(t, bc)
        C = sol.fields["C"]
        assert C.shape == (mesh.n_nodes,)
        # Should be bounded
        assert C.min() >= -0.1
        assert C.max() <= 1.1

    def test_transient_transport_with_velocity(self):
        """Transient transport with advection."""
        t, mesh = _make_transport()
        v = np.zeros((mesh.n_cells, 2))
        v[:, 0] = 1e-4  # flow in x-direction
        t.set_velocity(v)

        bc = BoundaryConditions()
        bc.add(Dirichlet(field="C", value=1.0, where=left()))
        bc.add(Dirichlet(field="C", value=0.0, where=right()))

        time = Stepper(t_end=1000.0, dt=500.0)
        solver = FiPyBackend()
        sol = solver.solve(t, bc, time=time, initial_condition={"C": 0.0})
        C = sol.fields["C"]
        assert C.shape == (mesh.n_nodes,)
        assert sol.times is not None
        assert len(sol.times) == 3  # t=0, t=500, t=1000


class TestTransportResidual:
    def test_residual_shape(self):
        t, mesh = _make_transport()
        C = np.zeros(mesh.n_nodes)
        res = t.residual(C)
        assert res.shape == (mesh.n_nodes,)
