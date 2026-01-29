"""Comprehensive tests for the Richards equation module."""

import numpy as np
import pytest

from pygeotech.geometry.primitives import Rectangle
from pygeotech.materials.base import Material, assign
from pygeotech.materials.constitutive import VanGenuchten, BrooksCorey
from pygeotech.physics.richards import Richards
from pygeotech.boundaries.base import BoundaryConditions, Dirichlet, Neumann
from pygeotech.boundaries.locators import left, right, top, bottom
from pygeotech.solvers.fipy_backend import FiPyBackend
from pygeotech.solvers.base import Solution


def _make_richards(resolution=2.0, Lx=10.0, Ly=5.0):
    """Create a Richards problem with Van Genuchten retention."""
    domain = Rectangle(Lx=Lx, Ly=Ly)
    mesh = domain.generate_mesh(resolution=resolution)
    mat = Material(
        name="sandy_loam",
        hydraulic_conductivity=1e-5,
        porosity=0.40,
    )
    materials = assign(mesh, {"default": mat})
    vg = VanGenuchten(alpha=0.01, n=1.5, theta_r=0.05, theta_s=0.40)
    return Richards(mesh, materials, retention_model=vg), mesh, vg


class TestRichardsProperties:
    def test_name_and_field(self):
        r, _, _ = _make_richards()
        assert r.name == "richards"
        assert r.primary_field == "H"

    def test_is_transient(self):
        r, _, _ = _make_richards()
        assert r.is_transient is True

    def test_validate_no_retention(self):
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=2.0)
        mat = Material(name="test", hydraulic_conductivity=1e-5, porosity=0.35)
        materials = assign(mesh, {"default": mat})
        richards = Richards(mesh, materials)
        issues = richards.validate()
        assert any("retention" in s.lower() for s in issues)

    def test_validate_with_retention(self):
        r, _, _ = _make_richards()
        issues = r.validate()
        assert not any("retention" in s.lower() for s in issues)


class TestRichardsCoefficients:
    def test_coefficients_keys(self):
        r, _, _ = _make_richards()
        coeff = r.coefficients()
        assert "conductivity" in coeff
        assert "porosity" in coeff

    def test_effective_conductivity_saturated(self):
        """When H >> z (saturated), K_eff ~ K_sat."""
        r, mesh, _ = _make_richards()
        # High total head means positive pressure head (saturated)
        H = np.full(mesh.n_nodes, 100.0)
        K_eff = r.effective_conductivity(H)
        assert K_eff.shape == (mesh.n_cells,)
        # Should be close to K_sat
        np.testing.assert_allclose(K_eff, 1e-5, rtol=0.01)

    def test_effective_conductivity_unsaturated(self):
        """When h < 0, K_eff < K_sat."""
        r, mesh, _ = _make_richards()
        # Low total head means negative pressure head
        z = mesh.nodes[:, -1]
        H = z - 5.0  # h = -5 m everywhere
        K_eff = r.effective_conductivity(H)
        assert np.all(K_eff < 1e-5 + 1e-15)
        assert np.all(K_eff > 0)

    def test_effective_conductivity_no_retention(self):
        """Without retention model, K_eff = K_sat."""
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=2.0)
        mat = Material(name="test", hydraulic_conductivity=1e-5, porosity=0.35)
        materials = assign(mesh, {"default": mat})
        richards = Richards(mesh, materials)
        H = np.zeros(mesh.n_nodes)
        K_eff = richards.effective_conductivity(H)
        np.testing.assert_allclose(K_eff, 1e-5)


class TestRichardsMoistureCapacity:
    def test_capacity_saturated(self):
        """C(h) = 0 for h >= 0 (saturated)."""
        r, mesh, _ = _make_richards()
        H = np.full(mesh.n_nodes, 100.0)  # saturated
        C = r.moisture_capacity(H)
        np.testing.assert_allclose(C, 0.0, atol=1e-10)

    def test_capacity_unsaturated_positive(self):
        """C(h) > 0 for h < 0 (unsaturated)."""
        r, mesh, _ = _make_richards()
        z = mesh.nodes[:, -1]
        H = z - 2.0  # h = -2 m
        C = r.moisture_capacity(H)
        assert np.all(C >= 0)

    def test_capacity_no_retention(self):
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=2.0)
        mat = Material(name="test", hydraulic_conductivity=1e-5, porosity=0.35)
        materials = assign(mesh, {"default": mat})
        richards = Richards(mesh, materials)
        H = np.zeros(mesh.n_nodes)
        C = richards.moisture_capacity(H)
        np.testing.assert_allclose(C, 0.0)


class TestRichardsWaterContent:
    def test_water_content_saturated(self):
        r, mesh, vg = _make_richards()
        H = np.full(mesh.n_nodes, 100.0)
        theta = r.water_content_at(H)
        np.testing.assert_allclose(theta, vg.theta_s, atol=1e-10)

    def test_water_content_range(self):
        r, mesh, vg = _make_richards()
        z = mesh.nodes[:, -1]
        H = z - 3.0
        theta = r.water_content_at(H)
        assert np.all(theta >= vg.theta_r - 1e-10)
        assert np.all(theta <= vg.theta_s + 1e-10)


class TestRichardsAssembly:
    def test_stiffness_shape(self):
        r, mesh, _ = _make_richards()
        K_eff = np.full(mesh.n_cells, 1e-5)
        A = r.assemble_stiffness(K_eff)
        assert A.shape == (mesh.n_nodes, mesh.n_nodes)

    def test_stiffness_symmetric(self):
        r, mesh, _ = _make_richards()
        K_eff = np.full(mesh.n_cells, 1e-5)
        A = r.assemble_stiffness(K_eff)
        diff = abs(A - A.T).max()
        assert diff < 1e-12

    def test_mass_matrix_diagonal(self):
        r, mesh, _ = _make_richards()
        capacity = np.ones(mesh.n_nodes)
        M = r.assemble_mass(capacity)
        assert M.shape == (mesh.n_nodes, mesh.n_nodes)
        # Lumped mass matrix is diagonal
        M_dense = M.toarray()
        np.testing.assert_allclose(
            M_dense, np.diag(np.diag(M_dense)), atol=1e-15
        )

    def test_assemble_system_returns_tuple(self):
        r, mesh, _ = _make_richards()
        bc = BoundaryConditions()
        bc.add(Dirichlet(field="H", value=10.0, where=left()))
        bc.add(Dirichlet(field="H", value=5.0, where=right()))
        H = np.full(mesh.n_nodes, 7.5)
        A, rhs, d_nodes, d_values = r.assemble_system(
            bc, H_current=H, H_old=H, dt=100.0,
        )
        assert A.shape == (mesh.n_nodes, mesh.n_nodes)
        assert rhs.shape == (mesh.n_nodes,)
        assert len(d_nodes) > 0
        assert len(d_values) > 0


class TestRichardsSolve:
    def test_steady_state_solve(self):
        """Steady-state Richards should behave like Darcy when saturated."""
        r, mesh, _ = _make_richards()
        bc = BoundaryConditions()
        bc.add(Dirichlet(field="H", value=10.0, where=left()))
        bc.add(Dirichlet(field="H", value=5.0, where=right()))

        solver = FiPyBackend(picard_max_iter=30)
        sol = solver.solve(r, bc, initial_condition={"H": 10.0})
        H = sol.fields["H"]
        assert H.shape == (mesh.n_nodes,)
        # Head should be bounded
        assert H.min() >= 4.5
        assert H.max() <= 10.5


class TestRichardsResidual:
    def test_constant_head_residual(self):
        """Constant head in saturated medium should give near-zero residual."""
        r, mesh, _ = _make_richards()
        H = np.full(mesh.n_nodes, 10.0)
        res = r.residual(H)
        assert np.allclose(res, 0.0, atol=1e-8)
