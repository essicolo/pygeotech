"""Comprehensive tests for the solid mechanics module."""

import numpy as np
import pytest

from pygeotech.geometry.primitives import Rectangle
from pygeotech.materials.base import Material, assign
from pygeotech.physics.mechanics import Mechanics
from pygeotech.boundaries.base import BoundaryConditions, Dirichlet
from pygeotech.boundaries.locators import left, right, top, bottom
from pygeotech.solvers.fipy_backend import FiPyBackend
from pygeotech.solvers.base import Solution


def _make_mechanics(resolution=2.0, Lx=10.0, Ly=5.0):
    """Create a mechanics problem."""
    domain = Rectangle(Lx=Lx, Ly=Ly)
    mesh = domain.generate_mesh(resolution=resolution)
    mat = Material(
        name="clay",
        youngs_modulus=50e6,
        poissons_ratio=0.3,
        dry_density=1600.0,
        hydraulic_conductivity=1e-8,
        porosity=0.40,
    )
    materials = assign(mesh, {"default": mat})
    return Mechanics(mesh, materials), mesh


class TestMechanicsProperties:
    def test_name_and_field(self):
        m, _ = _make_mechanics()
        assert m.name == "mechanics"
        assert m.primary_field == "u"

    def test_is_not_transient(self):
        m, _ = _make_mechanics()
        assert m.is_transient is False

    def test_coefficients(self):
        m, _ = _make_mechanics()
        coeff = m.coefficients()
        assert "youngs_modulus" in coeff
        assert "poissons_ratio" in coeff
        assert "density" in coeff

    def test_set_pore_pressure(self):
        m, mesh = _make_mechanics()
        p = np.ones(mesh.n_nodes) * 1000.0
        m.set_pore_pressure(p)
        assert m._pore_pressure is not None
        np.testing.assert_allclose(m._pore_pressure, 1000.0)


class TestElasticStiffness2D:
    def test_plane_strain_shape(self):
        m, _ = _make_mechanics()
        D = m.elastic_stiffness_2d(E=50e6, nu=0.3, plane_strain=True)
        assert D.shape == (3, 3)

    def test_plane_stress_shape(self):
        m, _ = _make_mechanics()
        D = m.elastic_stiffness_2d(E=50e6, nu=0.3, plane_strain=False)
        assert D.shape == (3, 3)

    def test_symmetric(self):
        m, _ = _make_mechanics()
        D = m.elastic_stiffness_2d(E=50e6, nu=0.3, plane_strain=True)
        np.testing.assert_allclose(D, D.T)

    def test_positive_definite(self):
        m, _ = _make_mechanics()
        D = m.elastic_stiffness_2d(E=50e6, nu=0.3, plane_strain=True)
        eigenvalues = np.linalg.eigvals(D)
        assert np.all(eigenvalues > 0)

    def test_plane_strain_vs_stress(self):
        """Plane strain should be stiffer than plane stress."""
        m, _ = _make_mechanics()
        D_strain = m.elastic_stiffness_2d(E=50e6, nu=0.3, plane_strain=True)
        D_stress = m.elastic_stiffness_2d(E=50e6, nu=0.3, plane_strain=False)
        # D11 (plane strain) > D11 (plane stress)
        assert D_strain[0, 0] > D_stress[0, 0]


class TestMechanicsAssembly:
    def test_stiffness_shape(self):
        m, mesh = _make_mechanics()
        K = m.assemble_stiffness()
        n_dof = 2 * mesh.n_nodes
        assert K.shape == (n_dof, n_dof)

    def test_stiffness_symmetric(self):
        m, mesh = _make_mechanics()
        K = m.assemble_stiffness()
        diff = abs(K - K.T).max()
        assert diff < 1e-6

    def test_body_force_shape(self):
        m, mesh = _make_mechanics()
        f = m.assemble_body_force()
        assert f.shape == (2 * mesh.n_nodes,)

    def test_body_force_downward(self):
        """Gravity should produce only negative y-forces."""
        m, mesh = _make_mechanics()
        f = m.assemble_body_force()
        # y-components (odd indices) should be negative
        fy = f[1::2]
        assert np.all(fy <= 0)
        # x-components should be zero
        fx = f[0::2]
        np.testing.assert_allclose(fx, 0.0, atol=1e-10)

    def test_pore_pressure_force(self):
        m, mesh = _make_mechanics()
        # No pore pressure set -> zero force
        f0 = m.assemble_pore_pressure_force()
        np.testing.assert_allclose(f0, 0.0)

        # With pore pressure
        m.set_pore_pressure(np.ones(mesh.n_nodes) * 10000.0)
        f = m.assemble_pore_pressure_force()
        assert f.shape == (2 * mesh.n_nodes,)
        # Should not be all zero
        assert np.any(np.abs(f) > 0)

    def test_assemble_system(self):
        m, mesh = _make_mechanics()
        bc = BoundaryConditions()
        bc.add(Dirichlet(field="u", value=0.0, where=bottom()))

        K, rhs, d_dofs, d_vals = m.assemble_system(bc)
        n_dof = 2 * mesh.n_nodes
        assert K.shape == (n_dof, n_dof)
        assert rhs.shape == (n_dof,)
        assert len(d_dofs) > 0


class TestMechanicsSolve:
    def test_gravity_loading(self):
        """Simple gravity loading: fix bottom, gravity load."""
        m, mesh = _make_mechanics(Lx=5.0, Ly=5.0)
        bc = BoundaryConditions()
        bc.add(Dirichlet(field="ux", value=0.0, where=left()))
        bc.add(Dirichlet(field="ux", value=0.0, where=right()))
        bc.add(Dirichlet(field="u", value=0.0, where=bottom()))

        solver = FiPyBackend()
        sol = solver.solve(m, bc)
        u = sol.fields["u"]
        ux = sol.fields["ux"]
        uy = sol.fields["uy"]
        assert u.shape == (2 * mesh.n_nodes,)
        assert ux.shape == (mesh.n_nodes,)
        assert uy.shape == (mesh.n_nodes,)
        # Under gravity, vertical displacement should be negative
        # (downward) for interior nodes
        top_mask = mesh.nodes[:, 1] > 4.0
        assert uy[top_mask].mean() < 0

    def test_with_pore_pressure(self):
        """Pore pressure should reduce effective stress."""
        m, mesh = _make_mechanics(Lx=5.0, Ly=5.0)
        bc = BoundaryConditions()
        bc.add(Dirichlet(field="u", value=0.0, where=bottom()))
        bc.add(Dirichlet(field="ux", value=0.0, where=left()))
        bc.add(Dirichlet(field="ux", value=0.0, where=right()))

        # Solve without pore pressure
        solver = FiPyBackend()
        sol1 = solver.solve(m, bc)

        # Solve with pore pressure
        m.set_pore_pressure(np.ones(mesh.n_nodes) * 50000.0)
        sol2 = solver.solve(m, bc)

        # Displacements should differ
        u1 = sol1.fields["u"]
        u2 = sol2.fields["u"]
        assert not np.allclose(u1, u2, atol=1e-10)


class TestMechanicsResidual:
    def test_residual_shape(self):
        m, mesh = _make_mechanics()
        u = np.zeros(2 * mesh.n_nodes)
        res = m.residual(u)
        assert res.shape == (2 * mesh.n_nodes,)
