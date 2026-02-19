"""Tests for the physics module."""

import numpy as np
import pytest

from pygeotech.geometry.primitives import Rectangle
from pygeotech.materials.base import Material, assign
from pygeotech.physics.darcy import Darcy
from pygeotech.physics.richards import Richards
from pygeotech.physics.transport import Transport
from pygeotech.physics.heat import HeatTransfer
from pygeotech.physics.mechanics import Mechanics
from pygeotech.materials.constitutive import VanGenuchten


def _make_darcy():
    """Create a simple Darcy problem for testing."""
    domain = Rectangle(Lx=10, Ly=5)
    mesh = domain.generate_mesh(resolution=1.0)
    mat = Material(
        name="test",
        hydraulic_conductivity=1e-5,
        porosity=0.35,
    )
    materials = assign(mesh, {"default": mat})
    return Darcy(mesh, materials), mesh


class TestDarcy:
    def test_creation(self):
        darcy, mesh = _make_darcy()
        assert darcy.name == "darcy"
        assert darcy.primary_field == "H"
        assert darcy.dim == 2
        assert not darcy.is_transient

    def test_coefficients(self):
        darcy, mesh = _make_darcy()
        coeff = darcy.coefficients()
        K = coeff["conductivity"]
        assert K.shape == (mesh.n_cells,)
        np.testing.assert_allclose(K, 1e-5)

    def test_assemble_stiffness(self):
        darcy, mesh = _make_darcy()
        A = darcy.assemble_stiffness()
        assert A.shape == (mesh.n_nodes, mesh.n_nodes)
        # Stiffness matrix should be symmetric
        diff = abs(A - A.T).max()
        assert diff < 1e-12

    def test_residual(self):
        darcy, mesh = _make_darcy()
        H = np.ones(mesh.n_nodes)
        r = darcy.residual(H)
        # Constant head → zero residual (up to numerical precision)
        assert np.allclose(r, 0, atol=1e-10)


class TestRichards:
    def test_creation(self):
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=1.0)
        mat = Material(name="test", hydraulic_conductivity=1e-5, porosity=0.35)
        materials = assign(mesh, {"default": mat})
        vg = VanGenuchten(alpha=0.01, n=1.5, theta_r=0.05, theta_s=0.40)
        richards = Richards(mesh, materials, retention_model=vg)
        assert richards.is_transient
        assert richards.primary_field == "H"

    def test_validate_no_retention(self):
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=1.0)
        mat = Material(name="test", hydraulic_conductivity=1e-5, porosity=0.35)
        materials = assign(mesh, {"default": mat})
        richards = Richards(mesh, materials)
        issues = richards.validate()
        assert any("retention" in s.lower() for s in issues)


class TestTransport:
    def test_creation(self):
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=1.0)
        mat = Material(name="test", hydraulic_conductivity=1e-5, porosity=0.35)
        materials = assign(mesh, {"default": mat})
        transport = Transport(mesh, materials)
        assert transport.is_transient
        assert transport.primary_field == "C"

    def test_set_velocity(self):
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=1.0)
        mat = Material(name="test", hydraulic_conductivity=1e-5, porosity=0.35)
        materials = assign(mesh, {"default": mat})
        transport = Transport(mesh, materials)
        v = np.ones((mesh.n_cells, 2))
        transport.set_velocity(v)
        assert transport._velocity is not None


class TestHeatTransfer:
    def test_creation(self):
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=1.0)
        mat = Material(
            name="test",
            thermal_conductivity=2.0,
            dry_density=1600,
            specific_heat=800,
        )
        materials = assign(mesh, {"default": mat})
        heat = HeatTransfer(mesh, materials)
        assert heat.primary_field == "T"
        assert heat.is_transient


class TestMechanics:
    def test_elastic_stiffness_2d(self):
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=1.0)
        mat = Material(
            name="test",
            youngs_modulus=50e6,
            poissons_ratio=0.3,
            dry_density=1600,
        )
        materials = assign(mesh, {"default": mat})
        mech = Mechanics(mesh, materials)
        C = mech.elastic_stiffness_2d(E=50e6, nu=0.3, plane_strain=True)
        assert C.shape == (3, 3)
        # Symmetric
        np.testing.assert_allclose(C, C.T)
