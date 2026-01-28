"""Tests for the materials module."""

import numpy as np
import pytest

from pygeotech.materials.base import Material, MaterialMap, assign
from pygeotech.materials.library import sand, clay, silt, gravel, concrete, rock
from pygeotech.materials.constitutive import (
    MohrCoulomb, CamClay, VanGenuchten, BrooksCorey,
)
from pygeotech.materials.fields import GaussianField
from pygeotech.geometry.primitives import Rectangle


class TestMaterial:
    def test_creation(self):
        m = Material(name="test", hydraulic_conductivity=1e-5, porosity=0.35)
        assert m.name == "test"
        assert m["hydraulic_conductivity"] == 1e-5
        assert m["porosity"] == 0.35

    def test_setitem(self):
        m = Material(name="test")
        m["K"] = 1e-6
        assert m["K"] == 1e-6

    def test_contains(self):
        m = Material(name="test", K=1e-5)
        assert "K" in m
        assert "missing" not in m

    def test_get_default(self):
        m = Material(name="test")
        assert m.get("missing", 42) == 42

    def test_properties(self):
        m = Material(name="test", hydraulic_conductivity=1e-5, porosity=0.3)
        assert m.hydraulic_conductivity == 1e-5
        assert m.porosity == 0.3


class TestLibrary:
    def test_sand(self):
        assert sand.hydraulic_conductivity == 1e-4
        assert sand.name == "sand"

    def test_clay(self):
        assert clay.hydraulic_conductivity == 1e-9

    def test_all_materials_have_K(self):
        for mat in [sand, clay, silt, gravel, concrete, rock]:
            assert mat.hydraulic_conductivity is not None


class TestAssign:
    def test_assign_default(self):
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=1.0)
        mat = Material(name="uniform", hydraulic_conductivity=1e-5)
        mm = assign(mesh, {"default": mat})
        K = mm.cell_property("hydraulic_conductivity")
        assert K.shape == (mesh.n_cells,)
        np.testing.assert_allclose(K, 1e-5)

    def test_assign_subdomains(self):
        domain = Rectangle(Lx=10, Ly=10)
        sub = Rectangle(x0=3, y0=3, width=4, height=4)
        domain.add_subdomain("block", sub)
        mesh = domain.generate_mesh(resolution=1.0)
        bg = Material(name="bg", hydraulic_conductivity=1e-5)
        block = Material(name="block", hydraulic_conductivity=1e-8)
        mm = assign(mesh, {"default": bg, "block": block})
        K = mm.cell_property("hydraulic_conductivity")
        assert K.min() < K.max()

    def test_invalid_subdomain(self):
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=1.0)
        mat = Material(name="test", hydraulic_conductivity=1e-5)
        with pytest.raises(ValueError, match="not found"):
            assign(mesh, {"default": mat, "nonexistent": mat})


class TestMohrCoulomb:
    def test_shear_strength(self):
        mc = MohrCoulomb(cohesion=10e3, friction_angle=30)
        sigma = np.array([0, 50e3, 100e3])
        tau = mc.shear_strength(sigma)
        # τ = c + σ tan(φ)
        expected = 10e3 + sigma * np.tan(np.radians(30))
        np.testing.assert_allclose(tau, expected)


class TestVanGenuchten:
    def test_saturated(self):
        vg = VanGenuchten(alpha=0.01, n=1.5, theta_r=0.05, theta_s=0.40)
        theta = vg.water_content(np.array([0.0, 1.0]))
        np.testing.assert_allclose(theta, 0.40)

    def test_unsaturated(self):
        vg = VanGenuchten(alpha=0.01, n=1.5, theta_r=0.05, theta_s=0.40)
        theta = vg.water_content(np.array([-1.0]))
        assert 0.05 < theta[0] < 0.40

    def test_effective_saturation_range(self):
        vg = VanGenuchten()
        h = np.linspace(-100, 0, 100)
        Se = vg.effective_saturation(h)
        assert np.all(Se >= 0)
        assert np.all(Se <= 1)

    def test_relative_permeability_range(self):
        vg = VanGenuchten()
        h = np.linspace(-100, 0, 100)
        Kr = vg.relative_permeability(h)
        assert np.all(Kr >= 0)
        assert np.all(Kr <= 1.001)  # numerical tolerance


class TestBrooksCorey:
    def test_saturated(self):
        bc = BrooksCorey(h_b=0.5, lambda_bc=2.0, theta_r=0.05, theta_s=0.40)
        theta = bc.water_content(np.array([0.0, -0.3]))
        assert theta[0] == pytest.approx(0.40)


class TestGaussianField:
    def test_generate(self):
        gf = GaussianField(mean=0.0, std=1.0, correlation_length=5.0)
        coords = np.random.uniform(0, 10, (50, 2))
        field = gf.generate(coords, seed=42)
        assert field.shape == (50,)

    def test_log_transform(self):
        gf = GaussianField(mean=0.0, std=0.5, log_transform=True)
        coords = np.random.uniform(0, 10, (30, 2))
        field = gf.generate(coords, seed=123)
        assert np.all(field > 0)  # log-normal is always positive
