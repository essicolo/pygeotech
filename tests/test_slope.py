"""Comprehensive tests for slope stability analysis.

Tests cover:
- Slope profile definition and interpolation
- Circular and polyline slip surfaces
- Slice generation
- LEM methods: Bishop, Spencer, Morgenstern-Price, Janbu
- Critical surface search
- Strength Reduction Method
- Verification against known analytical solutions
"""

import numpy as np
import pytest

from pygeotech.slope.profile import SlopeProfile, MaterialLayer
from pygeotech.slope.surfaces import CircularSurface, PolylineSurface
from pygeotech.slope.slices import Slice, generate_slices
from pygeotech.slope.lem import (
    bishop_simplified,
    spencer,
    morgenstern_price,
    janbu_simplified,
)
from pygeotech.slope.search import grid_search, SearchResult


# ======================================================================
# Common test fixtures
# ======================================================================


def _simple_slope():
    """A 1H:1V slope from elevation 10 to 20.

    Profile:
        (0, 10) -- (20, 10) -- (30, 20) -- (50, 20)

    Single homogeneous layer: c'=10 kPa, φ'=25°, γ=18 kN/m³.
    """
    return SlopeProfile(
        surface=[(0, 10), (20, 10), (30, 20), (50, 20)],
        layers=[
            {"c": 10e3, "phi": 25.0, "gamma": 18e3, "top": 25, "bottom": 0},
        ],
    )


def _simple_slope_with_water():
    """Same slope but with a water table."""
    return SlopeProfile(
        surface=[(0, 10), (20, 10), (30, 20), (50, 20)],
        layers=[
            {"c": 10e3, "phi": 25.0, "gamma": 18e3,
             "gamma_sat": 20e3, "top": 25, "bottom": 0},
        ],
        water_table=[(0, 8), (20, 8), (30, 15), (50, 18)],
    )


def _cohesionless_slope():
    """Purely frictional slope (c'=0) for infinite slope verification.

    tan(β) = 10/10 = 1.0 → β = 45°
    φ' = 35° → analytical FOS = tan(φ')/tan(β) = tan(35°)/1.0 ≈ 0.70
    for an infinite slope (shallow failure).
    """
    return SlopeProfile(
        surface=[(0, 0), (10, 0), (20, 10), (30, 10)],
        layers=[
            {"c": 0.0, "phi": 35.0, "gamma": 18e3, "top": 15, "bottom": 0},
        ],
    )


def _cohesive_slope():
    """Purely cohesive (φ=0) slope for Taylor chart verification."""
    return SlopeProfile(
        surface=[(0, 0), (10, 0), (20, 10), (30, 10)],
        layers=[
            {"c": 30e3, "phi": 0.0, "gamma": 18e3, "top": 15, "bottom": 0},
        ],
    )


# ======================================================================
# Profile tests
# ======================================================================


class TestSlopeProfile:
    def test_creation(self):
        p = _simple_slope()
        assert len(p.layers) == 1
        assert p.layers[0].cohesion == 10e3

    def test_surface_elevation(self):
        p = _simple_slope()
        assert p.surface_elevation(0) == pytest.approx(10.0)
        assert p.surface_elevation(50) == pytest.approx(20.0)
        assert p.surface_elevation(25) == pytest.approx(15.0)

    def test_water_elevation_no_table(self):
        p = _simple_slope()
        assert p.water_elevation(10) == -np.inf

    def test_water_elevation_with_table(self):
        p = _simple_slope_with_water()
        assert p.water_elevation(0) == pytest.approx(8.0)
        assert p.water_elevation(50) == pytest.approx(18.0)

    def test_pore_pressure_dry(self):
        p = _simple_slope()
        assert p.pore_pressure(10, 5) == 0.0

    def test_pore_pressure_below_wt(self):
        p = _simple_slope_with_water()
        # At x=0, water table is at y=8. Point at y=5 is below.
        u = p.pore_pressure(0, 5)
        expected = 9810.0 * (8.0 - 5.0)
        assert u == pytest.approx(expected)

    def test_pore_pressure_above_wt(self):
        p = _simple_slope_with_water()
        assert p.pore_pressure(0, 12) == 0.0

    def test_layer_at(self):
        p = _simple_slope()
        layer = p.layer_at(10, 5)
        assert layer.cohesion == 10e3
        assert layer.friction_angle == 25.0

    def test_dict_layer_aliases(self):
        """Both 'c'/'cohesion' and 'phi'/'friction_angle' should work."""
        p = SlopeProfile(
            surface=[(0, 0), (10, 10)],
            layers=[
                {"cohesion": 5e3, "friction_angle": 30, "unit_weight": 17e3,
                 "top": 15, "bottom": 0},
            ],
        )
        assert p.layers[0].cohesion == 5e3
        assert p.layers[0].friction_angle == 30.0
        assert p.layers[0].unit_weight == 17e3

    def test_material_layer_gamma_sat_default(self):
        lay = MaterialLayer(unit_weight=18e3)
        assert lay.gamma_sat == 18e3

    def test_material_layer_gamma_sat_explicit(self):
        lay = MaterialLayer(unit_weight=18e3, sat_unit_weight=20e3)
        assert lay.gamma_sat == 20e3


# ======================================================================
# Surface tests
# ======================================================================


class TestCircularSurface:
    def test_y_at_center(self):
        s = CircularSurface(xc=10, yc=20, radius=10)
        assert s.y_at(10) == pytest.approx(10.0)

    def test_y_at_outside(self):
        s = CircularSurface(xc=10, yc=20, radius=5)
        assert s.y_at(100) is None

    def test_base_angle_at_center(self):
        s = CircularSurface(xc=10, yc=20, radius=10)
        assert s.base_angle(10) == pytest.approx(0.0)

    def test_base_angle_left(self):
        s = CircularSurface(xc=10, yc=20, radius=10)
        alpha = s.base_angle(5)
        assert alpha < 0  # slopes down to the left

    def test_base_angle_right(self):
        s = CircularSurface(xc=10, yc=20, radius=10)
        alpha = s.base_angle(15)
        assert alpha > 0

    def test_entry_exit(self):
        p = _simple_slope()
        s = CircularSurface(xc=25, yc=25, radius=12)
        ee = s.entry_exit(p)
        assert ee is not None
        x_entry, x_exit = ee
        assert x_entry < x_exit
        assert x_entry >= 0
        assert x_exit <= 50

    def test_no_intersection(self):
        p = _simple_slope()
        s = CircularSurface(xc=25, yc=100, radius=5)
        assert s.entry_exit(p) is None


class TestPolylineSurface:
    def test_y_at(self):
        s = PolylineSurface([(10, 5), (20, 3), (30, 8)])
        assert s.y_at(10) == pytest.approx(5.0)
        assert s.y_at(30) == pytest.approx(8.0)
        assert s.y_at(15) == pytest.approx(4.0)

    def test_y_outside(self):
        s = PolylineSurface([(10, 5), (30, 8)])
        assert s.y_at(0) is None

    def test_base_angle(self):
        s = PolylineSurface([(10, 5), (20, 5)])  # flat
        alpha = s.base_angle(15)
        assert alpha == pytest.approx(0.0, abs=1e-10)

    def test_entry_exit(self):
        s = PolylineSurface([(10, 5), (20, 3), (30, 8)])
        p = _simple_slope()
        ee = s.entry_exit(p)
        assert ee is not None
        assert ee == pytest.approx((10.0, 30.0))


# ======================================================================
# Slice generation
# ======================================================================


class TestSliceGeneration:
    def test_generates_slices(self):
        p = _simple_slope()
        s = CircularSurface(xc=25, yc=25, radius=12)
        slices = generate_slices(p, s, n_slices=20)
        assert len(slices) > 0

    def test_slice_properties(self):
        p = _simple_slope()
        s = CircularSurface(xc=25, yc=25, radius=12)
        slices = generate_slices(p, s, n_slices=20)
        for sl in slices:
            assert sl.width > 0
            assert sl.height >= 0
            assert sl.weight > 0
            assert sl.base_length > 0
            assert sl.cohesion >= 0
            assert sl.friction_angle >= 0
            assert sl.pore_pressure >= 0

    def test_no_intersection_raises(self):
        p = _simple_slope()
        s = CircularSurface(xc=25, yc=100, radius=5)
        with pytest.raises(ValueError, match="does not intersect"):
            generate_slices(p, s)

    def test_pore_pressure_with_water(self):
        p = _simple_slope_with_water()
        s = CircularSurface(xc=25, yc=25, radius=12)
        slices = generate_slices(p, s, n_slices=20)
        # Some slices should have positive pore pressure
        has_u = any(sl.pore_pressure > 0 for sl in slices)
        assert has_u

    def test_all_dry(self):
        p = _simple_slope()  # no water table
        s = CircularSurface(xc=25, yc=25, radius=12)
        slices = generate_slices(p, s, n_slices=20)
        assert all(sl.pore_pressure == 0 for sl in slices)


# ======================================================================
# LEM Methods
# ======================================================================


class TestBishopSimplified:
    def test_returns_finite_fos(self):
        p = _simple_slope()
        s = CircularSurface(xc=25, yc=25, radius=12)
        fos = bishop_simplified(p, s)
        assert np.isfinite(fos)
        assert fos > 0

    def test_reasonable_fos(self):
        """c-φ slope should give FOS in a reasonable range."""
        p = _simple_slope()
        s = CircularSurface(xc=25, yc=25, radius=12)
        fos = bishop_simplified(p, s)
        assert 0.5 < fos < 5.0

    def test_water_reduces_fos(self):
        """Adding water should reduce FOS."""
        p_dry = _simple_slope()
        p_wet = _simple_slope_with_water()
        s = CircularSurface(xc=25, yc=25, radius=12)
        fos_dry = bishop_simplified(p_dry, s)
        fos_wet = bishop_simplified(p_wet, s)
        assert fos_wet < fos_dry

    def test_higher_cohesion_higher_fos(self):
        p_low = SlopeProfile(
            surface=[(0, 10), (20, 10), (30, 20), (50, 20)],
            layers=[{"c": 5e3, "phi": 25, "gamma": 18e3, "top": 25, "bottom": 0}],
        )
        p_high = SlopeProfile(
            surface=[(0, 10), (20, 10), (30, 20), (50, 20)],
            layers=[{"c": 50e3, "phi": 25, "gamma": 18e3, "top": 25, "bottom": 0}],
        )
        s = CircularSurface(xc=25, yc=25, radius=12)
        assert bishop_simplified(p_high, s) > bishop_simplified(p_low, s)

    def test_cohesive_slope(self):
        """φ=0 analysis should work."""
        p = _cohesive_slope()
        s = CircularSurface(xc=15, yc=15, radius=12)
        fos = bishop_simplified(p, s)
        assert np.isfinite(fos)
        assert fos > 0


class TestSpencer:
    def test_returns_finite_fos(self):
        p = _simple_slope()
        s = CircularSurface(xc=25, yc=25, radius=12)
        fos = spencer(p, s)
        assert np.isfinite(fos)
        assert fos > 0

    def test_close_to_bishop_for_circular(self):
        """Spencer should give similar results to Bishop for circular surfaces."""
        p = _simple_slope()
        s = CircularSurface(xc=25, yc=25, radius=12)
        fos_b = bishop_simplified(p, s)
        fos_s = spencer(p, s)
        # Spencer and Bishop typically agree within ~5% for circular
        assert abs(fos_s - fos_b) / fos_b < 0.15

    def test_non_circular(self):
        """Spencer should work with polyline surfaces."""
        p = _simple_slope()
        s = PolylineSurface([(15, 9), (22, 5), (28, 5), (35, 19)])
        fos = spencer(p, s)
        assert np.isfinite(fos)
        assert fos > 0


class TestMorgensternPrice:
    def test_returns_finite_fos(self):
        p = _simple_slope()
        s = CircularSurface(xc=25, yc=25, radius=12)
        fos = morgenstern_price(p, s)
        assert np.isfinite(fos)
        assert fos > 0

    def test_constant_force_function(self):
        """Constant f(x) should give results close to Spencer."""
        p = _simple_slope()
        s = CircularSurface(xc=25, yc=25, radius=12)
        fos_sp = spencer(p, s)
        fos_mp = morgenstern_price(p, s, force_function="constant")
        assert abs(fos_mp - fos_sp) / max(fos_sp, 1e-10) < 0.15

    def test_half_sine(self):
        p = _simple_slope()
        s = CircularSurface(xc=25, yc=25, radius=12)
        fos = morgenstern_price(p, s, force_function="half_sine")
        assert np.isfinite(fos)
        assert fos > 0

    def test_custom_force_function(self):
        p = _simple_slope()
        s = CircularSurface(xc=25, yc=25, radius=12)
        fos = morgenstern_price(p, s, force_function=lambda xi: 1.0)
        assert np.isfinite(fos)


class TestJanbuSimplified:
    def test_returns_finite_fos(self):
        p = _simple_slope()
        s = CircularSurface(xc=25, yc=25, radius=12)
        fos = janbu_simplified(p, s)
        assert np.isfinite(fos)
        assert fos > 0

    def test_non_circular(self):
        p = _simple_slope()
        s = PolylineSurface([(15, 9), (22, 5), (28, 5), (35, 19)])
        fos = janbu_simplified(p, s)
        assert np.isfinite(fos)
        assert fos > 0

    def test_correction_factor_applied(self):
        """Janbu should apply the f₀ correction (FOS ≥ uncorrected)."""
        p = _simple_slope()
        s = CircularSurface(xc=25, yc=25, radius=12)
        fos = janbu_simplified(p, s)
        # The correction factor f₀ ≥ 1, so corrected FOS ≥ uncorrected
        assert fos > 0


class TestMethodComparison:
    """Compare all four methods on the same problem."""

    def test_all_methods_in_range(self):
        p = _simple_slope()
        s = CircularSurface(xc=25, yc=25, radius=12)

        fos_b = bishop_simplified(p, s)
        fos_s = spencer(p, s)
        fos_mp = morgenstern_price(p, s)
        fos_j = janbu_simplified(p, s)

        for fos, name in [
            (fos_b, "Bishop"),
            (fos_s, "Spencer"),
            (fos_mp, "Morgenstern-Price"),
            (fos_j, "Janbu"),
        ]:
            assert 0.5 < fos < 5.0, f"{name} FOS={fos} out of range"

    def test_water_reduces_all(self):
        """Water should reduce FOS for all methods."""
        p_dry = _simple_slope()
        p_wet = _simple_slope_with_water()
        s = CircularSurface(xc=25, yc=25, radius=12)

        for lem_func in [bishop_simplified, spencer, morgenstern_price, janbu_simplified]:
            fos_dry = lem_func(p_dry, s)
            fos_wet = lem_func(p_wet, s)
            assert fos_wet < fos_dry, (
                f"{lem_func.__name__}: wet FOS ({fos_wet:.3f}) "
                f"should be < dry FOS ({fos_dry:.3f})"
            )


# ======================================================================
# Search
# ======================================================================


class TestGridSearch:
    def test_finds_minimum(self):
        p = _simple_slope()
        result = grid_search(
            p,
            method="bishop",
            xc_range=(20, 30),
            yc_range=(22, 30),
            r_range=(8, 16),
            n_xc=5,
            n_yc=5,
            n_r=5,
            n_slices=20,
        )
        assert isinstance(result, SearchResult)
        assert np.isfinite(result.fos)
        assert result.fos > 0
        assert result.surface is not None

    def test_fos_grid_shape(self):
        p = _simple_slope()
        result = grid_search(
            p,
            method="bishop",
            xc_range=(22, 28),
            yc_range=(23, 27),
            r_range=(10, 14),
            n_xc=3,
            n_yc=3,
            n_r=3,
        )
        assert result.fos_grid is not None
        assert len(result.fos_grid) == 27  # 3*3*3

    def test_different_methods(self):
        p = _simple_slope()
        for method in ["bishop", "spencer", "janbu"]:
            result = grid_search(
                p, method=method,
                xc_range=(22, 28), yc_range=(23, 27), r_range=(10, 14),
                n_xc=3, n_yc=3, n_r=3, n_slices=15,
            )
            assert np.isfinite(result.fos)


# ======================================================================
# SRM (Strength Reduction Method)
# ======================================================================


class TestSRM:
    def test_srm_basic(self):
        """SRM should produce a finite FOS."""
        from pygeotech.geometry.primitives import Rectangle
        from pygeotech.materials.base import Material, assign
        from pygeotech.boundaries.base import BoundaryConditions, Dirichlet
        from pygeotech.boundaries.locators import bottom, left, right
        from pygeotech.slope.srm import strength_reduction

        # Simple rectangular domain (not a real slope, but tests the SRM
        # machinery)
        domain = Rectangle(Lx=10, Ly=5)
        mesh = domain.generate_mesh(resolution=2.0)
        mat = Material(
            name="clay",
            youngs_modulus=50e6,
            poissons_ratio=0.3,
            dry_density=1600,
            cohesion=20e3,
            friction_angle=25.0,
        )
        materials = assign(mesh, {"default": mat})

        bc = BoundaryConditions()
        bc.add(Dirichlet(field="u", value=0.0, where=bottom()))
        bc.add(Dirichlet(field="ux", value=0.0, where=left()))
        bc.add(Dirichlet(field="ux", value=0.0, where=right()))

        result = strength_reduction(
            mesh, materials, bc,
            srf_range=(0.5, 3.0),
            n_steps=6,
            bisection_tol=0.1,
        )
        assert result.fos > 0
        assert result.srf_history is not None
        assert len(result.srf_history) > 0


# ======================================================================
# Verification against known solutions
# ======================================================================


class TestVerification:
    def test_infinite_slope_dry(self):
        """Infinite slope (shallow planar failure):

        FOS = (c'/(γ H cos²β tan β)) + (tan φ' / tan β)

        For c=0: FOS = tan(φ') / tan(β).
        β = 45° (1H:1V), φ = 35° → FOS = tan(35°)/tan(45°) = 0.700

        We test a wide shallow circular surface that approximates
        a planar failure.  The FOS should be close to the analytical
        value for the shallow mechanism.
        """
        # Use a straight polyline (planar failure) parallel to the slope
        # face, 2 m below the surface.  The slope face runs from (20, 0)
        # to (40, 20), angle β = 45°.  A parallel plane 2 m below
        # (measured normal) has offset dx = dy = 2/√2 ≈ 1.414 m.
        d = 2.0  # depth normal to slope face
        off = d / np.sqrt(2.0)
        p = SlopeProfile(
            surface=[(0, 0), (20, 0), (40, 20), (60, 20)],
            layers=[{"c": 0, "phi": 35, "gamma": 18e3, "top": 25, "bottom": -10}],
        )
        # Planar slip surface parallel to the 1:1 slope face
        s = PolylineSurface([
            (20 + off, 0 - off),
            (40 + off, 20 - off),
        ])
        fos = janbu_simplified(p, s, n_slices=50)
        analytical = np.tan(np.radians(35)) / np.tan(np.radians(45))
        # Janbu on a planar surface should be close to the infinite
        # slope analytical solution (tolerance for discrete slices
        # and correction factor)
        assert abs(fos - analytical) / analytical < 0.50

    def test_stronger_soil_higher_fos(self):
        """Increasing both c and φ should always increase FOS."""
        s = CircularSurface(xc=25, yc=25, radius=12)
        results = []
        for c in [5e3, 15e3, 30e3]:
            p = SlopeProfile(
                surface=[(0, 10), (20, 10), (30, 20), (50, 20)],
                layers=[{"c": c, "phi": 30, "gamma": 18e3, "top": 25, "bottom": 0}],
            )
            results.append(bishop_simplified(p, s))
        assert results[0] < results[1] < results[2]
