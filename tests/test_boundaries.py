"""Tests for the boundaries module."""

import numpy as np
import pytest

from pygeotech.boundaries.base import (
    BoundaryConditions, Dirichlet, Neumann, Robin, Seepage,
)
from pygeotech.boundaries.locators import (
    BoundaryLocator, top, bottom, left, right,
    x_equals, x_less_than, x_greater_than, x_between,
    y_equals, y_less_than, y_greater_than, y_between,
    on_curve,
)
from pygeotech.boundaries.time_varying import Hydrograph, TimeVaryingBC
from pygeotech.boundaries.internal import Well, Drain


class TestDirichlet:
    def test_scalar_value(self):
        bc = Dirichlet(field="H", value=10.0)
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        values = bc.evaluate(coords)
        np.testing.assert_allclose(values, 10.0)

    def test_with_locator(self):
        bc = Dirichlet(field="H", value=10.0, where=x_less_than(5))
        coords = np.array([[2, 0], [6, 0]])
        mask = bc.apply_mask(coords)
        assert mask[0] and not mask[1]


class TestNeumann:
    def test_zero_flux(self):
        bc = Neumann(field="H", flux=0.0)
        coords = np.array([[0, 0]])
        values = bc.evaluate(coords)
        np.testing.assert_allclose(values, 0.0)


class TestSeepage:
    def test_evaluate_returns_elevation(self):
        bc = Seepage()
        coords = np.array([[0, 5], [10, 15]])
        values = bc.evaluate(coords)
        np.testing.assert_allclose(values, [5, 15])


class TestBoundaryConditions:
    def test_add_and_iterate(self):
        bcs = BoundaryConditions()
        bcs.add(Dirichlet(field="H", value=10.0))
        bcs.add(Neumann(field="H", flux=0.0))
        assert len(bcs) == 2

    def test_of_type(self):
        bcs = BoundaryConditions()
        bcs.add(Dirichlet(field="H", value=10.0))
        bcs.add(Neumann(field="H", flux=0.0))
        assert len(bcs.of_type(Dirichlet)) == 1
        assert len(bcs.of_type(Neumann)) == 1

    def test_for_field(self):
        bcs = BoundaryConditions()
        bcs.add(Dirichlet(field="H", value=10.0))
        bcs.add(Dirichlet(field="C", value=100.0))
        assert len(bcs.for_field("H")) == 1
        assert len(bcs.for_field("C")) == 1


class TestLocators:
    """Test boundary locator functions."""

    def setup_method(self):
        self.coords = np.array([
            [0, 0],    # bottom-left
            [10, 0],   # bottom-right
            [0, 5],    # top-left
            [10, 5],   # top-right
            [5, 0],    # bottom-mid
            [5, 5],    # top-mid
        ], dtype=float)

    def test_top(self):
        mask = top()(self.coords)
        expected = [False, False, True, True, False, True]
        np.testing.assert_array_equal(mask, expected)

    def test_bottom(self):
        mask = bottom()(self.coords)
        expected = [True, True, False, False, True, False]
        np.testing.assert_array_equal(mask, expected)

    def test_left(self):
        mask = left()(self.coords)
        expected = [True, False, True, False, False, False]
        np.testing.assert_array_equal(mask, expected)

    def test_right(self):
        mask = right()(self.coords)
        expected = [False, True, False, True, False, False]
        np.testing.assert_array_equal(mask, expected)

    def test_x_less_than(self):
        mask = x_less_than(3)(self.coords)
        expected = [True, False, True, False, False, False]
        np.testing.assert_array_equal(mask, expected)

    def test_x_greater_than(self):
        mask = x_greater_than(7)(self.coords)
        expected = [False, True, False, True, False, False]
        np.testing.assert_array_equal(mask, expected)

    def test_x_between(self):
        mask = x_between(3, 7)(self.coords)
        expected = [False, False, False, False, True, True]
        np.testing.assert_array_equal(mask, expected)

    def test_and_combinator(self):
        loc = top() & x_less_than(3)
        mask = loc(self.coords)
        expected = [False, False, True, False, False, False]
        np.testing.assert_array_equal(mask, expected)

    def test_or_combinator(self):
        loc = left() | right()
        mask = loc(self.coords)
        expected = [True, True, True, True, False, False]
        np.testing.assert_array_equal(mask, expected)

    def test_not_combinator(self):
        loc = ~top()
        mask = loc(self.coords)
        expected = [True, True, False, False, True, False]
        np.testing.assert_array_equal(mask, expected)

    def test_on_curve(self):
        verts = np.array([[0, 0], [10, 0]])
        loc = on_curve(verts, tol=0.5)
        mask = loc(self.coords)
        # Points on y=0 should be selected
        assert mask[0] and mask[1] and mask[4]


class TestHydrograph:
    def test_interpolation(self):
        h = Hydrograph(times=[0, 100, 200], values=[10, 20, 15])
        assert h(0) == 10.0
        assert h(100) == 20.0
        assert h(50) == pytest.approx(15.0)


class TestWell:
    def test_source_term(self):
        well = Well(location=(5, 5), rate=1.0, radius=2.0)
        coords = np.array([[5, 5], [5, 6], [50, 50]])
        source = well.source_term(coords)
        assert source[2] == 0.0  # far from well
        assert np.isclose(source.sum(), 1.0)  # total rate preserved
