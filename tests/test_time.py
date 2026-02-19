"""Tests for the time module."""

import numpy as np
import pytest

from pygeotech.time.stepper import Stepper, AdaptiveStepper
from pygeotech.time.schemes import Implicit, Explicit, CrankNicolson


class TestStepper:
    def test_n_steps(self):
        s = Stepper(t_end=100, dt=10)
        assert s.n_steps == 10

    def test_iteration(self):
        s = Stepper(t_end=100, dt=25)
        steps = list(s)
        assert len(steps) == 4
        assert steps[0] == (25.0, 25.0)
        assert steps[-1] == (100.0, 25.0)

    def test_times(self):
        s = Stepper(t_end=10, dt=2.5)
        times = s.times
        np.testing.assert_allclose(times, [0, 2.5, 5.0, 7.5, 10.0])


class TestAdaptiveStepper:
    def test_suggest_dt_increase(self):
        a = AdaptiveStepper(t_end=100, dt_init=1.0, tol=1e-4)
        dt_new = a.suggest_dt(error=1e-6, dt_current=1.0)
        assert dt_new > 1.0

    def test_suggest_dt_decrease(self):
        a = AdaptiveStepper(t_end=100, dt_init=1.0, tol=1e-4)
        dt_new = a.suggest_dt(error=1e-2, dt_current=1.0)
        assert dt_new < 1.0


class TestTimeSchemes:
    def test_implicit_theta(self):
        assert Implicit().theta == 1.0

    def test_explicit_theta(self):
        assert Explicit().theta == 0.0

    def test_crank_nicolson_theta(self):
        assert CrankNicolson().theta == 0.5

    def test_blend(self):
        cn = CrankNicolson()
        f_new = np.array([2.0, 4.0])
        f_old = np.array([0.0, 0.0])
        blended = cn.blend(f_new, f_old)
        np.testing.assert_allclose(blended, [1.0, 2.0])
