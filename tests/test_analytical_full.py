"""Comprehensive tests for the analytical solutions module."""

import numpy as np
import pytest

from pygeotech.solvers.analytical import AnalyticalSolver


class TestConfinedFlow:
    def test_endpoints(self):
        x = np.linspace(0, 10, 50)
        H = AnalyticalSolver.confined_1d(x, H_left=20.0, H_right=10.0, L=10.0)
        assert H[0] == pytest.approx(20.0)
        assert H[-1] == pytest.approx(10.0)

    def test_linearity(self):
        x = np.linspace(0, 10, 50)
        H = AnalyticalSolver.confined_1d(x, H_left=20.0, H_right=10.0, L=10.0)
        np.testing.assert_allclose(H, 20.0 - x)


class TestTheis:
    def test_drawdown_shape(self):
        r = np.array([1.0, 5.0, 10.0])
        t = np.array([100.0, 100.0, 100.0])
        s = AnalyticalSolver.theis(r, t, Q=0.01, T=1e-3, S=1e-4)
        assert s.shape == (3,)

    def test_drawdown_decreases_with_distance(self):
        r = np.array([1.0, 5.0, 10.0, 50.0])
        t = np.full(4, 3600.0)
        s = AnalyticalSolver.theis(r, t, Q=0.01, T=1e-3, S=1e-4)
        # Drawdown should decrease with distance
        for i in range(len(s) - 1):
            assert s[i] > s[i + 1]

    def test_drawdown_positive(self):
        s = AnalyticalSolver.theis(
            r=np.array([5.0]), t=np.array([3600.0]),
            Q=0.01, T=1e-3, S=1e-4,
        )
        assert s[0] > 0


class TestTerzaghi:
    def test_shape(self):
        z = np.linspace(0, 1, 50)
        u = AnalyticalSolver.terzaghi_consolidation(
            z, t=100, H_drain=1.0, cv=1e-3, delta_sigma=100e3,
        )
        assert u.shape == (50,)

    def test_non_negative(self):
        z = np.linspace(0, 1, 50)
        u = AnalyticalSolver.terzaghi_consolidation(
            z, t=100, H_drain=1.0, cv=1e-3, delta_sigma=100e3,
        )
        assert np.all(u >= -1e-6)

    def test_dissipation_over_time(self):
        """Excess pore pressure should dissipate over time."""
        z = np.array([0.5])
        u_early = AnalyticalSolver.terzaghi_consolidation(
            z, t=10, H_drain=1.0, cv=1e-3, delta_sigma=100e3,
        )
        u_late = AnalyticalSolver.terzaghi_consolidation(
            z, t=10000, H_drain=1.0, cv=1e-3, delta_sigma=100e3,
        )
        assert u_early[0] > u_late[0]


class TestOgataBanks:
    def test_inlet_concentration(self):
        """At x=0, C should be close to C0."""
        x = np.array([0.0])
        C = AnalyticalSolver.ogata_banks(x, t=1000.0, v=1e-4, D=1e-6, C0=1.0)
        assert C[0] == pytest.approx(1.0, abs=0.01)

    def test_zero_at_large_distance(self):
        """Far from inlet, C should be near zero."""
        x = np.array([1000.0])
        C = AnalyticalSolver.ogata_banks(x, t=100.0, v=1e-4, D=1e-6, C0=1.0)
        assert C[0] < 0.01

    def test_monotonic_decrease(self):
        """Concentration should decrease with distance."""
        x = np.linspace(0, 10, 100)
        C = AnalyticalSolver.ogata_banks(x, t=1000.0, v=1e-3, D=1e-5, C0=1.0)
        # Should be monotonically decreasing (approximately)
        assert C[0] > C[-1]

    def test_zero_at_t_zero(self):
        x = np.linspace(0, 10, 50)
        C = AnalyticalSolver.ogata_banks(x, t=0.0, v=1e-4, D=1e-6)
        np.testing.assert_allclose(C, 0.0)

    def test_bounded(self):
        x = np.linspace(0, 100, 200)
        C = AnalyticalSolver.ogata_banks(x, t=10000.0, v=1e-3, D=1e-5, C0=1.0)
        assert np.all(C >= -0.01)
        assert np.all(C <= 1.01)


class TestOgataBanksRetardation:
    def test_retardation_slows_front(self):
        """Higher R means the front moves more slowly."""
        x = np.linspace(0, 100, 200)
        t = 50000.0
        v = 1e-3
        D = 1e-5

        C1 = AnalyticalSolver.ogata_banks_retardation(x, t, v, D, R=1.0)
        C2 = AnalyticalSolver.ogata_banks_retardation(x, t, v, D, R=5.0)

        # C1 should have advanced further than C2
        # Find approximate front position (C = 0.5)
        idx1 = np.argmin(np.abs(C1 - 0.5))
        idx2 = np.argmin(np.abs(C2 - 0.5))
        assert x[idx1] > x[idx2]

    def test_decay_reduces_concentration(self):
        x = np.linspace(0, 100, 200)
        C_no_decay = AnalyticalSolver.ogata_banks_retardation(
            x, t=10000.0, v=1e-3, D=1e-5, decay=0.0,
        )
        C_decay = AnalyticalSolver.ogata_banks_retardation(
            x, t=10000.0, v=1e-3, D=1e-5, decay=1e-4,
        )
        assert C_no_decay.sum() > C_decay.sum()


class TestPhilipInfiltration:
    def test_cumulative_at_zero(self):
        I = AnalyticalSolver.philip_infiltration(
            t=np.array([0.0]), S=0.001, Ks=1e-6,
        )
        np.testing.assert_allclose(I, 0.0, atol=1e-15)

    def test_monotonic_increase(self):
        t = np.linspace(0, 3600, 100)
        I = AnalyticalSolver.philip_infiltration(t, S=0.001, Ks=1e-6)
        # Cumulative infiltration should be monotonically increasing
        assert np.all(np.diff(I) >= 0)

    def test_sorptivity_dominates_early(self):
        """At small times, S*sqrt(t) >> Ks*t."""
        t_early = np.array([1.0])
        S = 0.01
        Ks = 1e-7
        I = AnalyticalSolver.philip_infiltration(t_early, S=S, Ks=Ks)
        assert I[0] == pytest.approx(S * 1.0 + Ks * 1.0)

    def test_rate(self):
        t = np.array([100.0])
        S = 0.001
        Ks = 1e-6
        rate = AnalyticalSolver.philip_infiltration_rate(t, S=S, Ks=Ks)
        expected = S / (2.0 * np.sqrt(100.0)) + Ks
        np.testing.assert_allclose(rate, expected, rtol=1e-10)


class TestSrivastavaYeh:
    def test_shape(self):
        z = np.linspace(0, 1, 50)
        h = AnalyticalSolver.srivastava_yeh(
            z, t=3600.0, L=1.0, Ks=1e-5, alpha_vg=0.01,
            theta_s=0.40, theta_r=0.05,
        )
        assert h.shape == (50,)

    def test_no_infiltration_steady(self):
        """With q_top=0, should reach steady state (gravity drainage)."""
        z = np.linspace(0, 1, 50)
        h = AnalyticalSolver.srivastava_yeh(
            z, t=1e8, L=1.0, Ks=1e-5, alpha_vg=0.01,
            theta_s=0.40, theta_r=0.05, q_top=0.0,
        )
        # Pressure head should be finite
        assert np.all(np.isfinite(h))

    def test_infiltration_wetter_than_no_infiltration(self):
        """Infiltration should increase pressure head."""
        z = np.linspace(0.01, 0.99, 50)
        h_dry = AnalyticalSolver.srivastava_yeh(
            z, t=3600.0, L=1.0, Ks=1e-5, alpha_vg=0.5,
            theta_s=0.40, theta_r=0.05, q_top=0.0,
        )
        h_wet = AnalyticalSolver.srivastava_yeh(
            z, t=3600.0, L=1.0, Ks=1e-5, alpha_vg=0.5,
            theta_s=0.40, theta_r=0.05, q_top=0.5e-5,
        )
        # Wet case should have higher (less negative) head on average
        assert h_wet.mean() > h_dry.mean()


class TestSteadyConduction:
    def test_endpoints(self):
        x = np.linspace(0, 10, 50)
        T = AnalyticalSolver.steady_conduction_1d(x, T_left=100.0, T_right=20.0, L=10.0)
        assert T[0] == pytest.approx(100.0)
        assert T[-1] == pytest.approx(20.0)

    def test_linearity(self):
        x = np.linspace(0, 10, 50)
        T = AnalyticalSolver.steady_conduction_1d(x, T_left=100.0, T_right=20.0, L=10.0)
        expected = 100.0 - 8.0 * x
        np.testing.assert_allclose(T, expected)
