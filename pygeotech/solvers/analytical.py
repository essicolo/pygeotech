"""Analytical solutions for verification.

Provides closed-form solutions for canonical problems that can be
used to verify numerical implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pygeotech.solvers.base import Solver, Solution


class AnalyticalSolver(Solver):
    """Solver that returns analytical solutions for canonical problems.

    Currently supports:
    - 1-D confined flow (linear head distribution)
    - Theis transient well solution
    - Terzaghi consolidation
    """

    def solve(
        self,
        physics: Any,
        boundary_conditions: Any = None,
        time: Any | None = None,
        initial_condition: dict[str, float | np.ndarray] | None = None,
        **kwargs: Any,
    ) -> Solution:
        raise NotImplementedError(
            "Use the specific analytical methods directly, e.g. "
            "AnalyticalSolver.confined_1d(...)."
        )

    @staticmethod
    def confined_1d(
        x: ArrayLike,
        H_left: float,
        H_right: float,
        L: float,
    ) -> np.ndarray:
        """1-D confined steady-state flow: H(x) = H_left + (H_right - H_left) * x / L.

        Args:
            x: Array of x-coordinates.
            H_left: Head at x = 0.
            H_right: Head at x = L.
            L: Domain length.

        Returns:
            Head array.
        """
        x_arr = np.asarray(x, dtype=float)
        return H_left + (H_right - H_left) * x_arr / L

    @staticmethod
    def theis(
        r: ArrayLike,
        t: ArrayLike,
        Q: float,
        T: float,
        S: float,
    ) -> np.ndarray:
        """Theis solution for transient drawdown from a pumping well.

        s(r, t) = Q / (4 π T) * W(u)
        where u = r² S / (4 T t)

        Args:
            r: Radial distance(s) from the well (m).
            t: Time(s) since pumping started (s).
            Q: Pumping rate (m³/s, positive for extraction).
            T: Transmissivity (m²/s).
            S: Storativity (–).

        Returns:
            Drawdown array.
        """
        from scipy.special import exp1

        r_arr = np.asarray(r, dtype=float)
        t_arr = np.asarray(t, dtype=float)
        u = r_arr ** 2 * S / (4.0 * T * t_arr)
        return Q / (4.0 * np.pi * T) * exp1(u)

    @staticmethod
    def terzaghi_consolidation(
        z: ArrayLike,
        t: float,
        H_drain: float,
        cv: float,
        delta_sigma: float,
        n_terms: int = 100,
    ) -> np.ndarray:
        """Terzaghi 1-D consolidation solution.

        Excess pore pressure u(z, t) for a doubly-drained layer of
        thickness 2H.

        Args:
            z: Depth coordinates (0 at top, H_drain at mid-layer).
            t: Time since load application (s).
            H_drain: Drainage path length (half-thickness).
            cv: Coefficient of consolidation (m²/s).
            delta_sigma: Applied stress increment (Pa).
            n_terms: Number of Fourier series terms.

        Returns:
            Excess pore pressure array.
        """
        z_arr = np.asarray(z, dtype=float)
        u = np.zeros_like(z_arr)
        for m in range(n_terms):
            M = np.pi * (2 * m + 1) / 2.0
            Tv = cv * t / H_drain ** 2
            u += (
                2.0 * delta_sigma / M
                * np.sin(M * z_arr / H_drain)
                * np.exp(-(M ** 2) * Tv)
            )
        return u
