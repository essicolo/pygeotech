"""Analytical solutions for verification.

Provides closed-form solutions for canonical problems that can be
used to verify numerical implementations.

Solutions
---------
confined_1d
    1-D steady-state confined flow (linear head).
theis
    Theis transient well drawdown.
terzaghi_consolidation
    Terzaghi 1-D consolidation.
ogata_banks
    1-D advection-dispersion (Ogata-Banks).
philip_infiltration
    Philip two-term infiltration.
srivastava_yeh_richards
    Srivastava-Yeh 1-D transient Richards (exponential K(h)).
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
    - Ogata-Banks 1-D advection-dispersion
    - Philip two-term infiltration
    - Srivastava-Yeh 1-D Richards equation
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

    # ------------------------------------------------------------------
    # Transport: Ogata-Banks 1-D ADE
    # ------------------------------------------------------------------

    @staticmethod
    def ogata_banks(
        x: ArrayLike,
        t: float,
        v: float,
        D: float,
        C0: float = 1.0,
    ) -> np.ndarray:
        """Ogata-Banks (1961) analytical solution for 1-D ADE.

        Solves the 1-D advection-dispersion equation::

            ∂C/∂t = D ∂²C/∂x² - v ∂C/∂x

        with C(x, 0) = 0 and C(0, t) = C0 for t > 0.

        C(x, t) = C0/2 [erfc((x - v t)/(2 √(D t)))
                        + exp(v x / D) erfc((x + v t)/(2 √(D t)))]

        Args:
            x: Distance from the inlet (m), must be >= 0.
            t: Time (s), must be > 0.
            v: Pore-water velocity (m/s).
            D: Hydrodynamic dispersion coefficient (m²/s).
            C0: Inlet concentration.

        Returns:
            Concentration array C(x, t).
        """
        from scipy.special import erfc

        x_arr = np.asarray(x, dtype=float)
        if t <= 0:
            return np.zeros_like(x_arr)

        sqrt_Dt = np.sqrt(D * t)
        term1 = erfc((x_arr - v * t) / (2.0 * sqrt_Dt))

        # The second term can overflow for large v*x/D; use log-space
        vx_D = v * x_arr / D
        # Clip to avoid overflow in exp
        vx_D_clipped = np.minimum(vx_D, 500.0)
        term2 = np.exp(vx_D_clipped) * erfc((x_arr + v * t) / (2.0 * sqrt_Dt))

        return C0 / 2.0 * (term1 + term2)

    # ------------------------------------------------------------------
    # Transport: Ogata-Banks with retardation and decay
    # ------------------------------------------------------------------

    @staticmethod
    def ogata_banks_retardation(
        x: ArrayLike,
        t: float,
        v: float,
        D: float,
        R: float = 1.0,
        decay: float = 0.0,
        C0: float = 1.0,
    ) -> np.ndarray:
        """Ogata-Banks with linear retardation and first-order decay.

        Transforms variables: v_eff = v/R, D_eff = D/R, and applies
        exponential decay.

        Args:
            x: Distance from inlet (m).
            t: Time (s).
            v: Pore-water velocity (m/s).
            D: Dispersion coefficient (m²/s).
            R: Retardation factor.
            decay: First-order decay rate (1/s).
            C0: Inlet concentration.

        Returns:
            Concentration C(x, t).
        """
        v_eff = v / R
        D_eff = D / R
        C = AnalyticalSolver.ogata_banks(x, t, v_eff, D_eff, C0)
        if decay > 0:
            C *= np.exp(-decay * t)
        return C

    # ------------------------------------------------------------------
    # Richards: Philip two-term infiltration
    # ------------------------------------------------------------------

    @staticmethod
    def philip_infiltration(
        t: ArrayLike,
        S: float,
        Ks: float,
    ) -> np.ndarray:
        """Philip (1957) two-term infiltration equation.

        Cumulative infiltration::

            I(t) = S √t + Ks t

        Infiltration rate::

            i(t) = S / (2√t) + Ks

        Args:
            t: Time array (s), must be >= 0.
            S: Sorptivity (m/s^0.5).
            Ks: Saturated hydraulic conductivity (m/s).

        Returns:
            Cumulative infiltration I(t) in metres.
        """
        t_arr = np.asarray(t, dtype=float)
        return S * np.sqrt(t_arr) + Ks * t_arr

    @staticmethod
    def philip_infiltration_rate(
        t: ArrayLike,
        S: float,
        Ks: float,
    ) -> np.ndarray:
        """Philip infiltration rate i(t) = S/(2√t) + Ks.

        Args:
            t: Time array (s), must be > 0.
            S: Sorptivity (m/s^0.5).
            Ks: Saturated hydraulic conductivity (m/s).

        Returns:
            Infiltration rate (m/s).
        """
        t_arr = np.asarray(t, dtype=float)
        return S / (2.0 * np.sqrt(t_arr + 1e-300)) + Ks

    # ------------------------------------------------------------------
    # Richards: Srivastava-Yeh 1-D transient
    # ------------------------------------------------------------------

    @staticmethod
    def srivastava_yeh(
        z: ArrayLike,
        t: float,
        L: float,
        Ks: float,
        alpha_vg: float,
        theta_s: float,
        theta_r: float,
        q_top: float = 0.0,
        q_bot: float = 0.0,
        n_terms: int = 50,
    ) -> np.ndarray:
        """Srivastava & Yeh (1991) 1-D transient Richards solution.

        Assumes an exponential conductivity model:
            K(h) = Ks exp(α h)
            θ(h) = θ_r + (θ_s - θ_r) exp(α h)

        for a vertical soil column of depth L with specified flux at
        top (q_top, positive downward) and bottom (q_bot).

        Initial condition: steady-state profile under gravity drainage.

        Args:
            z: Depth from surface (positive downward, 0..L).
            t: Time (s).
            L: Column depth (m).
            Ks: Saturated hydraulic conductivity (m/s).
            alpha_vg: Gardner/exponential model parameter (1/m).
            theta_s: Saturated water content.
            theta_r: Residual water content.
            q_top: Surface flux (m/s, positive = infiltration).
            q_bot: Bottom flux (m/s).
            n_terms: Fourier series terms.

        Returns:
            Pressure head h(z, t) array.

        Note:
            This is a simplified form valid for the Gardner exponential
            model K = Ks exp(α h).  For Van Genuchten parameters, α_vg
            is used as an approximation of the Gardner α.
        """
        z_arr = np.asarray(z, dtype=float)

        alpha = alpha_vg
        delta_theta = theta_s - theta_r

        # Normalised depth (z/L) and time factor
        Z = z_arr / L
        T_factor = Ks * alpha / delta_theta

        # Steady-state profile (gravity drainage): h_ss = -z (hydrostatic)
        # Under gravity-only drainage: K(h) dh/dz = -Ks (unit gradient)
        # h_ss(z) = 0 (fully saturated) for the initial condition

        # For the simplified Gardner model, the transient solution for
        # a step change in surface flux is:
        # K(h) = Ks exp(alpha * h)
        # Introducing Kirchhoff variable: Φ = K(h)/alpha = (Ks/alpha) exp(alpha h)
        # The Richards equation linearises to:
        #   dΦ/dt = (Ks/delta_theta) [d²Φ/dz² - alpha dΦ/dz]

        # Initial steady state with gravity: Φ_i(z) = (Ks/alpha) * exp(-alpha * z)
        # For flux q_top at surface: Φ_f(z) = (q_top/alpha) * exp(-alpha * z) + ...

        # Simplified: compute the Kirchhoff potential
        Phi_init = (Ks / alpha) * np.exp(-alpha * z_arr)

        # Final steady-state under constant infiltration q_top
        if abs(q_top) < 1e-30:
            Phi_final = Phi_init.copy()
        else:
            Phi_final = (q_top / alpha) * np.exp(-alpha * (z_arr - L)) + \
                        (Ks - q_top) / alpha * np.exp(-alpha * z_arr)

        # Transient part: exponential decay of Fourier modes
        # Diffusivity for linearised equation
        D_eff = Ks / delta_theta

        Phi = Phi_final.copy()
        for n in range(1, n_terms + 1):
            mu_n = n * np.pi / L
            eigenvalue = D_eff * (mu_n ** 2 + (alpha / 2) ** 2)
            decay = np.exp(-eigenvalue * t)

            # Fourier coefficient (simplified sine series)
            An = 2.0 / L * np.trapezoid(
                (Phi_init - Phi_final) * np.exp(alpha * z_arr / 2) * np.sin(mu_n * z_arr),
                z_arr,
            ) if len(z_arr) > 1 else 0.0

            Phi += An * np.exp(-alpha * z_arr / 2) * np.sin(mu_n * z_arr) * decay

        # Convert back to pressure head: h = ln(Phi * alpha / Ks) / alpha
        Phi_safe = np.maximum(Phi, 1e-300)
        h = np.log(Phi_safe * alpha / Ks) / alpha

        return h

    # ------------------------------------------------------------------
    # Heat: 1-D steady-state conduction
    # ------------------------------------------------------------------

    @staticmethod
    def steady_conduction_1d(
        x: ArrayLike,
        T_left: float,
        T_right: float,
        L: float,
    ) -> np.ndarray:
        """1-D steady-state heat conduction T(x) = T_left + (T_right - T_left) x/L.

        Args:
            x: Position coordinates (m).
            T_left: Temperature at x = 0.
            T_right: Temperature at x = L.
            L: Domain length (m).

        Returns:
            Temperature array.
        """
        x_arr = np.asarray(x, dtype=float)
        return T_left + (T_right - T_left) * x_arr / L
