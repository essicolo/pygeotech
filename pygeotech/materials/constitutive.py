"""Constitutive models for geotechnical materials.

Classes
-------
MohrCoulomb
    Classical Mohr-Coulomb failure criterion.
CamClay
    Modified Cam-Clay critical-state model.
VanGenuchten
    Van Genuchten soil-water retention model.
BrooksCorey
    Brooks-Corey soil-water retention model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


# ======================================================================
# Failure / yield criteria
# ======================================================================


@dataclass
class MohrCoulomb:
    """Mohr-Coulomb failure criterion.

    Args:
        cohesion: Effective cohesion c' (Pa).
        friction_angle: Effective friction angle φ' (degrees).
        dilation_angle: Dilation angle ψ (degrees).  Defaults to 0.
    """

    cohesion: float = 0.0
    friction_angle: float = 30.0
    dilation_angle: float = 0.0

    @property
    def phi_rad(self) -> float:
        """Friction angle in radians."""
        return np.radians(self.friction_angle)

    @property
    def psi_rad(self) -> float:
        """Dilation angle in radians."""
        return np.radians(self.dilation_angle)

    def shear_strength(self, normal_stress: ArrayLike) -> np.ndarray:
        """Compute shear strength τ = c' + σ'_n tan(φ').

        Args:
            normal_stress: Effective normal stress σ'_n (Pa).

        Returns:
            Shear strength array (Pa).
        """
        sigma = np.asarray(normal_stress, dtype=float)
        return self.cohesion + sigma * np.tan(self.phi_rad)


@dataclass
class CamClay:
    """Modified Cam-Clay critical-state model.

    Args:
        M: Slope of critical state line in q-p' space.
        lambda_cc: Compression index (slope of NCL in e-ln(p') space).
        kappa: Swelling/recompression index.
        e_ref: Reference void ratio at p'_ref on NCL.
        p_ref: Reference mean effective stress (Pa).
    """

    M: float = 1.0
    lambda_cc: float = 0.1
    kappa: float = 0.02
    e_ref: float = 1.0
    p_ref: float = 100e3

    def yield_surface(
        self,
        p_prime: ArrayLike,
        q: ArrayLike,
        p0: float,
    ) -> np.ndarray:
        """Evaluate yield function f = q² / M² + p'(p' - p0).

        Args:
            p_prime: Mean effective stress.
            q: Deviatoric stress.
            p0: Pre-consolidation pressure.

        Returns:
            Yield function value (≤ 0 elastic, = 0 on yield surface).
        """
        p = np.asarray(p_prime, dtype=float)
        qq = np.asarray(q, dtype=float)
        return qq ** 2 / self.M ** 2 + p * (p - p0)

    def void_ratio_ncl(self, p_prime: ArrayLike) -> np.ndarray:
        """Void ratio on the normal consolidation line.

        Args:
            p_prime: Mean effective stress.

        Returns:
            Void ratio e.
        """
        p = np.asarray(p_prime, dtype=float)
        return self.e_ref - self.lambda_cc * np.log(p / self.p_ref)


# ======================================================================
# Soil-water retention
# ======================================================================


@dataclass
class VanGenuchten:
    """Van Genuchten (1980) soil-water retention model.

    θ(h) = θ_r + (θ_s − θ_r) / [1 + (α|h|)^n]^m
    K_r(Se) = Se^0.5 [1 − (1 − Se^(1/m))^m]²

    Args:
        alpha: Inverse of air-entry pressure (1/m).
        n: Shape parameter (>1).
        theta_r: Residual water content.
        theta_s: Saturated water content (defaults to porosity).
    """

    alpha: float = 0.01
    n: float = 1.5
    theta_r: float = 0.05
    theta_s: float = 0.40

    @property
    def m(self) -> float:
        """Van Genuchten m parameter: m = 1 - 1/n."""
        return 1.0 - 1.0 / self.n

    def water_content(self, h: ArrayLike) -> np.ndarray:
        """Volumetric water content θ(h).

        Args:
            h: Pressure head (m), negative in unsaturated zone.

        Returns:
            Volumetric water content.
        """
        h_arr = np.asarray(h, dtype=float)
        # Saturated where h >= 0
        Se = self.effective_saturation(h_arr)
        return self.theta_r + (self.theta_s - self.theta_r) * Se

    def effective_saturation(self, h: ArrayLike) -> np.ndarray:
        """Effective saturation Se(h) ∈ [0, 1].

        Args:
            h: Pressure head (m).

        Returns:
            Effective saturation.
        """
        h_arr = np.asarray(h, dtype=float)
        Se = np.where(
            h_arr >= 0.0,
            1.0,
            (1.0 + (self.alpha * np.abs(h_arr)) ** self.n) ** (-self.m),
        )
        return Se

    def relative_permeability(self, h: ArrayLike) -> np.ndarray:
        """Relative hydraulic conductivity Kr(h).

        Mualem-van Genuchten model:
            Kr = Se^0.5 * [1 - (1 - Se^(1/m))^m]²

        Args:
            h: Pressure head (m).

        Returns:
            Relative permeability ∈ [0, 1].
        """
        Se = self.effective_saturation(h)
        inner = 1.0 - (1.0 - Se ** (1.0 / self.m)) ** self.m
        return Se ** 0.5 * inner ** 2

    def specific_moisture_capacity(self, h: ArrayLike) -> np.ndarray:
        """Specific moisture capacity C(h) = dθ/dh.

        Args:
            h: Pressure head (m).

        Returns:
            C(h) array.
        """
        h_arr = np.asarray(h, dtype=float)
        abs_h = np.abs(h_arr)
        alpha_h_n = (self.alpha * abs_h) ** self.n
        denom = 1.0 + alpha_h_n
        C = np.where(
            h_arr >= 0.0,
            0.0,
            (self.theta_s - self.theta_r)
            * self.alpha
            * self.n
            * self.m
            * alpha_h_n
            / (abs_h + 1e-300)
            * denom ** (-(self.m + 1)),
        )
        return C


@dataclass
class BrooksCorey:
    """Brooks-Corey soil-water retention model.

    Se = (h_b / |h|)^λ   for |h| > h_b
    Se = 1                for |h| ≤ h_b

    Args:
        h_b: Bubbling pressure / air-entry value (m, positive).
        lambda_bc: Pore-size distribution index.
        theta_r: Residual water content.
        theta_s: Saturated water content.
    """

    h_b: float = 0.5
    lambda_bc: float = 2.0
    theta_r: float = 0.05
    theta_s: float = 0.40

    def effective_saturation(self, h: ArrayLike) -> np.ndarray:
        """Effective saturation Se(h)."""
        h_arr = np.asarray(h, dtype=float)
        abs_h = np.abs(h_arr)
        # Avoid computing the power for saturated elements (abs_h <= h_b)
        unsaturated = abs_h > self.h_b
        Se = np.ones_like(h_arr)
        if unsaturated.any():
            Se[unsaturated] = (self.h_b / abs_h[unsaturated]) ** self.lambda_bc
        return Se

    def water_content(self, h: ArrayLike) -> np.ndarray:
        """Volumetric water content θ(h)."""
        Se = self.effective_saturation(h)
        return self.theta_r + (self.theta_s - self.theta_r) * Se

    def relative_permeability(self, h: ArrayLike) -> np.ndarray:
        """Relative permeability Kr = Se^(3 + 2/λ)."""
        Se = self.effective_saturation(h)
        return Se ** (3.0 + 2.0 / self.lambda_bc)
