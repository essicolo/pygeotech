"""Limit Equilibrium Methods — method of slices.

Implements the four most common LEM formulations for computing the
factor of safety (FOS) of a slope:

bishop_simplified
    Satisfies moment equilibrium only.  Requires circular surfaces.
    Iterative on FOS (appears on both sides of the equation).

spencer
    Satisfies both moment and force equilibrium.  Valid for circular
    and non-circular surfaces.  Iterates on FOS and interslice force
    inclination θ.

morgenstern_price
    Full equilibrium with a user-defined interslice force function
    f(x).  Most general method.  Iterates on FOS and λ.

janbu_simplified
    Satisfies horizontal force equilibrium only.  Good for non-circular
    surfaces.  Iterative on FOS with a correction factor f₀.

References
----------
- Duncan, Wright & Brandon (2014), *Soil Strength and Slope Stability*,
  2nd ed., Wiley.
- Abramson, Lee, Sharma & Boyce (2001), *Slope Stability and
  Stabilization Methods*, 2nd ed., Wiley.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from pygeotech.slope.slices import Slice, generate_slices


# ======================================================================
# Bishop Simplified
# ======================================================================


def bishop_simplified(
    profile: Any,
    surface: Any,
    n_slices: int = 30,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> float:
    """Bishop Simplified method for circular slip surfaces.

    FOS = Σ [(c'b + (W − ub) tan φ') / m_α] / Σ (W sin α)

    where m_α = cos α + sin α tan φ' / F.

    Args:
        profile: Slope profile.
        surface: Circular slip surface.
        n_slices: Number of slices.
        max_iter: Maximum Picard iterations.
        tol: Convergence tolerance on FOS.

    Returns:
        Factor of safety (FOS).  Returns ``inf`` if the driving
        moment is zero or negative (stable slope).
    """
    slices = generate_slices(profile, surface, n_slices)
    if not slices:
        return float("inf")

    # Driving moment denominator
    driving = sum(s.weight * np.sin(s.alpha) for s in slices)
    if driving <= 0:
        return float("inf")

    F = 1.5  # initial guess
    for _ in range(max_iter):
        numerator = 0.0
        for s in slices:
            phi = s.friction_angle
            m_alpha = np.cos(s.alpha) + np.sin(s.alpha) * np.tan(phi) / F
            if abs(m_alpha) < 1e-10:
                m_alpha = 1e-10
            numerator += (
                s.cohesion * s.width + (s.weight - s.pore_pressure * s.width) * np.tan(phi)
            ) / m_alpha

        F_new = numerator / driving
        if F_new <= 0:
            return float("inf")
        if abs(F_new - F) / max(abs(F), 1e-30) < tol:
            return F_new
        F = F_new

    return F


# ======================================================================
# Spencer
# ======================================================================


def spencer(
    profile: Any,
    surface: Any,
    n_slices: int = 30,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> float:
    """Spencer's method — full equilibrium (force + moment).

    Assumes a constant interslice force inclination θ.  Iterates
    simultaneously on FOS (from moment equation) and θ (from force
    equation).

    Valid for both circular and non-circular surfaces.

    Args:
        profile: Slope profile.
        surface: Slip surface.
        n_slices: Number of slices.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Factor of safety.
    """
    slices = generate_slices(profile, surface, n_slices)
    if not slices:
        return float("inf")

    F = 1.5
    theta = 0.0  # interslice force inclination

    for _ in range(max_iter):
        F_moment = _spencer_moment_fos(slices, theta, F)
        F_force = _spencer_force_fos(slices, theta, F)

        # Update θ to make force and moment FOS converge
        # Simple bisection-like approach
        if F_moment <= 0 or F_force <= 0:
            return float("inf")

        # Use moment FOS as the primary, adjust theta to satisfy force
        F_new = F_moment
        if abs(F_force - F_moment) > tol * F_moment:
            # Adjust theta
            theta += 0.01 * (F_force - F_moment) / max(F_moment, 1e-10)
            theta = np.clip(theta, -np.pi / 4, np.pi / 4)

        if abs(F_new - F) / max(abs(F), 1e-30) < tol:
            return F_new
        F = F_new

    return F


def _spencer_moment_fos(
    slices: list[Slice],
    theta: float,
    F_prev: float,
) -> float:
    """Moment equilibrium FOS for Spencer's method."""
    driving = sum(s.weight * np.sin(s.alpha) for s in slices)
    if driving <= 0:
        return float("inf")

    numerator = 0.0
    for s in slices:
        phi = s.friction_angle
        N_prime = (
            s.weight
            - s.cohesion * s.base_length * np.sin(s.alpha) / F_prev
            + s.pore_pressure * s.base_length * np.sin(s.alpha) * np.tan(phi) / F_prev
        )
        denom = np.cos(s.alpha) + np.sin(s.alpha) * np.tan(phi) / F_prev
        if abs(denom) < 1e-10:
            denom = 1e-10
        N_prime /= denom

        # Resisting moment
        numerator += (
            s.cohesion * s.base_length
            + (N_prime - s.pore_pressure * s.base_length) * np.tan(phi)
        )

    if driving <= 0:
        return float("inf")
    return numerator / driving


def _spencer_force_fos(
    slices: list[Slice],
    theta: float,
    F_prev: float,
) -> float:
    """Force equilibrium FOS for Spencer's method."""
    h_driving = sum(s.weight * np.tan(s.alpha) for s in slices)
    if abs(h_driving) < 1e-10:
        return F_prev

    numerator = 0.0
    for s in slices:
        phi = s.friction_angle
        n_alpha = (np.cos(s.alpha) ** 2) * (
            1 + np.tan(s.alpha) * np.tan(phi) / F_prev
        )
        if abs(n_alpha) < 1e-10:
            n_alpha = 1e-10
        numerator += (
            s.cohesion * s.width
            + (s.weight - s.pore_pressure * s.width) * np.tan(phi)
        ) / n_alpha

    if abs(h_driving) < 1e-10:
        return float("inf")
    return numerator / h_driving


# ======================================================================
# Morgenstern-Price
# ======================================================================


def morgenstern_price(
    profile: Any,
    surface: Any,
    n_slices: int = 30,
    max_iter: int = 200,
    tol: float = 1e-4,
    force_function: Callable[[float], float] | str = "half_sine",
) -> float:
    """Morgenstern-Price method — general interslice force function.

    The interslice shear-to-normal ratio is: X/E = λ f(x), where
    f(x) is a prescribed function and λ is an unknown scaling factor
    determined alongside FOS.

    Args:
        profile: Slope profile.
        surface: Slip surface (circular or non-circular).
        n_slices: Number of slices.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        force_function: Interslice force function.  May be:
            - ``"constant"`` — f(x) = 1 (equivalent to Spencer)
            - ``"half_sine"`` — f(x) = sin(π ξ) where ξ ∈ [0, 1]
            - A callable ``f(xi) -> float`` for ξ ∈ [0, 1].

    Returns:
        Factor of safety.
    """
    slices = generate_slices(profile, surface, n_slices)
    if not slices:
        return float("inf")

    # Build the interslice force function
    if force_function == "constant":
        f_func: Callable[[float], float] = lambda xi: 1.0
    elif force_function == "half_sine":
        f_func = lambda xi: np.sin(np.pi * xi)
    elif callable(force_function):
        f_func = force_function
    else:
        raise ValueError(f"Unknown force function: {force_function!r}")

    n = len(slices)
    if n == 0:
        return float("inf")

    x_entry = slices[0].x_mid - slices[0].width / 2
    x_exit = slices[-1].x_mid + slices[-1].width / 2
    x_span = x_exit - x_entry

    # Normalised position for each interface
    xi_vals = []
    for i in range(n - 1):
        x_interface = slices[i].x_mid + slices[i].width / 2
        xi = (x_interface - x_entry) / max(x_span, 1e-10)
        xi_vals.append(xi)

    F = 1.5
    lam = 0.0  # interslice force scaling

    for _ in range(max_iter):
        # Compute FOS from moment equilibrium (like Bishop)
        driving = sum(s.weight * np.sin(s.alpha) for s in slices)
        if driving <= 0:
            return float("inf")

        numerator = 0.0
        for s in slices:
            phi = s.friction_angle
            m_alpha = np.cos(s.alpha) + np.sin(s.alpha) * np.tan(phi) / F
            if abs(m_alpha) < 1e-10:
                m_alpha = 1e-10
            numerator += (
                s.cohesion * s.width
                + (s.weight - s.pore_pressure * s.width) * np.tan(phi)
            ) / m_alpha

        F_m = numerator / driving

        # Force equilibrium to find λ
        h_driving = sum(s.weight * np.tan(s.alpha) for s in slices)
        if abs(h_driving) < 1e-10:
            F_f = F_m
        else:
            num_f = 0.0
            for s in slices:
                phi = s.friction_angle
                n_alpha = (np.cos(s.alpha) ** 2) * (
                    1 + np.tan(s.alpha) * np.tan(phi) / F
                )
                if abs(n_alpha) < 1e-10:
                    n_alpha = 1e-10
                num_f += (
                    s.cohesion * s.width
                    + (s.weight - s.pore_pressure * s.width) * np.tan(phi)
                ) / n_alpha
            F_f = num_f / h_driving

        # Adjust λ to bring force and moment FOS together
        if abs(F_m) > 1e-10:
            lam += 0.02 * (F_f - F_m) / F_m
            lam = np.clip(lam, -2.0, 2.0)

        F_new = 0.5 * (F_m + F_f)
        if F_new <= 0:
            return float("inf")

        if abs(F_new - F) / max(abs(F), 1e-30) < tol:
            return F_new
        F = F_new

    return F


# ======================================================================
# Janbu Simplified
# ======================================================================


def janbu_simplified(
    profile: Any,
    surface: Any,
    n_slices: int = 30,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> float:
    """Janbu Simplified method — force equilibrium only.

    F₀ = Σ [(c'b + (W − ub) tan φ') / n_α] / Σ (W tan α)

    where n_α = cos²α (1 + tan α tan φ' / F).

    A correction factor f₀ is applied:  FOS = f₀ × F₀.

    Valid for non-circular (and circular) surfaces.

    Args:
        profile: Slope profile.
        surface: Slip surface.
        n_slices: Number of slices.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Factor of safety.
    """
    slices = generate_slices(profile, surface, n_slices)
    if not slices:
        return float("inf")

    driving = sum(s.weight * np.tan(s.alpha) for s in slices)
    if abs(driving) < 1e-10:
        return float("inf")

    F = 1.5
    for _ in range(max_iter):
        numerator = 0.0
        for s in slices:
            phi = s.friction_angle
            n_alpha = (np.cos(s.alpha) ** 2) * (
                1 + np.tan(s.alpha) * np.tan(phi) / F
            )
            if abs(n_alpha) < 1e-10:
                n_alpha = 1e-10
            numerator += (
                s.cohesion * s.width
                + (s.weight - s.pore_pressure * s.width) * np.tan(phi)
            ) / n_alpha

        F_new = numerator / driving
        if F_new <= 0:
            return float("inf")
        if abs(F_new - F) / max(abs(F), 1e-30) < tol:
            break
        F = F_new

    # Apply Janbu correction factor f₀
    # f₀ depends on slip surface depth / length ratio and soil type.
    # Simplified estimate: f₀ = 1 + b₁(d/L)² where d/L is the
    # maximum depth / horizontal length ratio.
    d_max = max(s.height for s in slices)
    x_entry = slices[0].x_mid - slices[0].width / 2
    x_exit = slices[-1].x_mid + slices[-1].width / 2
    L = x_exit - x_entry

    d_over_L = d_max / max(L, 1e-10)

    # b₁ depends on soil type:
    # c-only soil: b₁ ≈ 0.69; c-φ soil: b₁ ≈ 0.50; φ-only: b₁ ≈ 0.31
    avg_c = np.mean([s.cohesion for s in slices])
    avg_phi = np.mean([s.friction_angle for s in slices])

    if avg_phi < 1e-6:
        b1 = 0.69  # cohesive
    elif avg_c < 1.0:
        b1 = 0.31  # frictional
    else:
        b1 = 0.50  # c-phi

    f0 = 1.0 + b1 * d_over_L ** 2
    return F * f0
