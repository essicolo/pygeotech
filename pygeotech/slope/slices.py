"""Slice generation for the method of slices.

Vertical slices are created between the entry and exit points of a slip
surface with the ground profile.  Each slice carries the geometric and
material data needed by the LEM solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class Slice:
    """A single vertical slice for limit-equilibrium analysis.

    All angles in radians; stresses/forces in consistent SI units.

    Attributes:
        x_mid: Horizontal midpoint of the slice.
        width: Slice width b (m).
        height: Average height of the slice above the base (m).
        alpha: Inclination of the slice base from horizontal (rad).
            Positive = base slopes upward to the right.
        base_length: Length of the slice base l = b / cos(α).
        weight: Total weight W = γ h b (N/m per unit depth).
        cohesion: Effective cohesion c' at the base (Pa).
        friction_angle: Effective friction angle φ' at the base (rad).
        pore_pressure: Pore-water pressure u at the base midpoint (Pa).
        y_base: Elevation of the slip surface at the midpoint.
        y_top: Ground surface elevation at the midpoint.
    """

    x_mid: float
    width: float
    height: float
    alpha: float
    base_length: float
    weight: float
    cohesion: float
    friction_angle: float
    pore_pressure: float
    y_base: float
    y_top: float


def generate_slices(
    profile: Any,
    surface: Any,
    n_slices: int = 30,
) -> list[Slice]:
    """Generate vertical slices along a slip surface.

    Args:
        profile: A :class:`~pygeotech.slope.profile.SlopeProfile`.
        surface: A slip surface (``CircularSurface`` or ``PolylineSurface``).
        n_slices: Number of slices to generate.

    Returns:
        List of :class:`Slice` objects, ordered left to right.

    Raises:
        ValueError: If the slip surface does not intersect the ground.
    """
    ee = surface.entry_exit(profile)
    if ee is None:
        raise ValueError(
            "Slip surface does not intersect the ground profile."
        )
    x_entry, x_exit = ee

    # Slice boundaries
    x_bounds = np.linspace(x_entry, x_exit, n_slices + 1)
    slices: list[Slice] = []

    for i in range(n_slices):
        x_left = x_bounds[i]
        x_right = x_bounds[i + 1]
        b = x_right - x_left
        x_mid = 0.5 * (x_left + x_right)

        y_top = profile.surface_elevation(x_mid)
        y_base = surface.y_at(x_mid)
        if y_base is None:
            continue

        # Skip if base is above ground surface
        if y_base >= y_top:
            continue

        h = y_top - y_base
        alpha = surface.base_angle(x_mid)
        base_length = b / max(np.cos(alpha), 1e-10)

        # Material at base
        layer = profile.layer_at(x_mid, y_base)

        # Weight: integrate unit weight through the slice height.
        # For simplicity with multiple layers, use the layer at the
        # midheight for the average unit weight.
        y_mid_height = 0.5 * (y_top + y_base)
        hw = profile.water_elevation(x_mid)

        # Weighted unit weight accounting for saturated portion
        if hw <= y_base:
            # Entirely above water table
            gamma_avg = layer.gamma
        elif hw >= y_top:
            # Entirely below water table
            gamma_avg = layer.gamma_sat
        else:
            # Partially submerged
            h_dry = y_top - hw
            h_wet = hw - y_base
            gamma_avg = (layer.gamma * h_dry + layer.gamma_sat * h_wet) / h

        W = gamma_avg * h * b

        # Pore pressure at base midpoint
        u = profile.pore_pressure(x_mid, y_base)

        slices.append(Slice(
            x_mid=x_mid,
            width=b,
            height=h,
            alpha=alpha,
            base_length=base_length,
            weight=W,
            cohesion=layer.cohesion,
            friction_angle=layer.phi_rad,
            pore_pressure=u,
            y_base=y_base,
            y_top=y_top,
        ))

    return slices
