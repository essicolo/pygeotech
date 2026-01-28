"""Soil-Vegetation-Atmosphere Transfer (SVAT) module.

Extends Richards equation with:
- Evapotranspiration (potential → actual ET)
- Root water uptake (Feddes / Jarvis model)
- Surface energy / water balance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pygeotech.physics.base import PhysicsModule


@dataclass
class RootUptakeParams:
    """Parameters for the Feddes root water uptake model.

    The sink term S(h) = α(h) · S_max where α(h) is a stress function
    varying between 0 and 1 depending on pressure head.

    Args:
        h1: Anaerobiosis point (m, negative).
        h2: Reduction point (wet side, m).
        h3: Reduction point (dry side, m).
        h4: Wilting point (m).
        root_depth: Maximum root depth (m).
        max_transpiration: Maximum transpiration rate (m/s).
    """

    h1: float = -0.1
    h2: float = -0.25
    h3: float = -8.0
    h4: float = -150.0
    root_depth: float = 1.0
    max_transpiration: float = 5e-8  # ~4 mm/day


class SVAT(PhysicsModule):
    """Soil-Vegetation-Atmosphere Transfer module.

    Provides source/sink terms for the Richards equation based on
    atmospheric forcing and vegetation parameters.

    Args:
        mesh: Computational mesh.
        materials: Material map.
        root_uptake: Root water uptake parameters.
        potential_et: Potential evapotranspiration rate (m/s), scalar or
            callable ``f(t)``.
    """

    name = "svat"
    primary_field = "H"

    def __init__(
        self,
        mesh: Any,
        materials: Any,
        root_uptake: RootUptakeParams | None = None,
        potential_et: float | Any = 0.0,
    ) -> None:
        super().__init__(mesh, materials)
        self.root_uptake = root_uptake or RootUptakeParams()
        self.potential_et = potential_et

    @property
    def is_transient(self) -> bool:
        return True

    def feddes_alpha(self, h: np.ndarray) -> np.ndarray:
        """Compute the Feddes stress reduction factor α(h).

        Args:
            h: Pressure head array (m).

        Returns:
            Stress factor α ∈ [0, 1].
        """
        rp = self.root_uptake
        alpha = np.zeros_like(h)
        # Wet side: linear ramp from h1 to h2
        mask1 = (h >= rp.h2) & (h <= rp.h1)
        alpha[mask1] = (h[mask1] - rp.h1) / (rp.h2 - rp.h1 + 1e-30)
        # Optimal zone
        mask2 = (h >= rp.h3) & (h < rp.h2)
        alpha[mask2] = 1.0
        # Dry side: linear ramp from h3 to h4
        mask3 = (h >= rp.h4) & (h < rp.h3)
        alpha[mask3] = (h[mask3] - rp.h4) / (rp.h3 - rp.h4 + 1e-30)
        return alpha

    def root_sink(
        self,
        field_values: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Compute root water uptake sink term.

        Args:
            field_values: Current head field H.
            t: Current time (s) for time-varying ET.

        Returns:
            Sink rate per node (m/s), shape ``(n_nodes,)``.
        """
        z = self.mesh.nodes[:, -1]
        h = field_values - z  # pressure head

        alpha = self.feddes_alpha(h)

        # Potential ET rate
        if callable(self.potential_et):
            pet = self.potential_et(t)
        else:
            pet = float(self.potential_et)

        # Distribute over root zone
        z_surface = z.max()
        depth = z_surface - z
        root_mask = depth <= self.root_uptake.root_depth
        sink = np.zeros_like(field_values)
        n_root = root_mask.sum()
        if n_root > 0:
            sink[root_mask] = alpha[root_mask] * pet / self.root_uptake.root_depth

        return sink

    def coefficients(self) -> dict[str, np.ndarray]:
        return self.materials.cell_property("porosity"), {}

    def residual(
        self,
        field_values: np.ndarray,
        field_values_old: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Placeholder: returns root sink as a source term."""
        return self.root_sink(field_values)
