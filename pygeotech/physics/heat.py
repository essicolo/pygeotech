"""Heat transfer in porous media.

Governing equation::

    ρc ∂T/∂t = ∇·(λ ∇T) + ρ_w c_w q·∇T + Q

where T is temperature, λ is thermal conductivity, q is Darcy
velocity (for advective transport), and Q is a heat source.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pygeotech.physics.base import PhysicsModule


class HeatTransfer(PhysicsModule):
    """Heat conduction and advection in porous media.

    Args:
        mesh: Computational mesh.
        materials: Material map providing ``thermal_conductivity``,
            ``specific_heat``, and ``dry_density``.
        water_density: Density of water (kg/m³).
        water_specific_heat: Specific heat of water (J/(kg·K)).
    """

    name = "heat"
    primary_field = "T"

    def __init__(
        self,
        mesh: Any,
        materials: Any,
        water_density: float = 1000.0,
        water_specific_heat: float = 4186.0,
    ) -> None:
        super().__init__(mesh, materials)
        self.water_density = water_density
        self.water_specific_heat = water_specific_heat
        self._velocity: np.ndarray | None = None

    @property
    def is_transient(self) -> bool:
        return True

    def set_velocity(self, velocity: np.ndarray) -> None:
        """Set the Darcy velocity field for advective heat transport.

        Args:
            velocity: Array of shape ``(n_cells, dim)``.
        """
        self._velocity = np.asarray(velocity, dtype=float)

    def coefficients(self) -> dict[str, np.ndarray]:
        lam = self.materials.cell_property("thermal_conductivity")
        rho = self.materials.cell_property("dry_density")
        cp = self.materials.cell_property("specific_heat")
        return {
            "thermal_conductivity": lam,
            "bulk_heat_capacity": rho * cp,
        }

    def residual(
        self,
        field_values: np.ndarray,
        field_values_old: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Placeholder residual for the heat equation."""
        return np.zeros(self.mesh.n_nodes)
