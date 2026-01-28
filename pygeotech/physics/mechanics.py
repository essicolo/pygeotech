"""Solid mechanics — quasi-static equilibrium.

Governing equation::

    ∇·σ + b = 0
    σ = C : ε − α p I   (effective stress with pore-pressure coupling)

where σ is the Cauchy stress tensor, ε is the strain tensor,
C is the constitutive stiffness, α is the Biot coefficient, and
p is pore pressure.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pygeotech.physics.base import PhysicsModule


class Mechanics(PhysicsModule):
    """Quasi-static solid mechanics / geomechanics.

    Args:
        mesh: Computational mesh.
        materials: Material map providing ``youngs_modulus`` and
            ``poissons_ratio``.
        biot_coefficient: Biot effective-stress coefficient α.
        gravity: Gravitational acceleration (m/s²).
    """

    name = "mechanics"
    primary_field = "u"  # displacement vector

    def __init__(
        self,
        mesh: Any,
        materials: Any,
        biot_coefficient: float = 1.0,
        gravity: float = 9.81,
    ) -> None:
        super().__init__(mesh, materials)
        self.biot_coefficient = biot_coefficient
        self.gravity = gravity
        self._pore_pressure: np.ndarray | None = None

    def set_pore_pressure(self, pressure: np.ndarray) -> None:
        """Set the pore-pressure field for effective stress coupling.

        Args:
            pressure: Nodal pore pressure, shape ``(n_nodes,)``.
        """
        self._pore_pressure = np.asarray(pressure, dtype=float)

    def coefficients(self) -> dict[str, np.ndarray]:
        E = self.materials.cell_property("youngs_modulus")
        nu = self.materials.cell_property("poissons_ratio")
        rho = self.materials.cell_property("dry_density")
        return {
            "youngs_modulus": E,
            "poissons_ratio": nu,
            "density": rho,
        }

    def elastic_stiffness_2d(
        self,
        E: float,
        nu: float,
        plane_strain: bool = True,
    ) -> np.ndarray:
        """2-D elastic constitutive matrix (plane strain or plane stress).

        Args:
            E: Young's modulus.
            nu: Poisson's ratio.
            plane_strain: If True, plane-strain; else plane-stress.

        Returns:
            3×3 constitutive matrix (Voigt notation).
        """
        if plane_strain:
            factor = E / ((1 + nu) * (1 - 2 * nu))
            C = factor * np.array([
                [1 - nu, nu, 0],
                [nu, 1 - nu, 0],
                [0, 0, 0.5 * (1 - 2 * nu)],
            ])
        else:
            factor = E / (1 - nu ** 2)
            C = factor * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, 0.5 * (1 - nu)],
            ])
        return C

    def residual(
        self,
        field_values: np.ndarray,
        field_values_old: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Placeholder residual for the mechanics equation."""
        return np.zeros(self.mesh.n_nodes * self.dim)

    @property
    def is_transient(self) -> bool:
        return False
