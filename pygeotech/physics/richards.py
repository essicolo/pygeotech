"""Unsaturated flow — Richards equation.

Governing equation (mixed form)::

    ∂θ/∂t = ∇·(K(h) ∇H) - S

where θ = θ(h) is the volumetric water content, K(h) is the
unsaturated hydraulic conductivity, H = h + z is total head, and S
is a source/sink term.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pygeotech.physics.base import PhysicsModule


class Richards(PhysicsModule):
    """Transient unsaturated flow (Richards equation).

    Args:
        mesh: Computational mesh.
        materials: Material map.
        retention_model: A soil-water retention model (e.g.
            :class:`~pygeotech.materials.constitutive.VanGenuchten`).
    """

    name = "richards"
    primary_field = "H"

    def __init__(
        self,
        mesh: Any,
        materials: Any,
        retention_model: Any = None,
    ) -> None:
        super().__init__(mesh, materials)
        self.retention_model = retention_model

    @property
    def is_transient(self) -> bool:
        return True

    def coefficients(self) -> dict[str, np.ndarray]:
        """Return per-cell conductivity and storage.

        For Richards' equation the conductivity is head-dependent:
        K_eff = K_sat * Kr(h).  This method returns the saturated
        conductivity; the nonlinear update is done in :meth:`residual`.
        """
        K_sat = self.materials.cell_property("hydraulic_conductivity")
        porosity = self.materials.cell_property("porosity")
        return {
            "conductivity": K_sat,
            "porosity": porosity,
        }

    def residual(
        self,
        field_values: np.ndarray,
        field_values_old: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Evaluate the Richards equation residual.

        This is a placeholder for the full nonlinear residual.  Actual
        solution requires a Newton-type iteration handled by the solver
        backend.

        Args:
            field_values: Current head field H, shape ``(n_nodes,)``.
            field_values_old: Previous time-step head.
            dt: Time step (s).

        Returns:
            Residual vector.
        """
        n = self.mesh.n_nodes
        residual = np.zeros(n)

        if self.retention_model is not None:
            # Compute pressure head h = H - z
            z = self.mesh.nodes[:, -1]
            h = field_values - z
            theta = self.retention_model.water_content(h)

            if field_values_old is not None and dt is not None:
                h_old = field_values_old - z
                theta_old = self.retention_model.water_content(h_old)
                residual += (theta - theta_old) / dt

        return residual

    def validate(self) -> list[str]:
        issues = super().validate()
        if self.retention_model is None:
            issues.append("No retention model specified for Richards equation.")
        return issues
