"""Solute transport — advection-dispersion equation.

Governing equation::

    ∂(θC)/∂t + ∇·(q C) = ∇·(θ D ∇C) + θ(λC + S)

where C is concentration, q is Darcy velocity from flow solution,
D is the hydrodynamic dispersion tensor, λ is a first-order decay
rate, and S is a source/sink.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pygeotech.physics.base import PhysicsModule


class Transport(PhysicsModule):
    """Advection-dispersion solute transport.

    This module is typically coupled with a flow module (Darcy or
    Richards) that provides the velocity field.

    Args:
        mesh: Computational mesh.
        materials: Material map.
        dispersion_longitudinal: Longitudinal dispersivity (m).
        dispersion_transverse: Transverse dispersivity (m).
        molecular_diffusion: Molecular diffusion coefficient (m²/s).
        retardation_factor: Retardation factor R (–).
        decay_rate: First-order decay rate λ (1/s).
    """

    name = "transport"
    primary_field = "C"

    def __init__(
        self,
        mesh: Any,
        materials: Any,
        dispersion_longitudinal: float = 1.0,
        dispersion_transverse: float = 0.1,
        molecular_diffusion: float = 1e-9,
        retardation_factor: float = 1.0,
        decay_rate: float = 0.0,
    ) -> None:
        super().__init__(mesh, materials)
        self.dispersion_longitudinal = dispersion_longitudinal
        self.dispersion_transverse = dispersion_transverse
        self.molecular_diffusion = molecular_diffusion
        self.retardation_factor = retardation_factor
        self.decay_rate = decay_rate
        self._velocity: np.ndarray | None = None

    @property
    def is_transient(self) -> bool:
        return True

    def set_velocity(self, velocity: np.ndarray) -> None:
        """Set the Darcy velocity field from a flow solution.

        Args:
            velocity: Array of shape ``(n_cells, dim)``.
        """
        self._velocity = np.asarray(velocity, dtype=float)

    def dispersion_tensor(self, velocity: np.ndarray) -> np.ndarray:
        """Compute the hydrodynamic dispersion tensor D per cell.

        For isotropic dispersivity:
        D_ij = (α_T |v| + D_m) δ_ij + (α_L - α_T) v_i v_j / |v|

        Args:
            velocity: Per-cell velocity, shape ``(n_cells, dim)``.

        Returns:
            Dispersion tensor, shape ``(n_cells, dim, dim)``.
        """
        v = np.asarray(velocity, dtype=float)
        n_cells, dim = v.shape
        v_mag = np.linalg.norm(v, axis=1, keepdims=True)  # (n_cells, 1)
        v_mag_safe = np.maximum(v_mag, 1e-30)

        D = np.zeros((n_cells, dim, dim))
        eye = np.eye(dim)

        for ic in range(n_cells):
            vmag = v_mag_safe[ic, 0]
            vi = v[ic]
            D[ic] = (
                (self.dispersion_transverse * vmag + self.molecular_diffusion) * eye
                + (self.dispersion_longitudinal - self.dispersion_transverse)
                * np.outer(vi, vi) / vmag
            )

        return D

    def coefficients(self) -> dict[str, np.ndarray]:
        porosity = self.materials.cell_property("porosity")
        return {
            "porosity": porosity,
            "retardation": np.full(self.mesh.n_cells, self.retardation_factor),
            "decay_rate": np.full(self.mesh.n_cells, self.decay_rate),
        }

    def residual(
        self,
        field_values: np.ndarray,
        field_values_old: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Placeholder residual for the transport equation.

        The full residual requires the velocity field from a flow
        solution.  Solvers should use :meth:`coefficients` and
        :meth:`dispersion_tensor` for assembly.
        """
        n = self.mesh.n_nodes
        return np.zeros(n)
