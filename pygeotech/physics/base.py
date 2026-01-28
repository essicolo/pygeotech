"""Abstract base class for physics modules.

All physics modules (Darcy, Richards, Transport, etc.) inherit from
:class:`PhysicsModule` and implement a common interface that solvers
use to assemble and solve the governing equations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class PhysicsModule(ABC):
    """Abstract physics module.

    A physics module encapsulates the governing PDE, its discretisation
    requirements, and the mapping from material properties to equation
    coefficients.

    Attributes:
        name: Short identifier (e.g. ``"darcy"``, ``"richards"``).
        mesh: The computational mesh.
        materials: A :class:`~pygeotech.materials.base.MaterialMap`.
        primary_field: Name of the unknown field (e.g. ``"H"``, ``"C"``).
        dim: Spatial dimension (inherited from mesh).
    """

    name: str
    primary_field: str

    def __init__(self, mesh: Any, materials: Any) -> None:
        self.mesh = mesh
        self.materials = materials
        self.dim = mesh.dim

    @abstractmethod
    def residual(
        self,
        field_values: np.ndarray,
        field_values_old: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Evaluate the discrete residual of the governing equation.

        For steady-state problems, *field_values_old* and *dt* are
        ``None``.

        Args:
            field_values: Current solution field, shape ``(n_nodes,)``.
            field_values_old: Solution at the previous time step.
            dt: Time-step size (s).

        Returns:
            Residual vector, shape ``(n_nodes,)``.
        """

    @abstractmethod
    def coefficients(self) -> dict[str, np.ndarray]:
        """Return PDE coefficient arrays needed by solvers.

        The exact keys depend on the physics.  Typical keys include:

        * ``"conductivity"`` — per-cell conductivity tensor or scalar.
        * ``"storage"`` — per-cell storage coefficient.
        * ``"source"`` — per-cell source/sink term.

        Returns:
            Dictionary of named coefficient arrays.
        """

    @property
    def is_transient(self) -> bool:
        """Whether this physics module involves time derivatives."""
        return False

    def validate(self) -> list[str]:
        """Run basic consistency checks.

        Returns:
            List of warning/error strings (empty if all OK).
        """
        issues: list[str] = []
        if self.mesh is None:
            issues.append("No mesh assigned.")
        if self.materials is None:
            issues.append("No materials assigned.")
        return issues

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(primary_field={self.primary_field!r}, "
            f"dim={self.dim})"
        )
