"""Boundary condition base classes.

Classes
-------
BoundaryCondition
    Abstract base for all BC types.
BoundaryConditions
    Collection of boundary conditions applied to a problem.
Dirichlet
    Fixed-value (essential) boundary condition.
Neumann
    Fixed-flux (natural) boundary condition.
Robin
    Mixed (Robin / third-type) boundary condition.
Seepage
    Seepage-face boundary condition (H = z where H > z).
Cauchy
    Cauchy-type boundary condition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike


class BoundaryCondition(ABC):
    """Abstract boundary condition.

    Every concrete BC stores the *field* it applies to (e.g. ``"H"``,
    ``"C"``, ``"T"``), the prescribed value or flux, and a *locator*
    that identifies where on the boundary the condition is active.
    """

    field: str
    where: Any  # BoundaryLocator or None

    @abstractmethod
    def apply_mask(self, coords: np.ndarray) -> np.ndarray:
        """Return a boolean mask for boundary nodes where this BC is active.

        Args:
            coords: Boundary node coordinates, shape ``(N, dim)``.

        Returns:
            Boolean array of shape ``(N,)``.
        """

    @abstractmethod
    def evaluate(self, coords: np.ndarray, t: float | None = None) -> np.ndarray:
        """Evaluate the BC value at active boundary locations.

        Args:
            coords: Coordinates of active boundary nodes.
            t: Current simulation time (for time-varying BCs).

        Returns:
            Array of BC values.
        """


class Dirichlet(BoundaryCondition):
    """Fixed-value (Dirichlet / essential) boundary condition.

    Args:
        field: Name of the solution field (e.g. ``"H"``).
        value: Prescribed value — scalar, array, or callable ``f(coords, t)``.
        where: Boundary locator (from :mod:`pygeotech.boundaries.locators`).
    """

    def __init__(
        self,
        field: str = "H",
        value: float | ArrayLike | Callable = 0.0,
        where: Any = None,
    ) -> None:
        self.field = field
        self.value = value
        self.where = where

    def apply_mask(self, coords: np.ndarray) -> np.ndarray:
        if self.where is None:
            return np.ones(len(coords), dtype=bool)
        return self.where(coords)

    def evaluate(self, coords: np.ndarray, t: float | None = None) -> np.ndarray:
        if callable(self.value) and not isinstance(self.value, (int, float)):
            return np.asarray(self.value(coords, t), dtype=float)
        return np.full(len(coords), float(self.value))

    def __repr__(self) -> str:
        return f"Dirichlet(field={self.field!r}, value={self.value!r})"


class Neumann(BoundaryCondition):
    """Fixed-flux (Neumann / natural) boundary condition.

    Args:
        field: Solution field name.
        flux: Normal flux value (scalar, array, or callable).
        where: Boundary locator.
    """

    def __init__(
        self,
        field: str = "H",
        flux: float | ArrayLike | Callable = 0.0,
        where: Any = None,
    ) -> None:
        self.field = field
        self.flux = flux
        self.where = where

    def apply_mask(self, coords: np.ndarray) -> np.ndarray:
        if self.where is None:
            return np.ones(len(coords), dtype=bool)
        return self.where(coords)

    def evaluate(self, coords: np.ndarray, t: float | None = None) -> np.ndarray:
        if callable(self.flux) and not isinstance(self.flux, (int, float)):
            return np.asarray(self.flux(coords, t), dtype=float)
        return np.full(len(coords), float(self.flux))

    def __repr__(self) -> str:
        return f"Neumann(field={self.field!r}, flux={self.flux!r})"


class Robin(BoundaryCondition):
    """Robin (mixed / third-type) boundary condition.

    The condition is: ``alpha * u + beta * du/dn = gamma``.

    Args:
        field: Solution field name.
        alpha: Coefficient of the field value.
        beta: Coefficient of the normal derivative.
        gamma: Right-hand side.
        where: Boundary locator.
    """

    def __init__(
        self,
        field: str = "H",
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        where: Any = None,
    ) -> None:
        self.field = field
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.where = where

    def apply_mask(self, coords: np.ndarray) -> np.ndarray:
        if self.where is None:
            return np.ones(len(coords), dtype=bool)
        return self.where(coords)

    def evaluate(self, coords: np.ndarray, t: float | None = None) -> np.ndarray:
        return np.full(len(coords), self.gamma)

    def __repr__(self) -> str:
        return (
            f"Robin(field={self.field!r}, alpha={self.alpha}, "
            f"beta={self.beta}, gamma={self.gamma})"
        )


class Seepage(BoundaryCondition):
    """Seepage face boundary condition.

    On a seepage face, H = z wherever H > z (i.e. the pore pressure
    cannot be positive on a free-draining surface).

    Args:
        where: Boundary locator identifying the seepage face.
    """

    def __init__(self, where: Any = None) -> None:
        self.field = "H"
        self.where = where

    def apply_mask(self, coords: np.ndarray) -> np.ndarray:
        if self.where is None:
            return np.ones(len(coords), dtype=bool)
        return self.where(coords)

    def evaluate(self, coords: np.ndarray, t: float | None = None) -> np.ndarray:
        # Seepage: H = z (elevation)
        return coords[:, -1].copy()

    def __repr__(self) -> str:
        return "Seepage()"


class Cauchy(BoundaryCondition):
    """Cauchy boundary condition (alias for Robin with specific semantics).

    Args:
        field: Solution field name.
        transfer_coeff: Transfer coefficient.
        reference_value: External reference value.
        where: Boundary locator.
    """

    def __init__(
        self,
        field: str = "H",
        transfer_coeff: float = 1.0,
        reference_value: float = 0.0,
        where: Any = None,
    ) -> None:
        self.field = field
        self.transfer_coeff = transfer_coeff
        self.reference_value = reference_value
        self.where = where

    def apply_mask(self, coords: np.ndarray) -> np.ndarray:
        if self.where is None:
            return np.ones(len(coords), dtype=bool)
        return self.where(coords)

    def evaluate(self, coords: np.ndarray, t: float | None = None) -> np.ndarray:
        return np.full(len(coords), self.reference_value)

    def __repr__(self) -> str:
        return (
            f"Cauchy(field={self.field!r}, "
            f"transfer_coeff={self.transfer_coeff}, "
            f"reference_value={self.reference_value})"
        )


# ======================================================================
# Collection
# ======================================================================


class BoundaryConditions:
    """Ordered collection of boundary conditions.

    Example::

        bc = BoundaryConditions()
        bc.add(Dirichlet(field="H", value=10.0, where=top()))
        bc.add(Neumann(field="H", flux=0.0, where=bottom()))
    """

    def __init__(self) -> None:
        self._conditions: list[BoundaryCondition] = []

    def add(self, condition: BoundaryCondition) -> None:
        """Append a boundary condition."""
        self._conditions.append(condition)

    def __iter__(self):
        return iter(self._conditions)

    def __len__(self) -> int:
        return len(self._conditions)

    def of_type(self, cls: type) -> list[BoundaryCondition]:
        """Return all conditions of a given type."""
        return [c for c in self._conditions if isinstance(c, cls)]

    def for_field(self, field: str) -> list[BoundaryCondition]:
        """Return all conditions for a given field name."""
        return [c for c in self._conditions if c.field == field]

    def __repr__(self) -> str:
        return f"BoundaryConditions({self._conditions})"
