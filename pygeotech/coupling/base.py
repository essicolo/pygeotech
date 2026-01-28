"""Base class for coupled multiphysics problems.

Classes
-------
CoupledProblem
    Container for multiple physics modules that exchange data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from pygeotech.physics.base import PhysicsModule


class CoupledProblem(ABC):
    """Abstract base class for multiphysics coupling.

    A coupled problem wraps multiple :class:`PhysicsModule` instances
    and defines how they exchange information (e.g. velocity from flow
    to transport).

    Attributes:
        modules: Ordered list of physics modules.
        coupling_strategy: Name of the coupling approach.
    """

    coupling_strategy: str

    def __init__(self, modules: Sequence[PhysicsModule]) -> None:
        self.modules = list(modules)

    @property
    def primary_fields(self) -> list[str]:
        """Names of all primary unknowns across modules."""
        return [m.primary_field for m in self.modules]

    @property
    def is_transient(self) -> bool:
        """True if any module is transient."""
        return any(m.is_transient for m in self.modules)

    @abstractmethod
    def step(
        self,
        solutions: dict[str, Any],
        dt: float | None = None,
        t: float | None = None,
    ) -> dict[str, Any]:
        """Advance the coupled system by one time step.

        Args:
            solutions: Current solution fields keyed by module name.
            dt: Time-step size.
            t: Current simulation time.

        Returns:
            Updated solutions dictionary.
        """

    def validate(self) -> list[str]:
        """Validate all modules and coupling consistency."""
        issues: list[str] = []
        for m in self.modules:
            issues.extend(m.validate())
        return issues

    def __repr__(self) -> str:
        names = [m.name for m in self.modules]
        return f"{type(self).__name__}(modules={names})"
