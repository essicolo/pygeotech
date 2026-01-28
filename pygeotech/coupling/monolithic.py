"""Monolithic (fully coupled) system assembly.

All physics modules are assembled into a single global system of
equations and solved simultaneously.
"""

from __future__ import annotations

from typing import Any, Sequence

from pygeotech.coupling.base import CoupledProblem
from pygeotech.physics.base import PhysicsModule


class Monolithic(CoupledProblem):
    """Fully coupled monolithic system.

    All modules are assembled into a single block system.  This gives
    the most robust convergence for tightly coupled problems (e.g.
    Biot consolidation) but requires a monolithic solver.

    Args:
        modules: Physics modules to couple.
    """

    coupling_strategy = "monolithic"

    def __init__(self, modules: Sequence[PhysicsModule]) -> None:
        super().__init__(modules)

    def step(
        self,
        solutions: dict[str, Any],
        dt: float | None = None,
        t: float | None = None,
    ) -> dict[str, Any]:
        """Assemble and solve the monolithic block system.

        This is a placeholder — actual assembly depends on the solver
        backend.

        Args:
            solutions: Current fields.
            dt: Time-step size.
            t: Current time.

        Returns:
            Updated solutions.
        """
        raise NotImplementedError(
            "Monolithic coupling requires a backend that supports block "
            "system assembly (e.g. FEniCS).  Use Sequential or Iterative "
            "coupling with FiPy or PINN backends."
        )
