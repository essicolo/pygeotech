"""Iterative coupling with convergence check.

The modules are solved repeatedly until the coupling residual
drops below a tolerance.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pygeotech.coupling.base import CoupledProblem
from pygeotech.coupling.sequential import Sequential
from pygeotech.physics.base import PhysicsModule


class Iterative(CoupledProblem):
    """Iterative coupling with convergence control.

    At each time step the modules are solved sequentially in a loop
    until the coupling residual is below tolerance.

    Args:
        modules: Physics modules.
        max_iter: Maximum coupling iterations per time step.
        tol: Convergence tolerance on the L2-norm of field change.
    """

    coupling_strategy = "iterative"

    def __init__(
        self,
        modules: Sequence[PhysicsModule],
        max_iter: int = 20,
        tol: float = 1e-6,
    ) -> None:
        super().__init__(modules)
        self.max_iter = max_iter
        self.tol = tol
        self._sequential = Sequential(modules)

    def step(
        self,
        solutions: dict[str, Any],
        dt: float | None = None,
        t: float | None = None,
    ) -> dict[str, Any]:
        """Iterate until convergence.

        Args:
            solutions: Current field values keyed by module name.
            dt: Time-step size.
            t: Current time.

        Returns:
            Converged solutions.
        """
        updated = dict(solutions)

        for iteration in range(self.max_iter):
            prev = {}
            for k, v in updated.items():
                if isinstance(v, np.ndarray):
                    prev[k] = v.copy()

            updated = self._sequential.step(updated, dt=dt, t=t)

            if self._converged(prev, updated):
                break

        return updated

    def _converged(
        self,
        old: dict[str, Any],
        new: dict[str, Any],
    ) -> bool:
        """Check convergence based on field change norms."""
        for key in old:
            o = old[key]
            n = new.get(key)
            if isinstance(o, np.ndarray) and isinstance(n, np.ndarray):
                norm_old = np.linalg.norm(o)
                if norm_old < 1e-30:
                    continue
                change = np.linalg.norm(n - o) / norm_old
                if change > self.tol:
                    return False
        return True
