"""FiPy finite-volume solver backend.

This backend uses FiPy (optional dependency) for solving PDEs via the
finite-volume method.  It also includes a built-in sparse direct solver
for the Darcy equation that works without FiPy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from pygeotech.solvers.base import Solver, Solution


class FiPyBackend(Solver):
    """Finite-volume solver backend.

    For steady-state Darcy flow, this backend uses a direct sparse
    solver (SciPy).  For more complex problems it delegates to FiPy
    when available.

    Args:
        use_fipy: If ``True``, attempt to use FiPy for transient /
            nonlinear problems.  If ``False``, use the built-in sparse
            solver only.
    """

    def __init__(self, use_fipy: bool = True) -> None:
        self.use_fipy = use_fipy

    def solve(
        self,
        physics: Any,
        boundary_conditions: Any = None,
        time: Any | None = None,
        initial_condition: dict[str, float | np.ndarray] | None = None,
        **kwargs: Any,
    ) -> Solution:
        """Solve the PDE system.

        Args:
            physics: Physics module or coupled problem.
            boundary_conditions: Boundary conditions.
            time: Time stepper (for transient).
            initial_condition: Initial values.

        Returns:
            Solution object.
        """
        from pygeotech.physics.darcy import Darcy
        from pygeotech.coupling.base import CoupledProblem

        if isinstance(physics, Darcy) and time is None:
            return self._solve_darcy_steady(physics, boundary_conditions)

        if isinstance(physics, CoupledProblem):
            return self._solve_coupled(
                physics, boundary_conditions, time, initial_condition
            )

        raise NotImplementedError(
            f"FiPyBackend does not yet support {type(physics).__name__} "
            f"{'transient' if time else 'steady-state'} problems without FiPy.  "
            "Install fipy for full support."
        )

    # ------------------------------------------------------------------
    # Darcy steady-state (built-in sparse solver)
    # ------------------------------------------------------------------

    def _solve_darcy_steady(
        self,
        darcy: Any,
        boundary_conditions: Any,
    ) -> Solution:
        """Solve ∇·(K∇H) = 0 with Dirichlet / Neumann BCs.

        Uses the finite-element stiffness matrix assembled by the Darcy
        module and a direct sparse solver.
        """
        A, rhs, d_nodes, d_values = darcy.assemble_system(boundary_conditions)

        n = darcy.mesh.n_nodes
        A_lil = A.tolil()

        # Apply Dirichlet BCs by row/column elimination
        for node, val in zip(d_nodes, d_values):
            # Move known contributions to RHS
            col = np.array(A_lil[node, :].todense()).flatten()
            rhs -= col * val
            rhs[node] = val

            # Zero out row and column
            A_lil[node, :] = 0
            A_lil[:, node] = 0
            A_lil[node, node] = 1.0

        A_csr = A_lil.tocsr()
        H = spsolve(A_csr, rhs)

        return Solution(
            fields={"H": H},
            mesh=darcy.mesh,
        )

    # ------------------------------------------------------------------
    # Coupled / transient (placeholder)
    # ------------------------------------------------------------------

    def _solve_coupled(
        self,
        coupled: Any,
        boundary_conditions: Any,
        time: Any,
        initial_condition: dict | None,
    ) -> Solution:
        """Solve a coupled problem (sequential or iterative).

        This is a placeholder for the full FiPy integration.
        """
        raise NotImplementedError(
            "Coupled / transient solving requires FiPy.  "
            "Install with: pip install fipy"
        )
