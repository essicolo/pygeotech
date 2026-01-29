"""FiPy finite-volume solver backend.

This backend uses a built-in sparse direct solver (SciPy) for all
physics types.  Despite the name, FiPy is not required -- the name
is kept for API compatibility.

Supports:
- Darcy steady-state
- Richards transient (Picard iteration)
- Transport transient (with sorption and decay)
- Heat transfer transient
- Mechanics (quasi-static)
- Coupled problems (sequential/iterative)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from pygeotech.solvers.base import Solver, Solution


class FiPyBackend(Solver):
    """Finite-element solver backend using SciPy sparse direct solver.

    Args:
        use_fipy: Unused; kept for API compatibility.
        picard_max_iter: Maximum Picard iterations for nonlinear problems.
        picard_tol: Convergence tolerance for Picard iteration.
    """

    def __init__(
        self,
        use_fipy: bool = True,
        picard_max_iter: int = 50,
        picard_tol: float = 1e-6,
    ) -> None:
        self.use_fipy = use_fipy
        self.picard_max_iter = picard_max_iter
        self.picard_tol = picard_tol

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
            boundary_conditions: Boundary conditions (single or dict).
            time: Time stepper (for transient).
            initial_condition: Initial values.

        Returns:
            Solution object.
        """
        from pygeotech.physics.darcy import Darcy
        from pygeotech.physics.richards import Richards
        from pygeotech.physics.transport import Transport
        from pygeotech.physics.heat import HeatTransfer
        from pygeotech.physics.mechanics import Mechanics
        from pygeotech.coupling.base import CoupledProblem

        if isinstance(physics, CoupledProblem):
            return self._solve_coupled(
                physics, boundary_conditions, time, initial_condition
            )

        if isinstance(physics, Darcy) and time is None:
            return self._solve_darcy_steady(physics, boundary_conditions)

        if isinstance(physics, Richards):
            return self._solve_richards(
                physics, boundary_conditions, time, initial_condition
            )

        if isinstance(physics, Transport):
            return self._solve_transport(
                physics, boundary_conditions, time, initial_condition
            )

        if isinstance(physics, HeatTransfer):
            return self._solve_heat(
                physics, boundary_conditions, time, initial_condition
            )

        if isinstance(physics, Mechanics):
            return self._solve_mechanics(physics, boundary_conditions)

        raise NotImplementedError(
            f"FiPyBackend does not yet support {type(physics).__name__} "
            f"{'transient' if time else 'steady-state'} problems."
        )

    # ------------------------------------------------------------------
    # Dirichlet BC application (shared helper)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_dirichlet(
        A: sparse.spmatrix,
        rhs: np.ndarray,
        d_nodes: np.ndarray,
        d_values: np.ndarray,
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Apply Dirichlet BCs via row/column elimination."""
        A_lil = A.tolil()
        for node, val in zip(d_nodes, d_values):
            col = np.array(A_lil[node, :].todense()).flatten()
            rhs -= col * val
            rhs[node] = val
            A_lil[node, :] = 0
            A_lil[:, node] = 0
            A_lil[node, node] = 1.0
        return A_lil.tocsr(), rhs

    # ------------------------------------------------------------------
    # Darcy steady-state
    # ------------------------------------------------------------------

    def _solve_darcy_steady(
        self,
        darcy: Any,
        boundary_conditions: Any,
    ) -> Solution:
        """Solve nabla.(K nabla H) = 0."""
        A, rhs, d_nodes, d_values = darcy.assemble_system(boundary_conditions)
        A_bc, rhs_bc = self._apply_dirichlet(A, rhs, d_nodes, d_values)
        H = spsolve(A_bc, rhs_bc)
        return Solution(fields={"H": H}, mesh=darcy.mesh)

    # ------------------------------------------------------------------
    # Richards transient (Picard iteration)
    # ------------------------------------------------------------------

    def _solve_richards(
        self,
        richards: Any,
        boundary_conditions: Any,
        time: Any | None = None,
        initial_condition: dict | None = None,
    ) -> Solution:
        """Solve Richards equation with Picard iteration per time step."""
        mesh = richards.mesh
        n_nodes = mesh.n_nodes
        ic = initial_condition or {}

        # Initial head
        H = self._init_field(ic, "H", n_nodes)

        if time is None:
            # Steady-state Richards = nonlinear Darcy
            for _ in range(self.picard_max_iter):
                A, rhs, d_n, d_v = richards.assemble_system(
                    boundary_conditions, H_current=H, H_old=None, dt=None,
                )
                A_bc, rhs_bc = self._apply_dirichlet(A, rhs, d_n, d_v)
                H_new = spsolve(A_bc, rhs_bc)
                if np.linalg.norm(H_new - H) / (np.linalg.norm(H) + 1e-30) < self.picard_tol:
                    H = H_new
                    break
                H = H_new
            return Solution(fields={"H": H}, mesh=mesh)

        # Transient
        times_list = [0.0]
        history: list[np.ndarray] = [H.copy()]

        for t, dt in time:
            H_old = H.copy()
            for _ in range(self.picard_max_iter):
                A, rhs, d_n, d_v = richards.assemble_system(
                    boundary_conditions, H_current=H, H_old=H_old, dt=dt,
                )
                A_bc, rhs_bc = self._apply_dirichlet(A, rhs, d_n, d_v)
                H_new = spsolve(A_bc, rhs_bc)
                if np.linalg.norm(H_new - H) / (np.linalg.norm(H) + 1e-30) < self.picard_tol:
                    H = H_new
                    break
                H = H_new
            times_list.append(t + dt)
            history.append(H.copy())

        return Solution(
            fields={"H": H},
            mesh=mesh,
            times=np.array(times_list),
            field_history={"H": history},
        )

    # ------------------------------------------------------------------
    # Transport transient
    # ------------------------------------------------------------------

    def _solve_transport(
        self,
        transport: Any,
        boundary_conditions: Any,
        time: Any | None = None,
        initial_condition: dict | None = None,
    ) -> Solution:
        """Solve the advection-dispersion equation."""
        mesh = transport.mesh
        n_nodes = mesh.n_nodes
        ic = initial_condition or {}

        C = self._init_field(ic, "C", n_nodes)

        if time is None:
            # Steady-state transport
            A, rhs, d_n, d_v = transport.assemble_system(
                boundary_conditions, C_current=C, C_old=None, dt=None,
            )
            A_bc, rhs_bc = self._apply_dirichlet(A, rhs, d_n, d_v)
            C = spsolve(A_bc, rhs_bc)
            return Solution(fields={"C": C}, mesh=mesh)

        times_list = [0.0]
        history: list[np.ndarray] = [C.copy()]

        for t, dt in time:
            C_old = C.copy()
            # For nonlinear sorption, iterate
            if transport.sorption in ("freundlich", "langmuir"):
                for _ in range(self.picard_max_iter):
                    A, rhs, d_n, d_v = transport.assemble_system(
                        boundary_conditions, C_current=C, C_old=C_old, dt=dt,
                    )
                    A_bc, rhs_bc = self._apply_dirichlet(A, rhs, d_n, d_v)
                    C_new = spsolve(A_bc, rhs_bc)
                    if np.linalg.norm(C_new - C) / (np.linalg.norm(C) + 1e-30) < self.picard_tol:
                        C = C_new
                        break
                    C = C_new
            else:
                A, rhs, d_n, d_v = transport.assemble_system(
                    boundary_conditions, C_current=C, C_old=C_old, dt=dt,
                )
                A_bc, rhs_bc = self._apply_dirichlet(A, rhs, d_n, d_v)
                C = spsolve(A_bc, rhs_bc)

            times_list.append(t + dt)
            history.append(C.copy())

        return Solution(
            fields={"C": C},
            mesh=mesh,
            times=np.array(times_list),
            field_history={"C": history},
        )

    # ------------------------------------------------------------------
    # Heat transfer transient
    # ------------------------------------------------------------------

    def _solve_heat(
        self,
        heat: Any,
        boundary_conditions: Any,
        time: Any | None = None,
        initial_condition: dict | None = None,
    ) -> Solution:
        """Solve the heat equation."""
        mesh = heat.mesh
        n_nodes = mesh.n_nodes
        ic = initial_condition or {}

        T = self._init_field(ic, "T", n_nodes)

        if time is None:
            # Steady-state
            A, rhs, d_n, d_v = heat.assemble_system(
                boundary_conditions, T_old=None, dt=None,
            )
            A_bc, rhs_bc = self._apply_dirichlet(A, rhs, d_n, d_v)
            T = spsolve(A_bc, rhs_bc)
            return Solution(fields={"T": T}, mesh=mesh)

        times_list = [0.0]
        history: list[np.ndarray] = [T.copy()]

        for t, dt in time:
            T_old = T.copy()
            A, rhs, d_n, d_v = heat.assemble_system(
                boundary_conditions, T_old=T_old, dt=dt,
            )
            A_bc, rhs_bc = self._apply_dirichlet(A, rhs, d_n, d_v)
            T = spsolve(A_bc, rhs_bc)
            times_list.append(t + dt)
            history.append(T.copy())

        return Solution(
            fields={"T": T},
            mesh=mesh,
            times=np.array(times_list),
            field_history={"T": history},
        )

    # ------------------------------------------------------------------
    # Mechanics (quasi-static)
    # ------------------------------------------------------------------

    def _solve_mechanics(
        self,
        mech: Any,
        boundary_conditions: Any,
    ) -> Solution:
        """Solve the elasticity system K u = f."""
        K, rhs, d_dofs, d_vals = mech.assemble_system(boundary_conditions)
        K_bc, rhs_bc = self._apply_dirichlet(K, rhs, d_dofs, d_vals)
        u = spsolve(K_bc, rhs_bc)

        n_nodes = mech.mesh.n_nodes
        ux = u[0::2]
        uy = u[1::2]
        return Solution(
            fields={"u": u, "ux": ux, "uy": uy},
            mesh=mech.mesh,
        )

    # ------------------------------------------------------------------
    # Coupled problems
    # ------------------------------------------------------------------

    def _solve_coupled(
        self,
        coupled: Any,
        boundary_conditions: Any,
        time: Any,
        initial_condition: dict | None,
    ) -> Solution:
        """Solve a coupled problem using the coupling strategy.

        *boundary_conditions* should be a dict keyed by module name.
        """
        ic = initial_condition or {}
        bc_dict = boundary_conditions if isinstance(boundary_conditions, dict) else {}

        # Attach BCs to each module
        for module in coupled.modules:
            bc_key = module.name
            if bc_key in bc_dict:
                module._boundary_conditions = bc_dict[bc_key]

        # Initialise solution fields
        solutions: dict[str, np.ndarray] = {}
        for module in coupled.modules:
            field_name = module.primary_field
            n = module.mesh.n_nodes
            if module.name == "mechanics":
                n = 2 * module.mesh.n_nodes
            if field_name in ic:
                val = ic[field_name]
                if isinstance(val, np.ndarray):
                    solutions[module.name] = val.copy()
                else:
                    solutions[module.name] = np.full(n, float(val))
            else:
                solutions[module.name] = np.zeros(n)

        if time is None:
            # Single steady-state pass
            solutions = coupled.step(solutions, dt=None, t=None)
            fields = {}
            for module in coupled.modules:
                sol = solutions.get(module.name)
                if isinstance(sol, np.ndarray):
                    fields[module.primary_field] = sol
            return Solution(fields=fields, mesh=coupled.modules[0].mesh)

        # Transient: time-step loop
        mesh = coupled.modules[0].mesh
        times_list = [0.0]
        all_history: dict[str, list[np.ndarray]] = {}
        for module in coupled.modules:
            all_history[module.primary_field] = [
                solutions[module.name].copy()
            ]

        for t, dt in time:
            solutions = coupled.step(solutions, dt=dt, t=t)
            times_list.append(t + dt)
            for module in coupled.modules:
                sol = solutions.get(module.name)
                if isinstance(sol, np.ndarray):
                    all_history[module.primary_field].append(sol.copy())

        fields = {}
        for module in coupled.modules:
            sol = solutions.get(module.name)
            if isinstance(sol, np.ndarray):
                fields[module.primary_field] = sol

        return Solution(
            fields=fields,
            mesh=mesh,
            times=np.array(times_list),
            field_history=all_history,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_field(
        ic: dict[str, float | np.ndarray],
        field_name: str,
        n_nodes: int,
    ) -> np.ndarray:
        """Initialise a field from initial_condition dict."""
        if field_name in ic:
            val = ic[field_name]
            if isinstance(val, np.ndarray):
                return val.copy()
            return np.full(n_nodes, float(val))
        return np.zeros(n_nodes)
