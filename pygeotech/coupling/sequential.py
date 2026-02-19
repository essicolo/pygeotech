"""Sequential (operator-splitting) coupling.

The modules are solved one after another in order.  Data is passed
from earlier modules to later ones (one-way within a time step).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pygeotech.coupling.base import CoupledProblem
from pygeotech.physics.base import PhysicsModule


class Sequential(CoupledProblem):
    """Sequential operator-splitting coupling.

    Modules are solved in the order given.  After each module is solved,
    its output (e.g. velocity field) is made available to subsequent
    modules.

    Args:
        modules: Ordered list of physics modules.

    Example::

        coupled = Sequential([flow, transport])
        # 1. Solve flow -> extract velocity
        # 2. Pass velocity to transport -> solve transport
    """

    coupling_strategy = "sequential"

    def __init__(self, modules: Sequence[PhysicsModule]) -> None:
        super().__init__(modules)

    def step(
        self,
        solutions: dict[str, np.ndarray],
        dt: float | None = None,
        t: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Solve each module sequentially, passing data forward.

        This method performs actual solves using each module's
        ``assemble_system`` and the sparse direct solver.

        Args:
            solutions: Current fields keyed by module name.
            dt: Time-step size.
            t: Current time.

        Returns:
            Updated solutions dict with field arrays.
        """
        from scipy.sparse.linalg import spsolve

        updated = dict(solutions)
        for module in self.modules:
            # Transfer data from earlier modules
            self._transfer_data(module, updated)

            # Solve this module
            field = self._solve_module(module, updated, dt, t)
            updated[module.name] = field

        return updated

    def _solve_module(
        self,
        module: PhysicsModule,
        solutions: dict[str, np.ndarray],
        dt: float | None,
        t: float | None,
    ) -> np.ndarray:
        """Solve a single physics module.

        Dispatches to the appropriate assembly method based on module type.
        """
        from scipy.sparse.linalg import spsolve

        field_name = module.primary_field
        field_old = solutions.get(module.name)
        if isinstance(field_old, dict):
            field_old = None

        # Assemble system
        A, rhs, d_nodes, d_values = self._assemble(
            module, field_old, dt
        )

        # Apply Dirichlet BCs
        n = A.shape[0]
        A_lil = A.tolil()
        for node, val in zip(d_nodes, d_values):
            col = np.array(A_lil[node, :].todense()).flatten()
            rhs -= col * val
            rhs[node] = val
            A_lil[node, :] = 0
            A_lil[:, node] = 0
            A_lil[node, node] = 1.0

        return spsolve(A_lil.tocsr(), rhs)

    def _assemble(
        self,
        module: PhysicsModule,
        field_old: np.ndarray | None,
        dt: float | None,
    ) -> tuple:
        """Call the module's assemble_system with correct signature."""
        from pygeotech.boundaries.base import BoundaryConditions

        # Get BCs from the module if stored, else empty
        bc = getattr(module, "_boundary_conditions", BoundaryConditions())

        from pygeotech.physics.darcy import Darcy
        from pygeotech.physics.richards import Richards
        from pygeotech.physics.transport import Transport
        from pygeotech.physics.heat import HeatTransfer
        from pygeotech.physics.mechanics import Mechanics

        if isinstance(module, Darcy):
            return module.assemble_system(bc)

        if isinstance(module, Richards):
            H_current = field_old
            if H_current is None:
                H_current = np.full(module.mesh.n_nodes, 0.0)
            return module.assemble_system(bc, H_current=H_current, H_old=field_old, dt=dt)

        if isinstance(module, Transport):
            return module.assemble_system(bc, C_current=field_old, C_old=field_old, dt=dt)

        if isinstance(module, HeatTransfer):
            return module.assemble_system(bc, T_old=field_old, dt=dt)

        if isinstance(module, Mechanics):
            return module.assemble_system(bc)

        raise NotImplementedError(
            f"Sequential coupling does not know how to assemble {type(module).__name__}."
        )

    def _transfer_data(
        self,
        target: PhysicsModule,
        solutions: dict[str, Any],
    ) -> None:
        """Transfer coupling data to the target module.

        Transfers velocity from flow modules to transport/heat, and
        pore pressure from flow to mechanics.
        """
        if hasattr(target, "set_velocity"):
            for m in self.modules:
                if m is target:
                    break
                if m.name in ("darcy", "richards"):
                    sol = solutions.get(m.name)
                    if sol is not None and isinstance(sol, np.ndarray):
                        # Compute velocity from head field
                        from pygeotech.postprocess.fields import compute_gradient
                        grad_H = compute_gradient(m.mesh, sol)
                        K = m.materials.cell_property("hydraulic_conductivity")
                        velocity = -K[:, np.newaxis] * grad_H
                        target.set_velocity(velocity)

        if hasattr(target, "set_pore_pressure"):
            for m in self.modules:
                if m is target:
                    break
                if m.name in ("darcy", "richards"):
                    sol = solutions.get(m.name)
                    if sol is not None and isinstance(sol, np.ndarray):
                        # Pore pressure = (H - z) * rho_w * g
                        z = m.mesh.nodes[:, -1]
                        pressure = (sol - z) * 9810.0  # Pa
                        target.set_pore_pressure(pressure)
