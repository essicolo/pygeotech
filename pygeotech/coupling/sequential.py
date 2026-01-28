"""Sequential (operator-splitting) coupling.

The modules are solved one after another in order.  Data is passed
from earlier modules to later ones (one-way within a time step).
"""

from __future__ import annotations

from typing import Any, Sequence

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
        # 1. Solve flow → extract velocity
        # 2. Pass velocity to transport → solve transport
    """

    coupling_strategy = "sequential"

    def __init__(self, modules: Sequence[PhysicsModule]) -> None:
        super().__init__(modules)

    def step(
        self,
        solutions: dict[str, Any],
        dt: float | None = None,
        t: float | None = None,
    ) -> dict[str, Any]:
        """Solve each module sequentially, passing data forward.

        Args:
            solutions: Current fields keyed by module name.
            dt: Time-step size.
            t: Current time.

        Returns:
            Updated solutions.

        Note:
            The actual solve is delegated to the solver backend; this
            method defines the *order* and *data transfer* between
            modules.
        """
        updated = dict(solutions)
        for module in self.modules:
            # Transfer data from earlier modules
            self._transfer_data(module, updated)
            # Mark for solving (actual solve happens in solver backend)
            updated[module.name] = {"module": module, "dt": dt, "t": t}
        return updated

    def _transfer_data(
        self,
        target: PhysicsModule,
        solutions: dict[str, Any],
    ) -> None:
        """Transfer coupling data to the target module.

        For example, velocity from a flow module to a transport module.
        """
        if hasattr(target, "set_velocity"):
            # Look for a flow module that has computed velocity
            for m in self.modules:
                if m is target:
                    break
                if m.name in ("darcy", "richards"):
                    sol = solutions.get(m.name)
                    if sol is not None and hasattr(sol, "get"):
                        velocity = sol.get("velocity")
                        if velocity is not None:
                            target.set_velocity(velocity)

        if hasattr(target, "set_pore_pressure"):
            for m in self.modules:
                if m is target:
                    break
                if m.name in ("darcy", "richards"):
                    sol = solutions.get(m.name)
                    if sol is not None and hasattr(sol, "get"):
                        pressure = sol.get("pressure")
                        if pressure is not None:
                            target.set_pore_pressure(pressure)
