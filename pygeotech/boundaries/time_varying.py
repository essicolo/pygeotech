"""Time-dependent boundary conditions.

Classes
-------
TimeVaryingBC
    Wraps any BC with a time-dependent value function.
Hydrograph
    Piecewise-linear time series (e.g. river stage, rainfall).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from pygeotech.boundaries.base import BoundaryCondition, Dirichlet, Neumann


class Hydrograph:
    """Piecewise-linear time series for boundary conditions.

    Args:
        times: Sequence of time values (s).
        values: Corresponding BC values.

    Example::

        hydro = Hydrograph(
            times=[0, 3600, 7200, 86400],
            values=[10.0, 15.0, 12.0, 10.0],
        )
        hydro(5000)  # linearly interpolated value
    """

    def __init__(
        self,
        times: Sequence[float],
        values: Sequence[float],
    ) -> None:
        self.times = np.asarray(times, dtype=float)
        self.values = np.asarray(values, dtype=float)
        if len(self.times) != len(self.values):
            raise ValueError("times and values must have the same length.")

    def __call__(self, t: float) -> float:
        """Interpolate the value at time *t*."""
        return float(np.interp(t, self.times, self.values))


class TimeVaryingBC:
    """Wrap a Dirichlet or Neumann BC with a time-varying value.

    Args:
        bc_type: ``"dirichlet"`` or ``"neumann"``.
        field: Solution field name.
        value_func: Callable ``f(t) -> float`` returning the BC value at
            time *t*, or a :class:`Hydrograph`.
        where: Boundary locator.

    Example::

        tv_bc = TimeVaryingBC(
            bc_type="dirichlet",
            field="H",
            value_func=Hydrograph([0, 86400], [10.0, 15.0]),
            where=left(),
        )
    """

    def __init__(
        self,
        bc_type: str = "dirichlet",
        field: str = "H",
        value_func: Callable[[float], float] | Hydrograph = lambda t: 0.0,
        where: object = None,
    ) -> None:
        self.bc_type = bc_type.lower()
        self.field = field
        self.value_func = value_func
        self.where = where

    def at_time(self, t: float) -> BoundaryCondition:
        """Return a static BC evaluated at time *t*.

        Args:
            t: Simulation time (s).

        Returns:
            Dirichlet or Neumann BC with the interpolated value.
        """
        val = self.value_func(t)
        if self.bc_type == "dirichlet":
            return Dirichlet(field=self.field, value=val, where=self.where)
        elif self.bc_type == "neumann":
            return Neumann(field=self.field, flux=val, where=self.where)
        raise ValueError(f"Unknown bc_type: {self.bc_type!r}")

    def __repr__(self) -> str:
        return (
            f"TimeVaryingBC(bc_type={self.bc_type!r}, field={self.field!r})"
        )
