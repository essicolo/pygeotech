"""Time stepping utilities.

Classes
-------
Stepper
    Fixed-size time stepping.
AdaptiveStepper
    Adaptive time stepping with error control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class Stepper:
    """Fixed time stepper.

    Args:
        t_end: End time (s).
        dt: Time-step size (s).
        t_start: Start time (s).  Defaults to 0.

    Example::

        stepper = Stepper(t_end=86400, dt=3600)  # 1 day, hourly
        for t, dt in stepper:
            print(f"t={t:.0f} s, dt={dt:.0f} s")
    """

    t_end: float
    dt: float
    t_start: float = 0.0

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return int(np.ceil((self.t_end - self.t_start) / self.dt))

    @property
    def times(self) -> np.ndarray:
        """Array of all time values (including start)."""
        return np.arange(self.t_start, self.t_end + 0.5 * self.dt, self.dt)

    def __iter__(self) -> Iterator[tuple[float, float]]:
        """Yield ``(t, dt)`` tuples."""
        t = self.t_start
        while t < self.t_end - 1e-12:
            step_dt = min(self.dt, self.t_end - t)
            t += step_dt
            yield t, step_dt

    def __repr__(self) -> str:
        return f"Stepper(t_end={self.t_end}, dt={self.dt}, t_start={self.t_start})"


@dataclass
class AdaptiveStepper:
    """Adaptive time stepper with error-based step-size control.

    The step size is adjusted based on a local truncation error
    estimate:  ``dt_new = dt * (tol / error)^(1/order)``.

    Args:
        t_end: End time (s).
        dt_init: Initial time-step size (s).
        dt_min: Minimum allowed step size (s).
        dt_max: Maximum allowed step size (s).
        tol: Error tolerance.
        safety: Safety factor (< 1).
        order: Method order for step-size formula.
        t_start: Start time.
    """

    t_end: float
    dt_init: float
    dt_min: float = 1e-6
    dt_max: float = 1e6
    tol: float = 1e-4
    safety: float = 0.9
    order: int = 2
    t_start: float = 0.0

    def suggest_dt(self, error: float, dt_current: float) -> float:
        """Suggest the next step size based on the error estimate.

        Args:
            error: Estimated local truncation error.
            dt_current: Current step size.

        Returns:
            Suggested new step size.
        """
        if error < 1e-30:
            return min(2.0 * dt_current, self.dt_max)
        ratio = self.tol / error
        factor = self.safety * ratio ** (1.0 / self.order)
        factor = max(0.1, min(factor, 5.0))  # clamp growth
        dt_new = dt_current * factor
        return float(np.clip(dt_new, self.dt_min, self.dt_max))

    def __repr__(self) -> str:
        return (
            f"AdaptiveStepper(t_end={self.t_end}, dt_init={self.dt_init}, "
            f"tol={self.tol})"
        )
