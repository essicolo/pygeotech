"""Temporal discretisation schemes.

Classes
-------
TimeScheme
    Abstract base for time-integration schemes.
Implicit
    Backward Euler (fully implicit, θ = 1).
Explicit
    Forward Euler (fully explicit, θ = 0).
CrankNicolson
    Crank-Nicolson (θ = 0.5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class TimeScheme(ABC):
    """Abstract time-integration scheme.

    The θ-method discretises the generic ODE M du/dt = F(u) as:

        M (u^{n+1} - u^n) / dt = θ F(u^{n+1}) + (1 - θ) F(u^n)

    Subclasses set the value of θ.
    """

    @property
    @abstractmethod
    def theta(self) -> float:
        """Implicit weighting parameter θ ∈ [0, 1]."""

    def blend(
        self,
        f_new: np.ndarray,
        f_old: np.ndarray,
    ) -> np.ndarray:
        """Weighted blend of new and old right-hand sides.

        Returns θ·f_new + (1-θ)·f_old.
        """
        return self.theta * f_new + (1.0 - self.theta) * f_old

    def __repr__(self) -> str:
        return f"{type(self).__name__}(theta={self.theta})"


class Implicit(TimeScheme):
    """Backward Euler (θ = 1).  Unconditionally stable."""

    @property
    def theta(self) -> float:
        return 1.0


class Explicit(TimeScheme):
    """Forward Euler (θ = 0).  Conditionally stable."""

    @property
    def theta(self) -> float:
        return 0.0


class CrankNicolson(TimeScheme):
    """Crank-Nicolson (θ = 0.5).  Second-order accurate."""

    @property
    def theta(self) -> float:
        return 0.5
