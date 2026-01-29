"""Solvers: backend implementations for solving PDEs."""

from pygeotech.solvers.base import Solver
from pygeotech.solvers.fipy_backend import FiPyBackend
from pygeotech.solvers.pinn_backend import PINNBackend
from pygeotech.solvers.analytical import AnalyticalSolver

__all__ = [
    "Solver",
    "FiPyBackend",
    "PINNBackend",
    "AnalyticalSolver",
]
