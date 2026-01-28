"""Solvers: backend implementations for solving PDEs."""

from pygeotech.solvers.base import Solver
from pygeotech.solvers.fipy_backend import FiPyBackend
from pygeotech.solvers.fenics_backend import FEniCSBackend
from pygeotech.solvers.pinn_backend import PINNBackend
from pygeotech.solvers.analytical import AnalyticalSolver

__all__ = [
    "Solver",
    "FiPyBackend",
    "FEniCSBackend",
    "PINNBackend",
    "AnalyticalSolver",
]
