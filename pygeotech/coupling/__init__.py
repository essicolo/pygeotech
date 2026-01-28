"""Coupling: multiphysics coupling strategies."""

from pygeotech.coupling.base import CoupledProblem
from pygeotech.coupling.sequential import Sequential
from pygeotech.coupling.iterative import Iterative
from pygeotech.coupling.monolithic import Monolithic

__all__ = [
    "CoupledProblem",
    "Sequential",
    "Iterative",
    "Monolithic",
]
