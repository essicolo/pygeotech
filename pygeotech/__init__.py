"""
pygeotech: Modular Python package for modeling coupled processes
in geotechnical engineering and geosciences.

Subpackages
-----------
geometry
    Domain and subdomain definition, meshing.
materials
    Material properties, constitutive models, spatial fields.
boundaries
    Boundary condition specification and locators.
physics
    Physics modules (Darcy, Richards, transport, heat, mechanics, SVAT).
coupling
    Multiphysics coupling strategies.
solvers
    Solver backends (FiPy, PINN, analytical).
time
    Time-stepping schemes.
postprocess
    Derived quantities, probes, integrals, export.
visualization
    2-D and 3-D plotting utilities.
stratigraphy
    Borehole classification and spatial interpolation of geological layers.
slope
    Slope stability analysis (LEM, SRM).
"""

from pygeotech import (
    geometry,
    materials,
    boundaries,
    physics,
    coupling,
    solvers,
    time,
    postprocess,
    visualization,
    stratigraphy,
    slope,
)

__version__ = "0.1.0"

__all__ = [
    "geometry",
    "materials",
    "boundaries",
    "physics",
    "coupling",
    "solvers",
    "time",
    "postprocess",
    "visualization",
    "stratigraphy",
    "slope",
]
