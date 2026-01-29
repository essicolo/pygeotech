"""Slope stability analysis.

Provides both Limit Equilibrium Methods (LEM) and Strength Reduction
Method (SRM) for computing factors of safety.

LEM — Method of Slices
~~~~~~~~~~~~~~~~~~~~~~
Classic approach: divide the soil mass above a trial slip surface into
vertical slices and apply equilibrium equations.

Methods:
    - :func:`bishop_simplified` — moment equilibrium, circular surfaces
    - :func:`spencer` — full equilibrium (force + moment)
    - :func:`morgenstern_price` — general interslice force function
    - :func:`janbu_simplified` — force equilibrium, non-circular surfaces

SRM — Strength Reduction
~~~~~~~~~~~~~~~~~~~~~~~~~
FEM-based approach: progressively reduce c' and tan(φ') by a Strength
Reduction Factor (SRF) until the FEM solution fails to converge.

Example::

    from pygeotech.slope import (
        SlopeProfile, CircularSurface, bishop_simplified,
        grid_search,
    )

    profile = SlopeProfile(
        surface=[(0, 10), (20, 10), (30, 20), (50, 20)],
        layers=[
            {"c": 10e3, "phi": 25, "gamma": 18e3, "top": 20, "bottom": 0},
        ],
    )

    # Single surface
    surface = CircularSurface(xc=25, yc=30, radius=15)
    fos = bishop_simplified(profile, surface)

    # Critical surface search
    result = grid_search(
        profile,
        method="bishop",
        xc_range=(10, 40), yc_range=(25, 45), r_range=(10, 25),
    )
    print(result.fos, result.surface)
"""

from pygeotech.slope.profile import SlopeProfile
from pygeotech.slope.surfaces import CircularSurface, PolylineSurface
from pygeotech.slope.slices import Slice, generate_slices
from pygeotech.slope.lem import (
    bishop_simplified,
    spencer,
    morgenstern_price,
    janbu_simplified,
)
from pygeotech.slope.search import grid_search, SearchResult
from pygeotech.slope.srm import strength_reduction

__all__ = [
    "SlopeProfile",
    "CircularSurface",
    "PolylineSurface",
    "Slice",
    "generate_slices",
    "bishop_simplified",
    "spencer",
    "morgenstern_price",
    "janbu_simplified",
    "grid_search",
    "SearchResult",
    "strength_reduction",
]
