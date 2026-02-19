"""Critical slip surface search algorithms.

Provides grid search and optimisation-based search for the critical
(lowest FOS) slip surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from pygeotech.slope.surfaces import CircularSurface


@dataclass
class SearchResult:
    """Result of a critical surface search.

    Attributes:
        fos: Factor of safety of the critical surface.
        surface: The critical slip surface.
        fos_grid: Full array of computed FOS values (grid search only).
        centers: Tested centre coordinates (grid search only).
    """

    fos: float
    surface: Any
    fos_grid: np.ndarray | None = None
    centers: np.ndarray | None = None


def grid_search(
    profile: Any,
    method: str = "bishop",
    xc_range: tuple[float, float] = (0, 50),
    yc_range: tuple[float, float] = (20, 50),
    r_range: tuple[float, float] = (5, 30),
    n_xc: int = 15,
    n_yc: int = 15,
    n_r: int = 10,
    n_slices: int = 30,
    **method_kwargs: Any,
) -> SearchResult:
    """Grid search for the critical circular slip surface.

    Evaluates FOS for a grid of trial circle centres and radii,
    returning the combination with the lowest FOS.

    Args:
        profile: Slope profile.
        method: LEM method name — ``"bishop"``, ``"spencer"``,
            ``"morgenstern_price"``, or ``"janbu"``.
        xc_range: (min, max) for circle centre x.
        yc_range: (min, max) for circle centre y.
        r_range: (min, max) for circle radius.
        n_xc: Grid points in x-direction.
        n_yc: Grid points in y-direction.
        n_r: Grid points for radius.
        n_slices: Number of slices per trial.
        **method_kwargs: Extra keyword arguments passed to the LEM
            method (e.g. ``force_function`` for Morgenstern-Price).

    Returns:
        :class:`SearchResult` with the critical surface and FOS.
    """
    from pygeotech.slope.lem import (
        bishop_simplified,
        spencer,
        morgenstern_price,
        janbu_simplified,
    )

    methods: dict[str, Callable] = {
        "bishop": bishop_simplified,
        "spencer": spencer,
        "morgenstern_price": morgenstern_price,
        "janbu": janbu_simplified,
    }
    if method not in methods:
        raise ValueError(
            f"Unknown method {method!r}. Choose from {list(methods)}"
        )
    lem_func = methods[method]

    xc_vals = np.linspace(xc_range[0], xc_range[1], n_xc)
    yc_vals = np.linspace(yc_range[0], yc_range[1], n_yc)
    r_vals = np.linspace(r_range[0], r_range[1], n_r)

    best_fos = float("inf")
    best_surface = None
    all_fos: list[float] = []
    all_centers: list[tuple[float, float, float]] = []

    for xc in xc_vals:
        for yc in yc_vals:
            for r in r_vals:
                surface = CircularSurface(xc=xc, yc=yc, radius=r)
                try:
                    fos = lem_func(
                        profile, surface, n_slices=n_slices, **method_kwargs
                    )
                except (ValueError, ZeroDivisionError):
                    fos = float("inf")

                all_fos.append(fos)
                all_centers.append((xc, yc, r))

                if fos < best_fos:
                    best_fos = fos
                    best_surface = surface

    return SearchResult(
        fos=best_fos,
        surface=best_surface,
        fos_grid=np.array(all_fos),
        centers=np.array(all_centers),
    )


def optimize_surface(
    profile: Any,
    method: str = "bishop",
    x0: tuple[float, float, float] | None = None,
    bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] | None = None,
    n_slices: int = 30,
    **method_kwargs: Any,
) -> SearchResult:
    """Optimisation-based search for the critical circular surface.

    Uses ``scipy.optimize.differential_evolution`` to find the circle
    centre and radius that minimise FOS.

    Args:
        profile: Slope profile.
        method: LEM method name.
        x0: Initial guess ``(xc, yc, R)`` (unused by DE, kept for API).
        bounds: Parameter bounds ``((xc_lo, xc_hi), (yc_lo, yc_hi),
            (R_lo, R_hi))``.  Required.
        n_slices: Number of slices.
        **method_kwargs: Extra keyword arguments for the LEM method.

    Returns:
        :class:`SearchResult`.
    """
    from scipy.optimize import differential_evolution

    from pygeotech.slope.lem import (
        bishop_simplified,
        spencer,
        morgenstern_price,
        janbu_simplified,
    )

    methods: dict[str, Callable] = {
        "bishop": bishop_simplified,
        "spencer": spencer,
        "morgenstern_price": morgenstern_price,
        "janbu": janbu_simplified,
    }
    if method not in methods:
        raise ValueError(f"Unknown method {method!r}")
    lem_func = methods[method]

    if bounds is None:
        raise ValueError("bounds must be provided for optimize_surface")

    def objective(params: np.ndarray) -> float:
        xc, yc, r = params
        surface = CircularSurface(xc=xc, yc=yc, radius=r)
        try:
            fos = lem_func(
                profile, surface, n_slices=n_slices, **method_kwargs
            )
        except (ValueError, ZeroDivisionError):
            return 1e6
        if not np.isfinite(fos):
            return 1e6
        return fos

    result = differential_evolution(
        objective,
        bounds=list(bounds),
        seed=42,
        maxiter=100,
        tol=1e-3,
        polish=True,
    )

    best_surface = CircularSurface(
        xc=result.x[0], yc=result.x[1], radius=result.x[2]
    )
    return SearchResult(fos=result.fun, surface=best_surface)
