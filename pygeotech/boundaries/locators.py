"""Boundary locator functions.

Locators are composable predicates that select subsets of boundary
nodes.  They support ``&`` (AND), ``|`` (OR), and ``~`` (NOT)
operators for building complex selections.

Example::

    loc = top() & x_less_than(40)
    mask = loc(boundary_coords)  # boolean array
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike


class BoundaryLocator:
    """Composable boundary predicate.

    Wraps a function ``f(coords) -> bool_array`` and supports boolean
    algebra via ``&``, ``|``, ``~``.
    """

    def __init__(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        self._func = func

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        return self._func(np.atleast_2d(np.asarray(coords, dtype=float)))

    def __and__(self, other: "BoundaryLocator") -> "BoundaryLocator":
        return BoundaryLocator(lambda c: self(c) & other(c))

    def __or__(self, other: "BoundaryLocator") -> "BoundaryLocator":
        return BoundaryLocator(lambda c: self(c) | other(c))

    def __invert__(self) -> "BoundaryLocator":
        return BoundaryLocator(lambda c: ~self(c))


# ------------------------------------------------------------------
# Named boundary faces (for rectangular / box domains)
# ------------------------------------------------------------------

_TOL = 1e-10


def top(tol: float = _TOL) -> BoundaryLocator:
    """Select nodes on the top boundary (max y in 2-D, max z in 3-D)."""
    def _select(coords: np.ndarray) -> np.ndarray:
        last = coords[:, -1]
        return np.abs(last - last.max()) < tol
    return BoundaryLocator(_select)


def bottom(tol: float = _TOL) -> BoundaryLocator:
    """Select nodes on the bottom boundary (min y in 2-D, min z in 3-D)."""
    def _select(coords: np.ndarray) -> np.ndarray:
        last = coords[:, -1]
        return np.abs(last - last.min()) < tol
    return BoundaryLocator(_select)


def left(tol: float = _TOL) -> BoundaryLocator:
    """Select nodes on the left boundary (min x)."""
    def _select(coords: np.ndarray) -> np.ndarray:
        x = coords[:, 0]
        return np.abs(x - x.min()) < tol
    return BoundaryLocator(_select)


def right(tol: float = _TOL) -> BoundaryLocator:
    """Select nodes on the right boundary (max x)."""
    def _select(coords: np.ndarray) -> np.ndarray:
        x = coords[:, 0]
        return np.abs(x - x.max()) < tol
    return BoundaryLocator(_select)


def front(tol: float = _TOL) -> BoundaryLocator:
    """Select nodes on the front boundary (min y in 3-D)."""
    def _select(coords: np.ndarray) -> np.ndarray:
        if coords.shape[1] < 3:
            raise ValueError("front() requires 3-D coordinates.")
        y = coords[:, 1]
        return np.abs(y - y.min()) < tol
    return BoundaryLocator(_select)


def back(tol: float = _TOL) -> BoundaryLocator:
    """Select nodes on the back boundary (max y in 3-D)."""
    def _select(coords: np.ndarray) -> np.ndarray:
        if coords.shape[1] < 3:
            raise ValueError("back() requires 3-D coordinates.")
        y = coords[:, 1]
        return np.abs(y - y.max()) < tol
    return BoundaryLocator(_select)


# ------------------------------------------------------------------
# Coordinate predicates
# ------------------------------------------------------------------


def x_equals(value: float, tol: float = _TOL) -> BoundaryLocator:
    """Select nodes where x ≈ *value*."""
    return BoundaryLocator(lambda c: np.abs(c[:, 0] - value) < tol)


def x_less_than(value: float) -> BoundaryLocator:
    """Select nodes where x < *value*."""
    return BoundaryLocator(lambda c: c[:, 0] < value)


def x_greater_than(value: float) -> BoundaryLocator:
    """Select nodes where x > *value*."""
    return BoundaryLocator(lambda c: c[:, 0] > value)


def x_between(a: float, b: float) -> BoundaryLocator:
    """Select nodes where a ≤ x ≤ b."""
    return BoundaryLocator(lambda c: (c[:, 0] >= a) & (c[:, 0] <= b))


def y_equals(value: float, tol: float = _TOL) -> BoundaryLocator:
    """Select nodes where y ≈ *value*."""
    return BoundaryLocator(lambda c: np.abs(c[:, 1] - value) < tol)


def y_less_than(value: float) -> BoundaryLocator:
    """Select nodes where y < *value*."""
    return BoundaryLocator(lambda c: c[:, 1] < value)


def y_greater_than(value: float) -> BoundaryLocator:
    """Select nodes where y > *value*."""
    return BoundaryLocator(lambda c: c[:, 1] > value)


def y_between(a: float, b: float) -> BoundaryLocator:
    """Select nodes where a ≤ y ≤ b."""
    return BoundaryLocator(lambda c: (c[:, 1] >= a) & (c[:, 1] <= b))


# ------------------------------------------------------------------
# Geometric locators
# ------------------------------------------------------------------


def on_curve(
    vertices: ArrayLike,
    tol: float = 0.1,
) -> BoundaryLocator:
    """Select nodes within *tol* of a polyline defined by *vertices*.

    Args:
        vertices: Array of shape ``(M, dim)`` — the polyline vertices.
        tol: Distance tolerance.

    Returns:
        BoundaryLocator.
    """
    verts = np.atleast_2d(np.asarray(vertices, dtype=float))

    def _select(coords: np.ndarray) -> np.ndarray:
        mask = np.zeros(len(coords), dtype=bool)
        for i in range(len(verts) - 1):
            a, b = verts[i], verts[i + 1]
            ab = b - a
            ab_len2 = np.dot(ab, ab)
            if ab_len2 < 1e-30:
                continue
            ap = coords - a
            t = np.clip(ap @ ab / ab_len2, 0.0, 1.0)
            proj = a + np.outer(t, ab)
            dist = np.linalg.norm(coords - proj, axis=1)
            mask |= dist < tol
        return mask

    return BoundaryLocator(_select)


def on_surface(
    geometry: object,
    tol: float = 0.1,
) -> BoundaryLocator:
    """Select nodes within *tol* of a geometry boundary.

    This is a convenience wrapper; for 2-D domains it delegates to edge
    distance checks.

    Args:
        geometry: A :class:`~pygeotech.geometry.primitives.Geometry` with
            an ``edges`` or ``bounding_box`` attribute.
        tol: Distance tolerance.

    Returns:
        BoundaryLocator.
    """
    if hasattr(geometry, "edges"):
        verts = np.array([e[0] for e in geometry.edges] + [geometry.edges[-1][1]])
        return on_curve(verts, tol=tol)

    raise NotImplementedError(
        "on_surface currently requires a Polygon-like geometry with edges."
    )
