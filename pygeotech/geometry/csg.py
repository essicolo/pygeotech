"""Constructive Solid Geometry (CSG) boolean operations.

Functions
---------
union
    Boolean union of two geometries.
intersection
    Boolean intersection.
difference
    Boolean difference (A minus B).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pygeotech.geometry.primitives import Geometry


class _CSGComposite(Geometry):
    """Internal composite geometry produced by a boolean operation."""

    def __init__(self, a: Geometry, b: Geometry, op: str) -> None:
        if a.dim != b.dim:
            raise ValueError("Both geometries must have the same dimension.")
        super().__init__(dim=a.dim)
        self._a = a
        self._b = b
        self._op = op

    def contains(self, points: ArrayLike) -> np.ndarray:
        in_a = self._a.contains(points)
        in_b = self._b.contains(points)
        if self._op == "union":
            return in_a | in_b
        elif self._op == "intersection":
            return in_a & in_b
        elif self._op == "difference":
            return in_a & ~in_b
        raise ValueError(f"Unknown CSG operation: {self._op}")

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        lo_a, hi_a = self._a.bounding_box()
        lo_b, hi_b = self._b.bounding_box()
        if self._op == "union":
            return np.minimum(lo_a, lo_b), np.maximum(hi_a, hi_b)
        elif self._op == "intersection":
            return np.maximum(lo_a, lo_b), np.minimum(hi_a, hi_b)
        elif self._op == "difference":
            return lo_a.copy(), hi_a.copy()
        raise ValueError(f"Unknown CSG operation: {self._op}")

    def area_or_volume(self) -> float:
        raise NotImplementedError(
            "Exact area/volume for CSG composites requires meshing.  "
            "Generate a mesh and compute from that."
        )

    def __repr__(self) -> str:
        return f"CSG({self._op}, {self._a!r}, {self._b!r})"


def union(a: Geometry, b: Geometry) -> Geometry:
    """Boolean union of two geometries.

    Args:
        a: First geometry.
        b: Second geometry.

    Returns:
        Composite geometry representing the union.
    """
    return _CSGComposite(a, b, "union")


def intersection(a: Geometry, b: Geometry) -> Geometry:
    """Boolean intersection of two geometries.

    Args:
        a: First geometry.
        b: Second geometry.

    Returns:
        Composite geometry representing the intersection.
    """
    return _CSGComposite(a, b, "intersection")


def difference(a: Geometry, b: Geometry) -> Geometry:
    """Boolean difference A \\ B.

    Args:
        a: Base geometry.
        b: Geometry to subtract.

    Returns:
        Composite geometry representing the difference.
    """
    return _CSGComposite(a, b, "difference")
