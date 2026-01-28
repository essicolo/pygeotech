"""Geometric primitives for 2-D and 3-D domain definition.

Classes
-------
Geometry
    Abstract base class for all geometric objects.
Rectangle, Polygon, Circle
    2-D primitives.
Box, Cylinder
    3-D primitives.
Line
    Utility line segment for cross-sections and probes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike


class Geometry(ABC):
    """Abstract base class for all geometric primitives.

    Attributes:
        dim: Spatial dimension (2 or 3).
        subdomains: Named subregions within the geometry.
    """

    dim: int
    subdomains: dict[str, "Geometry"]

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.subdomains: dict[str, Geometry] = {}

    # ------------------------------------------------------------------
    # Subdomain management
    # ------------------------------------------------------------------

    def add_subdomain(self, name: str, geometry: "Geometry") -> None:
        """Register a named subdomain inside this geometry.

        Args:
            name: Unique identifier for the subdomain.
            geometry: Geometric object describing the subdomain.

        Raises:
            ValueError: If the subdomain dimension does not match.
        """
        if geometry.dim != self.dim:
            raise ValueError(
                f"Subdomain dimension ({geometry.dim}) must match "
                f"domain dimension ({self.dim})."
            )
        self.subdomains[name] = geometry

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @abstractmethod
    def contains(self, points: ArrayLike) -> np.ndarray:
        """Test whether each point lies inside the geometry.

        Args:
            points: Array of shape ``(N, dim)``.

        Returns:
            Boolean array of shape ``(N,)``.
        """

    @abstractmethod
    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box.

        Returns:
            Tuple ``(min_corner, max_corner)`` each of shape ``(dim,)``.
        """

    @abstractmethod
    def area_or_volume(self) -> float:
        """Area (2-D) or volume (3-D) of the geometry."""

    # ------------------------------------------------------------------
    # Meshing convenience
    # ------------------------------------------------------------------

    def generate_mesh(
        self,
        resolution: float = 1.0,
        refine: dict[str, float] | None = None,
        algorithm: str = "delaunay",
    ) -> "Mesh":
        """Generate a mesh for this geometry.

        Args:
            resolution: Default element size.
            refine: Per-subdomain element sizes.
            algorithm: Meshing algorithm (``"delaunay"``, ``"frontal"``,
                ``"structured"``).

        Returns:
            A :class:`~pygeotech.geometry.mesh.Mesh` instance.
        """
        from pygeotech.geometry.mesh import Mesh

        return Mesh.from_geometry(
            self,
            resolution=resolution,
            refine=refine or {},
            algorithm=algorithm,
        )


# ======================================================================
# 2-D Primitives
# ======================================================================


class Rectangle(Geometry):
    """Axis-aligned rectangle.

    Args:
        Lx: Width (x-extent).
        Ly: Height (y-extent).
        origin: Bottom-left corner ``(x0, y0)``.  Defaults to ``(0, 0)``.
        x0: Alternative: left x coordinate.
        y0: Alternative: bottom y coordinate.
        width: Alternative name for *Lx* (used together with *x0*/*y0*).
        height: Alternative name for *Ly*.
    """

    def __init__(
        self,
        Lx: float | None = None,
        Ly: float | None = None,
        origin: tuple[float, float] = (0.0, 0.0),
        *,
        x0: float | None = None,
        y0: float | None = None,
        width: float | None = None,
        height: float | None = None,
    ) -> None:
        super().__init__(dim=2)

        # Resolve convenience kwargs
        if x0 is not None and y0 is not None:
            origin = (x0, y0)
        _w = width if width is not None else Lx
        _h = height if height is not None else Ly
        if _w is None or _h is None:
            raise ValueError("Must provide (Lx, Ly) or (width, height).")

        self.origin = np.asarray(origin, dtype=float)
        self.Lx = float(_w)
        self.Ly = float(_h)

    # Properties for boundary helpers
    @property
    def x_min(self) -> float:
        return float(self.origin[0])

    @property
    def x_max(self) -> float:
        return float(self.origin[0] + self.Lx)

    @property
    def y_min(self) -> float:
        return float(self.origin[1])

    @property
    def y_max(self) -> float:
        return float(self.origin[1] + self.Ly)

    # Geometry interface ------------------------------------------------

    def contains(self, points: ArrayLike) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        x, y = pts[:, 0], pts[:, 1]
        return (
            (x >= self.x_min) & (x <= self.x_max)
            & (y >= self.y_min) & (y <= self.y_max)
        )

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        return self.origin.copy(), self.origin + np.array([self.Lx, self.Ly])

    def area_or_volume(self) -> float:
        return self.Lx * self.Ly

    def __repr__(self) -> str:
        return (
            f"Rectangle(Lx={self.Lx}, Ly={self.Ly}, "
            f"origin=({self.origin[0]}, {self.origin[1]}))"
        )


class Polygon(Geometry):
    """Arbitrary 2-D polygon defined by its vertices.

    Args:
        vertices: Sequence of ``(x, y)`` coordinate pairs.  The polygon
            is automatically closed (do not repeat the first vertex).
    """

    def __init__(self, vertices: Sequence[tuple[float, float]]) -> None:
        super().__init__(dim=2)
        self.vertices = np.asarray(vertices, dtype=float)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError("vertices must have shape (N, 2).")
        if len(self.vertices) < 3:
            raise ValueError("A polygon requires at least 3 vertices.")

    @property
    def edges(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return list of ``(start, end)`` vertex pairs for each edge."""
        n = len(self.vertices)
        return [
            (self.vertices[i], self.vertices[(i + 1) % n])
            for i in range(n)
        ]

    @property
    def downstream_face(self) -> np.ndarray:
        """Heuristic: the right-most edge (max mean-x). Returns ``(2, 2)``."""
        best_edge = max(self.edges, key=lambda e: 0.5 * (e[0][0] + e[1][0]))
        return np.array(best_edge)

    def contains(self, points: ArrayLike) -> np.ndarray:
        """Ray-casting algorithm for point-in-polygon test."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        n = len(self.vertices)
        inside = np.zeros(len(pts), dtype=bool)
        vx = self.vertices[:, 0]
        vy = self.vertices[:, 1]
        for i in range(n):
            j = (i + 1) % n
            yi, yj = vy[i], vy[j]
            xi, xj = vx[i], vx[j]
            cond = ((yi > pts[:, 1]) != (yj > pts[:, 1])) & (
                pts[:, 0] < (xj - xi) * (pts[:, 1] - yi) / (yj - yi + 1e-300) + xi
            )
            inside ^= cond
        return inside

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def area_or_volume(self) -> float:
        """Shoelace formula."""
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def __repr__(self) -> str:
        return f"Polygon(n_vertices={len(self.vertices)})"


class Circle(Geometry):
    """Circle in 2-D.

    Args:
        center: ``(x, y)`` coordinates of the centre.
        radius: Radius of the circle.
    """

    def __init__(
        self,
        center: tuple[float, float] = (0.0, 0.0),
        radius: float = 1.0,
    ) -> None:
        super().__init__(dim=2)
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

    def contains(self, points: ArrayLike) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        dist = np.linalg.norm(pts - self.center, axis=1)
        return dist <= self.radius

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        r = np.array([self.radius, self.radius])
        return self.center - r, self.center + r

    def area_or_volume(self) -> float:
        return float(np.pi * self.radius ** 2)

    def __repr__(self) -> str:
        return f"Circle(center={tuple(self.center)}, radius={self.radius})"


# ======================================================================
# 3-D Primitives
# ======================================================================


class Box(Geometry):
    """Axis-aligned 3-D box.

    Args:
        Lx, Ly, Lz: Dimensions along each axis.
        origin: Bottom-left-front corner ``(x0, y0, z0)``.
    """

    def __init__(
        self,
        Lx: float = 1.0,
        Ly: float = 1.0,
        Lz: float = 1.0,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        super().__init__(dim=3)
        self.origin = np.asarray(origin, dtype=float)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.Lz = float(Lz)

    def contains(self, points: ArrayLike) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        lo = self.origin
        hi = self.origin + np.array([self.Lx, self.Ly, self.Lz])
        return np.all((pts >= lo) & (pts <= hi), axis=1)

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        return self.origin.copy(), self.origin + np.array([self.Lx, self.Ly, self.Lz])

    def area_or_volume(self) -> float:
        return self.Lx * self.Ly * self.Lz

    def __repr__(self) -> str:
        return (
            f"Box(Lx={self.Lx}, Ly={self.Ly}, Lz={self.Lz}, "
            f"origin={tuple(self.origin)})"
        )


class Cylinder(Geometry):
    """Vertical cylinder in 3-D.

    Args:
        center: ``(x, y)`` centre of the base circle.
        radius: Radius.
        z_min: Bottom elevation.
        z_max: Top elevation.
    """

    def __init__(
        self,
        center: tuple[float, float] = (0.0, 0.0),
        radius: float = 1.0,
        z_min: float = 0.0,
        z_max: float = 1.0,
    ) -> None:
        super().__init__(dim=3)
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

    def contains(self, points: ArrayLike) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        dist_xy = np.linalg.norm(pts[:, :2] - self.center, axis=1)
        return (dist_xy <= self.radius) & (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        r = self.radius
        lo = np.array([self.center[0] - r, self.center[1] - r, self.z_min])
        hi = np.array([self.center[0] + r, self.center[1] + r, self.z_max])
        return lo, hi

    def area_or_volume(self) -> float:
        return float(np.pi * self.radius ** 2 * (self.z_max - self.z_min))

    def __repr__(self) -> str:
        return (
            f"Cylinder(center={tuple(self.center)}, radius={self.radius}, "
            f"z=[{self.z_min}, {self.z_max}])"
        )


# ======================================================================
# Utility
# ======================================================================


@dataclass
class Line:
    """Line segment defined by two endpoints.

    Useful for cross-section queries and probe definitions.

    Args:
        start: Starting point.
        end: Ending point.
    """

    start: tuple[float, ...]
    end: tuple[float, ...]

    def __post_init__(self) -> None:
        self.start = tuple(float(v) for v in self.start)
        self.end = tuple(float(v) for v in self.end)
        if len(self.start) != len(self.end):
            raise ValueError("start and end must have the same dimension.")

    @property
    def dim(self) -> int:
        return len(self.start)

    def sample(self, n: int = 100) -> np.ndarray:
        """Return *n* evenly spaced points along the line.

        Returns:
            Array of shape ``(n, dim)``.
        """
        t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
        s = np.asarray(self.start)
        e = np.asarray(self.end)
        return s + t * (e - s)
