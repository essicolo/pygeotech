"""Point, line, and surface probes for extracting time series.

Classes
-------
PointProbe
    Sample field values at a single point.
LineProbe
    Sample field values along a line.
SurfaceProbe
    Integrate or average over a surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class PointProbe:
    """Extract field values at a specific location.

    Args:
        location: Coordinates ``(x, y)`` or ``(x, y, z)``.
    """

    location: tuple[float, ...]

    def sample(self, mesh: Any, field: np.ndarray) -> float:
        """Sample the field at this probe location.

        Uses nearest-node interpolation.

        Args:
            mesh: Computational mesh.
            field: Nodal field values.

        Returns:
            Interpolated value.
        """
        loc = np.asarray(self.location, dtype=float)
        dist = np.linalg.norm(mesh.nodes - loc, axis=1)
        return float(field[np.argmin(dist)])


@dataclass
class LineProbe:
    """Sample field values along a line.

    Args:
        start: Start point.
        end: End point.
        n_points: Number of sample points.
    """

    start: tuple[float, ...]
    end: tuple[float, ...]
    n_points: int = 100

    def sample(self, mesh: Any, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample the field along this line.

        Args:
            mesh: Computational mesh.
            field: Nodal field values.

        Returns:
            Tuple ``(distances, values)`` where *distances* is the
            arc-length coordinate along the line.
        """
        s = np.asarray(self.start, dtype=float)
        e = np.asarray(self.end, dtype=float)
        t = np.linspace(0.0, 1.0, self.n_points)
        points = s + np.outer(t, e - s)
        distances = t * np.linalg.norm(e - s)

        values = np.empty(self.n_points)
        for i, pt in enumerate(points):
            dist = np.linalg.norm(mesh.nodes - pt, axis=1)
            values[i] = field[np.argmin(dist)]

        return distances, values


@dataclass
class SurfaceProbe:
    """Average or integrate a field over a surface region.

    Args:
        locator: A :class:`~pygeotech.boundaries.locators.BoundaryLocator`.
    """

    locator: Any

    def average(self, mesh: Any, field: np.ndarray) -> float:
        """Compute the average field value over the surface.

        Args:
            mesh: Computational mesh.
            field: Nodal field values.

        Returns:
            Mean value on the surface.
        """
        b_idx = mesh.boundary_nodes()
        b_coords = mesh.nodes[b_idx]
        mask = self.locator(b_coords)
        if not mask.any():
            return 0.0
        return float(field[b_idx[mask]].mean())
