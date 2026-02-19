"""Slip surface definitions.

A slip surface describes the failure plane through a slope.  Two types
are provided:

CircularSurface
    Circular arc defined by centre and radius (standard for Bishop).
PolylineSurface
    Arbitrary non-circular surface defined by vertices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CircularSurface:
    """Circular slip surface.

    Args:
        xc: x-coordinate of the arc centre.
        yc: y-coordinate of the arc centre.
        radius: Arc radius (m).
    """

    xc: float
    yc: float
    radius: float

    def y_at(self, x: float) -> float | None:
        """Return the y-coordinate of the *lower* arc at *x*.

        Returns ``None`` if *x* is outside the arc span.
        """
        dx = x - self.xc
        if abs(dx) > self.radius:
            return None
        return self.yc - np.sqrt(self.radius ** 2 - dx ** 2)

    def base_angle(self, x: float) -> float:
        """Inclination angle α of the arc base at *x* (radians).

        Positive when the base slopes upward to the right.
        dy/dx = (x - xc) / sqrt(R² - (x - xc)²)
        α = arctan(dy/dx)
        """
        dx = x - self.xc
        R2 = self.radius ** 2
        denom = R2 - dx ** 2
        if denom <= 0:
            return np.pi / 2 if dx > 0 else -np.pi / 2
        return np.arctan(dx / np.sqrt(denom))

    def entry_exit(self, profile) -> tuple[float, float] | None:
        """Find entry (left) and exit (right) x of the arc with the
        ground surface.

        Args:
            profile: A :class:`SlopeProfile` instance.

        Returns:
            Tuple ``(x_entry, x_exit)`` or ``None`` if no valid
            intersection.
        """
        xs = np.linspace(
            max(profile.x_min, self.xc - self.radius),
            min(profile.x_max, self.xc + self.radius),
            500,
        )
        diff = []
        for x in xs:
            y_surface = profile.surface_elevation(x)
            y_arc = self.y_at(x)
            if y_arc is None:
                diff.append(np.nan)
            else:
                diff.append(y_surface - y_arc)

        diff = np.array(diff)
        valid = ~np.isnan(diff)
        if not valid.any():
            return None

        # Find sign changes (surface crosses arc)
        crossings = []
        for i in range(len(diff) - 1):
            if valid[i] and valid[i + 1] and diff[i] * diff[i + 1] < 0:
                # Linear interpolation for crossing x
                x_cross = xs[i] - diff[i] * (xs[i + 1] - xs[i]) / (diff[i + 1] - diff[i])
                crossings.append(x_cross)

        if len(crossings) < 2:
            return None

        return (crossings[0], crossings[-1])


@dataclass
class PolylineSurface:
    """Non-circular slip surface defined by a polyline.

    Args:
        vertices: Slip surface vertices ``[(x, y), ...]`` from left
            (entry) to right (exit).
    """

    vertices: np.ndarray

    def __init__(self, vertices: Sequence[tuple[float, float]]) -> None:
        self.vertices = np.array(vertices, dtype=float)

    def y_at(self, x: float) -> float | None:
        """Interpolate slip surface y at *x*."""
        vx = self.vertices[:, 0]
        if x < vx[0] or x > vx[-1]:
            return None
        return float(np.interp(x, vx, self.vertices[:, 1]))

    def base_angle(self, x: float) -> float:
        """Base angle at *x* by central-difference of the polyline."""
        vx = self.vertices[:, 0]
        vy = self.vertices[:, 1]
        if x <= vx[0]:
            dx = vx[1] - vx[0]
            dy = vy[1] - vy[0]
        elif x >= vx[-1]:
            dx = vx[-1] - vx[-2]
            dy = vy[-1] - vy[-2]
        else:
            idx = np.searchsorted(vx, x) - 1
            idx = max(0, min(idx, len(vx) - 2))
            dx = vx[idx + 1] - vx[idx]
            dy = vy[idx + 1] - vy[idx]
        return np.arctan2(dy, dx)

    def entry_exit(self, profile) -> tuple[float, float] | None:
        """Return entry and exit x-coordinates."""
        vx = self.vertices[:, 0]
        if len(vx) < 2:
            return None
        return (float(vx[0]), float(vx[-1]))
