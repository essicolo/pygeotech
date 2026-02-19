"""Internal boundary conditions (sources and sinks).

Classes
-------
InternalBoundary
    Abstract base for internal BCs.
Well
    Point source/sink (injection or extraction well).
Drain
    Line drain with specified head or flux.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


class InternalBoundary(ABC):
    """Abstract base class for internal sources and sinks."""

    @abstractmethod
    def source_term(
        self,
        coords: np.ndarray,
        t: float | None = None,
    ) -> np.ndarray:
        """Evaluate source/sink contribution at given coordinates.

        Args:
            coords: Cell centre coordinates, shape ``(N, dim)``.
            t: Current simulation time (s).

        Returns:
            Source/sink rate array, shape ``(N,)``.
            Positive = injection, negative = extraction.
        """


@dataclass
class Well(InternalBoundary):
    """Point source or sink (pumping/injection well).

    Args:
        location: Well coordinates ``(x, y)`` or ``(x, y, z)``.
        rate: Volumetric flow rate (m³/s).  Positive = injection.
        radius: Influence radius for smearing the source over nearby cells.
    """

    location: tuple[float, ...]
    rate: float = 0.0
    radius: float = 1.0

    def source_term(
        self,
        coords: np.ndarray,
        t: float | None = None,
    ) -> np.ndarray:
        loc = np.asarray(self.location, dtype=float)
        dist = np.linalg.norm(coords - loc, axis=1)
        weight = np.where(dist <= self.radius, 1.0, 0.0)
        total = weight.sum()
        if total > 0:
            weight /= total
        return self.rate * weight


@dataclass
class Drain(InternalBoundary):
    """Line drain (e.g. French drain, tile drain).

    The drain is represented as a line segment from *start* to *end*.
    Cells within *radius* of the segment receive a sink proportional
    to the head excess above *drain_level*.

    Args:
        start: Start coordinate.
        end: End coordinate.
        drain_level: Drain elevation (m).  Flow occurs when H > drain_level.
        conductance: Drain conductance (m²/s).
        radius: Influence radius (m).
    """

    start: tuple[float, ...]
    end: tuple[float, ...]
    drain_level: float = 0.0
    conductance: float = 1e-4
    radius: float = 1.0

    def source_term(
        self,
        coords: np.ndarray,
        t: float | None = None,
    ) -> np.ndarray:
        """Drain acts as a sink where head exceeds drain_level.

        Returns a *template* source array; the actual head values must be
        supplied by the solver at runtime.  This method returns the
        spatial mask × conductance as a rate per unit head excess.
        """
        a = np.asarray(self.start, dtype=float)
        b = np.asarray(self.end, dtype=float)
        ab = b - a
        ab_len2 = np.dot(ab, ab)
        if ab_len2 < 1e-30:
            dist = np.linalg.norm(coords - a, axis=1)
        else:
            ap = coords - a
            t_param = np.clip(ap @ ab / ab_len2, 0.0, 1.0)
            proj = a + np.outer(t_param, ab)
            dist = np.linalg.norm(coords - proj, axis=1)

        mask = dist <= self.radius
        # Negative = extraction (sink)
        return np.where(mask, -self.conductance, 0.0)
