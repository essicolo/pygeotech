"""Stratified geology: horizontal and dipping layers.

Classes
-------
LayeredProfile
    A sequence of geological layers (horizontal or dipping) that can be
    converted into subdomains on a mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class Layer:
    """A single geological layer.

    Args:
        name: Identifier for the layer.
        top: Top elevation (or callable for dipping surfaces).
        bottom: Bottom elevation (or callable for dipping surfaces).
    """

    name: str
    top: float | callable
    bottom: float | callable

    def elevation_top(self, x: ArrayLike) -> np.ndarray:
        """Evaluate top elevation at horizontal coordinate(s) *x*."""
        if callable(self.top):
            return np.asarray(self.top(np.asarray(x, dtype=float)), dtype=float)
        return np.full_like(np.asarray(x, dtype=float), self.top)

    def elevation_bottom(self, x: ArrayLike) -> np.ndarray:
        """Evaluate bottom elevation at horizontal coordinate(s) *x*."""
        if callable(self.bottom):
            return np.asarray(self.bottom(np.asarray(x, dtype=float)), dtype=float)
        return np.full_like(np.asarray(x, dtype=float), self.bottom)


class LayeredProfile:
    """Stratified geological profile.

    Build a profile by adding layers from top to bottom.

    Example::

        profile = LayeredProfile()
        profile.add("topsoil", top=10.0, bottom=8.0)
        profile.add("clay", top=8.0, bottom=3.0)
        profile.add("bedrock", top=3.0, bottom=0.0)
    """

    def __init__(self) -> None:
        self.layers: list[Layer] = []

    def add(
        self,
        name: str,
        top: float | callable,
        bottom: float | callable,
    ) -> None:
        """Append a layer to the profile.

        Args:
            name: Unique layer name.
            top: Top elevation (constant or ``f(x)`` for dipping).
            bottom: Bottom elevation (constant or ``f(x)``).
        """
        self.layers.append(Layer(name=name, top=top, bottom=bottom))

    def identify(self, points: ArrayLike) -> np.ndarray:
        """Return the layer name for each point.

        Args:
            points: Array of shape ``(N, dim)`` — uses the last coordinate as
                elevation.

        Returns:
            Array of layer names (object dtype) of shape ``(N,)``.  Points
            outside all layers receive the value ``""``.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        z = pts[:, -1]
        # Use the first horizontal coordinate for dipping evaluations
        x = pts[:, 0]
        result = np.full(len(pts), "", dtype=object)
        for layer in self.layers:
            zt = layer.elevation_top(x)
            zb = layer.elevation_bottom(x)
            mask = (z <= zt) & (z >= zb) & (result == "")
            result[mask] = layer.name
        return result

    def __repr__(self) -> str:
        names = [l.name for l in self.layers]
        return f"LayeredProfile(layers={names})"
