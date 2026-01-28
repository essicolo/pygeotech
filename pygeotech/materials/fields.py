"""Spatially variable property fields.

Classes
-------
RandomField
    Abstract base for random field generation.
GaussianField
    Gaussian random field with specified correlation structure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


class RandomField(ABC):
    """Abstract base class for spatially variable property fields."""

    @abstractmethod
    def generate(
        self,
        coordinates: ArrayLike,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate field realisations at given coordinates.

        Args:
            coordinates: Array of shape ``(N, dim)``.
            seed: Random seed for reproducibility.

        Returns:
            1-D array of property values at the given coordinates.
        """


@dataclass
class GaussianField(RandomField):
    """Gaussian random field with isotropic exponential covariance.

    C(r) = σ² exp(−r / l)

    Args:
        mean: Mean value of the field.
        std: Standard deviation.
        correlation_length: Spatial correlation length.
        log_transform: If ``True``, generate a log-normal field
            (useful for hydraulic conductivity).
    """

    mean: float = 0.0
    std: float = 1.0
    correlation_length: float = 10.0
    log_transform: bool = False

    def generate(
        self,
        coordinates: ArrayLike,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate a realisation via Cholesky decomposition.

        Args:
            coordinates: Shape ``(N, dim)`` point coordinates.
            seed: Random seed.

        Returns:
            1-D array of length *N*.

        Note:
            For large *N* this is O(N³).  For production use, consider
            circulant embedding or turning-bands methods.
        """
        rng = np.random.default_rng(seed)
        coords = np.atleast_2d(np.asarray(coordinates, dtype=float))
        n = len(coords)

        # Distance matrix
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))

        # Covariance matrix
        cov = self.std ** 2 * np.exp(-dist / self.correlation_length)
        # Regularise for numerical stability
        cov += 1e-10 * np.eye(n)

        L = np.linalg.cholesky(cov)
        z = rng.standard_normal(n)
        field = self.mean + L @ z

        if self.log_transform:
            field = np.exp(field)

        return field
