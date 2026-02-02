"""Spatial interpolation of stratigraphic boundaries.

Interpolates layer boundary surfaces from borehole data using:

* **RBF** — :class:`scipy.interpolate.RBFInterpolator`
* **Linear** — :class:`scipy.interpolate.LinearNDInterpolator` (2-D)
  or :func:`scipy.interpolate.interp1d` (1-D)
* **Cubic** — :class:`scipy.interpolate.CloughTocher2DInterpolator`
  (2-D) or :class:`scipy.interpolate.CubicSpline` (1-D)
* **Kriging** — built-in ordinary kriging with auto-fitted variogram

Non-crossing constraints:

* **sequential** — top-down clipping: each surface is clipped so it
  never rises above the surface directly above it.
* ``None`` — no constraint (surfaces may cross).

References:

* Hillier et al. (2014) — RBF with inequality constraints
* Geostatistics Lessons — stratigraphic coordinate transformation
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.spatial.distance import cdist

from pygeotech.stratigraphy.borehole import BoreholeSet


class StratigraphicModel:
    """3-D stratigraphic model built from borehole data.

    After calling :meth:`interpolate`, the model provides:

    * :meth:`unit_at` — query the unit at any (x, [y,] z) point
    * :meth:`evaluate_surfaces` — evaluate all boundary elevations
    * :meth:`cross_section` — extract a 2-D section along a line
    * :meth:`tag_mesh` — assign unit tags to a pygeotech mesh

    Args:
        boreholes: Borehole dataset with assigned unit labels.
    """

    def __init__(self, boreholes: BoreholeSet) -> None:
        self.boreholes = boreholes
        self._column = boreholes.stratigraphic_column()
        self._surface_order: list[str] = []
        self._surfaces: dict[str, Callable] = {}
        self._interpolated = False

    @property
    def column(self) -> list[str]:
        """Stratigraphic column from top to bottom."""
        return list(self._column)

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate(
        self,
        method: str = "rbf",
        kernel: str = "thin_plate_spline",
        smoothing: float = 0.0,
        non_crossing: str | None = "sequential",
        min_thickness: float = 0.01,
        variogram: str = "exponential",
    ) -> None:
        """Interpolate all stratigraphic boundary surfaces.

        Args:
            method: ``"rbf"``, ``"linear"``, ``"cubic"``, or ``"kriging"``.
            kernel: RBF kernel (see :class:`scipy.interpolate.RBFInterpolator`).
            smoothing: Smoothing parameter for RBF.
            non_crossing: ``"sequential"`` or ``None``.
            min_thickness: Minimum layer thickness for non-crossing (m).
            variogram: Variogram model for kriging
                (``"exponential"``, ``"spherical"``, ``"gaussian"``).
        """
        raw = self.boreholes.all_interfaces()
        is_2d = self.boreholes.dim == 2

        # Build interpolators for each surface
        for name, pts in raw.items():
            xy = pts[:, 0:1] if is_2d else pts[:, :2]
            z = pts[:, 2]

            if len(z) < 2:
                # Not enough data — constant surface
                self._surfaces[name] = _ConstantSurface(float(z[0]) if len(z) else 0.0)
                continue

            interp = _build_interpolator(
                xy, z,
                method=method,
                kernel=kernel,
                smoothing=smoothing,
                variogram=variogram,
                is_2d=is_2d,
            )
            self._surfaces[name] = interp

        # Determine ordered surface list
        column = self._column
        surface_order = ["topography"]
        for i in range(len(column) - 1):
            surface_order.append(f"{column[i]}/{column[i + 1]}")
        surface_order.append("base")
        self._surface_order = [s for s in surface_order if s in self._surfaces]

        # Enforce non-crossing
        if non_crossing == "sequential":
            self._enforce_sequential(min_thickness)

        self._interpolated = True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def unit_at(self, *coords: float) -> str | None:
        """Return the stratigraphic unit at a point.

        Call as ``unit_at(x, z)`` for 2-D models or
        ``unit_at(x, y, z)`` for 3-D models.
        """
        self._check_interpolated()

        if len(coords) == 2:
            x, z = coords
            query = np.array([[x]])
        elif len(coords) == 3:
            x, y, z = coords
            query = np.array([[x, y]])
        else:
            raise ValueError("Expected 2 or 3 coordinates")

        elevations = self._evaluate_at(query)
        column = self._column

        for i, unit in enumerate(column):
            if i < len(self._surface_order) - 1:
                z_top = float(elevations[self._surface_order[i]][0])
                z_bot = float(elevations[self._surface_order[i + 1]][0])
                if z_bot <= z <= z_top:
                    return unit

        return None

    def evaluate_surfaces(
        self, x: np.ndarray, y: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Evaluate all boundary surfaces at given plan locations.

        Args:
            x: x-coordinates, shape ``(n,)``.
            y: y-coordinates, shape ``(n,)`` (3-D only).

        Returns:
            Dict mapping surface names to elevation arrays.
        """
        self._check_interpolated()

        x = np.atleast_1d(np.asarray(x, dtype=float))
        if y is not None:
            y = np.atleast_1d(np.asarray(y, dtype=float))
            query = np.column_stack([x, y])
        else:
            query = x.reshape(-1, 1)

        return self._evaluate_at(query)

    def cross_section(
        self,
        p0: tuple[float, float],
        p1: tuple[float, float],
        n_points: int = 100,
    ) -> dict[str, np.ndarray]:
        """Evaluate stratigraphy along a cross-section line.

        Args:
            p0: Start point (x0, y0).
            p1: End point (x1, y1).
            n_points: Number of sample points.

        Returns:
            Dict with ``"distance"``, ``"x"``, ``"y"`` arrays and one
            elevation array per surface.
        """
        self._check_interpolated()

        xs = np.linspace(p0[0], p1[0], n_points)
        ys = np.linspace(p0[1], p1[1], n_points)
        dist = np.sqrt((xs - xs[0]) ** 2 + (ys - ys[0]) ** 2)

        query = np.column_stack([xs, ys])
        if self.boreholes.dim == 2:
            query = xs.reshape(-1, 1)

        elevations = self._evaluate_at(query)
        result: dict[str, np.ndarray] = {
            "distance": dist,
            "x": xs,
            "y": ys,
        }
        result.update(elevations)
        return result

    def tag_mesh(self, mesh: Any) -> Any:
        """Assign stratigraphic unit tags to a pygeotech mesh.

        Each cell is tagged based on its centroid elevation relative
        to the interpolated surfaces.

        Args:
            mesh: A :class:`~pygeotech.geometry.mesh.Mesh` instance.

        Returns:
            The mesh (modified in place) with updated ``cell_tags``
            and ``subdomain_map``.
        """
        self._check_interpolated()

        centers = mesh.cell_centers()
        unit_map = {name: i for i, name in enumerate(self._column)}
        tags = np.full(mesh.n_cells, -1, dtype=int)

        # Build query coordinates
        if mesh.dim == 2:
            # 2-D mesh: columns are (x, z)
            if self.boreholes.dim == 2:
                query = centers[:, 0:1]
            else:
                query = centers[:, :2]
            z_vals = centers[:, -1]
        else:
            query = centers[:, :2]
            z_vals = centers[:, 2]

        # Evaluate all surfaces at cell centers
        elevations = self._evaluate_at(query)
        surface_z = np.column_stack([
            elevations[s] for s in self._surface_order
        ])  # shape (n_cells, n_surfaces)

        # Assign each cell to the unit whose top/bottom bounds contain it
        for ui, unit in enumerate(self._column):
            if ui + 1 < len(self._surface_order):
                z_top = surface_z[:, ui]
                z_bot = surface_z[:, ui + 1]
                mask = (z_vals <= z_top) & (z_vals >= z_bot) & (tags == -1)
                tags[mask] = unit_map[unit]

        mesh.cell_tags = tags
        mesh.subdomain_map = {name: idx for name, idx in unit_map.items()}
        return mesh

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate_at(self, query: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate all surfaces at query locations."""
        result: dict[str, np.ndarray] = {}
        for name in self._surface_order:
            interp = self._surfaces[name]
            result[name] = np.asarray(interp(query)).ravel()
        return result

    def _enforce_sequential(self, min_thickness: float) -> None:
        """Wrap interpolators with top-down clipping."""
        for i in range(1, len(self._surface_order)):
            above = self._surfaces[self._surface_order[i - 1]]
            current = self._surfaces[self._surface_order[i]]
            self._surfaces[self._surface_order[i]] = _ClippedSurface(
                current, above, min_thickness
            )

    def _check_interpolated(self) -> None:
        if not self._interpolated:
            raise RuntimeError(
                "Call .interpolate() before querying the model."
            )

    def __repr__(self) -> str:
        state = "interpolated" if self._interpolated else "not interpolated"
        return (
            f"StratigraphicModel(units={self._column}, {state})"
        )


# ======================================================================
# Interpolator builders
# ======================================================================

def _build_interpolator(
    xy: np.ndarray,
    z: np.ndarray,
    method: str,
    kernel: str,
    smoothing: float,
    variogram: str,
    is_2d: bool,
) -> Callable:
    """Create an interpolator callable: f(xy) → z."""
    if method == "rbf":
        from scipy.interpolate import RBFInterpolator
        return RBFInterpolator(xy, z, kernel=kernel, smoothing=smoothing)

    elif method == "linear":
        if is_2d:
            from scipy.interpolate import interp1d
            order = np.argsort(xy.ravel())
            return _Interp1DWrapper(
                interp1d(
                    xy.ravel()[order], z[order],
                    kind="linear", fill_value="extrapolate",
                )
            )
        else:
            from scipy.interpolate import LinearNDInterpolator
            return _NDWrapper(LinearNDInterpolator(xy, z), z.mean())

    elif method == "cubic":
        if is_2d:
            from scipy.interpolate import CubicSpline
            order = np.argsort(xy.ravel())
            return _Interp1DWrapper(
                CubicSpline(xy.ravel()[order], z[order], extrapolate=True)
            )
        else:
            from scipy.interpolate import CloughTocher2DInterpolator
            return _NDWrapper(CloughTocher2DInterpolator(xy, z), z.mean())

    elif method == "kriging":
        return OrdinaryKriging(xy, z, variogram=variogram)

    else:
        raise ValueError(f"Unknown interpolation method: {method!r}")


# ======================================================================
# Ordinary Kriging (built-in, scipy-based)
# ======================================================================

class OrdinaryKriging:
    """Simple ordinary kriging with auto-fitted variogram.

    Uses :func:`scipy.spatial.distance.cdist` for distance computation
    and :func:`numpy.linalg.solve` for the kriging system.

    Args:
        xy: Data locations, shape ``(n, d)``.
        z: Data values, shape ``(n,)``.
        variogram: Model type — ``"exponential"``, ``"spherical"``,
            or ``"gaussian"``.
        range_: Variogram range.  Auto-fitted if ``None``.
        sill: Variogram sill.  Auto-fitted if ``None``.
        nugget: Variogram nugget.
    """

    def __init__(
        self,
        xy: np.ndarray,
        z: np.ndarray,
        variogram: str = "exponential",
        range_: float | None = None,
        sill: float | None = None,
        nugget: float = 0.0,
    ) -> None:
        self.xy = np.atleast_2d(xy)
        self.z = np.asarray(z, dtype=float)
        self.variogram_model = variogram
        self.nugget = nugget
        n = len(z)

        # Auto-fit range and sill from data
        dist_matrix = cdist(self.xy, self.xy)
        if range_ is None:
            range_ = float(dist_matrix.max()) / 3.0
        if sill is None:
            sill = float(np.var(z)) if np.var(z) > 0 else 1.0
        self.range_ = range_
        self.sill = sill

        # Pre-compute kriging matrix (n+1 × n+1)
        gamma = self._gamma(dist_matrix)
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = gamma
        A[n, :n] = 1.0
        A[:n, n] = 1.0
        self._A = A

    def _gamma(self, h: np.ndarray) -> np.ndarray:
        """Evaluate the variogram model at lag distances *h*."""
        s, r, c0 = self.sill, self.range_, self.nugget
        if self.variogram_model == "exponential":
            return c0 + (s - c0) * (1.0 - np.exp(-3.0 * h / r))
        elif self.variogram_model == "spherical":
            hr = np.minimum(h / r, 1.0)
            return c0 + (s - c0) * (1.5 * hr - 0.5 * hr ** 3)
        elif self.variogram_model == "gaussian":
            return c0 + (s - c0) * (1.0 - np.exp(-3.0 * (h / r) ** 2))
        else:
            raise ValueError(f"Unknown variogram: {self.variogram_model!r}")

    def __call__(self, xy_new: np.ndarray) -> np.ndarray:
        xy_new = np.atleast_2d(xy_new)
        dist = cdist(xy_new, self.xy)
        gamma_new = self._gamma(dist)

        n = len(self.z)
        m = len(xy_new)
        B = np.zeros((n + 1, m))
        B[:n, :] = gamma_new.T
        B[n, :] = 1.0

        weights = np.linalg.solve(self._A, B)  # (n+1, m)
        return weights[:n, :].T @ self.z  # (m,)

    def __repr__(self) -> str:
        return (
            f"OrdinaryKriging(variogram={self.variogram_model!r}, "
            f"range={self.range_:.1f}, sill={self.sill:.3f})"
        )


# ======================================================================
# Wrapper utilities
# ======================================================================

class _ClippedSurface:
    """Surface wrapper that clips to stay below a ceiling surface."""

    def __init__(
        self,
        surface: Callable,
        ceiling: Callable,
        min_gap: float,
    ) -> None:
        self._surface = surface
        self._ceiling = ceiling
        self._min_gap = min_gap

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        z = np.asarray(self._surface(xy)).ravel()
        z_ceil = np.asarray(self._ceiling(xy)).ravel()
        return np.minimum(z, z_ceil - self._min_gap)


class _ConstantSurface:
    """Constant-elevation surface (fallback for single-point data)."""

    def __init__(self, z: float) -> None:
        self._z = z

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        return np.full(np.atleast_2d(xy).shape[0], self._z)


class _Interp1DWrapper:
    """Wrap a 1-D interpolator to accept (n, 1) arrays."""

    def __init__(self, interp: Callable) -> None:
        self._interp = interp

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(self._interp(np.asarray(xy).ravel()))


class _NDWrapper:
    """Wrap an ND interpolator, replacing NaN with a fallback value."""

    def __init__(self, interp: Callable, fallback: float) -> None:
        self._interp = interp
        self._fallback = fallback

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        result = np.asarray(self._interp(xy))
        result = np.where(np.isnan(result), self._fallback, result)
        return result
