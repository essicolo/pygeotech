"""Abstract solver interface.

All solver backends implement the :class:`Solver` interface, which
provides a uniform ``solve()`` method regardless of the underlying
numerical method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Solution:
    """Container for solver output.

    Attributes:
        fields: Dictionary of solution fields, e.g. ``{"H": array}``.
        mesh: The mesh used for the solution.
        times: Time values (for transient problems).
        field_history: Time series of field snapshots (transient).
    """

    def __init__(
        self,
        fields: dict[str, np.ndarray],
        mesh: Any,
        times: np.ndarray | None = None,
        field_history: dict[str, list[np.ndarray]] | None = None,
    ) -> None:
        self.fields = fields
        self.mesh = mesh
        self.times = times
        self.field_history = field_history or {}

    def __getitem__(self, key: str) -> np.ndarray:
        return self.fields[key]

    def compute_velocity(self) -> np.ndarray:
        """Compute Darcy velocity from the head field.

        v = -K grad(H)

        Returns:
            Velocity array of shape ``(n_cells, dim)``.
        """
        from pygeotech.postprocess.fields import compute_velocity
        return compute_velocity(self)

    def integrate_flux(self, locator: Any) -> float:
        """Integrate flux across a boundary defined by *locator*.

        Args:
            locator: A boundary locator specifying the integration surface.

        Returns:
            Total flux (m³/s for 2-D, m²/s per unit depth).
        """
        from pygeotech.postprocess.integrals import integrate_flux
        return integrate_flux(self, locator)

    def plot(
        self,
        field: str = "H",
        time: float | None = None,
        contours: int = 20,
        streamlines: bool = False,
        colorbar: bool = True,
        title: str | None = None,
        ax: Any = None,
    ) -> Any:
        """Quick 2-D field plot.

        Args:
            field: Field name to plot.
            time: Time for transient solutions (s).
            contours: Number of contour levels.
            streamlines: Overlay streamlines (requires velocity).
            colorbar: Show colour bar.
            title: Plot title.
            ax: Matplotlib axes (creates new figure if None).

        Returns:
            Matplotlib axes.
        """
        from pygeotech.visualization.plot2d import plot_field
        data = self._field_at_time(field, time)
        return plot_field(
            self.mesh, data,
            contours=contours,
            streamlines=streamlines,
            colorbar=colorbar,
            title=title or field,
            ax=ax,
        )

    def plot_cross_section(
        self,
        y: float | None = None,
        x: float | None = None,
        fields: list[str] | None = None,
    ) -> Any:
        """Plot field values along a cross-section.

        Args:
            y: Y-coordinate for a horizontal cross-section.
            x: X-coordinate for a vertical cross-section.
            fields: Field names to plot.

        Returns:
            Matplotlib axes.
        """
        from pygeotech.visualization.plot2d import plot_contours
        fields = fields or list(self.fields.keys())
        return plot_contours(self, y=y, x=x, fields=fields)

    def export_vtk(self, filename: str) -> None:
        """Export solution to VTK format.

        Args:
            filename: Output file path.
        """
        from pygeotech.postprocess.export import export_vtk
        export_vtk(self, filename)

    def export_csv(
        self,
        filename: str,
        along: Any = None,
    ) -> None:
        """Export solution to CSV.

        Args:
            filename: Output file path.
            along: Optional Line for sampling.
        """
        from pygeotech.postprocess.export import export_csv
        export_csv(self, filename, along=along)

    def _field_at_time(self, field: str, time: float | None) -> np.ndarray:
        """Retrieve field data, optionally at a specific time."""
        if time is None or self.times is None:
            return self.fields[field]
        idx = int(np.argmin(np.abs(self.times - time)))
        return self.field_history[field][idx]

    def __repr__(self) -> str:
        fields = list(self.fields.keys())
        n = self.mesh.n_nodes if self.mesh else "?"
        return f"Solution(fields={fields}, n_nodes={n})"


class Solver(ABC):
    """Abstract solver backend.

    Subclasses implement specific numerical methods (FVM, FEM, PINN)
    while exposing a uniform interface.
    """

    @abstractmethod
    def solve(
        self,
        physics: Any,
        boundary_conditions: Any,
        time: Any | None = None,
        initial_condition: dict[str, float | np.ndarray] | None = None,
        **kwargs: Any,
    ) -> Solution:
        """Solve the problem.

        Args:
            physics: A :class:`~pygeotech.physics.base.PhysicsModule` or
                :class:`~pygeotech.coupling.base.CoupledProblem`.
            boundary_conditions: Boundary conditions (single
                :class:`BoundaryConditions` or dict for coupled problems).
            time: Time stepper (for transient problems).
            initial_condition: Initial values keyed by field name.
            **kwargs: Backend-specific options.

        Returns:
            A :class:`Solution` object.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
