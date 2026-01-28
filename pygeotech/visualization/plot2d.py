"""2-D plotting utilities.

Functions
---------
plot_field
    Plot a scalar field on a triangular mesh.
plot_contours
    Plot cross-section profiles.
plot_streamlines
    Overlay streamlines on a field plot.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def plot_field(
    mesh: Any,
    field: np.ndarray,
    contours: int = 20,
    streamlines: bool = False,
    colorbar: bool = True,
    title: str = "",
    ax: Any = None,
    cmap: str = "viridis",
) -> Any:
    """Plot a scalar field on a 2-D triangular mesh.

    Args:
        mesh: Computational mesh.
        field: Nodal field values, shape ``(n_nodes,)``.
        contours: Number of contour levels.
        streamlines: If True, overlay velocity streamlines.
        colorbar: Show colour bar.
        title: Plot title.
        ax: Matplotlib axes (creates new figure if None).
        cmap: Matplotlib colour map name.

    Returns:
        Matplotlib axes.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    nodes = mesh.nodes
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], mesh.cells)

    cs = ax.tricontourf(triang, field, levels=contours, cmap=cmap)
    if colorbar:
        plt.colorbar(cs, ax=ax, label=title)

    ax.tricontour(triang, field, levels=contours, colors="k", linewidths=0.3)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    if streamlines:
        _add_streamlines(mesh, field, ax)

    return ax


def _add_streamlines(mesh: Any, H: np.ndarray, ax: Any) -> None:
    """Overlay Darcy velocity streamlines."""
    from pygeotech.postprocess.fields import compute_gradient

    grad_H = compute_gradient(mesh, H)
    centers = mesh.cell_centers()

    # Interpolate to regular grid for streamplot
    import matplotlib.pyplot as plt

    x_min, x_max = mesh.nodes[:, 0].min(), mesh.nodes[:, 0].max()
    y_min, y_max = mesh.nodes[:, 1].min(), mesh.nodes[:, 1].max()
    nx, ny = 50, 20
    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_min, y_max, ny)
    Xi, Yi = np.meshgrid(xi, yi)

    from scipy.interpolate import griddata

    vx = griddata(centers, -grad_H[:, 0], (Xi, Yi), method="linear", fill_value=0.0)
    vy = griddata(centers, -grad_H[:, 1], (Xi, Yi), method="linear", fill_value=0.0)

    ax.streamplot(xi, yi, vx, vy, color="white", linewidth=0.5, density=1.5)


def plot_contours(
    solution: Any,
    y: float | None = None,
    x: float | None = None,
    fields: list[str] | None = None,
) -> Any:
    """Plot field values along a cross-section.

    Args:
        solution: Solution object.
        y: Y-coordinate for a horizontal cross-section.
        x: X-coordinate for a vertical cross-section.
        fields: Field names to plot.

    Returns:
        Matplotlib axes.
    """
    import matplotlib.pyplot as plt

    fields = fields or list(solution.fields.keys())
    mesh = solution.mesh
    nodes = mesh.nodes

    fig, axes = plt.subplots(len(fields), 1, figsize=(10, 3 * len(fields)), squeeze=False)

    for i, fname in enumerate(fields):
        ax = axes[i, 0]
        field = solution.fields[fname]

        if y is not None:
            # Horizontal cross-section: find nodes near y
            tol = (nodes[:, 1].max() - nodes[:, 1].min()) / 50
            mask = np.abs(nodes[:, 1] - y) < tol
            ax.plot(nodes[mask, 0], field[mask], "b-", linewidth=1.5)
            ax.set_xlabel("x (m)")
        elif x is not None:
            tol = (nodes[:, 0].max() - nodes[:, 0].min()) / 50
            mask = np.abs(nodes[:, 0] - x) < tol
            ax.plot(nodes[mask, 1], field[mask], "b-", linewidth=1.5)
            ax.set_xlabel("y (m)")

        ax.set_ylabel(fname)
        ax.set_title(f"{fname} cross-section")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return axes


def plot_streamlines(
    solution: Any,
    ax: Any = None,
) -> Any:
    """Standalone streamline plot.

    Args:
        solution: Solution with "H" field.
        ax: Matplotlib axes.

    Returns:
        Matplotlib axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    _add_streamlines(solution.mesh, solution.fields["H"], ax)
    ax.set_aspect("equal")
    ax.set_title("Streamlines")
    return ax
