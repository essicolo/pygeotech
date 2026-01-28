"""3-D visualization using PyVista.

Functions
---------
plot_3d
    Interactive 3-D field plot.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def plot_3d(
    solution: Any,
    field: str = "H",
    show_edges: bool = True,
    cmap: str = "viridis",
    **kwargs: Any,
) -> Any:
    """Interactive 3-D field visualisation using PyVista.

    Args:
        solution: Solution object.
        field: Field name to display.
        show_edges: Show mesh edges.
        cmap: Colour map name.
        **kwargs: Additional keyword arguments passed to PyVista plotter.

    Returns:
        PyVista plotter object.

    Raises:
        ImportError: If PyVista is not installed.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "pyvista is required for 3-D visualisation.  "
            "Install with: pip install pyvista"
        ) from exc

    mesh = solution.mesh
    nodes = mesh.nodes
    cells = mesh.cells

    # Build PyVista mesh
    if nodes.shape[1] == 2:
        points = np.column_stack([nodes, np.zeros(len(nodes))])
    else:
        points = nodes

    if cells.shape[1] == 3:
        # Triangles: prepend count (3) per cell
        faces = np.column_stack([
            np.full(len(cells), 3, dtype=int),
            cells,
        ]).ravel()
        pv_mesh = pv.PolyData(points, faces)
    elif cells.shape[1] == 4:
        faces = np.column_stack([
            np.full(len(cells), 4, dtype=int),
            cells,
        ]).ravel()
        pv_mesh = pv.PolyData(points, faces)
    else:
        raise ValueError(f"Unsupported cell type: {cells.shape[1]} nodes per cell.")

    pv_mesh.point_data[field] = solution.fields[field]

    plotter = pv.Plotter()
    plotter.add_mesh(
        pv_mesh,
        scalars=field,
        show_edges=show_edges,
        cmap=cmap,
    )
    plotter.show()
    return plotter
