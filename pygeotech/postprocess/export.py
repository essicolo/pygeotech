"""Export solution data to various file formats.

Functions
---------
export_vtk
    Export to VTK (legacy or XML) format.
export_xdmf
    Export to XDMF/HDF5 format.
export_csv
    Export field data to CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def export_vtk(solution: Any, filename: str) -> None:
    """Export the solution to VTK format.

    Requires *meshio*.

    Args:
        solution: Solution object.
        filename: Output file path.
    """
    try:
        import meshio
    except ImportError as exc:
        raise ImportError(
            "meshio is required for VTK export.  "
            "Install with: pip install meshio"
        ) from exc

    mesh = solution.mesh
    cells = [("triangle", mesh.cells)] if mesh.cells.shape[1] == 3 else [("quad", mesh.cells)]

    points = mesh.nodes
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])

    point_data = {}
    for name, field in solution.fields.items():
        if field.shape == (mesh.n_nodes,):
            point_data[name] = field

    m = meshio.Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
    )
    m.write(filename)


def export_xdmf(solution: Any, filename: str) -> None:
    """Export the solution to XDMF format.

    Args:
        solution: Solution object.
        filename: Output file path.
    """
    export_vtk(solution, filename)  # meshio handles XDMF too


def export_csv(
    solution: Any,
    filename: str,
    along: Any = None,
) -> None:
    """Export solution data to CSV.

    Args:
        solution: Solution object.
        filename: Output file path.
        along: Optional :class:`~pygeotech.geometry.primitives.Line` for
            sampling along a cross-section.
    """
    mesh = solution.mesh

    if along is not None:
        # Sample along a line
        points = along.sample(100)
        header_parts = ["x", "y"] if mesh.dim == 2 else ["x", "y", "z"]
        data_cols = [points[:, i] for i in range(mesh.dim)]

        for name, field in solution.fields.items():
            values = np.empty(len(points))
            for ip, pt in enumerate(points):
                dist = np.linalg.norm(mesh.nodes - pt, axis=1)
                values[ip] = field[np.argmin(dist)]
            data_cols.append(values)
            header_parts.append(name)

        data = np.column_stack(data_cols)
        header = ",".join(header_parts)
    else:
        header_parts = ["x", "y"] if mesh.dim == 2 else ["x", "y", "z"]
        data_cols = [mesh.nodes[:, i] for i in range(mesh.dim)]

        for name, field in solution.fields.items():
            if field.shape == (mesh.n_nodes,):
                data_cols.append(field)
                header_parts.append(name)

        data = np.column_stack(data_cols)
        header = ",".join(header_parts)

    np.savetxt(filename, data, delimiter=",", header=header, comments="")
