"""Derived field computation.

Functions
---------
compute_velocity
    Compute Darcy velocity from a head solution.
compute_gradient
    Compute the gradient of a nodal field.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_gradient(mesh: Any, nodal_field: np.ndarray) -> np.ndarray:
    """Compute per-cell gradient of a nodal field.

    Uses linear interpolation within triangular elements.

    Args:
        mesh: A :class:`~pygeotech.geometry.mesh.Mesh`.
        nodal_field: Field values at nodes, shape ``(n_nodes,)``.

    Returns:
        Gradient array of shape ``(n_cells, dim)``.
    """
    nodes = mesh.nodes
    cells = mesh.cells
    dim = mesh.dim
    grad = np.zeros((len(cells), dim))

    for ic, cell in enumerate(cells):
        if len(cell) == 3 and dim == 2:
            i, j, k = cell
            xi, yi = nodes[i]
            xj, yj = nodes[j]
            xk, yk = nodes[k]

            area2 = abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
            if area2 < 1e-30:
                continue

            # Shape function gradients
            b = np.array([yj - yk, yk - yi, yi - yj]) / area2
            c = np.array([xk - xj, xi - xk, xj - xi]) / area2

            vals = nodal_field[cell]
            grad[ic, 0] = np.dot(b, vals)
            grad[ic, 1] = np.dot(c, vals)

    return grad


def compute_velocity(solution: Any) -> np.ndarray:
    """Compute Darcy velocity v = -K grad(H).

    Args:
        solution: A :class:`~pygeotech.solvers.base.Solution` with
            ``"H"`` field and a mesh with material map.

    Returns:
        Velocity array of shape ``(n_cells, dim)``.
    """
    H = solution.fields["H"]
    mesh = solution.mesh
    grad_H = compute_gradient(mesh, H)

    # Try to get per-cell K; fall back to uniform K=1
    try:
        K = solution.materials.cell_property("hydraulic_conductivity")
    except (AttributeError, KeyError):
        K = np.ones(mesh.n_cells)

    if K.ndim == 1:
        velocity = -K[:, np.newaxis] * grad_H
    else:
        velocity = -np.einsum("cij,cj->ci", K, grad_H)

    return velocity
