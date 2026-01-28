"""Flux integrals and mass balance computations.

Functions
---------
integrate_flux
    Integrate normal flux across a boundary or cross-section.
mass_balance
    Compute overall mass balance for a solution.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def integrate_flux(
    solution: Any,
    locator: Any,
) -> float:
    """Integrate the normal flux across a boundary.

    Uses the Darcy velocity field and a midpoint rule over boundary
    segments.

    Args:
        solution: A :class:`~pygeotech.solvers.base.Solution`.
        locator: A boundary locator identifying the integration surface.

    Returns:
        Total flux (m³/s per unit depth for 2-D).
    """
    mesh = solution.mesh
    b_idx = mesh.boundary_nodes()
    b_coords = mesh.nodes[b_idx]
    mask = locator(b_coords)
    active_idx = b_idx[mask]

    if len(active_idx) < 2:
        return 0.0

    # Compute gradient and velocity at cells
    from pygeotech.postprocess.fields import compute_gradient

    H = solution.fields.get("H")
    if H is None:
        return 0.0

    grad_H = compute_gradient(mesh, H)
    # Average velocity at boundary nodes (nearest cell)
    centers = mesh.cell_centers()

    total_flux = 0.0
    active_coords = mesh.nodes[active_idx]

    # Sort along primary axis of the boundary
    if active_coords[:, 0].ptp() > active_coords[:, -1].ptp():
        order = np.argsort(active_coords[:, 0])
    else:
        order = np.argsort(active_coords[:, -1])

    sorted_coords = active_coords[order]

    for i in range(len(sorted_coords) - 1):
        mid = 0.5 * (sorted_coords[i] + sorted_coords[i + 1])
        seg = sorted_coords[i + 1] - sorted_coords[i]
        seg_len = np.linalg.norm(seg)

        # Normal (90° rotation in 2-D)
        if mesh.dim == 2:
            normal = np.array([-seg[1], seg[0]]) / (seg_len + 1e-30)
        else:
            normal = np.zeros(mesh.dim)
            normal[0] = 1.0  # placeholder for 3-D

        # Nearest cell
        dist = np.linalg.norm(centers - mid, axis=1)
        ic = np.argmin(dist)

        # Flux = -K grad(H) · n * ds
        K = 1.0  # default; ideally from materials
        flux = -K * np.dot(grad_H[ic], normal) * seg_len
        total_flux += flux

    return float(total_flux)


def mass_balance(solution: Any) -> dict[str, float]:
    """Compute overall mass balance for the solution.

    Returns a dictionary with inflow, outflow, and balance error.

    Args:
        solution: Solution object.

    Returns:
        Dictionary with keys ``"inflow"``, ``"outflow"``, ``"balance_error"``.
    """
    # Placeholder: full implementation requires flux integration on all boundaries
    return {
        "inflow": 0.0,
        "outflow": 0.0,
        "balance_error": 0.0,
    }
