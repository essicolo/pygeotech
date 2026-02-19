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
    segments.  The per-cell hydraulic conductivity is retrieved from
    the material map when available.

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

    # Retrieve per-cell conductivity
    K_cell = _get_cell_conductivity(solution)

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
        K = float(K_cell[ic])
        flux = -K * np.dot(grad_H[ic], normal) * seg_len
        total_flux += flux

    return float(total_flux)


def mass_balance(solution: Any) -> dict[str, float]:
    """Compute overall mass balance for the solution.

    Integrates flux over each boundary face (left, right, top, bottom)
    and computes the balance error.

    Args:
        solution: Solution object with ``"H"`` field and mesh.

    Returns:
        Dictionary with keys ``"inflow"``, ``"outflow"``, ``"balance_error"``.
    """
    from pygeotech.boundaries.locators import left, right, top, bottom

    if "H" not in solution.fields:
        return {"inflow": 0.0, "outflow": 0.0, "balance_error": 0.0}

    locators = [left(), right(), top(), bottom()]
    inflow = 0.0
    outflow = 0.0

    for loc in locators:
        flux = integrate_flux(solution, loc)
        if flux > 0:
            inflow += flux
        else:
            outflow += abs(flux)

    balance_error = inflow - outflow
    return {
        "inflow": inflow,
        "outflow": outflow,
        "balance_error": balance_error,
    }


def _get_cell_conductivity(solution: Any) -> np.ndarray:
    """Extract per-cell hydraulic conductivity from the solution.

    Falls back to K=1.0 if the material map is not available.
    """
    mesh = solution.mesh

    # Try getting from materials stored on the mesh
    try:
        if hasattr(solution, "materials"):
            return solution.materials.cell_property("hydraulic_conductivity")
    except (AttributeError, KeyError):
        pass

    # Try from the mesh's material map
    try:
        if hasattr(mesh, "materials"):
            return mesh.materials.cell_property("hydraulic_conductivity")
    except (AttributeError, KeyError):
        pass

    return np.ones(mesh.n_cells)
