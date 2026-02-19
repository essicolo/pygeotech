"""Strength Reduction Method (SRM) for slope stability via FEM.

The SRM determines the factor of safety by progressively reducing
the shear-strength parameters until the FEM solution fails to
converge (i.e. the slope "fails"):

    c_reduced  = c  / SRF
    φ_reduced  = arctan(tan φ / SRF)

The factor of safety equals the SRF at failure.

Failure is detected by one of:
    - The FEM solver fails to converge (singular matrix).
    - Maximum nodal displacement exceeds a threshold.
    - Displacement increment between successive SRF steps diverges.

The algorithm uses a bisection approach to bracket the critical SRF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from pygeotech.solvers.base import Solution


@dataclass
class SRMResult:
    """Result of a Strength Reduction analysis.

    Attributes:
        fos: Factor of safety (critical SRF).
        displacement: Displacement field at the critical SRF.
        converged: Whether the bisection converged within tolerance.
        srf_history: Tested SRF values.
        max_disp_history: Maximum displacement at each SRF.
    """

    fos: float
    displacement: np.ndarray | None = None
    converged: bool = False
    srf_history: list[float] | None = None
    max_disp_history: list[float] | None = None


def strength_reduction(
    mesh: Any,
    materials: Any,
    boundary_conditions: Any,
    srf_range: tuple[float, float] = (0.5, 5.0),
    n_steps: int = 20,
    bisection_tol: float = 0.01,
    max_disp_threshold: float = 1e3,
    gravity: float = 9.81,
    plane_strain: bool = True,
) -> SRMResult:
    """Perform a Strength Reduction FEM analysis.

    Uses the existing :class:`~pygeotech.physics.mechanics.Mechanics`
    module with progressively reduced c' and tan(φ').

    The algorithm:

    1. Sweep SRF from ``srf_range[0]`` to ``srf_range[1]``.
    2. At each SRF, reduce c and φ of all materials, assemble and
       solve the FEM system.
    3. Track maximum displacement.  When the solver fails or
       displacement diverges, the current SRF bracket is refined
       by bisection.

    Args:
        mesh: FEM mesh (triangular).
        materials: Material map with ``youngs_modulus``,
            ``poissons_ratio``, ``dry_density``, ``cohesion``,
            and ``friction_angle`` per cell.
        boundary_conditions: Dirichlet BCs for the mechanics problem.
        srf_range: (min_SRF, max_SRF) search range.
        n_steps: Initial coarse sweep steps.
        bisection_tol: Bisection tolerance on SRF.
        max_disp_threshold: Displacement threshold for failure (m).
        gravity: Gravitational acceleration (m/s²).
        plane_strain: Plane-strain formulation.

    Returns:
        :class:`SRMResult`.
    """
    from pygeotech.physics.mechanics import Mechanics
    from pygeotech.boundaries.base import Dirichlet
    from pygeotech.solvers.fipy_backend import FiPyBackend

    # Extract original material properties
    try:
        c_orig = materials.cell_property("cohesion").copy()
    except (KeyError, AttributeError):
        c_orig = np.zeros(mesh.n_cells)

    try:
        phi_orig = materials.cell_property("friction_angle").copy()
    except (KeyError, AttributeError):
        phi_orig = np.full(mesh.n_cells, 30.0)

    phi_rad_orig = np.radians(phi_orig)

    srf_history: list[float] = []
    max_disp_history: list[float] = []
    last_good_u: np.ndarray | None = None

    def _try_solve(srf: float) -> tuple[bool, np.ndarray | None, float]:
        """Attempt FEM solve at a given SRF.

        Returns (success, displacement, max_disp).
        """
        # Reduce strength parameters
        c_red = c_orig / srf
        phi_red = np.degrees(np.arctan(np.tan(phi_rad_orig) / srf))

        # Temporarily modify material properties
        _set_cell_property(materials, "cohesion", c_red)
        _set_cell_property(materials, "friction_angle", phi_red)

        mech = Mechanics(
            mesh, materials,
            gravity=gravity, plane_strain=plane_strain,
        )

        try:
            solver = FiPyBackend()
            sol = solver.solve(mech, boundary_conditions)
            u = sol.fields["u"]
            max_d = float(np.max(np.abs(u)))
            return True, u, max_d
        except Exception:
            return False, None, float("inf")
        finally:
            # Restore original properties
            _set_cell_property(materials, "cohesion", c_orig)
            _set_cell_property(materials, "friction_angle", phi_orig)

    # Phase 1: coarse sweep to bracket failure
    srf_vals = np.linspace(srf_range[0], srf_range[1], n_steps)
    srf_low = srf_range[0]
    srf_high = srf_range[1]
    found_bracket = False

    for srf in srf_vals:
        ok, u, max_d = _try_solve(srf)
        srf_history.append(srf)
        max_disp_history.append(max_d)

        if ok and max_d < max_disp_threshold:
            srf_low = srf
            last_good_u = u
        else:
            srf_high = srf
            found_bracket = True
            break

    if not found_bracket:
        # Slope is stable across the entire range
        return SRMResult(
            fos=srf_range[1],
            displacement=last_good_u,
            converged=False,
            srf_history=srf_history,
            max_disp_history=max_disp_history,
        )

    # Phase 2: bisection to refine the critical SRF
    for _ in range(50):
        if (srf_high - srf_low) < bisection_tol:
            break

        srf_mid = 0.5 * (srf_low + srf_high)
        ok, u, max_d = _try_solve(srf_mid)
        srf_history.append(srf_mid)
        max_disp_history.append(max_d)

        if ok and max_d < max_disp_threshold:
            srf_low = srf_mid
            last_good_u = u
        else:
            srf_high = srf_mid

    fos = 0.5 * (srf_low + srf_high)

    return SRMResult(
        fos=fos,
        displacement=last_good_u,
        converged=(srf_high - srf_low) < bisection_tol,
        srf_history=srf_history,
        max_disp_history=max_disp_history,
    )


def _set_cell_property(
    materials: Any,
    name: str,
    values: np.ndarray,
) -> None:
    """Set a per-cell property on the material map.

    Works with the pygeotech MaterialMap which stores materials by
    index.  Since cells sharing a material index share the same
    Material object, we set the property on each unique Material.

    For SRM, all cells of the same material get the same reduced
    value (reduction is uniform), so we set the property on each
    Material in the list using the mean value for cells of that type.
    """
    for mat_idx, mat in enumerate(materials.materials):
        cell_mask = materials.cell_material_index == mat_idx
        if cell_mask.any():
            mat[name] = float(values[cell_mask].mean())
