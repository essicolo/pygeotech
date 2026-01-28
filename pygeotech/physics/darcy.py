"""Saturated groundwater flow — Darcy / Laplace equation.

Governing equation (steady-state)::

    ∇·(K ∇H) = 0

where *K* is hydraulic conductivity and *H* is total hydraulic head.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from pygeotech.physics.base import PhysicsModule


class Darcy(PhysicsModule):
    """Saturated steady-state groundwater flow.

    Uses a simple finite-volume / finite-difference discretisation on
    triangular meshes.

    Args:
        mesh: Computational mesh.
        materials: Material map providing ``hydraulic_conductivity``.
    """

    name = "darcy"
    primary_field = "H"

    def __init__(self, mesh: Any, materials: Any) -> None:
        super().__init__(mesh, materials)

    def coefficients(self) -> dict[str, np.ndarray]:
        """Return per-cell hydraulic conductivity."""
        K = self.materials.cell_property("hydraulic_conductivity")
        return {"conductivity": K}

    def residual(
        self,
        field_values: np.ndarray,
        field_values_old: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Evaluate ∇·(K∇H) at each node using the assembled system.

        Args:
            field_values: Current head field, shape ``(n_nodes,)``.

        Returns:
            Residual vector.
        """
        A = self.assemble_stiffness()
        return A @ field_values

    def assemble_stiffness(self) -> sparse.csr_matrix:
        """Assemble the global stiffness matrix for ∇·(K∇H) = 0.

        Uses linear triangular finite elements (constant-strain triangle)
        for 2-D meshes.

        Returns:
            Sparse stiffness matrix of shape ``(n_nodes, n_nodes)``.
        """
        mesh = self.mesh
        nodes = mesh.nodes
        cells = mesh.cells
        n_nodes = mesh.n_nodes
        K_cell = self.materials.cell_property("hydraulic_conductivity")

        rows, cols, vals = [], [], []

        for ic, cell in enumerate(cells):
            if len(cell) == 3:
                # Linear triangle
                i, j, k = cell
                xi, yi = nodes[i]
                xj, yj = nodes[j]
                xk, yk = nodes[k]

                area = 0.5 * abs(
                    (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)
                )
                if area < 1e-30:
                    continue

                # Shape function gradients
                b = np.array([yj - yk, yk - yi, yi - yj]) / (2.0 * area)
                c = np.array([xk - xj, xi - xk, xj - xi]) / (2.0 * area)

                K = float(K_cell[ic])
                local_indices = [i, j, k]

                for a in range(3):
                    for bb in range(3):
                        val = K * (b[a] * b[bb] + c[a] * c[bb]) * area
                        rows.append(local_indices[a])
                        cols.append(local_indices[bb])
                        vals.append(val)
            else:
                raise NotImplementedError(
                    f"Element type with {len(cell)} nodes not supported."
                )

        A = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_nodes, n_nodes)
        )
        return A

    def assemble_system(
        self,
        boundary_conditions: Any,
    ) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
        """Assemble the linear system with boundary conditions applied.

        Args:
            boundary_conditions: A
                :class:`~pygeotech.boundaries.base.BoundaryConditions`
                instance.

        Returns:
            Tuple of ``(A, rhs, dirichlet_nodes, dirichlet_values)``
            where *A* is the modified stiffness matrix and *rhs*
            incorporates Neumann contributions.
        """
        from pygeotech.boundaries.base import Dirichlet, Neumann

        A = self.assemble_stiffness()
        rhs = np.zeros(self.mesh.n_nodes)

        boundary_idx = self.mesh.boundary_nodes()
        boundary_coords = self.mesh.nodes[boundary_idx]

        dirichlet_nodes_list: list[int] = []
        dirichlet_values_list: list[float] = []

        for bc in boundary_conditions:
            if isinstance(bc, Dirichlet) and bc.field == self.primary_field:
                mask = bc.apply_mask(boundary_coords)
                active_idx = boundary_idx[mask]
                active_coords = boundary_coords[mask]
                values = bc.evaluate(active_coords)
                dirichlet_nodes_list.extend(active_idx.tolist())
                dirichlet_values_list.extend(values.tolist())

            elif isinstance(bc, Neumann) and bc.field == self.primary_field:
                mask = bc.apply_mask(boundary_coords)
                active_idx = boundary_idx[mask]
                active_coords = boundary_coords[mask]
                fluxes = bc.evaluate(active_coords)
                rhs[active_idx] += fluxes

        d_nodes = np.array(dirichlet_nodes_list, dtype=int)
        d_values = np.array(dirichlet_values_list, dtype=float)

        return A, rhs, d_nodes, d_values

    @property
    def is_transient(self) -> bool:
        return False
