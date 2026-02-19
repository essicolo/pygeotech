"""Solid mechanics — quasi-static equilibrium.

Governing equation::

    ∇·σ + b = 0
    σ = C : ε − α p I   (effective stress with pore-pressure coupling)

where σ is the Cauchy stress tensor, ε is the strain tensor,
C is the constitutive stiffness, α is the Biot coefficient, and
p is pore pressure.

Uses 2-D linear triangular CST (constant-strain triangle) elements
with 2 DOFs per node (u_x, u_y).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from pygeotech.physics.base import PhysicsModule


class Mechanics(PhysicsModule):
    """Quasi-static solid mechanics / geomechanics.

    Args:
        mesh: Computational mesh.
        materials: Material map providing ``youngs_modulus`` and
            ``poissons_ratio``.
        biot_coefficient: Biot effective-stress coefficient α.
        gravity: Gravitational acceleration (m/s²).
        plane_strain: Use plane-strain formulation (default ``True``).
    """

    name = "mechanics"
    primary_field = "u"  # displacement vector

    def __init__(
        self,
        mesh: Any,
        materials: Any,
        biot_coefficient: float = 1.0,
        gravity: float = 9.81,
        plane_strain: bool = True,
    ) -> None:
        super().__init__(mesh, materials)
        self.biot_coefficient = biot_coefficient
        self.gravity = gravity
        self.plane_strain = plane_strain
        self._pore_pressure: np.ndarray | None = None

    def set_pore_pressure(self, pressure: np.ndarray) -> None:
        """Set the pore-pressure field for effective stress coupling.

        Args:
            pressure: Nodal pore pressure, shape ``(n_nodes,)``.
        """
        self._pore_pressure = np.asarray(pressure, dtype=float)

    def coefficients(self) -> dict[str, np.ndarray]:
        E = self.materials.cell_property("youngs_modulus")
        nu = self.materials.cell_property("poissons_ratio")
        rho = self.materials.cell_property("dry_density")
        return {
            "youngs_modulus": E,
            "poissons_ratio": nu,
            "density": rho,
        }

    def elastic_stiffness_2d(
        self,
        E: float,
        nu: float,
        plane_strain: bool = True,
    ) -> np.ndarray:
        """2-D elastic constitutive matrix (plane strain or plane stress).

        Args:
            E: Young's modulus.
            nu: Poisson's ratio.
            plane_strain: If True, plane-strain; else plane-stress.

        Returns:
            3×3 constitutive matrix (Voigt notation: σ_xx, σ_yy, τ_xy).
        """
        if plane_strain:
            factor = E / ((1 + nu) * (1 - 2 * nu))
            C = factor * np.array([
                [1 - nu, nu, 0],
                [nu, 1 - nu, 0],
                [0, 0, 0.5 * (1 - 2 * nu)],
            ])
        else:
            factor = E / (1 - nu ** 2)
            C = factor * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, 0.5 * (1 - nu)],
            ])
        return C

    def assemble_stiffness(self) -> sparse.csr_matrix:
        """Assemble the global stiffness matrix for 2-D elasticity.

        Uses the CST (constant-strain triangle) element with
        2 DOFs per node.

        Returns:
            Sparse stiffness matrix, shape ``(2*n_nodes, 2*n_nodes)``.
        """
        mesh = self.mesh
        nodes = mesh.nodes
        cells = mesh.cells
        n_nodes = mesh.n_nodes
        n_dof = 2 * n_nodes

        E_cell = self.materials.cell_property("youngs_modulus")
        nu_cell = self.materials.cell_property("poissons_ratio")

        rows, cols, vals = [], [], []

        for ic, cell in enumerate(cells):
            if len(cell) != 3:
                raise NotImplementedError("Only triangular elements supported.")

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

            # B matrix (3×6):  [dN/dx,    0  ]
            #                  [  0,   dN/dy ]
            #                  [dN/dy, dN/dx ]
            B = np.zeros((3, 6))
            for n_local in range(3):
                B[0, 2 * n_local] = b[n_local]
                B[1, 2 * n_local + 1] = c[n_local]
                B[2, 2 * n_local] = c[n_local]
                B[2, 2 * n_local + 1] = b[n_local]

            D = self.elastic_stiffness_2d(
                float(E_cell[ic]), float(nu_cell[ic]), self.plane_strain
            )

            # Local stiffness: k_e = B^T D B * area
            k_local = (B.T @ D @ B) * area

            # Global DOF indices
            dof_idx = []
            for n_local in [i, j, k]:
                dof_idx.extend([2 * n_local, 2 * n_local + 1])

            for a in range(6):
                for bb in range(6):
                    rows.append(dof_idx[a])
                    cols.append(dof_idx[bb])
                    vals.append(k_local[a, bb])

        return sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_dof, n_dof)
        )

    def assemble_body_force(self) -> np.ndarray:
        """Assemble gravity body force vector.

        Returns:
            Force vector, shape ``(2*n_nodes,)``.
            f_y = -ρ g for each node.
        """
        mesh = self.mesh
        nodes = mesh.nodes
        cells = mesh.cells
        n_nodes = mesh.n_nodes
        n_dof = 2 * n_nodes

        rho = self.materials.cell_property("dry_density")
        f = np.zeros(n_dof)

        for ic, cell in enumerate(cells):
            if len(cell) != 3:
                continue
            i, j, k = cell
            xi, yi = nodes[i]
            xj, yj = nodes[j]
            xk, yk = nodes[k]

            area = 0.5 * abs(
                (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)
            )
            # Distributed gravity: f_y = -ρ g * area / 3
            fy = -float(rho[ic]) * self.gravity * area / 3.0
            for n_local in [i, j, k]:
                f[2 * n_local + 1] += fy

        return f

    def assemble_pore_pressure_force(self) -> np.ndarray:
        """Assemble pore-pressure coupling force: α ∫ B^T m p dA.

        The vector m = [1, 1, 0]^T in 2-D (Voigt notation for identity).

        Returns:
            Force vector, shape ``(2*n_nodes,)``.
        """
        if self._pore_pressure is None:
            return np.zeros(2 * self.mesh.n_nodes)

        mesh = self.mesh
        nodes = mesh.nodes
        cells = mesh.cells
        n_nodes = mesh.n_nodes
        n_dof = 2 * n_nodes
        f = np.zeros(n_dof)
        m = np.array([1.0, 1.0, 0.0])  # Voigt identity

        for ic, cell in enumerate(cells):
            if len(cell) != 3:
                continue
            i, j, k = cell
            xi, yi = nodes[i]
            xj, yj = nodes[j]
            xk, yk = nodes[k]

            area = 0.5 * abs(
                (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)
            )
            if area < 1e-30:
                continue

            b = np.array([yj - yk, yk - yi, yi - yj]) / (2.0 * area)
            c = np.array([xk - xj, xi - xk, xj - xi]) / (2.0 * area)

            B = np.zeros((3, 6))
            for n_local in range(3):
                B[0, 2 * n_local] = b[n_local]
                B[1, 2 * n_local + 1] = c[n_local]
                B[2, 2 * n_local] = c[n_local]
                B[2, 2 * n_local + 1] = b[n_local]

            # Average pore pressure in element
            p_avg = self._pore_pressure[cell].mean()

            # Force: α * B^T * m * p * area
            f_local = self.biot_coefficient * (B.T @ m) * p_avg * area

            dof_idx = []
            for n_local in [i, j, k]:
                dof_idx.extend([2 * n_local, 2 * n_local + 1])

            for a in range(6):
                f[dof_idx[a]] += f_local[a]

        return f

    def assemble_system(
        self,
        boundary_conditions: Any,
    ) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
        """Assemble the full elasticity system K u = f.

        Args:
            boundary_conditions: BoundaryConditions instance.

        Returns:
            Tuple of ``(K, rhs, dirichlet_dofs, dirichlet_values)``.
        """
        from pygeotech.boundaries.base import Dirichlet, Neumann

        K = self.assemble_stiffness()
        rhs = self.assemble_body_force() + self.assemble_pore_pressure_force()

        mesh = self.mesh
        n_dof = 2 * mesh.n_nodes
        boundary_idx = mesh.boundary_nodes()
        boundary_coords = mesh.nodes[boundary_idx]

        dirichlet_dofs_list: list[int] = []
        dirichlet_vals_list: list[float] = []

        for bc in boundary_conditions:
            if isinstance(bc, Dirichlet) and bc.field in ("u", "ux", "uy"):
                mask = bc.apply_mask(boundary_coords)
                active_idx = boundary_idx[mask]
                active_coords = boundary_coords[mask]
                values = bc.evaluate(active_coords)

                for ii, (node_idx, val) in enumerate(zip(active_idx, values)):
                    if bc.field == "u" or bc.field == "ux":
                        dirichlet_dofs_list.append(2 * node_idx)
                        dirichlet_vals_list.append(float(val))
                    if bc.field == "u" or bc.field == "uy":
                        dirichlet_dofs_list.append(2 * node_idx + 1)
                        dirichlet_vals_list.append(float(val))

        d_dofs = np.array(dirichlet_dofs_list, dtype=int)
        d_vals = np.array(dirichlet_vals_list, dtype=float)

        return K, rhs, d_dofs, d_vals

    def residual(
        self,
        field_values: np.ndarray,
        field_values_old: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Evaluate the equilibrium residual R = K u - f."""
        K = self.assemble_stiffness()
        f = self.assemble_body_force() + self.assemble_pore_pressure_force()
        return K @ field_values - f

    @property
    def is_transient(self) -> bool:
        return False
