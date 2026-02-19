"""Unsaturated flow — Richards equation.

Governing equation (mixed form)::

    ∂θ/∂t = ∇·(K(h) ∇H) - S

where θ = θ(h) is the volumetric water content, K(h) is the
unsaturated hydraulic conductivity, H = h + z is total head, and S
is a source/sink term.

The nonlinearity in K(h) and θ(h) is handled by Picard (fixed-point)
iteration within each time step.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from pygeotech.physics.base import PhysicsModule


class Richards(PhysicsModule):
    """Transient unsaturated flow (Richards equation).

    Uses linear triangular finite elements with Picard linearisation.
    At each time step, the nonlinear conductivity K(h) and moisture
    capacity C(h) are evaluated at the current iterate and the
    system is re-assembled until convergence.

    Args:
        mesh: Computational mesh.
        materials: Material map.
        retention_model: A soil-water retention model (e.g.
            :class:`~pygeotech.materials.constitutive.VanGenuchten`).
    """

    name = "richards"
    primary_field = "H"

    def __init__(
        self,
        mesh: Any,
        materials: Any,
        retention_model: Any = None,
    ) -> None:
        super().__init__(mesh, materials)
        self.retention_model = retention_model

    @property
    def is_transient(self) -> bool:
        return True

    def coefficients(self) -> dict[str, np.ndarray]:
        """Return per-cell saturated conductivity and porosity."""
        K_sat = self.materials.cell_property("hydraulic_conductivity")
        porosity = self.materials.cell_property("porosity")
        return {
            "conductivity": K_sat,
            "porosity": porosity,
        }

    def effective_conductivity(self, H: np.ndarray) -> np.ndarray:
        """Compute per-cell effective conductivity K_eff = K_sat * Kr(h).

        Args:
            H: Total hydraulic head at nodes, shape ``(n_nodes,)``.

        Returns:
            Per-cell effective conductivity, shape ``(n_cells,)``.
        """
        K_sat = self.materials.cell_property("hydraulic_conductivity")
        if self.retention_model is None:
            return K_sat

        z = self.mesh.nodes[:, -1]
        h = H - z  # pressure head at nodes
        Kr_nodes = self.retention_model.relative_permeability(h)

        # Average over element nodes to get per-cell value
        cells = self.mesh.cells
        Kr_cell = Kr_nodes[cells].mean(axis=1)
        return K_sat * Kr_cell

    def moisture_capacity(self, H: np.ndarray) -> np.ndarray:
        """Compute per-node specific moisture capacity C(h) = dθ/dh.

        Args:
            H: Total head at nodes, shape ``(n_nodes,)``.

        Returns:
            Capacity at nodes, shape ``(n_nodes,)``.
        """
        if self.retention_model is None:
            return np.zeros(self.mesh.n_nodes)

        z = self.mesh.nodes[:, -1]
        h = H - z
        if hasattr(self.retention_model, "specific_moisture_capacity"):
            return self.retention_model.specific_moisture_capacity(h)
        # Numerical approximation
        eps = 1e-6
        theta_p = self.retention_model.water_content(h + eps)
        theta_m = self.retention_model.water_content(h - eps)
        return (theta_p - theta_m) / (2.0 * eps)

    def water_content_at(self, H: np.ndarray) -> np.ndarray:
        """Volumetric water content at nodes.

        Args:
            H: Total head, shape ``(n_nodes,)``.

        Returns:
            θ at nodes, shape ``(n_nodes,)``.
        """
        if self.retention_model is None:
            porosity = self.materials.cell_property("porosity")
            # Use mean porosity
            return np.full(self.mesh.n_nodes, porosity.mean())
        z = self.mesh.nodes[:, -1]
        h = H - z
        return self.retention_model.water_content(h)

    def assemble_stiffness(self, K_eff: np.ndarray) -> sparse.csr_matrix:
        """Assemble global stiffness matrix for ∇·(K_eff ∇H) = 0.

        Uses the same linear triangular element as Darcy, but with
        K_eff = K_sat * Kr(h) varying per element.

        Args:
            K_eff: Per-cell effective conductivity, shape ``(n_cells,)``.

        Returns:
            Sparse stiffness matrix, shape ``(n_nodes, n_nodes)``.
        """
        mesh = self.mesh
        nodes = mesh.nodes
        cells = mesh.cells
        n_nodes = mesh.n_nodes

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

            b = np.array([yj - yk, yk - yi, yi - yj]) / (2.0 * area)
            c = np.array([xk - xj, xi - xk, xj - xi]) / (2.0 * area)

            K = float(K_eff[ic])
            local_indices = [i, j, k]

            for a in range(3):
                for bb in range(3):
                    val = K * (b[a] * b[bb] + c[a] * c[bb]) * area
                    rows.append(local_indices[a])
                    cols.append(local_indices[bb])
                    vals.append(val)

        return sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_nodes, n_nodes)
        )

    def assemble_mass(self, capacity: np.ndarray) -> sparse.csr_matrix:
        """Assemble lumped mass matrix M_ii = Σ (C_avg * area / 3).

        For the modified Picard method the mass matrix uses the specific
        moisture capacity C(h) = dθ/dh.

        Args:
            capacity: Per-node moisture capacity, shape ``(n_nodes,)``.

        Returns:
            Diagonal sparse mass matrix, shape ``(n_nodes, n_nodes)``.
        """
        mesh = self.mesh
        nodes = mesh.nodes
        cells = mesh.cells
        n_nodes = mesh.n_nodes

        diag = np.zeros(n_nodes)

        for cell in cells:
            if len(cell) != 3:
                continue
            i, j, k = cell
            xi, yi = nodes[i]
            xj, yj = nodes[j]
            xk, yk = nodes[k]

            area = 0.5 * abs(
                (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)
            )
            C_avg = capacity[cell].mean()
            contrib = C_avg * area / 3.0
            diag[i] += contrib
            diag[j] += contrib
            diag[k] += contrib

        return sparse.diags(diag)

    def assemble_storage_rhs(
        self,
        H: np.ndarray,
        H_old: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Assemble right-hand side contribution from storage term.

        Uses the mixed form: (θ(H) - θ(H_old)) / dt, assembled as
        lumped nodal values.

        Args:
            H: Current head iterate, shape ``(n_nodes,)``.
            H_old: Head at previous time step, shape ``(n_nodes,)``.
            dt: Time step size (s).

        Returns:
            Storage RHS vector, shape ``(n_nodes,)``.
        """
        theta_new = self.water_content_at(H)
        theta_old = self.water_content_at(H_old)

        mesh = self.mesh
        nodes = mesh.nodes
        cells = mesh.cells
        n_nodes = mesh.n_nodes
        rhs = np.zeros(n_nodes)

        for cell in cells:
            if len(cell) != 3:
                continue
            i, j, k = cell
            xi, yi = nodes[i]
            xj, yj = nodes[j]
            xk, yk = nodes[k]

            area = 0.5 * abs(
                (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)
            )
            dtheta = theta_new[cell] - theta_old[cell]
            contrib = dtheta.mean() * area / (3.0 * dt)
            rhs[i] -= contrib
            rhs[j] -= contrib
            rhs[k] -= contrib

        return rhs

    def assemble_system(
        self,
        boundary_conditions: Any,
        H_current: np.ndarray | None = None,
        H_old: np.ndarray | None = None,
        dt: float | None = None,
    ) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
        """Assemble the linearised system for one Picard iteration.

        For steady state (dt=None): K * H = rhs
        For transient: (M/dt + K) * H = M/dt * H_old + rhs_bc

        Args:
            boundary_conditions: BoundaryConditions instance.
            H_current: Current Picard iterate (for K(h) evaluation).
            H_old: Head at previous time step.
            dt: Time step size.

        Returns:
            Tuple of ``(A, rhs, dirichlet_nodes, dirichlet_values)``.
        """
        from pygeotech.boundaries.base import Dirichlet, Neumann

        n_nodes = self.mesh.n_nodes

        # Evaluate nonlinear conductivity at current iterate
        if H_current is not None:
            K_eff = self.effective_conductivity(H_current)
        else:
            K_eff = self.materials.cell_property("hydraulic_conductivity")

        A = self.assemble_stiffness(K_eff)
        rhs = np.zeros(n_nodes)

        # Transient terms
        if dt is not None and H_old is not None and H_current is not None:
            capacity = self.moisture_capacity(H_current)
            # Ensure minimum capacity for numerical stability
            capacity = np.maximum(capacity, 1e-10)
            M = self.assemble_mass(capacity)
            A = A + M / dt
            rhs += (M / dt) @ H_old

        # Boundary conditions
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

    def residual(
        self,
        field_values: np.ndarray,
        field_values_old: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Evaluate the Richards equation residual.

        Computes: R = K(H) @ H + storage_term

        Args:
            field_values: Current head field H, shape ``(n_nodes,)``.
            field_values_old: Previous time-step head.
            dt: Time step (s).

        Returns:
            Residual vector.
        """
        K_eff = self.effective_conductivity(field_values)
        A = self.assemble_stiffness(K_eff)
        residual = A @ field_values

        if field_values_old is not None and dt is not None:
            residual += self.assemble_storage_rhs(
                field_values, field_values_old, dt
            )

        return residual

    def validate(self) -> list[str]:
        issues = super().validate()
        if self.retention_model is None:
            issues.append("No retention model specified for Richards equation.")
        return issues
