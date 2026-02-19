"""Heat transfer in porous media.

Governing equation::

    ρc ∂T/∂t = ∇·(λ ∇T) - ρ_w c_w q·∇T + Q

where T is temperature, λ is effective thermal conductivity, q is
Darcy velocity (for advective transport), ρc is the bulk volumetric
heat capacity, and Q is a heat source.

Uses linear triangular FEM with SUPG stabilisation for the advective
term (analogous to solute transport).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from pygeotech.physics.base import PhysicsModule


class HeatTransfer(PhysicsModule):
    """Heat conduction and advection in porous media.

    Args:
        mesh: Computational mesh.
        materials: Material map providing ``thermal_conductivity``,
            ``specific_heat``, and ``dry_density``.
        water_density: Density of water (kg/m³).
        water_specific_heat: Specific heat of water (J/(kg·K)).
    """

    name = "heat"
    primary_field = "T"

    def __init__(
        self,
        mesh: Any,
        materials: Any,
        water_density: float = 1000.0,
        water_specific_heat: float = 4186.0,
    ) -> None:
        super().__init__(mesh, materials)
        self.water_density = water_density
        self.water_specific_heat = water_specific_heat
        self._velocity: np.ndarray | None = None

    @property
    def is_transient(self) -> bool:
        return True

    def set_velocity(self, velocity: np.ndarray) -> None:
        """Set the Darcy velocity field for advective heat transport.

        Args:
            velocity: Array of shape ``(n_cells, dim)``.
        """
        self._velocity = np.asarray(velocity, dtype=float)

    def coefficients(self) -> dict[str, np.ndarray]:
        lam = self.materials.cell_property("thermal_conductivity")
        rho = self.materials.cell_property("dry_density")
        cp = self.materials.cell_property("specific_heat")
        return {
            "thermal_conductivity": lam,
            "bulk_heat_capacity": rho * cp,
        }

    def assemble_system(
        self,
        boundary_conditions: Any,
        T_old: np.ndarray | None = None,
        dt: float | None = None,
    ) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
        """Assemble the heat transfer system.

        Backward-Euler discretisation:

            (ρc M / dt + K_cond + K_adv) T^{n+1} = ρc M / dt · T^n + rhs_bc

        Args:
            boundary_conditions: BoundaryConditions instance.
            T_old: Temperature at previous time step.
            dt: Time step size (s).

        Returns:
            Tuple of ``(A, rhs, dirichlet_nodes, dirichlet_values)``.
        """
        from pygeotech.boundaries.base import Dirichlet, Neumann

        mesh = self.mesh
        nodes = mesh.nodes
        cells = mesh.cells
        n_nodes = mesh.n_nodes
        dim = mesh.dim

        lam = self.materials.cell_property("thermal_conductivity")
        rho = self.materials.cell_property("dry_density")
        cp = self.materials.cell_property("specific_heat")
        rho_c = rho * cp  # bulk volumetric heat capacity per cell

        if self._velocity is not None:
            velocity = self._velocity
        else:
            velocity = np.zeros((mesh.n_cells, dim))

        rho_w_cw = self.water_density * self.water_specific_heat

        rows, cols, vals = [], [], []
        rhs = np.zeros(n_nodes)
        mass_diag = np.zeros(n_nodes)

        for ic, cell in enumerate(cells):
            if len(cell) != 3 or dim != 2:
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
            c_arr = np.array([xk - xj, xi - xk, xj - xi]) / (2.0 * area)

            lambda_e = float(lam[ic])
            v = velocity[ic]
            local_idx = [i, j, k]

            # ---- Conduction: λ ∫ (∇N)^T (∇N) dA ----
            for a in range(3):
                for bi in range(3):
                    val = lambda_e * (b[a] * b[bi] + c_arr[a] * c_arr[bi]) * area
                    rows.append(local_idx[a])
                    cols.append(local_idx[bi])
                    vals.append(val)

            # ---- Advection: ρ_w c_w ∫ N_a (v · ∇N_b) dA ----
            v_mag = np.linalg.norm(v)
            h_e = np.sqrt(2.0 * area)
            Pe_local = rho_w_cw * v_mag * h_e / (2.0 * lambda_e + 1e-30)

            if Pe_local > 1e-2:
                coth_val = 1.0 / np.tanh(Pe_local)
                tau_supg = h_e / (2.0 * v_mag + 1e-30) * max(coth_val - 1.0 / Pe_local, 0.0)
            else:
                tau_supg = 0.0

            for a in range(3):
                grad_a = np.array([b[a], c_arr[a]])
                w_a = 1.0 / 3.0
                w_supg_a = tau_supg * float(v @ grad_a)

                for bi in range(3):
                    grad_b = np.array([b[bi], c_arr[bi]])
                    v_dot_grad_b = float(v @ grad_b)
                    val = rho_w_cw * (w_a + w_supg_a) * v_dot_grad_b * area
                    rows.append(local_idx[a])
                    cols.append(local_idx[bi])
                    vals.append(val)

            # ---- Lumped mass ----
            mc = float(rho_c[ic]) * area / 3.0
            mass_diag[i] += mc
            mass_diag[j] += mc
            mass_diag[k] += mc

        A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
        M = sparse.diags(mass_diag)

        if dt is not None and T_old is not None:
            A = A + M / dt
            rhs += (M / dt) @ T_old

        # Boundary conditions
        boundary_idx = mesh.boundary_nodes()
        boundary_coords = nodes[boundary_idx]

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
        """Evaluate the heat equation residual."""
        from pygeotech.boundaries.base import BoundaryConditions
        A, rhs, _, _ = self.assemble_system(
            BoundaryConditions(), T_old=field_values_old, dt=dt,
        )
        return A @ field_values - rhs
