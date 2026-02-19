"""Solute transport — advection-dispersion equation.

Governing equation::

    R θ ∂C/∂t + ∇·(q C) = ∇·(θ D ∇C) - θ λ R C + θ S

where C is concentration, q is Darcy velocity from the flow solution,
D is the hydrodynamic dispersion tensor, R is the retardation factor,
λ is a first-order decay rate, and S is a source/sink.

Sorption models:
    - Linear:     R = 1 + ρ_b Kd / θ
    - Freundlich: sorbed = Kf · C^(1/n)
    - Langmuir:   sorbed = S_max · Kl · C / (1 + Kl · C)

FEM discretisation uses SUPG (Streamline Upwind Petrov-Galerkin)
to stabilise advection-dominated transport.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from pygeotech.physics.base import PhysicsModule


class Transport(PhysicsModule):
    """Advection-dispersion solute transport with sorption and decay.

    Args:
        mesh: Computational mesh.
        materials: Material map.
        dispersion_longitudinal: Longitudinal dispersivity α_L (m).
        dispersion_transverse: Transverse dispersivity α_T (m).
        molecular_diffusion: Effective molecular diffusion D_m (m²/s).
        retardation_factor: Linear retardation R (--). Ignored if a
            sorption model is provided.
        decay_rate: First-order decay rate λ (1/s).
        sorption: Sorption model -- ``None`` (use *retardation_factor*),
            ``"linear"``, ``"freundlich"``, or ``"langmuir"``.
        sorption_params: Dict of sorption parameters.  Keys depend on
            the model:

            - linear: ``{"Kd": float, "bulk_density": float}``
            - freundlich: ``{"Kf": float, "nf": float, "bulk_density": float}``
            - langmuir: ``{"Kl": float, "S_max": float, "bulk_density": float}``
    """

    name = "transport"
    primary_field = "C"

    def __init__(
        self,
        mesh: Any,
        materials: Any,
        dispersion_longitudinal: float = 1.0,
        dispersion_transverse: float = 0.1,
        molecular_diffusion: float = 1e-9,
        retardation_factor: float = 1.0,
        decay_rate: float = 0.0,
        sorption: str | None = None,
        sorption_params: dict[str, float] | None = None,
    ) -> None:
        super().__init__(mesh, materials)
        self.dispersion_longitudinal = dispersion_longitudinal
        self.dispersion_transverse = dispersion_transverse
        self.molecular_diffusion = molecular_diffusion
        self.retardation_factor = retardation_factor
        self.decay_rate = decay_rate
        self.sorption = sorption
        self.sorption_params = sorption_params or {}
        self._velocity: np.ndarray | None = None

    @property
    def is_transient(self) -> bool:
        return True

    def set_velocity(self, velocity: np.ndarray) -> None:
        """Set the Darcy velocity field from a flow solution.

        Args:
            velocity: Array of shape ``(n_cells, dim)``.
        """
        self._velocity = np.asarray(velocity, dtype=float)

    # ------------------------------------------------------------------
    # Sorption
    # ------------------------------------------------------------------

    def compute_retardation(
        self,
        C: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute retardation factor R at nodes.

        Args:
            C: Current concentration at nodes (needed for nonlinear
               sorption).

        Returns:
            Retardation array, shape ``(n_nodes,)``.
        """
        n_nodes = self.mesh.n_nodes
        porosity = self.materials.cell_property("porosity")
        theta_avg = porosity.mean()

        if self.sorption is None:
            return np.full(n_nodes, self.retardation_factor)

        sp = self.sorption_params
        rho_b = sp.get("bulk_density", 1600.0)

        if self.sorption == "linear":
            Kd = sp.get("Kd", 0.0)
            R = 1.0 + rho_b * Kd / theta_avg
            return np.full(n_nodes, R)

        if self.sorption == "freundlich":
            Kf = sp.get("Kf", 0.0)
            nf = sp.get("nf", 1.0)
            if C is None or nf == 1.0:
                R = 1.0 + rho_b * Kf / theta_avg
                return np.full(n_nodes, R)
            C_safe = np.maximum(np.abs(C), 1e-30)
            dS_dC = Kf * (1.0 / nf) * C_safe ** (1.0 / nf - 1.0)
            return 1.0 + rho_b * dS_dC / theta_avg

        if self.sorption == "langmuir":
            Kl = sp.get("Kl", 0.0)
            S_max = sp.get("S_max", 0.0)
            if C is None:
                R = 1.0 + rho_b * Kl * S_max / theta_avg
                return np.full(n_nodes, R)
            dS_dC = Kl * S_max / (1.0 + Kl * np.abs(C)) ** 2
            return 1.0 + rho_b * dS_dC / theta_avg

        return np.full(n_nodes, self.retardation_factor)

    # ------------------------------------------------------------------
    # Dispersion tensor
    # ------------------------------------------------------------------

    def dispersion_tensor(self, velocity: np.ndarray) -> np.ndarray:
        """Compute the hydrodynamic dispersion tensor D per cell.

        D_ij = (α_T |v| + D_m) δ_ij + (α_L - α_T) v_i v_j / |v|

        Args:
            velocity: Per-cell velocity, shape ``(n_cells, dim)``.

        Returns:
            Dispersion tensor, shape ``(n_cells, dim, dim)``.
        """
        v = np.asarray(velocity, dtype=float)
        n_cells, dim = v.shape
        v_mag = np.linalg.norm(v, axis=1, keepdims=True)
        v_mag_safe = np.maximum(v_mag, 1e-30)

        D = np.zeros((n_cells, dim, dim))
        eye = np.eye(dim)

        for ic in range(n_cells):
            vmag = v_mag_safe[ic, 0]
            vi = v[ic]
            D[ic] = (
                (self.dispersion_transverse * vmag + self.molecular_diffusion) * eye
                + (self.dispersion_longitudinal - self.dispersion_transverse)
                * np.outer(vi, vi) / vmag
            )

        return D

    # ------------------------------------------------------------------
    # FEM assembly
    # ------------------------------------------------------------------

    def assemble_system(
        self,
        boundary_conditions: Any,
        C_current: np.ndarray | None = None,
        C_old: np.ndarray | None = None,
        dt: float | None = None,
    ) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
        """Assemble the transport system with SUPG stabilisation.

        The discretised system (backward Euler):

            (R θ M / dt + K_disp + K_adv + λ R θ M) C^{n+1}
                = R θ M / dt · C^n + rhs_bc

        Args:
            boundary_conditions: BoundaryConditions instance.
            C_current: Current concentration iterate.
            C_old: Concentration at previous time step.
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

        porosity = self.materials.cell_property("porosity")

        if self._velocity is not None:
            velocity = self._velocity
        else:
            velocity = np.zeros((mesh.n_cells, dim))

        D_tensor = self.dispersion_tensor(velocity)
        R = self.compute_retardation(C_current)

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

            theta = float(porosity[ic])
            D = D_tensor[ic]
            v = velocity[ic]
            local_idx = [i, j, k]

            # ---- Dispersion stiffness: θ ∫ (∇N)^T D (∇N) dA ----
            for a in range(3):
                grad_a = np.array([b[a], c_arr[a]])
                for bi in range(3):
                    grad_b = np.array([b[bi], c_arr[bi]])
                    val = theta * float(grad_a @ D @ grad_b) * area
                    rows.append(local_idx[a])
                    cols.append(local_idx[bi])
                    vals.append(val)

            # ---- Advection: ∫ N_a (v · ∇N_b) dA ----
            # With SUPG: test = N_a + τ (v · ∇N_a)
            v_mag = np.linalg.norm(v)
            h_e = np.sqrt(2.0 * area)
            D_eff = max(np.trace(D) / dim, 1e-30)
            Pe_local = v_mag * h_e / (2.0 * D_eff) if D_eff > 1e-30 else 0.0

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
                    val = (w_a + w_supg_a) * v_dot_grad_b * area
                    rows.append(local_idx[a])
                    cols.append(local_idx[bi])
                    vals.append(val)

            # ---- Lumped mass ----
            R_avg = R[cell].mean()
            mass_contrib = R_avg * theta * area / 3.0
            mass_diag[i] += mass_contrib
            mass_diag[j] += mass_contrib
            mass_diag[k] += mass_contrib

        A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
        M = sparse.diags(mass_diag)

        # First-order decay
        if self.decay_rate > 0.0:
            A = A + self.decay_rate * M

        # Transient storage
        if dt is not None and C_old is not None:
            A = A + M / dt
            rhs += (M / dt) @ C_old

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

    def coefficients(self) -> dict[str, np.ndarray]:
        porosity = self.materials.cell_property("porosity")
        return {
            "porosity": porosity,
            "retardation": np.full(self.mesh.n_cells, self.retardation_factor),
            "decay_rate": np.full(self.mesh.n_cells, self.decay_rate),
        }

    def residual(
        self,
        field_values: np.ndarray,
        field_values_old: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Evaluate the transport equation residual.

        Assembles the system matrix and computes A @ C - rhs.
        """
        from pygeotech.boundaries.base import BoundaryConditions
        A, rhs, _, _ = self.assemble_system(
            BoundaryConditions(),
            C_current=field_values,
            C_old=field_values_old,
            dt=dt,
        )
        return A @ field_values - rhs
