"""Physics-Informed Neural Network (PINN) solver backend.

The network learns the solution field (e.g. H(x,y,t)) by minimising::

    L = L_pde + λ_bc · L_bc + λ_ic · L_ic

where:
- L_pde: PDE residual evaluated at interior collocation points
- L_bc: boundary condition residual
- L_ic: initial condition residual (transient problems)

Supports all pygeotech physics modules:

- **Darcy**: ∇·(K∇H) = 0
- **Richards**: ∇·[K(h)(∇h + ∇z)] = C(h) ∂h/∂t  (van Genuchten)
- **Transport**: D∇²C − v·∇C = Rθ ∂C/∂t + λRC
- **Heat**: λ∇²T − ρ_w c_w v·∇T = ρc ∂T/∂t
- **Mechanics**: ∇·σ + b = 0 (plane strain / plane stress elasticity)

Requires PyTorch (``pip install torch``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pygeotech.solvers.base import Solver, Solution


class PINNBackend(Solver):
    """Physics-Informed Neural Network solver using PyTorch.

    Args:
        layers: Hidden layer sizes, e.g. ``[64, 128, 64]``.
        activation: Activation function (``"tanh"``, ``"relu"``, ``"gelu"``).
        learning_rate: Optimiser learning rate.
        epochs: Number of training epochs.
        collocation_points: Number of interior collocation points.
        bc_weight: Weight λ_bc for boundary loss.
        data_weight: Weight λ_data for data-fitting loss.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
    """

    def __init__(
        self,
        layers: list[int] | None = None,
        activation: str = "tanh",
        learning_rate: float = 1e-3,
        epochs: int = 10000,
        collocation_points: int = 10000,
        bc_weight: float = 10.0,
        data_weight: float = 1.0,
        device: str = "auto",
    ) -> None:
        self.layers = layers or [64, 64, 64]
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.collocation_points = collocation_points
        self.bc_weight = bc_weight
        self.data_weight = data_weight
        self.device_name = device

    def solve(
        self,
        physics: Any,
        boundary_conditions: Any = None,
        time: Any | None = None,
        initial_condition: dict[str, float | np.ndarray] | None = None,
        **kwargs: Any,
    ) -> Solution:
        """Train the PINN and return the solution.

        Args:
            physics: Physics module defining the PDE.
            boundary_conditions: Boundary conditions.
            time: Time stepper (for transient).
            initial_condition: Initial values.

        Returns:
            Solution with fields evaluated at mesh nodes.
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError(
                "PINNBackend requires PyTorch.  "
                "Install with: pip install torch"
            ) from exc

        device = self._resolve_device(torch)
        mesh = physics.mesh
        dim = mesh.dim
        is_transient = time is not None
        physics_name = getattr(physics, "name", "darcy")

        input_dim = dim + (1 if is_transient else 0)

        # Mechanics outputs a displacement vector (2 components in 2-D)
        if physics_name == "mechanics":
            output_dim = dim
        else:
            output_dim = 1

        net = self._build_network(torch, nn, input_dim, output_dim).to(device)
        optimiser = torch.optim.Adam(net.parameters(), lr=self.learning_rate)

        # Collocation points
        x_col = self._sample_collocation_points(
            mesh, device, torch, is_transient, time
        )

        # Boundary points
        x_bc, bc_values = self._extract_bc_data(
            physics, boundary_conditions, device, torch
        )

        # Initial condition points (for transient)
        x_ic, ic_values = (None, None)
        if is_transient and initial_condition is not None:
            x_ic, ic_values = self._extract_ic_data(
                physics, mesh, initial_condition, device, torch
            )

        # Training loop
        for epoch in range(self.epochs):
            optimiser.zero_grad()

            # PDE residual
            loss_pde = self._compute_pde_residual(
                physics, net, x_col, torch
            )

            # BC residual
            loss_bc = self._compute_bc_residual(
                net, x_bc, bc_values, torch, output_dim
            )

            loss = loss_pde + self.bc_weight * loss_bc

            # IC residual for transient problems
            if x_ic is not None and ic_values is not None:
                loss_ic = self._compute_ic_residual(
                    net, x_ic, ic_values, torch, output_dim
                )
                loss = loss + self.bc_weight * loss_ic

            loss.backward()
            optimiser.step()

        # Evaluate on mesh nodes
        with torch.no_grad():
            x_eval = torch.tensor(
                mesh.nodes, dtype=torch.float32, device=device
            )
            if is_transient and time is not None:
                t_final = float(time.t_end)
                t_col = torch.full(
                    (x_eval.shape[0], 1), t_final,
                    dtype=torch.float32, device=device,
                )
                x_eval = torch.cat([x_eval, t_col], dim=1)

            output = net(x_eval).cpu().numpy()

        if physics_name == "mechanics":
            # Interleave: [ux0, uy0, ux1, uy1, ...]
            u = output.flatten()
            return Solution(fields={physics.primary_field: u}, mesh=mesh)

        return Solution(
            fields={physics.primary_field: output.flatten()},
            mesh=mesh,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self, torch: Any) -> Any:
        if self.device_name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device_name)

    def _build_network(
        self,
        torch: Any,
        nn: Any,
        input_dim: int,
        output_dim: int,
    ) -> Any:
        """Construct a fully connected neural network."""
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }
        act_cls = activations.get(self.activation, nn.Tanh)

        layer_list: list[Any] = []
        prev = input_dim
        for width in self.layers:
            layer_list.append(nn.Linear(prev, width))
            layer_list.append(act_cls())
            prev = width
        layer_list.append(nn.Linear(prev, output_dim))

        return nn.Sequential(*layer_list)

    def _sample_collocation_points(
        self,
        mesh: Any,
        device: Any,
        torch: Any,
        is_transient: bool = False,
        time: Any | None = None,
    ) -> Any:
        """Sample random collocation points inside the domain."""
        lo, hi = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
        pts = np.random.uniform(lo, hi, size=(self.collocation_points, mesh.dim))

        if is_transient and time is not None:
            t_start = float(time.t_start) if hasattr(time, "t_start") else 0.0
            t_end = float(time.t_end)
            t_col = np.random.uniform(
                t_start, t_end, size=(self.collocation_points, 1)
            )
            pts = np.hstack([pts, t_col])

        x = torch.tensor(
            pts, dtype=torch.float32, device=device, requires_grad=True
        )
        return x

    def _extract_bc_data(
        self,
        physics: Any,
        boundary_conditions: Any,
        device: Any,
        torch: Any,
    ) -> tuple[Any, Any]:
        """Extract boundary coordinates and values."""
        from pygeotech.boundaries.base import Dirichlet

        mesh = physics.mesh
        b_idx = mesh.boundary_nodes()
        b_coords = mesh.nodes[b_idx]

        # Determine which field names to match
        target_fields = {physics.primary_field}
        physics_name = getattr(physics, "name", "")
        if physics_name == "mechanics":
            target_fields.update({"u", "ux", "uy"})

        bc_coords_list: list[np.ndarray] = []
        bc_vals_list: list[np.ndarray] = []

        if boundary_conditions is not None:
            for bc in boundary_conditions:
                if isinstance(bc, Dirichlet) and bc.field in target_fields:
                    mask = bc.apply_mask(b_coords)
                    active_coords = b_coords[mask]
                    vals = bc.evaluate(active_coords)
                    if len(active_coords) > 0:
                        bc_coords_list.append(active_coords)
                        bc_vals_list.append(vals)

        if bc_coords_list:
            all_coords = np.vstack(bc_coords_list)
            all_vals = np.concatenate(bc_vals_list)
        else:
            all_coords = np.empty((0, mesh.dim))
            all_vals = np.empty(0)

        x_bc = torch.tensor(all_coords, dtype=torch.float32, device=device)
        v_bc = torch.tensor(all_vals, dtype=torch.float32, device=device)
        return x_bc, v_bc

    def _extract_ic_data(
        self,
        physics: Any,
        mesh: Any,
        initial_condition: dict[str, Any] | None,
        device: Any,
        torch: Any,
    ) -> tuple[Any, Any]:
        """Extract initial condition data for transient problems."""
        if initial_condition is None:
            return None, None

        field = physics.primary_field
        if field not in initial_condition:
            return None, None

        ic_val = initial_condition[field]
        n_nodes = mesh.n_nodes
        coords = mesh.nodes

        # Build coordinates at t=0
        t_zero = np.zeros((n_nodes, 1))
        x_ic = np.hstack([coords, t_zero])

        if np.isscalar(ic_val):
            vals = np.full(n_nodes, float(ic_val))
        else:
            vals = np.asarray(ic_val, dtype=float).flatten()

        x_ic_t = torch.tensor(x_ic, dtype=torch.float32, device=device)
        v_ic_t = torch.tensor(vals, dtype=torch.float32, device=device)
        return x_ic_t, v_ic_t

    # ------------------------------------------------------------------
    # PDE residual dispatch
    # ------------------------------------------------------------------

    def _compute_pde_residual(
        self,
        physics: Any,
        net: Any,
        x: Any,
        torch: Any,
    ) -> Any:
        """Compute PDE residual using automatic differentiation.

        Dispatches to physics-specific residual methods based on
        ``physics.name``.
        """
        name = getattr(physics, "name", "darcy")

        if name == "darcy":
            return self._residual_darcy(physics, net, x, torch)
        elif name == "richards":
            return self._residual_richards(physics, net, x, torch)
        elif name == "transport":
            return self._residual_transport(physics, net, x, torch)
        elif name == "heat":
            return self._residual_heat(physics, net, x, torch)
        elif name == "mechanics":
            return self._residual_mechanics(physics, net, x, torch)
        else:
            # Fallback: generic Laplacian
            return self._residual_laplacian(net, x, torch)

    # ------------------------------------------------------------------
    # Physics-specific residuals
    # ------------------------------------------------------------------

    def _residual_darcy(
        self,
        physics: Any,
        net: Any,
        x: Any,
        torch: Any,
    ) -> Any:
        """Darcy: K ∇²H = 0."""
        H = net(x)
        laplacian = self._compute_laplacian(H, x, torch, n_dims=physics.dim)
        return (laplacian ** 2).mean()

    def _residual_richards(
        self,
        physics: Any,
        net: Any,
        x: Any,
        torch: Any,
    ) -> Any:
        """Richards: ∇·[K(h)(∇h + ∇z)] = C(h) ∂h/∂t.

        When a retention model (e.g. van Genuchten) is available, the
        full nonlinear equation is enforced with K(h) and C(h) computed
        in differentiable PyTorch operations so that autodiff correctly
        propagates through the conductivity function.

        Without a retention model, falls back to the linearised form
        K_sat ∇²H = n ∂H/∂t.
        """
        dim = physics.dim
        H = net(x)

        grads_H = torch.autograd.grad(
            H, x,
            grad_outputs=torch.ones_like(H),
            create_graph=True,
        )[0]

        K_sat = float(physics.materials.cell_property("hydraulic_conductivity").mean())
        retention = getattr(physics, "retention_model", None)

        if retention is not None:
            # --- Full nonlinear Richards via van Genuchten ----------
            alpha = retention.alpha
            n_vg = retention.n
            m_vg = retention.m
            theta_r = retention.theta_r
            theta_s = retention.theta_s

            # Pressure head: h = H - z  (z is last spatial coordinate)
            z = x[:, dim - 1 : dim]
            h = H - z

            # Effective saturation Se(h) ∈ [0, 1]
            abs_h = torch.abs(h)
            Se = torch.where(
                h >= 0,
                torch.ones_like(h),
                (1.0 + (alpha * abs_h) ** n_vg) ** (-m_vg),
            )
            Se = torch.clamp(Se, 1e-10, 1.0)

            # Mualem–van Genuchten relative permeability
            inner = 1.0 - (1.0 - Se ** (1.0 / m_vg)) ** m_vg
            Kr = Se ** 0.5 * inner ** 2
            K_h = K_sat * Kr  # (N, 1)

            # Flux divergence ∇·[K(h) ∇H] via autodiff
            divergence = torch.zeros_like(H)
            for d in range(dim):
                q_d = K_h * grads_H[:, d : d + 1]
                dq_d = torch.autograd.grad(
                    q_d, x,
                    grad_outputs=torch.ones_like(q_d),
                    create_graph=True,
                )[0][:, d : d + 1]
                divergence += dq_d

            # Specific moisture capacity C(h) = dθ/dh
            alpha_h_n = (alpha * abs_h) ** n_vg
            denom = 1.0 + alpha_h_n
            C_h = torch.where(
                h >= 0,
                torch.zeros_like(h),
                (theta_s - theta_r)
                * alpha * n_vg * m_vg
                * alpha_h_n
                / (abs_h + 1e-30)
                * denom ** (-(m_vg + 1)),
            )

            # Time derivative ∂H/∂t (= ∂h/∂t since ∂z/∂t = 0)
            if x.shape[1] > dim:
                dH_dt = grads_H[:, dim : dim + 1]
                residual = divergence - C_h * dH_dt
            else:
                residual = divergence
        else:
            # --- Linearised fallback: K_sat ∇²H = n ∂H/∂t ----------
            laplacian = torch.zeros_like(H)
            for d in range(dim):
                grad_d = grads_H[:, d : d + 1]
                grad2 = torch.autograd.grad(
                    grad_d, x,
                    grad_outputs=torch.ones_like(grad_d),
                    create_graph=True,
                )[0][:, d : d + 1]
                laplacian += grad2

            if x.shape[1] > dim:
                dH_dt = grads_H[:, dim : dim + 1]
                porosity = float(
                    physics.materials.cell_property("porosity").mean()
                )
                residual = K_sat * laplacian - porosity * dH_dt
            else:
                residual = K_sat * laplacian

        return (residual ** 2).mean()

    def _residual_transport(
        self,
        physics: Any,
        net: Any,
        x: Any,
        torch: Any,
    ) -> Any:
        """Transport: D∇²C − v·∇C = Rθ ∂C/∂t + λRC."""
        dim = physics.dim
        C = net(x)

        grads = torch.autograd.grad(
            C, x,
            grad_outputs=torch.ones_like(C),
            create_graph=True,
        )[0]

        # Spatial Laplacian
        laplacian = torch.zeros_like(C)
        for d in range(dim):
            grad_d = grads[:, d : d + 1]
            grad2 = torch.autograd.grad(
                grad_d, x,
                grad_outputs=torch.ones_like(grad_d),
                create_graph=True,
            )[0][:, d : d + 1]
            laplacian += grad2

        # Effective dispersion (isotropic scalar approximation)
        D_eff = physics.molecular_diffusion + physics.dispersion_longitudinal

        # Advection: v · ∇C
        advection = torch.zeros_like(C)
        if hasattr(physics, "_velocity") and physics._velocity is not None:
            v_mean = physics._velocity.mean(axis=0)
            for d in range(dim):
                advection = advection + float(v_mean[d]) * grads[:, d : d + 1]

        # Decay
        decay = physics.decay_rate * C

        # Storage: Rθ ∂C/∂t
        porosity = float(physics.materials.cell_property("porosity").mean())
        R = physics.retardation_factor

        if x.shape[1] > dim:
            dC_dt = grads[:, dim : dim + 1]
            residual = D_eff * laplacian - advection - decay - R * porosity * dC_dt
        else:
            residual = D_eff * laplacian - advection - decay

        return (residual ** 2).mean()

    def _residual_heat(
        self,
        physics: Any,
        net: Any,
        x: Any,
        torch: Any,
    ) -> Any:
        """Heat: λ∇²T − ρ_w c_w v·∇T = ρc ∂T/∂t."""
        dim = physics.dim
        T = net(x)

        grads = torch.autograd.grad(
            T, x,
            grad_outputs=torch.ones_like(T),
            create_graph=True,
        )[0]

        # Spatial Laplacian
        laplacian = torch.zeros_like(T)
        for d in range(dim):
            grad_d = grads[:, d : d + 1]
            grad2 = torch.autograd.grad(
                grad_d, x,
                grad_outputs=torch.ones_like(grad_d),
                create_graph=True,
            )[0][:, d : d + 1]
            laplacian += grad2

        lam = float(physics.materials.cell_property("thermal_conductivity").mean())

        # Advection: ρ_w c_w v · ∇T
        advection = torch.zeros_like(T)
        rho_w_cw = physics.water_density * physics.water_specific_heat
        if hasattr(physics, "_velocity") and physics._velocity is not None:
            v_mean = physics._velocity.mean(axis=0)
            for d in range(dim):
                advection = advection + float(v_mean[d]) * grads[:, d : d + 1]
        advection = rho_w_cw * advection

        # Volumetric heat capacity
        rho = float(physics.materials.cell_property("dry_density").mean())
        cp = float(physics.materials.cell_property("specific_heat").mean())
        rho_c = rho * cp

        if x.shape[1] > dim:
            dT_dt = grads[:, dim : dim + 1]
            residual = lam * laplacian - advection - rho_c * dT_dt
        else:
            residual = lam * laplacian - advection

        return (residual ** 2).mean()

    def _residual_mechanics(
        self,
        physics: Any,
        net: Any,
        x: Any,
        torch: Any,
    ) -> Any:
        """Mechanics: ∇·σ + b = 0 (2-D plane strain / plane stress).

        The network outputs (u_x, u_y).  Strains are computed via
        autodiff, stresses via the constitutive matrix, and the
        equilibrium residual is the divergence of stress plus body
        forces.
        """
        u = net(x)  # shape: (N, 2)
        ux = u[:, 0:1]
        uy = u[:, 1:2]

        # First derivatives
        dux = torch.autograd.grad(
            ux, x, grad_outputs=torch.ones_like(ux), create_graph=True
        )[0]
        duy = torch.autograd.grad(
            uy, x, grad_outputs=torch.ones_like(uy), create_graph=True
        )[0]

        dux_dx = dux[:, 0:1]
        dux_dy = dux[:, 1:2]
        duy_dx = duy[:, 0:1]
        duy_dy = duy[:, 1:2]

        # Strains (Voigt: ε_xx, ε_yy, γ_xy)
        eps_xx = dux_dx
        eps_yy = duy_dy
        gamma_xy = dux_dy + duy_dx

        # Constitutive matrix (average properties)
        E = float(physics.materials.cell_property("youngs_modulus").mean())
        nu = float(physics.materials.cell_property("poissons_ratio").mean())

        if physics.plane_strain:
            factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
            D00 = factor * (1.0 - nu)
            D01 = factor * nu
            D22 = factor * 0.5 * (1.0 - 2.0 * nu)
        else:
            factor = E / (1.0 - nu ** 2)
            D00 = factor
            D01 = factor * nu
            D22 = factor * 0.5 * (1.0 - nu)

        # Stresses
        sig_xx = D00 * eps_xx + D01 * eps_yy
        sig_yy = D01 * eps_xx + D00 * eps_yy
        tau_xy = D22 * gamma_xy

        # Divergence of stress tensor
        dsig_xx = torch.autograd.grad(
            sig_xx, x, grad_outputs=torch.ones_like(sig_xx), create_graph=True
        )[0]
        dsig_yy = torch.autograd.grad(
            sig_yy, x, grad_outputs=torch.ones_like(sig_yy), create_graph=True
        )[0]
        dtau_xy = torch.autograd.grad(
            tau_xy, x, grad_outputs=torch.ones_like(tau_xy), create_graph=True
        )[0]

        dsig_xx_dx = dsig_xx[:, 0:1]
        dsig_yy_dy = dsig_yy[:, 1:2]
        dtau_xy_dx = dtau_xy[:, 0:1]
        dtau_xy_dy = dtau_xy[:, 1:2]

        # Body force (gravity in negative y-direction)
        rho = float(physics.materials.cell_property("dry_density").mean())
        g = physics.gravity

        # Equilibrium:
        #   ∂σ_xx/∂x + ∂τ_xy/∂y = 0       (x-direction, no body force)
        #   ∂τ_xy/∂x + ∂σ_yy/∂y = ρg       (y-direction, gravity)
        res_x = dsig_xx_dx + dtau_xy_dy
        res_y = dtau_xy_dx + dsig_yy_dy - rho * g

        return (res_x ** 2).mean() + (res_y ** 2).mean()

    def _residual_laplacian(
        self,
        net: Any,
        x: Any,
        torch: Any,
    ) -> Any:
        """Generic Laplacian residual (fallback for unknown physics)."""
        H = net(x)
        laplacian = self._compute_laplacian(H, x, torch)
        return (laplacian ** 2).mean()

    # ------------------------------------------------------------------
    # Autodiff utilities
    # ------------------------------------------------------------------

    def _compute_laplacian(
        self,
        u: Any,
        x: Any,
        torch: Any,
        n_dims: int | None = None,
    ) -> Any:
        """Compute the Laplacian ∇²u using automatic differentiation.

        Args:
            u: Network output, shape ``(N, 1)``.
            x: Input coordinates, shape ``(N, d)``.
            torch: The torch module.
            n_dims: Number of spatial dimensions to sum over.
                Defaults to all input dimensions.

        Returns:
            Laplacian, shape ``(N, 1)``.
        """
        if n_dims is None:
            n_dims = x.shape[1]

        grads = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]

        laplacian = torch.zeros_like(u)
        for d in range(n_dims):
            grad_d = grads[:, d : d + 1]
            grad2 = torch.autograd.grad(
                grad_d, x,
                grad_outputs=torch.ones_like(grad_d),
                create_graph=True,
            )[0][:, d : d + 1]
            laplacian += grad2

        return laplacian

    def _compute_bc_residual(
        self,
        net: Any,
        x_bc: Any,
        bc_values: Any,
        torch: Any,
        output_dim: int = 1,
    ) -> Any:
        """Compute boundary condition residual."""
        if len(x_bc) == 0:
            return torch.tensor(0.0)

        pred = net(x_bc)

        if output_dim == 1:
            return ((pred.squeeze() - bc_values) ** 2).mean()

        # For vector fields (mechanics): apply BC value to all components
        target = bc_values.unsqueeze(-1).expand_as(pred)
        return ((pred - target) ** 2).mean()

    def _compute_ic_residual(
        self,
        net: Any,
        x_ic: Any,
        ic_values: Any,
        torch: Any,
        output_dim: int = 1,
    ) -> Any:
        """Compute initial condition residual for transient problems."""
        if x_ic is None or len(x_ic) == 0:
            return torch.tensor(0.0)

        pred = net(x_ic)

        if output_dim == 1:
            return ((pred.squeeze() - ic_values) ** 2).mean()

        target = ic_values.unsqueeze(-1).expand_as(pred)
        return ((pred - target) ** 2).mean()

    def __repr__(self) -> str:
        return (
            f"PINNBackend(layers={self.layers}, epochs={self.epochs}, "
            f"lr={self.learning_rate})"
        )
