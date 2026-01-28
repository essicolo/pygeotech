"""Physics-Informed Neural Network (PINN) solver backend.

The network learns the solution field (e.g. H(x,y,t)) by minimising::

    L = L_pde + λ_bc · L_bc + λ_data · L_data

where:
- L_pde: PDE residual (e.g. ∇·(K∇H) = 0 for Darcy)
- L_bc: boundary condition residual
- L_data: optional measurement data fitting

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

        input_dim = dim + (1 if is_transient else 0)
        output_dim = 1  # single scalar field

        net = self._build_network(torch, nn, input_dim, output_dim).to(device)
        optimiser = torch.optim.Adam(net.parameters(), lr=self.learning_rate)

        # Collocation points
        x_col = self._sample_collocation_points(mesh, device, torch)

        # Boundary points
        x_bc, bc_values = self._extract_bc_data(
            physics, boundary_conditions, device, torch
        )

        # Training loop
        for epoch in range(self.epochs):
            optimiser.zero_grad()

            # PDE residual
            loss_pde = self._compute_pde_residual(
                physics, net, x_col, torch
            )

            # BC residual
            loss_bc = self._compute_bc_residual(net, x_bc, bc_values, torch)

            loss = loss_pde + self.bc_weight * loss_bc
            loss.backward()
            optimiser.step()

        # Evaluate on mesh nodes
        with torch.no_grad():
            x_eval = torch.tensor(
                mesh.nodes, dtype=torch.float32, device=device
            )
            H = net(x_eval).cpu().numpy().flatten()

        return Solution(
            fields={physics.primary_field: H},
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
    ) -> Any:
        """Sample random collocation points inside the domain."""
        lo, hi = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
        pts = np.random.uniform(lo, hi, size=(self.collocation_points, mesh.dim))
        x = torch.tensor(pts, dtype=torch.float32, device=device, requires_grad=True)
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

        bc_coords_list: list[np.ndarray] = []
        bc_vals_list: list[np.ndarray] = []

        if boundary_conditions is not None:
            for bc in boundary_conditions:
                if isinstance(bc, Dirichlet) and bc.field == physics.primary_field:
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

    def _compute_pde_residual(
        self,
        physics: Any,
        net: Any,
        x: Any,
        torch: Any,
    ) -> Any:
        """Compute PDE residual using automatic differentiation.

        For Darcy: residual = K * (∂²H/∂x² + ∂²H/∂y²)
        """
        H = net(x)

        # First derivatives
        grads = torch.autograd.grad(
            H, x,
            grad_outputs=torch.ones_like(H),
            create_graph=True,
        )[0]

        # Laplacian (sum of second derivatives)
        laplacian = torch.zeros_like(H)
        for d in range(x.shape[1]):
            grad_d = grads[:, d:d+1]
            grad2 = torch.autograd.grad(
                grad_d, x,
                grad_outputs=torch.ones_like(grad_d),
                create_graph=True,
            )[0][:, d:d+1]
            laplacian += grad2

        return (laplacian ** 2).mean()

    def _compute_bc_residual(
        self,
        net: Any,
        x_bc: Any,
        bc_values: Any,
        torch: Any,
    ) -> Any:
        """Compute boundary condition residual."""
        if len(x_bc) == 0:
            return torch.tensor(0.0)
        H_pred = net(x_bc).squeeze()
        return ((H_pred - bc_values) ** 2).mean()

    def __repr__(self) -> str:
        return (
            f"PINNBackend(layers={self.layers}, epochs={self.epochs}, "
            f"lr={self.learning_rate})"
        )
