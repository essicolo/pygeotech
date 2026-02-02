# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 03 — Contaminant Transport with Sorption and Decay
#
# Simulates advective–dispersive transport of a dissolved contaminant
# through a homogeneous aquifer, with optional linear sorption and
# first-order decay.
#
# **Governing equation:**
#
# $$R \frac{\partial C}{\partial t}
#   = \nabla \cdot (D \nabla C) - \nabla \cdot (\mathbf{v} C)
#   - \lambda R C$$
#
# where $C$ is concentration, $R$ is the retardation factor,
# $D$ is the hydrodynamic dispersion tensor, $\mathbf{v}$ is the
# pore-water velocity, and $\lambda$ is the first-order decay rate.
#
# **Physics**: `pygeotech.physics.Transport` (coupled with `Darcy`)

# %%
import numpy as np
from pygeotech import geometry, materials, boundaries, physics, solvers, time
from pygeotech import postprocess

# %% [markdown]
# ## 1. Aquifer Domain
#
# 200 m long × 40 m wide rectangular aquifer.

# %%
aquifer = geometry.Rectangle(Lx=200, Ly=40, origin=(0, 0))
mesh = aquifer.generate_mesh(resolution=3.0)
print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_cells} cells")

# %% [markdown]
# ## 2. Materials

# %%
sand = materials.Material(
    name="aquifer_sand",
    hydraulic_conductivity=5e-4,
    porosity=0.30,
)
mat_map = materials.assign(mesh, {"aquifer_sand": sand})

# %% [markdown]
# ## 3. Flow Field (Steady-State Darcy)
#
# Establish a uniform flow from left to right.

# %%
darcy = physics.Darcy(mesh, mat_map)

flow_bcs = boundaries.BoundaryConditions([
    boundaries.Dirichlet(field="H", value=42.0, where=boundaries.left()),
    boundaries.Dirichlet(field="H", value=40.0, where=boundaries.right()),
    boundaries.Neumann(field="H", flux=0.0, where=boundaries.top()),
    boundaries.Neumann(field="H", flux=0.0, where=boundaries.bottom()),
])

flow_solver = solvers.FiPyBackend()
flow_solution = flow_solver.solve(darcy, flow_bcs)
velocity = postprocess.compute_velocity(flow_solution)
print(f"Mean velocity magnitude: {np.linalg.norm(velocity, axis=1).mean():.2e} m/s")

# %% [markdown]
# ## 4. Transport Physics
#
# Set up transport with longitudinal dispersivity $\alpha_L = 5$ m,
# transverse dispersivity $\alpha_T = 0.5$ m, linear sorption with
# $R = 2$, and first-order decay $\lambda = 1 \times 10^{-7}$ s⁻¹.

# %%
transport = physics.Transport(
    mesh, mat_map,
    dispersion_longitudinal=5.0,
    dispersion_transverse=0.5,
    molecular_diffusion=1e-9,
    retardation_factor=2.0,
    decay_rate=1e-7,
)
transport.set_velocity(velocity)

# %% [markdown]
# ## 5. Transport Boundary Conditions
#
# A continuous source of concentration $C_0 = 1$ mg/L applied on the
# left boundary between $y = 15$ m and $y = 25$ m.

# %%
source_locator = boundaries.left() & boundaries.y_between(15.0, 25.0)

transport_bcs = boundaries.BoundaryConditions([
    boundaries.Dirichlet(field="C", value=1.0, where=source_locator),
    boundaries.Dirichlet(field="C", value=0.0,
                         where=boundaries.left() & ~boundaries.y_between(15.0, 25.0)),
    boundaries.Neumann(field="C", flux=0.0, where=boundaries.right()),
    boundaries.Neumann(field="C", flux=0.0, where=boundaries.top()),
    boundaries.Neumann(field="C", flux=0.0, where=boundaries.bottom()),
])

# %% [markdown]
# ## 6. Time Stepping and Solution
#
# Simulate 30 days.

# %%
stepper = time.Stepper(t_end=30 * 86400, dt=3600)
print(f"Time steps: {stepper.n_steps}")

# %%
solver = solvers.FiPyBackend()
solution = solver.solve(
    transport, transport_bcs,
    time=stepper,
    initial_condition={"C": np.zeros(mesh.n_nodes)},
)

# %% [markdown]
# ## 7. Plume Snapshots

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, t_day in zip(axes, [5, 15, 30]):
    t_sec = t_day * 86400
    idx = np.argmin(np.abs(np.array(solution.times) - t_sec))
    C = solution.field_history["C"][idx]
    solution.plot(field="C", contours=15)
    ax.set_title(f"Day {t_day}")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Breakthrough Curve at a Monitoring Well

# %%
well_probe = postprocess.PointProbe((100, 20))
concentrations = []
for H_snap in solution.field_history["C"]:
    concentrations.append(well_probe.sample(mesh, H_snap))

fig, ax = plt.subplots(figsize=(7, 4))
t_days = np.array(solution.times) / 86400
ax.plot(t_days, concentrations, "r-")
ax.set_xlabel("Time (days)")
ax.set_ylabel("C / C₀")
ax.set_title("Breakthrough curve at (100 m, 20 m)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Takeaways
#
# - Transport is one-way coupled to flow: the Darcy velocity field
#   drives advection.
# - Sorption ($R > 1$) retards the plume front; decay reduces peak
#   concentrations.
# - The dispersivity ratio $\alpha_L / \alpha_T = 10$ produces an
#   elongated plume in the flow direction.
