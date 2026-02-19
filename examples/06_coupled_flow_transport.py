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
# # 06 — Coupled Flow and Heat Transport
#
# This example demonstrates one-way sequential coupling between
# steady-state Darcy flow and transient heat transport with advection.
#
# The scenario: warm water is injected into a cold aquifer, creating
# a thermal plume that is advected by groundwater flow.
#
# **Coupling strategy**: `pygeotech.coupling.Sequential`
# (Darcy velocity drives heat advection — no feedback)

# %%
import numpy as np
from pygeotech import (
    geometry, materials, boundaries, physics,
    solvers, time, coupling, postprocess,
)

# %% [markdown]
# ## 1. Domain and Mesh

# %%
aquifer = geometry.Rectangle(Lx=100, Ly=30, origin=(0, 0))
mesh = aquifer.generate_mesh(resolution=1.5)
print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_cells} cells")

# %% [markdown]
# ## 2. Material Properties

# %%
sand = materials.Material(
    name="sand",
    hydraulic_conductivity=2e-4,
    porosity=0.35,
    thermal_conductivity=2.0,
    dry_density=1650,
    specific_heat=800,
)
mat_map = materials.assign(mesh, {"sand": sand})

# %% [markdown]
# ## 3. Steady-State Flow

# %%
darcy = physics.Darcy(mesh, mat_map)

flow_bcs = boundaries.BoundaryConditions([
    boundaries.Dirichlet(field="H", value=32.0, where=boundaries.left()),
    boundaries.Dirichlet(field="H", value=30.0, where=boundaries.right()),
    boundaries.Neumann(field="H", flux=0.0, where=boundaries.top()),
    boundaries.Neumann(field="H", flux=0.0, where=boundaries.bottom()),
])

flow_solver = solvers.FiPyBackend()
flow_solution = flow_solver.solve(darcy, flow_bcs)
velocity = postprocess.compute_velocity(flow_solution)
print(f"Mean |v|: {np.linalg.norm(velocity, axis=1).mean():.2e} m/s")

# %% [markdown]
# ## 4. Heat Transport with Advection
#
# Warm water at 50 °C enters at the left boundary between
# $y = 12$ m and $y = 18$ m. The initial aquifer temperature is 10 °C.

# %%
heat = physics.HeatTransfer(mesh, mat_map)
heat.set_velocity(velocity)

injection_zone = boundaries.left() & boundaries.y_between(12.0, 18.0)

heat_bcs = boundaries.BoundaryConditions([
    boundaries.Dirichlet(field="T", value=50.0, where=injection_zone),
    boundaries.Dirichlet(
        field="T", value=10.0,
        where=boundaries.left() & ~boundaries.y_between(12.0, 18.0),
    ),
    boundaries.Neumann(field="T", flux=0.0, where=boundaries.right()),
    boundaries.Neumann(field="T", flux=0.0, where=boundaries.top()),
    boundaries.Neumann(field="T", flux=0.0, where=boundaries.bottom()),
])

# %% [markdown]
# ## 5. Sequential Coupling Object
#
# Although we solve manually here, the `Sequential` coupling class
# formalises the one-way dependency.

# %%
coupled = coupling.Sequential([darcy, heat])
print(f"Coupled fields: {coupled.primary_fields}")
print(f"Transient: {coupled.is_transient}")

# %% [markdown]
# ## 6. Time Stepping — 60 Days

# %%
stepper = time.Stepper(t_end=60 * 86400, dt=7200)
initial_T = np.full(mesh.n_nodes, 10.0)

solver = solvers.FiPyBackend()
solution = solver.solve(
    heat, heat_bcs,
    time=stepper,
    initial_condition={"T": initial_T},
)

# %% [markdown]
# ## 7. Thermal Plume Evolution

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for ax, t_day in zip(axes, [10, 30, 60]):
    t_sec = t_day * 86400
    idx = np.argmin(np.abs(np.array(solution.times) - t_sec))
    T = solution.field_history["T"][idx]
    solution.plot(field="T", contours=15)
    ax.set_title(f"Temperature — Day {t_day}")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Temperature Breakthrough

# %%
probe = postprocess.PointProbe((50, 15))
temps = [probe.sample(mesh, T_snap) for T_snap in solution.field_history["T"]]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.array(solution.times) / 86400, temps, "r-")
ax.set_xlabel("Time (days)")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Thermal breakthrough at (50 m, 15 m)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Takeaways
#
# - Sequential coupling: flow is solved once, then the velocity field
#   is passed to the heat transport physics.
# - For bidirectional coupling (e.g. temperature-dependent viscosity),
#   use `coupling.Iterative` instead.
# - The thermal plume shape depends on the ratio of advective to
#   conductive heat transport (thermal Peclet number).
