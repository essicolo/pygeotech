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
# # 04 — Transient Heat Conduction in a Soil Mass
#
# Simulates thermal response of a soil mass to a surface temperature
# change (e.g. seasonal variation or geothermal pile operation).
#
# **Governing equation:**
#
# $$\rho c_p \frac{\partial T}{\partial t}
#   = \nabla \cdot (\lambda \nabla T)$$
#
# where $T$ is temperature, $\lambda$ is thermal conductivity, $\rho$
# is dry density, and $c_p$ is specific heat capacity.
#
# **Physics**: `pygeotech.physics.HeatTransfer`

# %%
import numpy as np
from pygeotech import geometry, materials, boundaries, physics, solvers, time

# %% [markdown]
# ## 1. Domain
#
# A 10 m wide × 20 m deep soil cross-section.

# %%
domain = geometry.Rectangle(Lx=10, Ly=20, origin=(0, 0))
mesh = domain.generate_mesh(resolution=0.5)
print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_cells} cells")

# %% [markdown]
# ## 2. Material Properties
#
# | Property               | Value     |
# |------------------------|-----------|
# | Thermal conductivity   | 1.5 W/(m K)|
# | Dry density            | 1700 kg/m³|
# | Specific heat capacity | 900 J/(kg K)|

# %%
clay = materials.Material(
    name="clay",
    thermal_conductivity=1.5,
    dry_density=1700,
    specific_heat=900,
    porosity=0.40,
)
mat_map = materials.assign(mesh, {"clay": clay})

# %% [markdown]
# ## 3. Boundary Conditions
#
# | Boundary | Condition                             |
# |----------|---------------------------------------|
# | Top      | Dirichlet $T = 40$ °C (heat source)   |
# | Bottom   | Dirichlet $T = 12$ °C (deep ground)   |
# | Sides    | No-flux (adiabatic)                    |

# %%
bcs = boundaries.BoundaryConditions([
    boundaries.Dirichlet(field="T", value=40.0, where=boundaries.top()),
    boundaries.Dirichlet(field="T", value=12.0, where=boundaries.bottom()),
    boundaries.Neumann(field="T", flux=0.0, where=boundaries.left()),
    boundaries.Neumann(field="T", flux=0.0, where=boundaries.right()),
])

# %% [markdown]
# ## 4. Physics and Initial Condition
#
# Initial temperature is a uniform 12 °C (undisturbed ground).

# %%
heat = physics.HeatTransfer(mesh, mat_map)
print(f"Primary field: {heat.primary_field}, transient: {heat.is_transient}")

initial_T = np.full(mesh.n_nodes, 12.0)

# %% [markdown]
# ## 5. Solve — 30 Days of Heating

# %%
stepper = time.Stepper(t_end=30 * 86400, dt=3600)

solver = solvers.FiPyBackend()
solution = solver.solve(
    heat, bcs,
    time=stepper,
    initial_condition={"T": initial_T},
)

print(f"Solved {len(solution.times)} time steps")

# %% [markdown]
# ## 6. Temperature Profiles at Selected Times

# %%
import matplotlib.pyplot as plt
from pygeotech import postprocess

probe = postprocess.LineProbe((5.0, 0.0), (5.0, 20.0), n_points=200)

fig, ax = plt.subplots(figsize=(5, 7))
for t_day in [0, 1, 5, 15, 30]:
    t_sec = t_day * 86400
    idx = np.argmin(np.abs(np.array(solution.times) - t_sec))
    T_snap = solution.field_history["T"][idx]
    dist, vals = probe.sample(mesh, T_snap)
    ax.plot(vals, dist, label=f"Day {t_day}")

ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Depth from bottom (m)")
ax.set_title("Temperature profiles at x = 5 m")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Final Temperature Field

# %%
solution.plot(field="T", contours=20)

# %% [markdown]
# ## Key Takeaways
#
# - Pure conduction produces a smooth diffusion front that propagates
#   downward from the heated surface.
# - The thermal diffusivity $\alpha = \lambda / (\rho c_p)$ governs
#   the rate of front propagation.
# - For geothermal pile design, coupling with advection (groundwater
#   flow) would be the natural next step — see Example 06.
