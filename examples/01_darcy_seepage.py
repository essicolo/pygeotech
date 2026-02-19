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
# # 01 — Steady-State Darcy Seepage Through a Dam
#
# This example solves steady-state saturated groundwater flow (Darcy's law)
# through a rectangular earth dam with a head difference between upstream
# and downstream faces.
#
# **Governing equation:**
#
# $$\nabla \cdot (K \nabla H) = 0$$
#
# where $H$ is the total hydraulic head and $K$ is the saturated
# hydraulic conductivity.
#
# **Physics**: `pygeotech.physics.Darcy`
# **Solver**: `pygeotech.solvers.FiPyBackend` (direct sparse solver)

# %%
import numpy as np
from pygeotech import geometry, materials, boundaries, physics, solvers

# %% [markdown]
# ## 1. Define the Domain
#
# A 60 m wide × 20 m tall rectangular dam cross-section.

# %%
dam = geometry.Rectangle(Lx=60, Ly=20, origin=(0, 0))
mesh = dam.generate_mesh(resolution=1.5)
print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_cells} cells")

# %% [markdown]
# ## 2. Material Properties
#
# Homogeneous isotropic dam fill with $K = 1 \times 10^{-5}$ m/s.

# %%
fill = materials.Material(
    name="dam_fill",
    hydraulic_conductivity=1e-5,
    porosity=0.30,
)
mat_map = materials.assign(mesh, {"dam_fill": fill})

# %% [markdown]
# ## 3. Boundary Conditions
#
# | Boundary | Condition                     |
# |----------|-------------------------------|
# | Left     | $H = 18$ m (upstream level)   |
# | Right    | $H = 4$ m (downstream level)  |
# | Top      | No-flow (Neumann, $q_n = 0$)  |
# | Bottom   | No-flow (Neumann, $q_n = 0$)  |

# %%
bcs = boundaries.BoundaryConditions([
    boundaries.Dirichlet(field="H", value=18.0, where=boundaries.left()),
    boundaries.Dirichlet(field="H", value=4.0, where=boundaries.right()),
    boundaries.Neumann(field="H", flux=0.0, where=boundaries.top()),
    boundaries.Neumann(field="H", flux=0.0, where=boundaries.bottom()),
])

# %% [markdown]
# ## 4. Physics Module

# %%
darcy = physics.Darcy(mesh, mat_map)
print(f"Primary field: {darcy.primary_field}")

# %% [markdown]
# ## 5. Solve

# %%
solver = solvers.FiPyBackend()
solution = solver.solve(darcy, bcs)

H = solution["H"]
print(f"Head range: {H.min():.2f} – {H.max():.2f} m")

# %% [markdown]
# ## 6. Post-processing
#
# Plot the head field with flow streamlines and compute the total seepage
# flux through the downstream face.

# %%
from pygeotech import postprocess

velocity = postprocess.compute_velocity(solution)
print(f"Max Darcy velocity: {np.linalg.norm(velocity, axis=1).max():.2e} m/s")

# %%
# Contour plot of total head with streamlines
solution.plot(field="H", contours=20, streamlines=True)

# %%
# Compute seepage flux through the right (downstream) face
flux_out = postprocess.integrate_flux(solution, locator=boundaries.right())
print(f"Seepage discharge: {flux_out:.4e} m³/s per metre of dam")

# %% [markdown]
# ## 7. Head Profile Along the Base
#
# Extract head values along the bottom of the dam.

# %%
probe = postprocess.LineProbe((0, 0), (60, 0), n_points=100)
distances, h_values = probe.sample(mesh, H)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(distances, h_values, "b-")
ax.set_xlabel("Distance along base (m)")
ax.set_ylabel("Total head H (m)")
ax.set_title("Head distribution along the dam base")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Takeaways
#
# - For homogeneous isotropic media the head distribution is linear
#   between the two fixed-head boundaries.
# - The Darcy module solves a single sparse linear system — no iteration
#   needed.
# - `postprocess.compute_velocity` computes cell-centred Darcy velocity
#   $\mathbf{v} = -K \nabla H$.
