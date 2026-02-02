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
# # 05 — Elastic Settlement Under a Strip Footing
#
# Computes the displacement field in a soil mass loaded by a rigid
# strip footing using the plane-strain elasticity module.
#
# **Governing equation (quasi-static equilibrium):**
#
# $$\nabla \cdot \boldsymbol{\sigma} + \mathbf{b} = \mathbf{0}$$
#
# with linear-elastic constitutive law:
#
# $$\boldsymbol{\sigma} = \mathbf{C} : \boldsymbol{\varepsilon}$$
#
# **Physics**: `pygeotech.physics.Mechanics`

# %%
import numpy as np
from pygeotech import geometry, materials, boundaries, physics, solvers

# %% [markdown]
# ## 1. Domain
#
# A 30 m wide × 15 m deep soil mass. The strip footing is 4 m wide,
# centered at $x = 15$ m.

# %%
domain = geometry.Rectangle(Lx=30, Ly=15, origin=(0, 0))
mesh = domain.generate_mesh(resolution=0.8)
print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_cells} cells")

footing_half = 2.0  # half-width of footing
footing_x_min = 15 - footing_half
footing_x_max = 15 + footing_half

# %% [markdown]
# ## 2. Material — Stiff Clay
#
# | Property          | Value       |
# |-------------------|-------------|
# | Young's modulus   | 30 MPa      |
# | Poisson's ratio   | 0.35        |
# | Dry density       | 1800 kg/m³  |

# %%
clay = materials.Material(
    name="stiff_clay",
    youngs_modulus=30e6,
    poissons_ratio=0.35,
    dry_density=1800,
)
mat_map = materials.assign(mesh, {"stiff_clay": clay})

# %% [markdown]
# ## 3. Boundary Conditions
#
# | Boundary  | Condition                                    |
# |-----------|----------------------------------------------|
# | Bottom    | Fixed ($u_x = u_y = 0$)                       |
# | Left      | Roller ($u_x = 0$, free in $y$)               |
# | Right     | Roller ($u_x = 0$, free in $y$)               |
# | Top       | Vertical stress under footing ($q = 150$ kPa) |
# | Top       | Traction-free outside footing                  |

# %%
footing_locator = boundaries.top() & boundaries.x_between(footing_x_min, footing_x_max)

bcs = boundaries.BoundaryConditions([
    # Bottom: fully fixed
    boundaries.Dirichlet(field="ux", value=0.0, where=boundaries.bottom()),
    boundaries.Dirichlet(field="uy", value=0.0, where=boundaries.bottom()),
    # Sides: roller (horizontal fixed)
    boundaries.Dirichlet(field="ux", value=0.0, where=boundaries.left()),
    boundaries.Dirichlet(field="ux", value=0.0, where=boundaries.right()),
    # Footing: vertical traction (downward = negative)
    boundaries.Neumann(field="uy", flux=-150e3, where=footing_locator),
])

# %% [markdown]
# ## 4. Physics — Plane-Strain Elasticity (with gravity)

# %%
mech = physics.Mechanics(
    mesh, mat_map,
    gravity=9.81,
    plane_strain=True,
)
print(f"Primary field: {mech.primary_field}")

# %% [markdown]
# ## 5. Solve

# %%
solver = solvers.FiPyBackend()
solution = solver.solve(mech, bcs)

u = solution["u"]  # displacement vector, shape (2 * n_nodes,)
ux = u[0::2]
uy = u[1::2]
print(f"Max vertical settlement: {uy.min():.4f} m ({uy.min()*1e3:.1f} mm)")
print(f"Max horizontal disp:     {np.abs(ux).max():.4f} m")

# %% [markdown]
# ## 6. Settlement Contours

# %%
solution.plot(field="u", contours=20)

# %% [markdown]
# ## 7. Settlement Profile Under the Footing

# %%
import matplotlib.pyplot as plt
from pygeotech import postprocess

probe = postprocess.LineProbe((0, 15), (30, 15), n_points=200)
dist, uy_surface = probe.sample(mesh, uy)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(dist, uy_surface * 1e3, "b-")
ax.axvspan(footing_x_min, footing_x_max, alpha=0.15, color="gray",
           label="Footing")
ax.set_xlabel("x (m)")
ax.set_ylabel("Vertical settlement (mm)")
ax.set_title("Surface settlement profile")
ax.legend()
ax.grid(True, alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Takeaways
#
# - The settlement bowl is centred under the footing and decays
#   laterally.
# - Gravity body forces produce a self-weight stress field; the
#   footing load is superimposed.
# - For layered soils, use `geometry.LayeredProfile` to assign
#   different stiffness to each stratum.
