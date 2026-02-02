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
# # 08 — Critical Failure Surface Search
#
# Finds the circular failure surface with the minimum factor of safety
# using a grid search over the circle center and radius space.
#
# Also demonstrates the Strength Reduction Method (SRM) as an
# alternative FEM-based approach.
#
# **Module**: `pygeotech.slope`

# %%
import numpy as np
from pygeotech.slope import (
    SlopeProfile,
    CircularSurface,
    bishop_simplified,
    grid_search,
)

# %% [markdown]
# ## 1. Slope Profile (same as Example 07)

# %%
surface_pts = [
    (0, 0), (15, 0), (75, 30), (85, 30), (100, 30),
]

profile = SlopeProfile(
    surface=surface_pts,
    layers=[
        {"c": 5e3, "phi": 28, "gamma": 19e3, "top": 30, "bottom": 15},
        {"c": 20e3, "phi": 22, "gamma": 18e3, "top": 15, "bottom": 0},
    ],
    water_table=None,
)

# %% [markdown]
# ## 2. Grid Search
#
# Search over a grid of circle centers and radii. The search space
# should cover plausible failure geometries:
#
# - $x_c$: 20 to 60 m (behind the slope face)
# - $y_c$: 30 to 55 m (above the slope crest)
# - $r$: 15 to 40 m (intersects the slope)

# %%
result = grid_search(
    profile,
    method="bishop",
    xc_range=(20, 60),
    yc_range=(30, 55),
    r_range=(15, 40),
    n_xc=20,
    n_yc=15,
    n_r=12,
)

print(f"Critical FoS:  {result.fos:.3f}")
print(f"Critical surface: xc={result.surface.xc:.1f} m, "
      f"yc={result.surface.yc:.1f} m, r={result.surface.radius:.1f} m")
print(f"Total surfaces evaluated: {len(result.all_results)}")

# %% [markdown]
# ## 3. Visualise the Search Results

# %%
import matplotlib.pyplot as plt

# Extract all (fos, surface) pairs
all_fos = [r[0] for r in result.all_results if np.isfinite(r[0])]
all_xc = [r[1].xc for r in result.all_results if np.isfinite(r[0])]
all_yc = [r[1].yc for r in result.all_results if np.isfinite(r[0])]

fig, ax = plt.subplots(figsize=(9, 5))

# Plot slope surface
sx = [p[0] for p in surface_pts]
sy = [p[1] for p in surface_pts]
ax.fill_between(sx, 0, sy, alpha=0.2, color="brown", label="Slope")
ax.plot(sx, sy, "k-", linewidth=2)

# Plot critical circle
crit = result.surface
theta = np.linspace(0, 2 * np.pi, 200)
cx = crit.xc + crit.radius * np.cos(theta)
cy = crit.yc + crit.radius * np.sin(theta)
ax.plot(cx, cy, "r-", linewidth=2, label=f"Critical (FoS={result.fos:.3f})")
ax.plot(crit.xc, crit.yc, "r+", markersize=10)

# Search grid centers, colored by FoS
sc = ax.scatter(all_xc, all_yc, c=all_fos, cmap="RdYlGn",
                s=8, alpha=0.4, vmin=0.8, vmax=3.0)
plt.colorbar(sc, ax=ax, label="FoS")

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Critical surface search — Bishop simplified")
ax.legend(loc="upper left")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Strength Reduction Method (SRM) — FEM Approach
#
# The SRM progressively reduces $c'$ and $\tan \phi'$ until the
# FEM solution diverges. The critical SRF equals the FoS.

# %%
from pygeotech import geometry, materials, boundaries, physics
from pygeotech.slope import strength_reduction

# Build a mesh of the slope domain
domain = geometry.Rectangle(Lx=100, Ly=30, origin=(0, 0))
mesh = domain.generate_mesh(resolution=2.0)

soil = materials.Material(
    name="slope_soil",
    youngs_modulus=50e6,
    poissons_ratio=0.30,
    dry_density=1800,
    friction_angle=25,
    cohesion=15e3,
)
mat_map = materials.assign(mesh, {"slope_soil": soil})

mech_bcs = boundaries.BoundaryConditions([
    boundaries.Dirichlet(field="ux", value=0.0, where=boundaries.left()),
    boundaries.Dirichlet(field="ux", value=0.0, where=boundaries.right()),
    boundaries.Dirichlet(field="ux", value=0.0, where=boundaries.bottom()),
    boundaries.Dirichlet(field="uy", value=0.0, where=boundaries.bottom()),
])

srm_result = strength_reduction(
    mesh, mat_map, mech_bcs,
    srf_range=(0.5, 3.0),
    n_steps=15,
    bisection_tol=0.02,
)

print(f"\nSRM Factor of Safety: {srm_result.fos:.3f}")
print(f"Converged: {srm_result.converged}")

# %% [markdown]
# ## 5. SRM Displacement vs SRF

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(srm_result.srf_history, srm_result.max_disp_history, "ko-")
ax.axvline(srm_result.fos, color="r", linestyle="--", label=f"FoS = {srm_result.fos:.3f}")
ax.set_xlabel("Strength Reduction Factor (SRF)")
ax.set_ylabel("Max displacement (m)")
ax.set_title("SRM convergence — displacement vs SRF")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Takeaways
#
# - Grid search over $(x_c, y_c, r)$ is simple but effective for
#   circular surfaces. For non-circular, use polyline surfaces with
#   optimisation (future feature).
# - SRM provides a FoS without assuming a failure surface shape.
# - The SRM FoS and LEM FoS should be comparable for simple slopes.
# - SRM additionally provides the failure mechanism (displacement
#   field) as output.
