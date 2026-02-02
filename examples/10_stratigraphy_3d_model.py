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
# # 10 — 3-D Stratigraphic Model from Boreholes
#
# Builds a three-dimensional stratigraphic model from spatially
# distributed boreholes and extracts cross-sections.
#
# **Module**: `pygeotech.stratigraphy`

# %%
import numpy as np
import matplotlib.pyplot as plt
from pygeotech.stratigraphy import (
    Layer, Borehole, BoreholeSet,
    StratigraphicModel, OrdinaryKriging,
)

# %% [markdown]
# ## 1. Synthetic 3-D Borehole Data
#
# Nine boreholes on a 3 × 3 grid over a 200 m × 200 m site with
# three geological units: **topsoil**, **gravel**, **bedrock**.
# Layer elevations vary spatially to mimic a buried channel.

# %%
np.random.seed(42)

# Grid locations
xs = [0, 100, 200]
ys = [0, 100, 200]
bh_list = []
bh_count = 0

for xi in xs:
    for yi in ys:
        bh_count += 1
        # Surface elevation: gentle dome
        z_surface = 25 + 3 * np.sin(np.pi * xi / 200) * np.sin(np.pi * yi / 200)
        # Gravel top: buried channel deeper in the center
        z_gravel_top = z_surface - 3 - 2 * np.exp(-((xi - 100)**2 + (yi - 100)**2) / 5000)
        # Bedrock top: relatively flat with some undulation
        z_bedrock_top = 10 + 1.5 * np.sin(np.pi * xi / 200) + np.random.normal(0, 0.3)
        z_base = 0.0

        layers = [
            Layer(z_top=z_surface, z_bottom=z_gravel_top, unit="topsoil",
                  properties={"N_SPT": np.random.uniform(3, 8)}),
            Layer(z_top=z_gravel_top, z_bottom=z_bedrock_top, unit="gravel",
                  properties={"N_SPT": np.random.uniform(25, 50)}),
            Layer(z_top=z_bedrock_top, z_bottom=z_base, unit="bedrock",
                  properties={"N_SPT": np.random.uniform(50, 100)}),
        ]
        bh_list.append(Borehole(
            id=f"BH-{bh_count:02d}",
            x=float(xi), y=float(yi),
            layers=layers,
        ))

bhs = BoreholeSet(bh_list)
print(bhs)
print(f"Dimension: {bhs.dim}-D")
print(f"Column: {bhs.stratigraphic_column()}")

# %% [markdown]
# ## 2. Build and Interpolate the 3-D Model

# %%
model = StratigraphicModel(bhs)
model.interpolate(
    method="rbf",
    kernel="thin_plate_spline",
    non_crossing="sequential",
    min_thickness=0.2,
)
print(model)

# %% [markdown]
# ## 3. Evaluate Surfaces on a Regular Grid

# %%
nx, ny = 50, 50
x_grid = np.linspace(0, 200, nx)
y_grid = np.linspace(0, 200, ny)
X, Y = np.meshgrid(x_grid, y_grid)
x_flat = X.ravel()
y_flat = Y.ravel()

surfaces = model.evaluate_surfaces(x_flat, y_flat)

# %% [markdown]
# ## 4. Surface Maps

# %%
fig, axes = plt.subplots(1, len(surfaces), figsize=(5 * len(surfaces), 4))
if len(surfaces) == 1:
    axes = [axes]

for ax, (name, z) in zip(axes, surfaces.items()):
    Z = z.reshape(ny, nx)
    im = ax.contourf(X, Y, Z, levels=20, cmap="terrain")
    plt.colorbar(im, ax=ax, label="Elevation (m)")
    # Plot borehole locations
    locs = bhs.locations
    ax.plot(locs[:, 0], locs[:, 1], "ko", markersize=5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(name)
    ax.set_aspect("equal")

fig.suptitle("Interpolated boundary surfaces (3-D)", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Cross-Section

# %%
section = model.cross_section(
    p0=(0, 100),
    p1=(200, 100),
    n_points=150,
)

fig, ax = plt.subplots(figsize=(12, 4))

# Fill between surfaces
surface_names = [s for s in model._surface_order if s in section]
unit_colors = {"topsoil": "#8B4513", "gravel": "#DAA520", "bedrock": "#808080"}

column = model.column
for i, unit in enumerate(column):
    if i + 1 < len(surface_names):
        z_top = section[surface_names[i]]
        z_bot = section[surface_names[i + 1]]
        ax.fill_between(
            section["distance"], z_bot, z_top,
            alpha=0.6,
            color=unit_colors.get(unit, f"C{i}"),
            label=unit,
        )

# Plot surface lines
for name in surface_names:
    ax.plot(section["distance"], section[name], "k-", linewidth=0.5)

ax.set_xlabel("Distance along section (m)")
ax.set_ylabel("Elevation (m)")
ax.set_title("Cross-section at y = 100 m")
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Point Queries

# %%
test_points = [(100, 100, 20), (100, 100, 12), (100, 100, 5)]
for x, y, z in test_points:
    unit = model.unit_at(x, y, z)
    print(f"  ({x}, {y}, z={z}): {unit}")

# %% [markdown]
# ## 7. Kriging Comparison
#
# Use ordinary kriging with an exponential variogram for comparison.

# %%
model_krig = StratigraphicModel(bhs)
model_krig.interpolate(
    method="kriging",
    variogram="exponential",
    non_crossing="sequential",
)

surfaces_krig = model_krig.evaluate_surfaces(x_flat, y_flat)

# Compare topography
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (title, surfs) in zip(axes, [("RBF", surfaces), ("Kriging", surfaces_krig)]):
    Z = surfs["topography"].reshape(ny, nx)
    im = ax.contourf(X, Y, Z, levels=20, cmap="terrain")
    plt.colorbar(im, ax=ax, label="Elevation (m)")
    locs = bhs.locations
    ax.plot(locs[:, 0], locs[:, 1], "ko", markersize=5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Topography — {title}")
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Takeaways
#
# - 3-D models require boreholes with varying $x$ and $y$ coordinates.
# - RBF and kriging both produce smooth interpolated surfaces; kriging
#   can additionally estimate prediction uncertainty (extension).
# - Cross-sections can be extracted at any orientation through the
#   3-D model.
# - The built-in `OrdinaryKriging` auto-fits the variogram range and
#   sill from the data — suitable for quick estimates. For production
#   use, consider fitting the variogram experimentally.
