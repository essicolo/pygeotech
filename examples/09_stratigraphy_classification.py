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
# # 09 — Borehole Classification and Stratigraphic Modelling (2-D)
#
# This example demonstrates the stratigraphy workflow:
#
# 1. **Load** borehole data (from CSV or in-memory)
# 2. **Classify** intervals into stratigraphic units using clustering
# 3. **Interpolate** boundary surfaces (RBF, linear, cubic, kriging)
# 4. **Query** the model and tag a finite-element mesh
#
# **Module**: `pygeotech.stratigraphy`

# %%
import numpy as np
from pygeotech.stratigraphy import (
    Layer, Borehole, BoreholeSet,
    cluster_logs, StratigraphicModel,
)

# %% [markdown]
# ## 1. Synthetic Borehole Data
#
# Five boreholes along a 2-D cross-section (y = 0).  Each has three
# layers: **fill** (soft), **sand** (medium), **clay** (stiff),
# with varying thicknesses.

# %%
def make_boreholes():
    """Create a synthetic 2-D borehole dataset."""
    bh_data = [
        # (id, x, z_surface, [(z_top, z_bot, unit, N_SPT), ...])
        ("BH-1", 0, 10, [
            (10, 7, "fill", 5),
            (7, 3, "sand", 20),
            (3, -5, "clay", 8),
        ]),
        ("BH-2", 25, 12, [
            (12, 9, "fill", 4),
            (9, 4, "sand", 22),
            (4, -5, "clay", 10),
        ]),
        ("BH-3", 50, 11, [
            (11, 8, "fill", 3),
            (8, 2, "sand", 18),
            (2, -5, "clay", 7),
        ]),
        ("BH-4", 75, 13, [
            (13, 10, "fill", 6),
            (10, 5, "sand", 25),
            (5, -5, "clay", 12),
        ]),
        ("BH-5", 100, 10, [
            (10, 6, "fill", 4),
            (6, 1, "sand", 19),
            (1, -5, "clay", 9),
        ]),
    ]

    boreholes = []
    for bh_id, x, z_s, layers in bh_data:
        layer_objs = [
            Layer(z_top=zt, z_bottom=zb, unit=u, properties={"N_SPT": n})
            for zt, zb, u, n in layers
        ]
        boreholes.append(Borehole(id=bh_id, x=x, y=0.0, layers=layer_objs))
    return BoreholeSet(boreholes)

bhs = make_boreholes()
print(bhs)
print(f"Dimension: {bhs.dim}-D")
print(f"Units: {bhs.unit_names()}")
print(f"Stratigraphic column: {bhs.stratigraphic_column()}")

# %% [markdown]
# ## 2. Automated Classification (Optional)
#
# If intervals lack unit labels, `cluster_logs` classifies them using
# measured properties (here: N_SPT + depth).

# %%
# Create unlabelled copy for demonstration
bhs_unlabelled = make_boreholes()
for bh in bhs_unlabelled:
    for layer in bh.layers:
        layer.unit = ""  # clear labels

# Cluster into 3 units using GMM
unit_names = cluster_logs(
    bhs_unlabelled,
    n_units=3,
    method="gmm",
    features=["N_SPT"],
    depth_weight=0.8,
    stratigraphic_order=["fill", "sand", "clay"],
)
print(f"Assigned units (top → bottom): {unit_names}")

# Verify a single borehole
for bh in bhs_unlabelled:
    seq = bh.unit_sequence()
    print(f"  {bh.id}: {seq}")

# %% [markdown]
# ## 3. Build and Interpolate the Stratigraphic Model

# %%
model = StratigraphicModel(bhs)
print(f"Before interpolation: {model}")

model.interpolate(
    method="rbf",
    kernel="thin_plate_spline",
    smoothing=0.0,
    non_crossing="sequential",
    min_thickness=0.1,
)
print(f"After interpolation:  {model}")

# %% [markdown]
# ## 4. Evaluate Surfaces Along the Section

# %%
import matplotlib.pyplot as plt

x_query = np.linspace(0, 100, 200)
surfaces = model.evaluate_surfaces(x_query)

fig, ax = plt.subplots(figsize=(12, 5))

# Plot surfaces
colors = {"topography": "brown", "base": "gray"}
for name, z in surfaces.items():
    c = colors.get(name, None)
    ax.plot(x_query, z, label=name, color=c)

# Plot borehole sticks
for bh in bhs:
    for layer in bh.layers:
        ax.plot([bh.x, bh.x], [layer.z_bottom, layer.z_top],
                linewidth=4, alpha=0.6,
                color={"fill": "orange", "sand": "gold", "clay": "steelblue"}.get(layer.unit, "gray"))
    ax.plot(bh.x, bh.z_surface, "kv", markersize=8)

ax.set_xlabel("x (m)")
ax.set_ylabel("Elevation (m)")
ax.set_title("Interpolated stratigraphic boundaries (RBF)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Query the Model at a Point

# %%
test_x, test_z = 60.0, 6.0
unit = model.unit_at(test_x, test_z)
print(f"Unit at (x={test_x}, z={test_z}): {unit}")

# Query several depths
for z in [12, 9, 6, 3, 0, -3]:
    u = model.unit_at(50.0, z)
    print(f"  x=50, z={z:+3d}: {u}")

# %% [markdown]
# ## 6. Compare Interpolation Methods

# %%
methods = ["rbf", "linear", "cubic", "kriging"]
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)

for ax, m in zip(axes.flat, methods):
    mdl = StratigraphicModel(bhs)
    mdl.interpolate(method=m, non_crossing="sequential", min_thickness=0.1)
    surfs = mdl.evaluate_surfaces(x_query)

    for name, z in surfs.items():
        ax.plot(x_query, z, label=name)

    # Borehole data
    for bh in bhs:
        ax.plot([bh.x, bh.x], [bh.z_bottom, bh.z_surface],
                "k-", linewidth=2, alpha=0.3)
        ax.plot(bh.x, bh.z_surface, "kv", markersize=6)

    ax.set_title(m.upper())
    ax.grid(True, alpha=0.2)

axes[0, 0].legend(fontsize=7, loc="upper right")
axes[1, 0].set_xlabel("x (m)")
axes[1, 1].set_xlabel("x (m)")
axes[0, 0].set_ylabel("Elevation (m)")
axes[1, 0].set_ylabel("Elevation (m)")
fig.suptitle("Interpolation method comparison", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Tag a Mesh with Stratigraphic Units

# %%
from pygeotech import geometry, materials

domain = geometry.Rectangle(Lx=100, Ly=20, origin=(0, -5))
mesh = domain.generate_mesh(resolution=1.5)

model.tag_mesh(mesh)
print(f"Cell tags shape: {mesh.cell_tags.shape}")
print(f"Subdomain map: {mesh.subdomain_map}")

# Visualise
centers = mesh.cell_centers()
fig, ax = plt.subplots(figsize=(12, 4))
sc = ax.scatter(
    centers[:, 0], centers[:, 1],
    c=mesh.cell_tags, cmap="Set2", s=5, alpha=0.8,
)
cbar = plt.colorbar(sc, ax=ax, ticks=list(mesh.subdomain_map.values()))
cbar.ax.set_yticklabels(list(mesh.subdomain_map.keys()))
ax.set_xlabel("x (m)")
ax.set_ylabel("Elevation (m)")
ax.set_title("Mesh tagged with stratigraphic units")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Loading from CSV
#
# In practice, borehole data comes from a CSV file. The expected
# format:
#
# ```csv
# borehole_id,x,y,z_top,z_bottom,unit,N_SPT,water_content
# BH-1,0,0,10,7,fill,5,22.3
# BH-1,0,0,7,3,sand,20,15.1
# ...
# ```
#
# Load with:
#
# ```python
# bhs = BoreholeSet.from_csv(
#     "drill_logs.csv",
#     id_col="borehole_id",
#     x_col="x",
#     y_col="y",
#     z_top_col="z_top",
#     z_bottom_col="z_bottom",
#     unit_col="unit",
#     property_cols=["N_SPT", "water_content"],
# )
# ```

# %% [markdown]
# ## Key Takeaways
#
# - `cluster_logs` uses scikit-learn (GMM, K-Means, Agglomerative)
#   to auto-classify intervals. Depth weighting encourages
#   stratigraphically coherent clusters.
# - RBF interpolation with thin-plate splines gives smooth surfaces.
#   Kriging provides uncertainty estimates (future extension).
# - Non-crossing enforcement (`"sequential"`) clips each surface to
#   remain below the one above — essential for geological realism.
# - `tag_mesh` assigns unit indices to FEM cells for spatially varying
#   material assignment.
