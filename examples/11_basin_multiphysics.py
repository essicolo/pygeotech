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
# # 11 — Multi-Layer Basin: Coupled Seepage and Contaminant Transport
#
# A realistic scenario combining **stratigraphy**, **flow**, and
# **transport** in a multi-layer alluvial basin. The workflow:
#
# 1. Define stratigraphy from borehole data
# 2. Build a mesh and tag it with geological units
# 3. Assign spatially varying material properties
# 4. Solve steady-state seepage (Darcy)
# 5. Simulate contaminant migration
#
# This example shows how the different pygeotech modules connect.

# %%
import numpy as np
import matplotlib.pyplot as plt
from pygeotech import (
    geometry, materials, boundaries, physics,
    solvers, time, postprocess,
)
from pygeotech.stratigraphy import (
    Layer, Borehole, BoreholeSet, StratigraphicModel,
)

# %% [markdown]
# ## 1. Borehole Data — Alluvial Basin
#
# Four boreholes defining three units: **alluvium** (permeable),
# **lacustrine clay** (low-K aquitard), and **gravel aquifer** (deep).

# %%
bh_data = [
    ("BH-A", 0, [
        (20, 14, "alluvium", 1e-4),
        (14, 8, "clay", 1e-8),
        (8, 0, "gravel", 5e-3),
    ]),
    ("BH-B", 80, [
        (22, 16, "alluvium", 1e-4),
        (16, 10, "clay", 1e-8),
        (10, 0, "gravel", 5e-3),
    ]),
    ("BH-C", 160, [
        (18, 11, "alluvium", 8e-5),
        (11, 6, "clay", 5e-9),
        (6, 0, "gravel", 3e-3),
    ]),
    ("BH-D", 240, [
        (21, 15, "alluvium", 1e-4),
        (15, 9, "clay", 1e-8),
        (9, 0, "gravel", 5e-3),
    ]),
]

boreholes = []
for bh_id, x, layers in bh_data:
    layer_objs = [
        Layer(z_top=zt, z_bottom=zb, unit=u, properties={"K": k})
        for zt, zb, u, k in layers
    ]
    boreholes.append(Borehole(id=bh_id, x=x, y=0.0, layers=layer_objs))

bhs = BoreholeSet(boreholes)
print(bhs)

# %% [markdown]
# ## 2. Build Stratigraphic Model and Tag the Mesh

# %%
strat_model = StratigraphicModel(bhs)
strat_model.interpolate(method="cubic", non_crossing="sequential", min_thickness=0.2)

# Evaluate surfaces
x_plot = np.linspace(0, 240, 300)
surfs = strat_model.evaluate_surfaces(x_plot)

# %%
# Create the mesh
domain = geometry.Rectangle(Lx=240, Ly=25, origin=(0, 0))
mesh = domain.generate_mesh(resolution=2.0)

# Tag cells with stratigraphy
strat_model.tag_mesh(mesh)
print(f"Subdomain map: {mesh.subdomain_map}")

# %% [markdown]
# ## 3. Assign Spatially Varying Materials
#
# Each geological unit gets distinct hydraulic and transport properties.

# %%
alluvium = materials.Material(
    name="alluvium",
    hydraulic_conductivity=1e-4,
    porosity=0.35,
)
clay = materials.Material(
    name="clay",
    hydraulic_conductivity=1e-8,
    porosity=0.50,
)
gravel = materials.Material(
    name="gravel",
    hydraulic_conductivity=5e-3,
    porosity=0.30,
)

mat_map = materials.assign(mesh, {
    "alluvium": alluvium,
    "clay": clay,
    "gravel": gravel,
})

# Verify: conductivity varies by orders of magnitude
K_field = mat_map.cell_property("hydraulic_conductivity")
print(f"K range: {K_field.min():.1e} – {K_field.max():.1e} m/s")

# %% [markdown]
# ## 4. Steady-State Seepage
#
# River at left ($H = 20$ m), pumping well at right ($H = 15$ m).

# %%
darcy = physics.Darcy(mesh, mat_map)

flow_bcs = boundaries.BoundaryConditions([
    boundaries.Dirichlet(field="H", value=20.0, where=boundaries.left()),
    boundaries.Dirichlet(field="H", value=15.0, where=boundaries.right()),
    boundaries.Neumann(field="H", flux=0.0, where=boundaries.top()),
    boundaries.Neumann(field="H", flux=0.0, where=boundaries.bottom()),
])

solver = solvers.FiPyBackend()
flow_solution = solver.solve(darcy, flow_bcs)

H = flow_solution["H"]
velocity = postprocess.compute_velocity(flow_solution)
print(f"Head range: {H.min():.2f} – {H.max():.2f} m")

# %%
# Visualise head field
solution_plot = flow_solution.plot(field="H", contours=20, streamlines=True)

# %% [markdown]
# ## 5. Contaminant Transport
#
# A spill enters the alluvium at the left boundary between $z = 14$ m
# and $z = 20$ m. Track its migration for 10 years.

# %%
transport = physics.Transport(
    mesh, mat_map,
    dispersion_longitudinal=10.0,
    dispersion_transverse=1.0,
    molecular_diffusion=1e-9,
    retardation_factor=1.5,
    decay_rate=0.0,
)
transport.set_velocity(velocity)

spill_zone = boundaries.left() & boundaries.y_between(14.0, 20.0)

transport_bcs = boundaries.BoundaryConditions([
    boundaries.Dirichlet(field="C", value=1.0, where=spill_zone),
    boundaries.Dirichlet(field="C", value=0.0,
                         where=boundaries.left() & ~boundaries.y_between(14.0, 20.0)),
    boundaries.Neumann(field="C", flux=0.0, where=boundaries.right()),
    boundaries.Neumann(field="C", flux=0.0, where=boundaries.top()),
    boundaries.Neumann(field="C", flux=0.0, where=boundaries.bottom()),
])

stepper = time.Stepper(t_end=10 * 365.25 * 86400, dt=30 * 86400)

solution = solver.solve(
    transport, transport_bcs,
    time=stepper,
    initial_condition={"C": np.zeros(mesh.n_nodes)},
)

# %% [markdown]
# ## 6. Plume Evolution

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, t_yr in zip(axes.flat, [1, 3, 5, 10]):
    t_sec = t_yr * 365.25 * 86400
    idx = np.argmin(np.abs(np.array(solution.times) - t_sec))
    C = solution.field_history["C"][idx]
    solution.plot(field="C", contours=15)
    ax.set_title(f"Year {t_yr}")

plt.suptitle("Contaminant plume in a layered basin", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Stratigraphy Cross-Section with Flow Overlay

# %%
fig, ax = plt.subplots(figsize=(14, 5))

# Layer fills
surface_names = [s for s in strat_model._surface_order if s in surfs]
unit_colors = {"alluvium": "#E8C07D", "clay": "#7B9EA8", "gravel": "#C4956A"}

column = strat_model.column
for i, unit in enumerate(column):
    if i + 1 < len(surface_names):
        z_top = surfs[surface_names[i]]
        z_bot = surfs[surface_names[i + 1]]
        ax.fill_between(x_plot, z_bot, z_top,
                        alpha=0.4, color=unit_colors.get(unit, f"C{i}"),
                        label=unit)

for name in surface_names:
    ax.plot(x_plot, surfs[name], "k-", linewidth=0.5)

# Borehole locations
for bh in bhs:
    ax.plot([bh.x, bh.x], [bh.z_bottom, bh.z_surface], "k-", linewidth=3, alpha=0.4)
    ax.plot(bh.x, bh.z_surface, "kv", markersize=8)

ax.set_xlabel("x (m)")
ax.set_ylabel("Elevation (m)")
ax.set_title("Basin stratigraphy with borehole control")
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Takeaways
#
# - The stratigraphy module assigns spatially varying $K$ across the
#   mesh — the clay aquitard acts as a barrier to flow and transport.
# - Contaminant migration is strongly channelled through the higher-K
#   alluvium and gravel layers.
# - This workflow (stratigraphy → mesh tagging → material assignment →
#   physics) is the standard approach for real-site modelling.
# - For real projects, replace synthetic boreholes with a CSV import
#   using `BoreholeSet.from_csv()`.
