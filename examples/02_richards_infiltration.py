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
# # 02 — Transient Richards Infiltration in a Soil Column
#
# Simulates rainfall infiltration into an initially unsaturated soil
# column using the Richards equation with van Genuchten retention.
#
# **Governing equation:**
#
# $$\nabla \cdot \bigl[K(h)\,(\nabla h + \nabla z)\bigr] = C(h)\,\frac{\partial h}{\partial t}$$
#
# where $h$ is pressure head, $K(h)$ is the unsaturated hydraulic
# conductivity (Mualem–van Genuchten), and $C(h) = d\theta/dh$ is
# the specific moisture capacity.
#
# **Physics**: `pygeotech.physics.Richards`
# **Solver**: `pygeotech.solvers.FiPyBackend` (Picard iteration)

# %%
import numpy as np
from pygeotech import geometry, materials, boundaries, physics, solvers, time

# %% [markdown]
# ## 1. Domain — 1-D Soil Column
#
# A 2 m tall column (width 0.1 m for the 2-D mesh).

# %%
column = geometry.Rectangle(Lx=0.1, Ly=2.0, origin=(0, 0))
mesh = column.generate_mesh(resolution=0.04)
print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_cells} cells")

# %% [markdown]
# ## 2. Material — Sandy Loam with van Genuchten Parameters
#
# | Parameter        | Value                |
# |------------------|----------------------|
# | $K_s$            | $1.23 \times 10^{-5}$ m/s |
# | $\theta_r$       | 0.065                |
# | $\theta_s$       | 0.41                 |
# | $\alpha$         | 7.5 m⁻¹             |
# | $n$              | 1.89                 |

# %%
retention = materials.VanGenuchten(
    alpha=7.5,
    n=1.89,
    theta_r=0.065,
    theta_s=0.41,
)

soil = materials.Material(
    name="sandy_loam",
    hydraulic_conductivity=1.23e-5,
    porosity=0.41,
)
mat_map = materials.assign(mesh, {"sandy_loam": soil})

# %% [markdown]
# ## 3. Boundary Conditions
#
# | Boundary | Condition                                     |
# |----------|-----------------------------------------------|
# | Top      | Neumann flux $q = 5 \times 10^{-6}$ m/s (rain)|
# | Bottom   | Dirichlet $H = 0$ m (water table)              |
# | Sides    | No-flow                                        |

# %%
bcs = boundaries.BoundaryConditions([
    boundaries.Neumann(field="H", flux=-5e-6, where=boundaries.top()),
    boundaries.Dirichlet(field="H", value=0.0, where=boundaries.bottom()),
    boundaries.Neumann(field="H", flux=0.0, where=boundaries.left()),
    boundaries.Neumann(field="H", flux=0.0, where=boundaries.right()),
])

# %% [markdown]
# ## 4. Physics — Richards with Van Genuchten Retention

# %%
richards = physics.Richards(mesh, mat_map, retention_model=retention)
print(f"Primary field: {richards.primary_field}, transient: {richards.is_transient}")

# %% [markdown]
# ## 5. Time Stepping
#
# Simulate 24 hours with 15-minute time steps.

# %%
stepper = time.Stepper(t_end=86400, dt=900, t_start=0)
print(f"Number of time steps: {stepper.n_steps}")

# %% [markdown]
# ## 6. Initial Condition
#
# Hydrostatic profile: $H = z$ (pressure head $h = 0$ everywhere),
# giving a fully saturated initial state.  We start with a drier
# column: $H = z - 1$ (suction of 1 m).

# %%
node_y = mesh.boundary_node_coords()  # only needed for reference
initial_H = mesh.nodes[:, 1] - 1.0  # H = z - 1 (unsaturated)

# %% [markdown]
# ## 7. Solve

# %%
solver = solvers.FiPyBackend(picard_max_iter=50, picard_tol=1e-6)
solution = solver.solve(
    richards, bcs,
    time=stepper,
    initial_condition={"H": initial_H},
)

print(f"Solved {len(solution.times)} time steps")

# %% [markdown]
# ## 8. Visualise the Wetting Front
#
# Plot the pressure head profile at selected times.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 7))

# Sample at selected times
from pygeotech import postprocess

probe = postprocess.LineProbe((0.05, 0.0), (0.05, 2.0), n_points=200)

for t_hr in [0, 2, 6, 12, 24]:
    t_sec = t_hr * 3600
    # Find closest stored time
    idx = np.argmin(np.abs(np.array(solution.times) - t_sec))
    H_snap = solution.field_history["H"][idx]
    h_snap = H_snap - mesh.nodes[:, 1]  # pressure head = H - z
    dist, vals = probe.sample(mesh, h_snap)
    ax.plot(vals, dist, label=f"t = {t_hr} h")

ax.set_xlabel("Pressure head h (m)")
ax.set_ylabel("Elevation z (m)")
ax.set_title("Wetting front propagation")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Takeaways
#
# - Richards uses Picard iteration to handle the nonlinear $K(h)$ and
#   $C(h)$ terms at each time step.
# - The wetting front advances downward as rainfall infiltrates.
# - The van Genuchten model provides a smooth relationship between
#   pressure head and water content / conductivity.
