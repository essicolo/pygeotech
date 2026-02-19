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
# # 07 — Slope Stability: Limit Equilibrium Methods
#
# Evaluates the factor of safety of a simple slope using four
# limit equilibrium methods (LEM):
#
# | Method                     | Force / Moment   | Interslice assumptions |
# |----------------------------|------------------|------------------------|
# | **Bishop Simplified**      | Moment           | Horizontal forces only |
# | **Janbu Simplified**       | Force            | Zero interslice shear  |
# | **Spencer**                | Force + Moment   | Constant θ             |
# | **Morgenstern–Price**      | Force + Moment   | Half-sine f(x)         |
#
# **Module**: `pygeotech.slope`

# %%
import numpy as np
from pygeotech.slope import (
    SlopeProfile,
    CircularSurface,
    bishop_simplified,
    janbu_simplified,
    spencer,
    morgenstern_price,
)

# %% [markdown]
# ## 1. Define the Slope Profile
#
# A 30 m high slope inclined at 2H:1V with a 10 m flat crest and
# 15 m flat toe. Two soil layers:
#
# | Layer   | c' (kPa) | φ' (°) | γ (kN/m³) | Top (m) | Bottom (m) |
# |---------|----------|--------|-----------|---------|------------|
# | Fill    | 5        | 28     | 19        | 30      | 15         |
# | Clay    | 20       | 22     | 18        | 15      | 0          |

# %%
# Surface coordinates (x, y)
surface_pts = [
    (0, 0),
    (15, 0),     # toe
    (75, 30),    # crest (2H:1V slope over 60m horizontal, 30m vertical)
    (85, 30),    # flat crest
    (100, 30),
]

profile = SlopeProfile(
    surface=surface_pts,
    layers=[
        {"c": 5e3, "phi": 28, "gamma": 19e3, "top": 30, "bottom": 15},
        {"c": 20e3, "phi": 22, "gamma": 18e3, "top": 15, "bottom": 0},
    ],
    water_table=None,  # dry slope for simplicity
)

# %% [markdown]
# ## 2. Define a Trial Failure Surface
#
# A circular arc with center at $(40, 50)$ and radius $35$ m.

# %%
trial_surface = CircularSurface(xc=40, yc=50, radius=35)

# %% [markdown]
# ## 3. Compare LEM Methods

# %%
fos_bishop = bishop_simplified(profile, trial_surface, n_slices=30)
fos_janbu = janbu_simplified(profile, trial_surface, n_slices=30)
fos_spencer = spencer(profile, trial_surface, n_slices=30)
fos_mp = morgenstern_price(profile, trial_surface, n_slices=30)

print("Factor of Safety — single trial surface:")
print(f"  Bishop Simplified:      {fos_bishop:.3f}")
print(f"  Janbu Simplified:       {fos_janbu:.3f}")
print(f"  Spencer:                {fos_spencer:.3f}")
print(f"  Morgenstern–Price:      {fos_mp:.3f}")

# %% [markdown]
# ## 4. Effect of Water Table
#
# Add a water table and observe the reduction in FoS.

# %%
wt_pts = [
    (0, 0),
    (15, 5),    # 5 m above toe
    (50, 20),   # mid-slope
    (75, 28),   # near crest
    (100, 28),
]

profile_wet = SlopeProfile(
    surface=surface_pts,
    layers=[
        {"c": 5e3, "phi": 28, "gamma": 19e3, "top": 30, "bottom": 15},
        {"c": 20e3, "phi": 22, "gamma": 18e3, "top": 15, "bottom": 0},
    ],
    water_table=wt_pts,
)

fos_dry = bishop_simplified(profile, trial_surface, n_slices=30)
fos_wet = bishop_simplified(profile_wet, trial_surface, n_slices=30)

print(f"\nBishop FoS — dry:   {fos_dry:.3f}")
print(f"Bishop FoS — with WT: {fos_wet:.3f}")
print(f"Reduction: {(1 - fos_wet / fos_dry) * 100:.1f}%")

# %% [markdown]
# ## 5. Sensitivity to Slice Count

# %%
import matplotlib.pyplot as plt

n_values = [5, 10, 15, 20, 30, 50, 75, 100]
fos_values = [bishop_simplified(profile, trial_surface, n_slices=n) for n in n_values]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(n_values, fos_values, "bo-")
ax.set_xlabel("Number of slices")
ax.set_ylabel("Factor of Safety (Bishop)")
ax.set_title("Convergence with number of slices")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Takeaways
#
# - Bishop and Spencer (moment-based) typically give similar FoS.
# - Janbu simplified (force-based) is often slightly conservative.
# - Morgenstern–Price satisfies both force and moment equilibrium.
# - Water table presence reduces FoS by reducing effective stresses.
# - Convergence with slice count is rapid — 30 slices is usually
#   sufficient.
# - This analysis uses a **single trial surface**; see Example 08
#   for automated critical surface search.
