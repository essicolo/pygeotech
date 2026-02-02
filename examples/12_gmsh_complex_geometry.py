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
# # 12 — Complex Geometry with Gmsh (External Mesh Import)
#
# This example shows how to use **gmsh** to create complex 2-D and
# 3-D geometries and import them into pygeotech.
#
# ## Workflow
#
# ```
# gmsh (Python API) → .msh file → pygeotech.geometry.import_mesh()
# ```
#
# For 3-D models, you can also:
#
# ```
# FreeCAD → .step file → gmsh → .msh → pygeotech
# Blender  → .stl file → gmsh (surface remesh + tet) → .msh → pygeotech
# ```
#
# ---
#
# > **NOTE**: This example requires `gmsh` to be installed:
# > ```
# > pip install gmsh
# > ```
# > If gmsh is not available, the code cells that call the gmsh API
# > will not run, but the mesh import from a pre-generated `.msh` file
# > will still work.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Part A: 2-D Embankment Cross-Section
#
# A trapezoidal embankment on a layered foundation.
#
# ```
#        ___________
#       /           \       ← embankment crest (10m wide, 8m tall)
#      /    embank.   \
#     /       fill      \
#    /___________________\  ← ground level (z = 0)
#    |     alluvium       |
#    |____________________|  ← z = -5
#    |     bedrock        |
#    |____________________|  ← z = -15
# ```

# %%
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
    print("gmsh not installed — skipping mesh generation.")
    print("Install with: pip install gmsh")

# %%
if GMSH_AVAILABLE:
    gmsh.initialize()
    gmsh.model.add("embankment")

    # Characteristic mesh size
    lc_fine = 0.5    # near embankment
    lc_coarse = 2.0  # far field

    # Foundation outline (rectangle: 80m wide, 15m deep)
    p1 = gmsh.model.geo.addPoint(0, -15, 0, lc_coarse)
    p2 = gmsh.model.geo.addPoint(80, -15, 0, lc_coarse)
    p3 = gmsh.model.geo.addPoint(80, 0, 0, lc_coarse)
    p4 = gmsh.model.geo.addPoint(0, 0, 0, lc_coarse)

    # Embankment outline (trapezoid on top of foundation)
    p5 = gmsh.model.geo.addPoint(15, 0, 0, lc_fine)   # left toe
    p6 = gmsh.model.geo.addPoint(25, 8, 0, lc_fine)   # left crest
    p7 = gmsh.model.geo.addPoint(55, 8, 0, lc_fine)   # right crest
    p8 = gmsh.model.geo.addPoint(65, 0, 0, lc_fine)   # right toe

    # Alluvium/bedrock interface
    p9 = gmsh.model.geo.addPoint(0, -5, 0, lc_coarse)
    p10 = gmsh.model.geo.addPoint(80, -5, 0, lc_coarse)

    # --- Foundation lines ---
    l_bottom = gmsh.model.geo.addLine(p1, p2)
    l_right_bed = gmsh.model.geo.addLine(p2, p10)
    l_interface = gmsh.model.geo.addLine(p10, p9)
    l_left_bed = gmsh.model.geo.addLine(p9, p1)

    l_right_alluv = gmsh.model.geo.addLine(p10, p3)
    l_top_right = gmsh.model.geo.addLine(p3, p8)
    l_top_mid = gmsh.model.geo.addLine(p8, p5)
    l_top_left = gmsh.model.geo.addLine(p5, p4)
    l_left_alluv = gmsh.model.geo.addLine(p4, p9)

    # --- Embankment lines ---
    l_emb_left = gmsh.model.geo.addLine(p5, p6)
    l_emb_top = gmsh.model.geo.addLine(p6, p7)
    l_emb_right = gmsh.model.geo.addLine(p7, p8)

    # --- Surfaces ---
    # Bedrock
    cl_bedrock = gmsh.model.geo.addCurveLoop([l_bottom, l_right_bed, l_interface, l_left_bed])
    s_bedrock = gmsh.model.geo.addPlaneSurface([cl_bedrock])

    # Alluvium
    cl_alluvium = gmsh.model.geo.addCurveLoop([
        -l_interface, l_right_alluv, l_top_right, l_top_mid, l_top_left, l_left_alluv
    ])
    s_alluvium = gmsh.model.geo.addPlaneSurface([cl_alluvium])

    # Embankment
    cl_embankment = gmsh.model.geo.addCurveLoop([
        l_emb_left, l_emb_top, l_emb_right, -l_top_mid  # note: reversed
    ])
    # Swap sign to fix orientation: the base of the embankment goes from p5→p8,
    # but l_top_mid goes p8→p5, so -l_top_mid goes p5→p8... let's fix:
    cl_embankment = gmsh.model.geo.addCurveLoop([
        -l_top_mid, l_emb_left, l_emb_top, l_emb_right
    ])
    s_embankment = gmsh.model.geo.addPlaneSurface([cl_embankment])

    # Physical groups (used as subdomain tags in pygeotech)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [s_bedrock], tag=1, name="bedrock")
    gmsh.model.addPhysicalGroup(2, [s_alluvium], tag=2, name="alluvium")
    gmsh.model.addPhysicalGroup(2, [s_embankment], tag=3, name="embankment")

    # Generate mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # Save
    gmsh.write("/tmp/embankment.msh")
    print("Mesh saved to /tmp/embankment.msh")

    gmsh.finalize()

# %% [markdown]
# ## Import into pygeotech

# %%
from pygeotech.geometry import import_mesh

if GMSH_AVAILABLE:
    mesh = import_mesh("/tmp/embankment.msh")
    print(f"Imported mesh: {mesh.n_nodes} nodes, {mesh.n_cells} cells")
    print(f"Subdomain map: {mesh.subdomain_map}")
else:
    print("Skipping import — gmsh mesh not generated.")
    print("To run this example, install gmsh: pip install gmsh")

# %% [markdown]
# ## Assign Materials and Solve Seepage

# %%
if GMSH_AVAILABLE:
    bedrock_mat = materials.Material(
        name="bedrock", hydraulic_conductivity=1e-8, porosity=0.10,
    )
    alluvium_mat = materials.Material(
        name="alluvium", hydraulic_conductivity=1e-5, porosity=0.35,
    )
    embankment_mat = materials.Material(
        name="embankment", hydraulic_conductivity=5e-5, porosity=0.30,
    )

    from pygeotech import materials as mat_mod
    mat_map = mat_mod.assign(mesh, {
        "bedrock": bedrock_mat,
        "alluvium": alluvium_mat,
        "embankment": embankment_mat,
    })

    from pygeotech import physics, solvers, boundaries

    darcy = physics.Darcy(mesh, mat_map)

    bcs = boundaries.BoundaryConditions([
        boundaries.Dirichlet(field="H", value=7.0, where=boundaries.left()),
        boundaries.Dirichlet(field="H", value=2.0, where=boundaries.right()),
        boundaries.Neumann(field="H", flux=0.0, where=boundaries.top()),
        boundaries.Neumann(field="H", flux=0.0, where=boundaries.bottom()),
    ])

    solver = solvers.FiPyBackend()
    solution = solver.solve(darcy, bcs)
    solution.plot(field="H", contours=20, streamlines=True)

# %% [markdown]
# ## Part B: 3-D Geometry from FreeCAD / Blender
#
# For 3-D models, the recommended workflow is:
#
# ### Option 1: FreeCAD → STEP → gmsh
#
# ```python
# # In FreeCAD (or programmatically with cadquery):
# # Export the solid geometry as a STEP file.
#
# # Then in gmsh:
# gmsh.initialize()
# gmsh.model.add("dam_3d")
# gmsh.model.occ.importShapes("dam_geometry.step")
# gmsh.model.occ.synchronize()
# gmsh.model.mesh.generate(3)
# gmsh.write("dam_3d.msh")
# gmsh.finalize()
#
# # Import into pygeotech:
# mesh_3d = import_mesh("dam_3d.msh")
# ```
#
# ### Option 2: Blender → STL → gmsh
#
# ```python
# # Blender exports surface meshes (.stl). These need volumetric
# # meshing in gmsh:
#
# gmsh.initialize()
# gmsh.model.add("slope_3d")
# gmsh.merge("slope_surface.stl")
#
# # Classify and repair the surface mesh
# gmsh.model.mesh.classifySurfaces(
#     angle=40 * np.pi / 180,
#     boundary=True,
#     forReparametrization=True,
# )
# gmsh.model.mesh.createGeometry()
# gmsh.model.geo.synchronize()
#
# # Create a volume from the surface
# sl = gmsh.model.geo.addSurfaceLoop([1])
# gmsh.model.geo.addVolume([sl])
# gmsh.model.geo.synchronize()
#
# # Generate tetrahedral mesh
# gmsh.model.mesh.generate(3)
# gmsh.write("slope_3d.msh")
# gmsh.finalize()
#
# mesh_3d = import_mesh("slope_3d.msh")
# ```
#
# ---
#
# > **HELP NEEDED**: To run Part B, you would need to provide:
# >
# > 1. A `.step` file exported from FreeCAD (e.g. an earth dam or
# >    retaining wall geometry), **or**
# > 2. A `.stl` surface mesh exported from Blender (e.g. a natural
# >    slope terrain).
# >
# > I can then integrate these files into the workflow and generate
# > the volumetric mesh with gmsh.

# %% [markdown]
# ## Part C: Programmatic 3-D Geometry with gmsh OCC Kernel
#
# For simple 3-D shapes, gmsh's OpenCASCADE kernel can build the
# geometry directly.

# %%
if GMSH_AVAILABLE:
    gmsh.initialize()
    gmsh.model.add("box_with_tunnel")

    # A soil block with a cylindrical tunnel
    box = gmsh.model.occ.addBox(0, 0, 0, 50, 20, 30)
    tunnel = gmsh.model.occ.addCylinder(0, 10, 15, 50, 0, 0, 3)

    # Boolean difference: box minus tunnel
    result = gmsh.model.occ.cut([(3, box)], [(3, tunnel)])
    gmsh.model.occ.synchronize()

    # Mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 3.0)
    gmsh.model.mesh.generate(3)
    gmsh.write("/tmp/tunnel.msh")
    print("3-D tunnel mesh saved to /tmp/tunnel.msh")

    gmsh.finalize()

    mesh_3d = import_mesh("/tmp/tunnel.msh")
    print(f"3-D mesh: {mesh_3d.n_nodes} nodes, {mesh_3d.n_cells} cells, dim={mesh_3d.dim}")

# %% [markdown]
# ## Key Takeaways
#
# - **gmsh** is the recommended mesher for complex geometries. Its
#   Python API allows programmatic geometry construction and meshing.
# - Physical groups in gmsh map to `subdomain_map` in pygeotech,
#   enabling per-region material assignment.
# - The OpenCASCADE (OCC) kernel in gmsh supports Boolean operations
#   (union, cut, intersection) — useful for embedded structures
#   (tunnels, piles, walls).
# - For real projects, use **FreeCAD** (STEP export) or **Blender**
#   (STL export) for complex shapes, then mesh with gmsh.
# - `pygeotech.geometry.import_mesh()` reads `.msh`, `.vtk`, `.xdmf`
#   via meshio.
