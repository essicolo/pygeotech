# pygeotech

A modular Python package for modeling coupled processes in geotechnical engineering and geosciences.

## Features

- **Geometry** -- 2-D and 3-D domain definition with rectangles, polygons, circles, boxes, cylinders; CSG boolean operations; stratified geology layers; structured meshing with subdomain refinement.
- **Materials** -- Flexible material property containers; standard library (sand, clay, silt, gravel, concrete, rock); constitutive models (Mohr-Coulomb, Cam-Clay, van Genuchten, Brooks-Corey); spatially variable random fields.
- **Boundaries** -- Dirichlet, Neumann, Robin, Seepage, and Cauchy boundary conditions; composable locator predicates (`top() & x_less_than(40)`); time-varying BCs; internal sources (wells, drains).
- **Physics** -- Darcy (saturated steady-state), Richards (unsaturated transient), solute transport, heat transfer, solid mechanics, SVAT.
- **Coupling** -- Sequential, iterative, and monolithic multiphysics coupling strategies.
- **Solvers** -- FiPy (finite volume), FEniCS (finite element), PINN (physics-informed neural networks), analytical solutions; built-in sparse direct solver for Darcy.
- **Time stepping** -- Fixed and adaptive time steppers; implicit, explicit, and Crank-Nicolson schemes.
- **Post-processing** -- Gradient / velocity computation, point and line probes, flux integrals, VTK/CSV export.
- **Visualization** -- 2-D contour plots with streamlines; 3-D interactive visualization via PyVista.

## Installation

```bash
# Core (numpy, scipy, matplotlib)
pip install -e .

# With optional backends
pip install -e ".[fipy]"       # FiPy finite-volume solver
pip install -e ".[pinn]"       # PyTorch PINN solver
pip install -e ".[meshing]"    # gmsh + meshio
pip install -e ".[viz3d]"      # PyVista 3-D visualization
pip install -e ".[dev]"        # pytest + coverage
pip install -e ".[all]"        # everything
```

Requires Python 3.10+.

## Quickstart

```python
import pygeotech as pgt

# 1. Geometry
domain = pgt.geometry.Rectangle(Lx=120, Ly=20, origin=(0, 0))
cutoff_wall = pgt.geometry.Rectangle(x0=58, y0=8, width=4, height=12)
domain.add_subdomain("cutoff", cutoff_wall)

# 2. Mesh
mesh = domain.generate_mesh(resolution=1.0)

# 3. Materials
foundation = pgt.materials.Material(
    name="sandy_silt",
    hydraulic_conductivity=1e-5,
    porosity=0.35,
)
concrete = pgt.materials.Material(
    name="concrete",
    hydraulic_conductivity=1e-10,
    porosity=0.05,
)
materials = pgt.materials.assign(mesh, {
    "default": foundation,
    "cutoff": concrete,
})

# 4. Physics
flow = pgt.physics.Darcy(mesh, materials)

# 5. Boundary conditions
bc = pgt.boundaries.BoundaryConditions()
bc.add(pgt.boundaries.Dirichlet(
    field="H", value=27.0,
    where=pgt.boundaries.top() & pgt.boundaries.x_less_than(40),
))
bc.add(pgt.boundaries.Dirichlet(
    field="H", value=20.0,
    where=pgt.boundaries.top() & pgt.boundaries.x_greater_than(80),
))

# 6. Solve
solver = pgt.solvers.FiPyBackend()
solution = solver.solve(flow, bc)

# 7. Inspect results
print(solution.fields["H"].min(), solution.fields["H"].max())

# 8. Export
solution.export_csv("head_profile.csv")
```

## Running the dam seepage example

```bash
python -m pygeotech.examples.dam_seepage.run
```

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

## Architecture

```
pygeotech/
├── geometry/       # Domain, subdomains, meshing
├── materials/      # Properties, constitutive models, random fields
├── boundaries/     # BCs, locators, time-varying, internal sources
├── physics/        # Darcy, Richards, transport, heat, mechanics, SVAT
├── coupling/       # Sequential, iterative, monolithic coupling
├── solvers/        # FiPy, FEniCS, PINN, analytical backends
├── time/           # Steppers and temporal schemes
├── postprocess/    # Derived fields, probes, integrals, export
├── visualization/  # 2-D and 3-D plotting
└── examples/       # Documented examples
```

### Design principles

1. **Separation of concerns** -- Geometry, physics, and solver are independent modules.
2. **Declarative API** -- Problems are defined through readable Python objects, not procedural code.
3. **Dimension-agnostic** -- The same API works for 2-D and 3-D problems.
4. **Solver-agnostic** -- Switch backends without changing the problem definition.
5. **Extensible** -- Add new physics modules, constitutive models, or solver backends with minimal effort.

## License

MIT
