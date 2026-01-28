"""Seepage under a dam with cutoff wall.

This example demonstrates the core pygeotech workflow:

1. Define geometry (rectangular domain with dam and cutoff wall subdomains)
2. Generate a mesh with subdomain-based refinement
3. Assign materials with contrasting hydraulic conductivity
4. Set up Darcy steady-state flow physics
5. Apply boundary conditions (upstream/downstream heads)
6. Solve using the built-in sparse solver
7. Visualise the head field

Run::

    python -m pygeotech.examples.dam_seepage.run
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    """Run the dam seepage example."""
    import pygeotech as pgt

    # -----------------------------------------------------------------
    # 1. Geometry
    # -----------------------------------------------------------------
    domain = pgt.geometry.Rectangle(Lx=120, Ly=20, origin=(0, 0))

    # Cutoff wall (low-K concrete)
    cutoff_wall = pgt.geometry.Rectangle(x0=58, y0=8, width=4, height=12)
    domain.add_subdomain("cutoff", cutoff_wall)

    # Dam body on top of the foundation
    dam_base = pgt.geometry.Polygon([
        (40, 20), (50, 25), (70, 25), (80, 20),
    ])
    domain.add_subdomain("dam", dam_base)

    print(f"Domain: {domain}")
    print(f"Subdomains: {list(domain.subdomains.keys())}")

    # -----------------------------------------------------------------
    # 2. Mesh
    # -----------------------------------------------------------------
    mesh = domain.generate_mesh(
        resolution=1.0,
        refine={"cutoff": 0.5, "dam": 0.5},
        algorithm="delaunay",
    )
    print(f"Mesh: {mesh}")

    # -----------------------------------------------------------------
    # 3. Materials
    # -----------------------------------------------------------------
    foundation = pgt.materials.Material(
        name="sandy_silt",
        hydraulic_conductivity=1e-5,
        porosity=0.35,
        dry_density=1600,
    )
    dam_fill = pgt.materials.Material(
        name="compacted_fill",
        hydraulic_conductivity=1e-4,
        porosity=0.30,
    )
    concrete = pgt.materials.Material(
        name="concrete",
        hydraulic_conductivity=1e-10,
        porosity=0.05,
    )

    materials = pgt.materials.assign(mesh, {
        "default": foundation,
        "dam": dam_fill,
        "cutoff": concrete,
    })
    print(f"Materials: {materials}")

    # -----------------------------------------------------------------
    # 4. Physics
    # -----------------------------------------------------------------
    flow = pgt.physics.Darcy(mesh, materials)
    print(f"Physics: {flow}")

    # -----------------------------------------------------------------
    # 5. Boundary conditions
    # -----------------------------------------------------------------
    bc = pgt.boundaries.BoundaryConditions()

    # Upstream reservoir: H = 27 m on top-left (x < 40)
    bc.add(pgt.boundaries.Dirichlet(
        field="H",
        value=27.0,
        where=pgt.boundaries.top() & pgt.boundaries.x_less_than(40),
    ))

    # Downstream: H = 20 m on top-right (x > 80)
    bc.add(pgt.boundaries.Dirichlet(
        field="H",
        value=20.0,
        where=pgt.boundaries.top() & pgt.boundaries.x_greater_than(80),
    ))

    # Bottom impermeable (default: zero Neumann flux)

    print(f"Boundary conditions: {len(bc)} conditions")

    # -----------------------------------------------------------------
    # 6. Solve
    # -----------------------------------------------------------------
    solver = pgt.solvers.FiPyBackend()
    solution = solver.solve(flow, bc)
    print(f"Solution: {solution}")

    H = solution.fields["H"]
    print(f"Head range: {H.min():.2f} to {H.max():.2f} m")

    # -----------------------------------------------------------------
    # 7. Post-processing
    # -----------------------------------------------------------------
    # Export to CSV
    solution.export_csv("dam_seepage_head.csv")
    print("Exported: dam_seepage_head.csv")

    # Plot (only if display is available)
    try:
        solution.plot("H", contours=20, title="Hydraulic Head (m)")
        import matplotlib.pyplot as plt
        plt.savefig("dam_seepage_head.png", dpi=150, bbox_inches="tight")
        print("Saved: dam_seepage_head.png")
        plt.close()
    except Exception as exc:
        print(f"Plotting skipped: {exc}")

    print("\nDam seepage example completed successfully!")


if __name__ == "__main__":
    main()
