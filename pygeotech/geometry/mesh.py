"""Mesh generation and import.

Classes
-------
Mesh
    Container for nodes, cells, and subdomain tags, with convenience
    methods for structured mesh generation.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike


class Mesh:
    """Unstructured or structured mesh.

    Attributes:
        nodes: Node coordinates, shape ``(n_nodes, dim)``.
        cells: Cell connectivity, shape ``(n_cells, nodes_per_cell)``.
        cell_tags: Integer tag per cell (for subdomain identification).
        dim: Spatial dimension.
        subdomain_map: Mapping from subdomain name to integer tag.
    """

    def __init__(
        self,
        nodes: np.ndarray,
        cells: np.ndarray,
        cell_tags: np.ndarray | None = None,
        subdomain_map: dict[str, int] | None = None,
    ) -> None:
        self.nodes = np.asarray(nodes, dtype=float)
        self.cells = np.asarray(cells, dtype=int)
        self.dim = self.nodes.shape[1]
        self.cell_tags = (
            np.asarray(cell_tags, dtype=int)
            if cell_tags is not None
            else np.zeros(len(self.cells), dtype=int)
        )
        self.subdomain_map: dict[str, int] = subdomain_map or {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return len(self.cells)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def cell_centers(self) -> np.ndarray:
        """Compute centroids of all cells.

        Returns:
            Array of shape ``(n_cells, dim)``.
        """
        return self.nodes[self.cells].mean(axis=1)

    def boundary_nodes(self) -> np.ndarray:
        """Return indices of nodes on the boundary.

        Uses a simple heuristic for 2-D triangular meshes: an edge that
        appears only once is a boundary edge.

        Returns:
            Sorted 1-D integer array of boundary node indices.
        """
        if self.cells.shape[1] == 3:
            # Triangular mesh
            edges: dict[tuple[int, int], int] = {}
            for tri in self.cells:
                for i in range(3):
                    e = tuple(sorted((tri[i], tri[(i + 1) % 3])))
                    edges[e] = edges.get(e, 0) + 1
            boundary = set()
            for (a, b), count in edges.items():
                if count == 1:
                    boundary.add(a)
                    boundary.add(b)
            return np.array(sorted(boundary), dtype=int)
        elif self.cells.shape[1] == 4:
            # Quad mesh — edges from (i, i+1) cyclically
            edges: dict[tuple[int, int], int] = {}
            for quad in self.cells:
                for i in range(4):
                    e = tuple(sorted((quad[i], quad[(i + 1) % 4])))
                    edges[e] = edges.get(e, 0) + 1
            boundary = set()
            for (a, b), count in edges.items():
                if count == 1:
                    boundary.add(a)
                    boundary.add(b)
            return np.array(sorted(boundary), dtype=int)
        raise NotImplementedError("boundary_nodes only supports triangular and quad meshes.")

    def boundary_node_coords(self) -> np.ndarray:
        """Coordinates of boundary nodes, shape ``(n_boundary, dim)``."""
        return self.nodes[self.boundary_nodes()]

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_geometry(
        cls,
        geometry: Any,
        resolution: float = 1.0,
        refine: dict[str, float] | None = None,
        algorithm: str = "delaunay",
    ) -> "Mesh":
        """Build a mesh from a :class:`~pygeotech.geometry.primitives.Geometry`.

        For now, this creates a structured triangular mesh for
        :class:`Rectangle` domains.  External meshing via *gmsh* can be
        added later.

        Args:
            geometry: Geometry to mesh.
            resolution: Default element size.
            refine: Per-subdomain element sizes (used for future gmsh integration).
            algorithm: Meshing algorithm name.

        Returns:
            Mesh instance.
        """
        from pygeotech.geometry.primitives import Rectangle

        if isinstance(geometry, Rectangle):
            return cls._structured_rect(geometry, resolution, refine or {})

        raise NotImplementedError(
            f"Auto-meshing not yet supported for {type(geometry).__name__}.  "
            "Use import_mesh() to load a mesh from gmsh."
        )

    @classmethod
    def _structured_rect(
        cls,
        rect: Any,
        resolution: float,
        refine: dict[str, float],
    ) -> "Mesh":
        """Create a structured triangular mesh for a Rectangle."""
        nx = max(2, int(np.ceil(rect.Lx / resolution)) + 1)
        ny = max(2, int(np.ceil(rect.Ly / resolution)) + 1)
        x = np.linspace(rect.x_min, rect.x_max, nx)
        y = np.linspace(rect.y_min, rect.y_max, ny)
        xx, yy = np.meshgrid(x, y)
        nodes = np.column_stack([xx.ravel(), yy.ravel()])

        # Triangulate using two triangles per quad
        cells = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                n0 = j * nx + i
                n1 = n0 + 1
                n2 = n0 + nx
                n3 = n2 + 1
                cells.append([n0, n1, n2])
                cells.append([n1, n3, n2])
        cells = np.array(cells, dtype=int)

        # Tag cells by subdomain
        centroids_arr = nodes[cells].mean(axis=1)
        tags = np.zeros(len(cells), dtype=int)
        subdomain_map: dict[str, int] = {}
        for idx, (name, geom) in enumerate(rect.subdomains.items(), start=1):
            subdomain_map[name] = idx
            inside = geom.contains(centroids_arr)
            tags[inside] = idx

        return cls(
            nodes=nodes,
            cells=cells,
            cell_tags=tags,
            subdomain_map=subdomain_map,
        )

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Mesh(n_nodes={self.n_nodes}, n_cells={self.n_cells}, "
            f"dim={self.dim}, subdomains={list(self.subdomain_map.keys())})"
        )


def import_mesh(filename: str) -> Mesh:
    """Import a mesh from an external file using *meshio*.

    Supported formats include ``.msh`` (gmsh), ``.vtk``, ``.xdmf``, etc.

    Args:
        filename: Path to the mesh file.

    Returns:
        Mesh instance.

    Raises:
        ImportError: If *meshio* is not installed.
    """
    try:
        import meshio
    except ImportError as exc:
        raise ImportError(
            "meshio is required for mesh import.  "
            "Install it with: pip install meshio"
        ) from exc

    m = meshio.read(filename)
    nodes = m.points[:, : (3 if m.points.shape[1] == 3 else 2)]

    # Take the first cell block
    cell_block = m.cells[0]
    cells = cell_block.data

    # Try to extract cell tags from cell_data
    cell_tags = None
    if m.cell_data:
        for key in m.cell_data:
            cell_tags = np.asarray(m.cell_data[key][0], dtype=int)
            break

    return Mesh(nodes=nodes, cells=cells, cell_tags=cell_tags)
