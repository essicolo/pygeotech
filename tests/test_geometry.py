"""Tests for the geometry module."""

import numpy as np
import pytest

from pygeotech.geometry.primitives import (
    Rectangle, Polygon, Circle, Box, Cylinder, Line, Geometry,
)
from pygeotech.geometry.csg import union, intersection, difference
from pygeotech.geometry.layers import LayeredProfile
from pygeotech.geometry.mesh import Mesh


# ======================================================================
# Primitives
# ======================================================================

class TestRectangle:
    def test_basic_creation(self):
        r = Rectangle(Lx=10, Ly=5)
        assert r.Lx == 10.0
        assert r.Ly == 5.0
        assert r.dim == 2
        np.testing.assert_array_equal(r.origin, [0, 0])

    def test_creation_with_origin(self):
        r = Rectangle(Lx=10, Ly=5, origin=(2, 3))
        assert r.x_min == 2.0
        assert r.x_max == 12.0
        assert r.y_min == 3.0
        assert r.y_max == 8.0

    def test_creation_with_x0_y0(self):
        r = Rectangle(x0=5, y0=10, width=20, height=8)
        assert r.Lx == 20.0
        assert r.x_min == 5.0
        assert r.y_min == 10.0

    def test_contains(self):
        r = Rectangle(Lx=10, Ly=5)
        pts = np.array([[5, 2.5], [0, 0], [10, 5], [-1, 0], [11, 3]])
        mask = r.contains(pts)
        np.testing.assert_array_equal(mask, [True, True, True, False, False])

    def test_bounding_box(self):
        r = Rectangle(Lx=10, Ly=5, origin=(1, 2))
        lo, hi = r.bounding_box()
        np.testing.assert_array_equal(lo, [1, 2])
        np.testing.assert_array_equal(hi, [11, 7])

    def test_area(self):
        r = Rectangle(Lx=10, Ly=5)
        assert r.area_or_volume() == 50.0

    def test_add_subdomain(self):
        domain = Rectangle(Lx=100, Ly=20)
        sub = Rectangle(x0=40, y0=5, width=20, height=10)
        domain.add_subdomain("block", sub)
        assert "block" in domain.subdomains

    def test_subdomain_dim_mismatch(self):
        domain = Rectangle(Lx=10, Ly=5)
        box = Box(Lx=1, Ly=1, Lz=1)
        with pytest.raises(ValueError, match="dimension"):
            domain.add_subdomain("mismatch", box)


class TestPolygon:
    def test_triangle(self):
        p = Polygon([(0, 0), (10, 0), (5, 5)])
        assert p.dim == 2
        assert len(p.vertices) == 3

    def test_contains(self):
        p = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        pts = np.array([[5, 5], [-1, -1], [10, 10]])
        mask = p.contains(pts)
        assert mask[0] is np.True_
        assert mask[1] is np.False_

    def test_area(self):
        p = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        assert abs(p.area_or_volume() - 100.0) < 1e-10

    def test_edges(self):
        p = Polygon([(0, 0), (1, 0), (1, 1)])
        assert len(p.edges) == 3

    def test_min_vertices(self):
        with pytest.raises(ValueError):
            Polygon([(0, 0), (1, 1)])


class TestCircle:
    def test_creation(self):
        c = Circle(center=(5, 5), radius=3)
        assert c.dim == 2
        assert c.radius == 3.0

    def test_contains(self):
        c = Circle(center=(0, 0), radius=1)
        pts = np.array([[0, 0], [0.5, 0.5], [2, 0]])
        mask = c.contains(pts)
        assert mask[0] and mask[1] and not mask[2]

    def test_area(self):
        c = Circle(radius=1)
        assert abs(c.area_or_volume() - np.pi) < 1e-10


class TestBox:
    def test_volume(self):
        b = Box(Lx=2, Ly=3, Lz=4)
        assert b.area_or_volume() == 24.0
        assert b.dim == 3

    def test_contains(self):
        b = Box(Lx=1, Ly=1, Lz=1)
        pts = np.array([[0.5, 0.5, 0.5], [2, 0.5, 0.5]])
        mask = b.contains(pts)
        assert mask[0] and not mask[1]


class TestCylinder:
    def test_volume(self):
        cy = Cylinder(radius=1, z_min=0, z_max=10)
        assert abs(cy.area_or_volume() - np.pi * 10) < 1e-10

    def test_contains(self):
        cy = Cylinder(center=(0, 0), radius=1, z_min=0, z_max=5)
        pts = np.array([[0, 0, 2.5], [2, 0, 2.5], [0, 0, 6]])
        mask = cy.contains(pts)
        assert mask[0] and not mask[1] and not mask[2]


class TestLine:
    def test_sample(self):
        line = Line(start=(0, 0), end=(10, 0))
        pts = line.sample(11)
        assert pts.shape == (11, 2)
        np.testing.assert_allclose(pts[0], [0, 0])
        np.testing.assert_allclose(pts[-1], [10, 0])


# ======================================================================
# CSG
# ======================================================================

class TestCSG:
    def test_union(self):
        a = Rectangle(Lx=5, Ly=5)
        b = Rectangle(Lx=5, Ly=5, origin=(3, 0))
        u = union(a, b)
        pts = np.array([[1, 1], [7, 1], [-1, -1]])
        mask = u.contains(pts)
        assert mask[0] and mask[1] and not mask[2]

    def test_intersection(self):
        a = Rectangle(Lx=5, Ly=5)
        b = Rectangle(Lx=5, Ly=5, origin=(3, 0))
        inter = intersection(a, b)
        pts = np.array([[4, 1], [1, 1], [7, 1]])
        mask = inter.contains(pts)
        assert mask[0] and not mask[1] and not mask[2]

    def test_difference(self):
        a = Rectangle(Lx=10, Ly=10)
        b = Rectangle(Lx=3, Ly=3, origin=(3, 3))
        d = difference(a, b)
        pts = np.array([[1, 1], [4, 4]])
        mask = d.contains(pts)
        assert mask[0] and not mask[1]


# ======================================================================
# Layers
# ======================================================================

class TestLayeredProfile:
    def test_identify(self):
        profile = LayeredProfile()
        profile.add("topsoil", top=10.0, bottom=8.0)
        profile.add("clay", top=8.0, bottom=3.0)
        profile.add("bedrock", top=3.0, bottom=0.0)

        pts = np.array([[5, 9], [5, 5], [5, 1]])
        names = profile.identify(pts)
        assert names[0] == "topsoil"
        assert names[1] == "clay"
        assert names[2] == "bedrock"


# ======================================================================
# Mesh
# ======================================================================

class TestMesh:
    def test_structured_rect(self):
        r = Rectangle(Lx=10, Ly=5)
        mesh = r.generate_mesh(resolution=1.0)
        assert mesh.n_nodes > 0
        assert mesh.n_cells > 0
        assert mesh.dim == 2
        assert mesh.cells.shape[1] == 3  # triangles

    def test_cell_centers(self):
        r = Rectangle(Lx=2, Ly=2)
        mesh = r.generate_mesh(resolution=1.0)
        centers = mesh.cell_centers()
        assert centers.shape == (mesh.n_cells, 2)

    def test_boundary_nodes(self):
        r = Rectangle(Lx=4, Ly=2)
        mesh = r.generate_mesh(resolution=1.0)
        b_nodes = mesh.boundary_nodes()
        assert len(b_nodes) > 0
        # Boundary nodes should be on edges
        coords = mesh.nodes[b_nodes]
        on_edge = (
            np.isclose(coords[:, 0], 0) | np.isclose(coords[:, 0], 4)
            | np.isclose(coords[:, 1], 0) | np.isclose(coords[:, 1], 2)
        )
        assert on_edge.all()

    def test_subdomain_tags(self):
        domain = Rectangle(Lx=10, Ly=10)
        sub = Rectangle(x0=3, y0=3, width=4, height=4)
        domain.add_subdomain("block", sub)
        mesh = domain.generate_mesh(resolution=1.0)
        assert "block" in mesh.subdomain_map
        assert (mesh.cell_tags == mesh.subdomain_map["block"]).any()
