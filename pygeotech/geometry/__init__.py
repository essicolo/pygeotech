"""Geometry: domain and subdomain definition, meshing."""

from pygeotech.geometry.primitives import (
    Geometry,
    Rectangle,
    Polygon,
    Circle,
    Box,
    Cylinder,
    Line,
)
from pygeotech.geometry.csg import union, intersection, difference
from pygeotech.geometry.layers import LayeredProfile
from pygeotech.geometry.mesh import Mesh, import_mesh

__all__ = [
    "Geometry",
    "Rectangle",
    "Polygon",
    "Circle",
    "Box",
    "Cylinder",
    "Line",
    "union",
    "intersection",
    "difference",
    "LayeredProfile",
    "Mesh",
    "import_mesh",
]
