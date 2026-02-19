"""Stratigraphy: borehole classification and spatial interpolation.

Workflow::

    boreholes = BoreholeSet.from_csv("drill_logs.csv")

    # Optional: auto-classify intervals
    cluster_logs(boreholes, n_units=4, features=["N_SPT"])

    # Interpolate boundaries
    model = StratigraphicModel(boreholes)
    model.interpolate(method="rbf", non_crossing="sequential")

    # Query
    unit = model.unit_at(x=50, y=30, z=-5)

    # Tag an existing mesh
    model.tag_mesh(mesh)
"""

from pygeotech.stratigraphy.borehole import Borehole, BoreholeSet, Layer
from pygeotech.stratigraphy.classify import cluster_logs
from pygeotech.stratigraphy.interpolate import (
    OrdinaryKriging,
    StratigraphicModel,
)

__all__ = [
    "Layer",
    "Borehole",
    "BoreholeSet",
    "cluster_logs",
    "StratigraphicModel",
    "OrdinaryKriging",
]
