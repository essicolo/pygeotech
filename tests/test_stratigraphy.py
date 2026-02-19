"""Tests for the stratigraphy subpackage.

Covers borehole data containers, clustering, interpolation,
non-crossing enforcement, unit queries, kriging, and mesh tagging.
"""

from __future__ import annotations

import csv
import os
import tempfile

import numpy as np
import pytest

from pygeotech.stratigraphy.borehole import Borehole, BoreholeSet, Layer
from pygeotech.stratigraphy.interpolate import (
    OrdinaryKriging,
    StratigraphicModel,
)


# ======================================================================
# Fixtures — synthetic borehole data
# ======================================================================

def _make_boreholes_2d():
    """Create a simple 2-D cross-section with 4 boreholes and 3 units."""
    boreholes = []
    for i, x in enumerate([0, 30, 60, 100]):
        fill_top = 10.0 - 0.02 * x
        fill_bot = 8.0 - 0.02 * x
        clay_bot = 4.0 - 0.01 * x
        sand_bot = -5.0
        boreholes.append(Borehole(
            id=f"BH-{i+1}", x=float(x), y=0.0,
            layers=[
                Layer(z_top=fill_top, z_bottom=fill_bot, unit="fill",
                      properties={"N_SPT": 3.0 + i, "water_content": 0.15}),
                Layer(z_top=fill_bot, z_bottom=clay_bot, unit="clay",
                      properties={"N_SPT": 6.0 + i, "water_content": 0.35}),
                Layer(z_top=clay_bot, z_bottom=sand_bot, unit="sand",
                      properties={"N_SPT": 25.0 + 2 * i, "water_content": 0.20}),
            ],
        ))
    return BoreholeSet(boreholes)


def _make_boreholes_3d():
    """Create a 3-D dataset with 6 boreholes."""
    boreholes = []
    locs = [(0, 0), (50, 0), (100, 0), (0, 50), (50, 50), (100, 50)]
    for i, (x, y) in enumerate(locs):
        topo = 20.0 - 0.05 * x - 0.02 * y
        clay_top = 15.0 - 0.04 * x - 0.01 * y
        sand_top = 8.0 - 0.02 * x
        base = 0.0
        boreholes.append(Borehole(
            id=f"BH-{i+1}", x=float(x), y=float(y),
            layers=[
                Layer(z_top=topo, z_bottom=clay_top, unit="fill",
                      properties={"N_SPT": 5.0}),
                Layer(z_top=clay_top, z_bottom=sand_top, unit="clay",
                      properties={"N_SPT": 10.0}),
                Layer(z_top=sand_top, z_bottom=base, unit="sand",
                      properties={"N_SPT": 30.0}),
            ],
        ))
    return BoreholeSet(boreholes)


# ======================================================================
# Layer / Borehole basics
# ======================================================================

class TestLayer:
    def test_thickness(self):
        layer = Layer(z_top=10, z_bottom=5, unit="clay")
        assert layer.thickness == pytest.approx(5.0)

    def test_z_mid(self):
        layer = Layer(z_top=10, z_bottom=6, unit="sand")
        assert layer.z_mid == pytest.approx(8.0)

    def test_properties(self):
        layer = Layer(z_top=5, z_bottom=0, properties={"N_SPT": 20.0})
        assert layer.properties["N_SPT"] == pytest.approx(20.0)


class TestBorehole:
    def test_unit_at(self):
        bh = Borehole(id="BH-1", x=0, y=0, layers=[
            Layer(z_top=10, z_bottom=5, unit="fill"),
            Layer(z_top=5, z_bottom=0, unit="clay"),
        ])
        assert bh.unit_at(7) == "fill"
        assert bh.unit_at(3) == "clay"
        assert bh.unit_at(12) is None

    def test_property_at(self):
        bh = Borehole(id="BH-1", x=0, y=0, layers=[
            Layer(z_top=10, z_bottom=5, unit="fill",
                  properties={"N_SPT": 4.0}),
        ])
        assert bh.property_at(7, "N_SPT") == pytest.approx(4.0)
        assert bh.property_at(7, "missing") is None

    def test_z_surface_inferred(self):
        bh = Borehole(id="BH-1", x=0, y=0, layers=[
            Layer(z_top=10, z_bottom=5, unit="fill"),
            Layer(z_top=5, z_bottom=0, unit="clay"),
        ])
        assert bh.z_surface == pytest.approx(10.0)

    def test_z_bottom(self):
        bh = Borehole(id="BH-1", x=0, y=0, layers=[
            Layer(z_top=10, z_bottom=5, unit="fill"),
            Layer(z_top=5, z_bottom=-2, unit="clay"),
        ])
        assert bh.z_bottom == pytest.approx(-2.0)

    def test_unit_sequence(self):
        bh = Borehole(id="BH-1", x=0, y=0, layers=[
            Layer(z_top=5, z_bottom=0, unit="clay"),
            Layer(z_top=10, z_bottom=5, unit="fill"),
        ])
        assert bh.unit_sequence() == ["fill", "clay"]


# ======================================================================
# BoreholeSet
# ======================================================================

class TestBoreholeSet:
    def test_len_iter(self):
        bs = _make_boreholes_2d()
        assert len(bs) == 4
        assert sum(1 for _ in bs) == 4

    def test_dim_2d(self):
        bs = _make_boreholes_2d()
        assert bs.dim == 2

    def test_dim_3d(self):
        bs = _make_boreholes_3d()
        assert bs.dim == 3

    def test_unit_names(self):
        bs = _make_boreholes_2d()
        assert set(bs.unit_names()) == {"fill", "clay", "sand"}

    def test_stratigraphic_column(self):
        bs = _make_boreholes_2d()
        col = bs.stratigraphic_column()
        assert col == ["fill", "clay", "sand"]

    def test_interface_points(self):
        bs = _make_boreholes_2d()
        pts = bs.interface_points("fill", "clay")
        assert pts.shape == (4, 3)
        # z values should be the fill bottom
        assert pts[0, 2] == pytest.approx(8.0)  # BH-1 at x=0

    def test_all_interfaces(self):
        bs = _make_boreholes_2d()
        ifaces = bs.all_interfaces()
        assert "topography" in ifaces
        assert "fill/clay" in ifaces
        assert "clay/sand" in ifaces
        assert "base" in ifaces

    def test_locations(self):
        bs = _make_boreholes_2d()
        locs = bs.locations
        assert locs.shape == (4, 2)
        assert locs[0, 0] == pytest.approx(0.0)

    def test_feature_matrix(self):
        bs = _make_boreholes_2d()
        X, labels = bs.feature_matrix(["N_SPT", "water_content"])
        assert X.shape == (12, 2)  # 4 boreholes × 3 layers
        assert len(labels) == 12

    def test_from_csv(self, tmp_path):
        csv_path = str(tmp_path / "logs.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["borehole_id", "x", "y", "z_top", "z_bottom",
                             "unit", "N_SPT"])
            writer.writerow(["BH-1", "0", "0", "10", "5", "fill", "4"])
            writer.writerow(["BH-1", "0", "0", "5", "0", "clay", "8"])
            writer.writerow(["BH-2", "50", "0", "9", "4", "fill", "5"])
            writer.writerow(["BH-2", "50", "0", "4", "-1", "clay", "10"])

        bs = BoreholeSet.from_csv(csv_path)
        assert len(bs) == 2
        assert bs[0].layers[0].unit == "fill"
        assert bs[0].layers[0].properties["N_SPT"] == pytest.approx(4.0)
        assert bs[1].x == pytest.approx(50.0)


# ======================================================================
# Clustering
# ======================================================================

class TestClusterLogs:
    def test_kmeans_separable(self):
        """Three well-separated clusters should be found."""
        from pygeotech.stratigraphy.classify import cluster_logs

        boreholes = []
        for i in range(5):
            boreholes.append(Borehole(
                id=f"BH-{i}", x=float(i * 20), y=0.0,
                layers=[
                    Layer(z_top=10, z_bottom=7, unit="",
                          properties={"N_SPT": 3.0 + np.random.randn() * 0.3}),
                    Layer(z_top=7, z_bottom=3, unit="",
                          properties={"N_SPT": 15.0 + np.random.randn() * 0.5}),
                    Layer(z_top=3, z_bottom=-2, unit="",
                          properties={"N_SPT": 40.0 + np.random.randn() * 1.0}),
                ],
            ))
        bs = BoreholeSet(boreholes)
        units = cluster_logs(bs, n_units=3, method="kmeans",
                             features=["N_SPT"],
                             stratigraphic_order=["soft", "medium", "dense"])

        assert len(units) == 3
        # Top layer should be "soft" (lowest N_SPT)
        assert bs[0].layers[0].unit == "soft"
        # Bottom layer should be "dense" (highest N_SPT)
        assert bs[0].layers[2].unit == "dense"

    def test_gmm(self):
        """GMM clustering should work."""
        from pygeotech.stratigraphy.classify import cluster_logs

        boreholes = [
            Borehole(id="BH-1", x=0, y=0, layers=[
                Layer(z_top=10, z_bottom=5, properties={"N_SPT": 5.0}),
                Layer(z_top=5, z_bottom=0, properties={"N_SPT": 30.0}),
            ]),
            Borehole(id="BH-2", x=50, y=0, layers=[
                Layer(z_top=9, z_bottom=4, properties={"N_SPT": 4.0}),
                Layer(z_top=4, z_bottom=-1, properties={"N_SPT": 28.0}),
            ]),
        ]
        bs = BoreholeSet(boreholes)
        units = cluster_logs(bs, n_units=2, method="gmm",
                             features=["N_SPT"])
        assert len(units) == 2
        # All top layers should share one unit
        assert bs[0].layers[0].unit == bs[1].layers[0].unit

    def test_agglomerative(self):
        """Agglomerative clustering basic check."""
        from pygeotech.stratigraphy.classify import cluster_logs

        boreholes = [
            Borehole(id="BH-1", x=0, y=0, layers=[
                Layer(z_top=10, z_bottom=5, properties={"N_SPT": 5.0}),
                Layer(z_top=5, z_bottom=0, properties={"N_SPT": 30.0}),
            ]),
        ]
        bs = BoreholeSet(boreholes)
        units = cluster_logs(bs, n_units=2, method="agglomerative",
                             features=["N_SPT"])
        assert len(units) == 2

    def test_auto_features(self):
        """Should auto-detect features if not specified."""
        from pygeotech.stratigraphy.classify import cluster_logs

        boreholes = [
            Borehole(id="BH-1", x=0, y=0, layers=[
                Layer(z_top=10, z_bottom=5,
                      properties={"N_SPT": 5.0, "w": 0.3}),
                Layer(z_top=5, z_bottom=0,
                      properties={"N_SPT": 30.0, "w": 0.1}),
            ]),
        ]
        bs = BoreholeSet(boreholes)
        units = cluster_logs(bs, n_units=2, method="kmeans")
        assert len(units) == 2


# ======================================================================
# Ordinary Kriging
# ======================================================================

class TestOrdinaryKriging:
    def test_exact_interpolation(self):
        """Kriging should exactly reproduce data points (no nugget)."""
        xy = np.array([[0], [10], [20], [30]])
        z = np.array([5.0, 7.0, 6.0, 8.0])
        krig = OrdinaryKriging(xy, z, nugget=0.0)
        z_pred = krig(xy)
        np.testing.assert_allclose(z_pred, z, atol=1e-6)

    def test_interpolation_smooth(self):
        """Predictions between data points should be reasonable."""
        xy = np.array([[0], [100]])
        z = np.array([10.0, 5.0])
        krig = OrdinaryKriging(xy, z)
        z_mid = krig(np.array([[50]]))[0]
        assert 4.0 < z_mid < 11.0

    def test_2d_locations(self):
        """Kriging with 2-D plan locations."""
        xy = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])
        z = np.array([10.0, 8.0, 9.0, 7.0])
        krig = OrdinaryKriging(xy, z)
        z_pred = krig(xy)
        np.testing.assert_allclose(z_pred, z, atol=1e-6)

    def test_variogram_models(self):
        xy = np.array([[0], [50], [100]])
        z = np.array([10.0, 8.0, 6.0])
        for model in ("exponential", "spherical", "gaussian"):
            krig = OrdinaryKriging(xy, z, variogram=model)
            z_pred = krig(xy)
            np.testing.assert_allclose(z_pred, z, atol=1e-5)


# ======================================================================
# StratigraphicModel — interpolation
# ======================================================================

class TestStratigraphicModel2D:
    def test_rbf_interpolation(self):
        bs = _make_boreholes_2d()
        model = StratigraphicModel(bs)
        model.interpolate(method="rbf", non_crossing="sequential")

        # Query at a borehole location — should match
        unit = model.unit_at(0, 9.0)
        assert unit == "fill"
        unit = model.unit_at(0, 6.0)
        assert unit == "clay"
        unit = model.unit_at(0, 0.0)
        assert unit == "sand"

    def test_linear_interpolation(self):
        bs = _make_boreholes_2d()
        model = StratigraphicModel(bs)
        model.interpolate(method="linear", non_crossing="sequential")
        assert model.unit_at(0, 9.0) == "fill"

    def test_cubic_interpolation(self):
        bs = _make_boreholes_2d()
        model = StratigraphicModel(bs)
        model.interpolate(method="cubic", non_crossing="sequential")
        assert model.unit_at(0, 9.0) == "fill"

    def test_kriging_interpolation(self):
        bs = _make_boreholes_2d()
        model = StratigraphicModel(bs)
        model.interpolate(method="kriging", non_crossing="sequential")
        assert model.unit_at(0, 9.0) == "fill"

    def test_evaluate_surfaces(self):
        bs = _make_boreholes_2d()
        model = StratigraphicModel(bs)
        model.interpolate(method="rbf")
        x = np.array([0, 30, 60, 100])
        surfaces = model.evaluate_surfaces(x)
        assert "topography" in surfaces
        assert len(surfaces["topography"]) == 4

    def test_non_crossing(self):
        """Surfaces should not cross with sequential enforcement."""
        bs = _make_boreholes_2d()
        model = StratigraphicModel(bs)
        model.interpolate(method="rbf", non_crossing="sequential",
                          min_thickness=0.01)
        x = np.linspace(0, 100, 50)
        surfaces = model.evaluate_surfaces(x)
        keys = model._surface_order
        for i in range(len(keys) - 1):
            above = surfaces[keys[i]]
            below = surfaces[keys[i + 1]]
            assert np.all(above >= below - 1e-6), (
                f"{keys[i]} crosses below {keys[i+1]}"
            )

    def test_cross_section(self):
        bs = _make_boreholes_2d()
        model = StratigraphicModel(bs)
        model.interpolate(method="rbf")
        section = model.cross_section((0, 0), (100, 0), n_points=20)
        assert "distance" in section
        assert "topography" in section
        assert len(section["distance"]) == 20


class TestStratigraphicModel3D:
    def test_rbf_3d(self):
        bs = _make_boreholes_3d()
        model = StratigraphicModel(bs)
        model.interpolate(method="rbf", non_crossing="sequential")

        # At (0, 0) topography is 20, clay_top=15, sand_top=8
        assert model.unit_at(0, 0, 19) == "fill"
        assert model.unit_at(0, 0, 12) == "clay"
        assert model.unit_at(0, 0, 4) == "sand"

    def test_kriging_3d(self):
        bs = _make_boreholes_3d()
        model = StratigraphicModel(bs)
        model.interpolate(method="kriging", non_crossing="sequential")
        assert model.unit_at(0, 0, 19) == "fill"

    def test_cross_section_3d(self):
        bs = _make_boreholes_3d()
        model = StratigraphicModel(bs)
        model.interpolate(method="rbf")
        section = model.cross_section((0, 0), (100, 50), n_points=30)
        assert len(section["distance"]) == 30
        assert "topography" in section

    def test_non_crossing_3d(self):
        bs = _make_boreholes_3d()
        model = StratigraphicModel(bs)
        model.interpolate(method="rbf", non_crossing="sequential",
                          min_thickness=0.01)
        x = np.linspace(0, 100, 20)
        y = np.linspace(0, 50, 20)
        surfaces = model.evaluate_surfaces(x, y)
        keys = model._surface_order
        for i in range(len(keys) - 1):
            above = surfaces[keys[i]]
            below = surfaces[keys[i + 1]]
            assert np.all(above >= below - 1e-6)


# ======================================================================
# Mesh tagging
# ======================================================================

class TestTagMesh:
    def test_tag_2d_mesh(self):
        """Tag a 2-D mesh with stratigraphy."""
        from pygeotech.geometry import Rectangle, Mesh

        bs = _make_boreholes_2d()
        model = StratigraphicModel(bs)
        model.interpolate(method="rbf", non_crossing="sequential")

        # Create a mesh covering the cross-section domain
        rect = Rectangle(100, 15, origin=(0, -5))
        mesh = rect.generate_mesh(resolution=5.0)

        model.tag_mesh(mesh)
        assert mesh.subdomain_map is not None
        assert "fill" in mesh.subdomain_map
        assert "clay" in mesh.subdomain_map
        assert "sand" in mesh.subdomain_map

        # Some cells should be tagged (not all -1)
        assert np.any(mesh.cell_tags >= 0)


# ======================================================================
# Edge cases
# ======================================================================

class TestEdgeCases:
    def test_single_borehole(self):
        """Model with a single borehole should still work."""
        bs = BoreholeSet([
            Borehole(id="BH-1", x=0, y=0, layers=[
                Layer(z_top=10, z_bottom=5, unit="fill"),
                Layer(z_top=5, z_bottom=0, unit="clay"),
            ]),
        ])
        model = StratigraphicModel(bs)
        model.interpolate(method="rbf")
        # Should return constant surfaces
        assert model.unit_at(50, 7) == "fill"
        assert model.unit_at(50, 3) == "clay"

    def test_not_interpolated_raises(self):
        bs = _make_boreholes_2d()
        model = StratigraphicModel(bs)
        with pytest.raises(RuntimeError, match="interpolate"):
            model.unit_at(0, 5)

    def test_repr(self):
        bs = _make_boreholes_2d()
        model = StratigraphicModel(bs)
        assert "not interpolated" in repr(model)
        model.interpolate(method="rbf")
        assert "interpolated" in repr(model)

    def test_column_order(self):
        bs = _make_boreholes_2d()
        model = StratigraphicModel(bs)
        assert model.column == ["fill", "clay", "sand"]
