"""Borehole data containers for stratigraphic modelling.

Provides :class:`Layer`, :class:`Borehole`, and :class:`BoreholeSet`
for loading, querying, and preparing drill-log data for stratigraphic
classification and interpolation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Layer:
    """A single stratigraphic interval in a borehole.

    Args:
        z_top: Elevation of the top of the interval (m).
        z_bottom: Elevation of the bottom of the interval (m).
        unit: Stratigraphic unit name (may be empty if not yet classified).
        properties: Measured properties (e.g. ``{"N_SPT": 12}``).
    """

    z_top: float
    z_bottom: float
    unit: str = ""
    properties: dict[str, float] = field(default_factory=dict)

    @property
    def thickness(self) -> float:
        return self.z_top - self.z_bottom

    @property
    def z_mid(self) -> float:
        return (self.z_top + self.z_bottom) / 2.0


@dataclass
class Borehole:
    """A single borehole with layered stratigraphy.

    Args:
        id: Unique borehole identifier.
        x: Easting / x-coordinate (m).
        y: Northing / y-coordinate (m).  Use 0 for 2-D cross-sections.
        layers: List of :class:`Layer` objects from top to bottom.
        z_surface: Ground surface elevation.  Inferred from the
            topmost layer if not given.
    """

    id: str
    x: float
    y: float
    layers: list[Layer]
    z_surface: float | None = None

    def __post_init__(self) -> None:
        if self.z_surface is None and self.layers:
            self.z_surface = max(l.z_top for l in self.layers)

    def unit_at(self, z: float) -> str | None:
        """Return the unit name at elevation *z*, or ``None``."""
        for layer in self.layers:
            if layer.z_bottom <= z <= layer.z_top:
                return layer.unit
        return None

    def property_at(self, z: float, name: str) -> float | None:
        """Return a named property at elevation *z*, or ``None``."""
        for layer in self.layers:
            if layer.z_bottom <= z <= layer.z_top:
                return layer.properties.get(name)
        return None

    @property
    def z_bottom(self) -> float:
        if not self.layers:
            return 0.0
        return min(l.z_bottom for l in self.layers)

    def unit_sequence(self) -> list[str]:
        """Ordered list of unit names from top to bottom."""
        sorted_layers = sorted(self.layers, key=lambda l: -l.z_top)
        return [l.unit for l in sorted_layers]


class BoreholeSet:
    """Collection of boreholes for stratigraphic analysis.

    Args:
        boreholes: List of :class:`Borehole` instances.
    """

    def __init__(self, boreholes: list[Borehole]) -> None:
        self.boreholes = list(boreholes)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        filename: str,
        id_col: str = "borehole_id",
        x_col: str = "x",
        y_col: str = "y",
        z_top_col: str = "z_top",
        z_bottom_col: str = "z_bottom",
        unit_col: str = "unit",
        property_cols: list[str] | None = None,
    ) -> BoreholeSet:
        """Load boreholes from a CSV file.

        Each row represents one interval.  Rows with the same
        *id_col* are grouped into one borehole.

        Args:
            filename: Path to CSV file.
            id_col: Column name for borehole ID.
            x_col: Column for x-coordinate.
            y_col: Column for y-coordinate (``"y"`` by default;
                set to ``None`` for 2-D data).
            z_top_col: Column for interval top elevation.
            z_bottom_col: Column for interval bottom elevation.
            unit_col: Column for unit label (may be absent).
            property_cols: Columns to read as numeric properties.
                If ``None``, all non-reserved columns are used.
        """
        borehole_data: dict[str, dict[str, Any]] = {}

        with open(filename, newline="") as fh:
            reader = csv.DictReader(fh)
            all_cols = reader.fieldnames or []

            reserved = {id_col, x_col, z_top_col, z_bottom_col}
            if y_col:
                reserved.add(y_col)
            if unit_col:
                reserved.add(unit_col)
            if property_cols is None:
                property_cols = [c for c in all_cols if c not in reserved]

            for row in reader:
                bh_id = row[id_col].strip()
                if bh_id not in borehole_data:
                    borehole_data[bh_id] = {
                        "x": float(row[x_col]),
                        "y": float(row[y_col]) if y_col and y_col in row else 0.0,
                        "layers": [],
                    }

                props: dict[str, float] = {}
                for pc in property_cols:
                    if pc in row and row[pc].strip():
                        try:
                            props[pc] = float(row[pc])
                        except (ValueError, TypeError):
                            pass

                unit = row.get(unit_col, "").strip() if unit_col else ""
                layer = Layer(
                    z_top=float(row[z_top_col]),
                    z_bottom=float(row[z_bottom_col]),
                    unit=unit,
                    properties=props,
                )
                borehole_data[bh_id]["layers"].append(layer)

        boreholes = [
            Borehole(id=bh_id, x=d["x"], y=d["y"], layers=d["layers"])
            for bh_id, d in borehole_data.items()
        ]
        return cls(boreholes)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __iter__(self):
        return iter(self.boreholes)

    def __len__(self) -> int:
        return len(self.boreholes)

    def __getitem__(self, idx: int) -> Borehole:
        return self.boreholes[idx]

    @property
    def locations(self) -> np.ndarray:
        """Plan-view locations, shape ``(n_boreholes, 2)``."""
        return np.array([(bh.x, bh.y) for bh in self.boreholes])

    @property
    def dim(self) -> int:
        """Spatial dimension of the borehole layout.

        Returns 2 if boreholes vary in *x* only, 3 if both *x* and *y*
        vary.
        """
        locs = self.locations
        if locs.shape[0] < 2:
            return 2
        return 3 if np.ptp(locs[:, 1]) > 1e-10 else 2

    def unit_names(self) -> list[str]:
        """Sorted list of unique unit names."""
        units: set[str] = set()
        for bh in self.boreholes:
            for layer in bh.layers:
                if layer.unit:
                    units.add(layer.unit)
        return sorted(units)

    def stratigraphic_column(self) -> list[str]:
        """Infer global stratigraphic order from mean elevation.

        Units are sorted from highest (shallowest) to lowest
        (deepest) mean mid-elevation.
        """
        elevations: dict[str, list[float]] = {}
        for bh in self.boreholes:
            for layer in bh.layers:
                if layer.unit:
                    elevations.setdefault(layer.unit, []).append(layer.z_mid)
        return sorted(
            elevations.keys(),
            key=lambda u: -float(np.mean(elevations[u])),
        )

    # ------------------------------------------------------------------
    # Interface extraction
    # ------------------------------------------------------------------

    def interface_points(
        self, unit_above: str, unit_below: str
    ) -> np.ndarray:
        """Extract (x, y, z) of interface between two adjacent units.

        Returns:
            Array of shape ``(n, 3)``.
        """
        points: list[list[float]] = []
        for bh in self.boreholes:
            sorted_layers = sorted(bh.layers, key=lambda l: -l.z_top)
            for i in range(len(sorted_layers) - 1):
                if (
                    sorted_layers[i].unit == unit_above
                    and sorted_layers[i + 1].unit == unit_below
                ):
                    points.append(
                        [bh.x, bh.y, sorted_layers[i].z_bottom]
                    )
        return np.array(points) if points else np.empty((0, 3))

    def all_interfaces(self) -> dict[str, np.ndarray]:
        """Extract all interface surfaces.

        Returns a dict mapping surface names to ``(n, 3)`` arrays.
        Names follow the pattern ``"unit_above/unit_below"`` plus
        ``"topography"`` and ``"base"``.
        """
        column = self.stratigraphic_column()
        interfaces: dict[str, np.ndarray] = {}

        # Topography
        topo = [[bh.x, bh.y, bh.z_surface] for bh in self.boreholes
                if bh.z_surface is not None]
        if topo:
            interfaces["topography"] = np.array(topo)

        # Inter-unit interfaces
        for i in range(len(column) - 1):
            pts = self.interface_points(column[i], column[i + 1])
            if len(pts) > 0:
                interfaces[f"{column[i]}/{column[i + 1]}"] = pts

        # Base
        base = []
        for bh in self.boreholes:
            if bh.layers:
                base.append([bh.x, bh.y, bh.z_bottom])
        if base:
            interfaces["base"] = np.array(base)

        return interfaces

    # ------------------------------------------------------------------
    # Feature extraction (for clustering)
    # ------------------------------------------------------------------

    def feature_matrix(
        self,
        features: list[str],
    ) -> tuple[np.ndarray, list[tuple[str, int]]]:
        """Build a feature matrix from all intervals.

        Args:
            features: Property names to include as columns.

        Returns:
            ``(X, labels)`` where *X* has shape ``(n_intervals, n_features)``
            and *labels* is a list of ``(borehole_id, layer_index)`` tuples.
        """
        rows: list[list[float]] = []
        labels: list[tuple[str, int]] = []
        for bh in self.boreholes:
            for li, layer in enumerate(bh.layers):
                row = []
                for feat in features:
                    val = layer.properties.get(feat, np.nan)
                    row.append(float(val) if val is not None else np.nan)
                rows.append(row)
                labels.append((bh.id, li))
        return np.array(rows, dtype=float), labels

    def __repr__(self) -> str:
        return (
            f"BoreholeSet(n_boreholes={len(self.boreholes)}, "
            f"units={self.unit_names()})"
        )
