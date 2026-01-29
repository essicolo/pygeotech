"""Slope profile definition.

A :class:`SlopeProfile` describes the ground surface, material layers,
and optional phreatic surface for a 2-D slope stability analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class MaterialLayer:
    """Material properties for a layer in a slope profile.

    All stresses and strengths use consistent SI units (Pa, m, N).

    Args:
        cohesion: Effective cohesion c' (Pa).
        friction_angle: Effective friction angle φ' (degrees).
        unit_weight: Total unit weight γ (N/m³).
        top: Top elevation (m).
        bottom: Bottom elevation (m).
        sat_unit_weight: Saturated unit weight (N/m³).  Defaults to
            ``unit_weight`` if not given.
    """

    cohesion: float = 0.0
    friction_angle: float = 30.0
    unit_weight: float = 18e3
    top: float = 100.0
    bottom: float = 0.0
    sat_unit_weight: float | None = None

    @property
    def gamma(self) -> float:
        return self.unit_weight

    @property
    def gamma_sat(self) -> float:
        return self.sat_unit_weight if self.sat_unit_weight is not None else self.unit_weight

    @property
    def phi_rad(self) -> float:
        return np.radians(self.friction_angle)


class SlopeProfile:
    """2-D slope profile for stability analysis.

    The profile is defined by a ground surface polyline and one or more
    material layers.  An optional water table polyline sets the phreatic
    surface for pore-pressure computation.

    Args:
        surface: Ground surface as ``[(x, y), ...]`` from left to right.
        layers: List of layer definitions.  Each may be a
            :class:`MaterialLayer` or a dict with keys
            ``c`` or ``cohesion``, ``phi`` or ``friction_angle``,
            ``gamma`` or ``unit_weight``, ``top``, ``bottom``, and
            optionally ``gamma_sat`` or ``sat_unit_weight``.
        water_table: Phreatic surface as ``[(x, y), ...]`` or ``None``
            for dry analysis.

    Example::

        profile = SlopeProfile(
            surface=[(0, 10), (20, 10), (30, 20), (50, 20)],
            layers=[
                {"c": 10e3, "phi": 25, "gamma": 18e3, "top": 20, "bottom": 0},
            ],
            water_table=[(0, 8), (50, 18)],
        )
    """

    def __init__(
        self,
        surface: Sequence[tuple[float, float]],
        layers: Sequence[MaterialLayer | dict[str, float]],
        water_table: Sequence[tuple[float, float]] | None = None,
    ) -> None:
        self.surface = np.array(surface, dtype=float)
        self.layers = [self._to_layer(lay) for lay in layers]
        self.water_table = (
            np.array(water_table, dtype=float)
            if water_table is not None
            else None
        )

        # Sort layers top-to-bottom
        self.layers.sort(key=lambda lay: -lay.top)

    @staticmethod
    def _to_layer(spec: MaterialLayer | dict[str, float]) -> MaterialLayer:
        if isinstance(spec, MaterialLayer):
            return spec
        return MaterialLayer(
            cohesion=spec.get("c", spec.get("cohesion", 0.0)),
            friction_angle=spec.get("phi", spec.get("friction_angle", 30.0)),
            unit_weight=spec.get("gamma", spec.get("unit_weight", 18e3)),
            top=spec.get("top", 100.0),
            bottom=spec.get("bottom", 0.0),
            sat_unit_weight=spec.get("gamma_sat", spec.get("sat_unit_weight")),
        )

    # ------------------------------------------------------------------
    # Interpolation helpers
    # ------------------------------------------------------------------

    def surface_elevation(self, x: float) -> float:
        """Interpolate ground surface elevation at *x*."""
        return float(np.interp(x, self.surface[:, 0], self.surface[:, 1]))

    def water_elevation(self, x: float) -> float:
        """Interpolate water-table elevation at *x*.

        Returns ``-inf`` if no water table is defined.
        """
        if self.water_table is None:
            return -np.inf
        return float(
            np.interp(x, self.water_table[:, 0], self.water_table[:, 1])
        )

    def layer_at(self, x: float, y: float) -> MaterialLayer:
        """Return the material layer at position *(x, y)*.

        Searches from top to bottom and returns the first layer whose
        elevation range contains *y*.  If no layer matches, returns the
        deepest layer.
        """
        for lay in self.layers:
            if lay.bottom <= y <= lay.top:
                return lay
        return self.layers[-1]

    def pore_pressure(self, x: float, y: float) -> float:
        """Pore-water pressure at *(x, y)*.

        u = γ_w (h_w − y)  where h_w is the water-table elevation
        at *x* and γ_w = 9810 N/m³.

        Returns 0 if the point is above the water table.
        """
        hw = self.water_elevation(x)
        if y >= hw:
            return 0.0
        return 9810.0 * (hw - y)

    @property
    def x_min(self) -> float:
        return float(self.surface[0, 0])

    @property
    def x_max(self) -> float:
        return float(self.surface[-1, 0])
