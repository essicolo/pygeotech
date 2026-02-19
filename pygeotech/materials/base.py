"""Base material class and assignment utilities.

Classes
-------
Material
    Container for material properties used by physics modules.

Functions
---------
assign
    Map materials to mesh subdomains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


class Material:
    """Generic material with named properties.

    Properties are stored in a dictionary and can be scalars, arrays (for
    anisotropic tensors), or callables (for nonlinear / state-dependent
    properties).

    Args:
        name: Human-readable material name.
        **kwargs: Arbitrary material properties.  Common keys include:

            * ``hydraulic_conductivity`` — K (m/s), scalar or 2×2/3×3 tensor.
            * ``porosity`` — n (–).
            * ``dry_density`` — ρ_d (kg/m³).
            * ``youngs_modulus`` — E (Pa).
            * ``poissons_ratio`` — ν (–).
            * ``thermal_conductivity`` — λ (W/(m·K)).
            * ``specific_heat`` — c (J/(kg·K)).

    Example::

        mat = Material(
            name="sandy_silt",
            hydraulic_conductivity=1e-5,
            porosity=0.35,
            dry_density=1600,
        )
        mat["hydraulic_conductivity"]  # 1e-5
    """

    def __init__(self, name: str = "unnamed", **kwargs: Any) -> None:
        self.name = name
        self._props: dict[str, Any] = dict(kwargs)

    # dict-like access -----------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._props[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._props[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._props

    def get(self, key: str, default: Any = None) -> Any:
        """Return property *key*, or *default* if absent."""
        return self._props.get(key, default)

    @property
    def properties(self) -> dict[str, Any]:
        """Read-only view of all stored properties."""
        return dict(self._props)

    # convenience property accessors ----------------------------------------

    @property
    def hydraulic_conductivity(self) -> Any:
        """Hydraulic conductivity K."""
        return self._props.get("hydraulic_conductivity")

    @property
    def porosity(self) -> Any:
        """Porosity n."""
        return self._props.get("porosity")

    def __repr__(self) -> str:
        props = ", ".join(f"{k}={v!r}" for k, v in self._props.items())
        return f"Material(name={self.name!r}, {props})"


class MaterialMap:
    """Mapping from mesh cells to :class:`Material` instances.

    Created by :func:`assign`.  Provides per-cell look-up of any material
    property.

    Attributes:
        materials: List of materials in tag order.
        cell_material_index: Integer index per cell into *materials*.
    """

    def __init__(
        self,
        materials: list[Material],
        cell_material_index: np.ndarray,
    ) -> None:
        self.materials = materials
        self.cell_material_index = np.asarray(cell_material_index, dtype=int)

    def cell_property(self, key: str) -> np.ndarray:
        """Return an array of property *key* for every cell.

        Scalar properties yield a 1-D array of shape ``(n_cells,)``.
        Tensor properties yield an array of shape ``(n_cells, ...)``.

        Raises:
            KeyError: If any material lacks the requested property.
        """
        values = [m[key] for m in self.materials]
        # Determine if scalar or array-valued
        sample = np.asarray(values[0])
        if sample.ndim == 0:
            out = np.empty(len(self.cell_material_index), dtype=float)
            for i, v in enumerate(values):
                out[self.cell_material_index == i] = float(v)
            return out
        else:
            shape = sample.shape
            out = np.empty((len(self.cell_material_index), *shape), dtype=float)
            for i, v in enumerate(values):
                out[self.cell_material_index == i] = np.asarray(v)
            return out

    def __repr__(self) -> str:
        names = [m.name for m in self.materials]
        return f"MaterialMap(materials={names}, n_cells={len(self.cell_material_index)})"


def assign(
    mesh: Any,
    mapping: dict[str, Material],
) -> MaterialMap:
    """Assign materials to mesh cells based on subdomain tags.

    Args:
        mesh: A :class:`~pygeotech.geometry.mesh.Mesh` instance.
        mapping: Dictionary whose keys are subdomain names (or
            ``"default"`` for untagged cells) and values are
            :class:`Material` instances.

    Returns:
        A :class:`MaterialMap` that can look up per-cell properties.

    Raises:
        ValueError: If a subdomain name in *mapping* is not found in the
            mesh and is not ``"default"``.
    """
    # Build ordered material list:  index 0 = default
    mat_list: list[Material] = []
    tag_to_index: dict[int, int] = {}

    if "default" in mapping:
        mat_list.append(mapping["default"])
    else:
        # Create a placeholder — should not be reached if mesh is well-formed
        mat_list.append(Material(name="_unassigned"))

    for name, mat in mapping.items():
        if name == "default":
            continue
        if name not in mesh.subdomain_map:
            raise ValueError(
                f"Subdomain '{name}' not found in mesh.  "
                f"Available: {list(mesh.subdomain_map.keys())}"
            )
        idx = len(mat_list)
        mat_list.append(mat)
        tag_to_index[mesh.subdomain_map[name]] = idx

    # Map cell tags → material index
    cell_idx = np.zeros(mesh.n_cells, dtype=int)
    for tag, midx in tag_to_index.items():
        cell_idx[mesh.cell_tags == tag] = midx

    return MaterialMap(materials=mat_list, cell_material_index=cell_idx)
