# pygeotech Examples

Jupytext percent-format notebooks (`.py`) covering the main use cases.
Open them in Jupyter with `jupytext --to notebook *.py`, or run directly
with `python 01_darcy_seepage.py`.

| #  | File                              | Topic                                     |
|----|-----------------------------------|-------------------------------------------|
| 01 | `01_darcy_seepage.py`             | Steady-state Darcy flow through a dam     |
| 02 | `02_richards_infiltration.py`     | Transient Richards infiltration (VG)      |
| 03 | `03_contaminant_transport.py`     | Advection–dispersion with sorption/decay  |
| 04 | `04_heat_conduction.py`           | Transient heat conduction in soil         |
| 05 | `05_elastic_settlement.py`        | Elastic settlement under a strip footing  |
| 06 | `06_coupled_flow_transport.py`    | Coupled Darcy + heat advection            |
| 07 | `07_slope_stability_lem.py`       | LEM: Bishop, Spencer, Janbu, M-P         |
| 08 | `08_critical_surface_search.py`   | Grid search + Strength Reduction Method   |
| 09 | `09_stratigraphy_classification.py` | Borehole clustering & 2-D interpolation |
| 10 | `10_stratigraphy_3d_model.py`     | 3-D stratigraphic model & cross-sections  |
| 11 | `11_basin_multiphysics.py`        | Multi-layer basin: stratigraphy + flow + transport |
| 12 | `12_gmsh_complex_geometry.py`     | Complex geometry with gmsh (requires `pip install gmsh`) |

## Prerequisites

```bash
pip install pygeotech matplotlib
pip install scikit-learn   # for example 09 (clustering)
pip install gmsh           # for example 12 (complex geometry)
```

## Notes

- Examples 01–06 use the built-in `Rectangle` mesher.
- Example 12 requires gmsh and shows workflows for FreeCAD/Blender
  geometry import. Some parts need user-provided `.step` or `.stl`
  files — see the notebook for details.
