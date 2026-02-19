"""Standard materials library.

Pre-configured :class:`~pygeotech.materials.base.Material` instances for
common geotechnical and structural materials.  Property values represent
typical textbook ranges (mid-range estimates).

Usage::

    from pygeotech.materials import sand, clay
    print(sand.hydraulic_conductivity)  # 1e-4 m/s
"""

from pygeotech.materials.base import Material

# ------------------------------------------------------------------
# Soils
# ------------------------------------------------------------------

sand = Material(
    name="sand",
    hydraulic_conductivity=1e-4,   # m/s
    porosity=0.35,
    dry_density=1600.0,            # kg/m³
    youngs_modulus=50e6,           # Pa
    poissons_ratio=0.30,
    friction_angle=33.0,           # degrees
    cohesion=0.0,                  # Pa
    thermal_conductivity=2.0,      # W/(m·K)
    specific_heat=800.0,           # J/(kg·K)
)

clay = Material(
    name="clay",
    hydraulic_conductivity=1e-9,
    porosity=0.45,
    dry_density=1400.0,
    youngs_modulus=10e6,
    poissons_ratio=0.40,
    friction_angle=20.0,
    cohesion=25e3,
    thermal_conductivity=1.5,
    specific_heat=900.0,
)

silt = Material(
    name="silt",
    hydraulic_conductivity=1e-6,
    porosity=0.40,
    dry_density=1500.0,
    youngs_modulus=20e6,
    poissons_ratio=0.35,
    friction_angle=28.0,
    cohesion=5e3,
    thermal_conductivity=1.8,
    specific_heat=850.0,
)

gravel = Material(
    name="gravel",
    hydraulic_conductivity=1e-2,
    porosity=0.30,
    dry_density=1800.0,
    youngs_modulus=100e6,
    poissons_ratio=0.25,
    friction_angle=38.0,
    cohesion=0.0,
    thermal_conductivity=2.5,
    specific_heat=750.0,
)

# ------------------------------------------------------------------
# Structural / geological
# ------------------------------------------------------------------

concrete = Material(
    name="concrete",
    hydraulic_conductivity=1e-10,
    porosity=0.05,
    dry_density=2400.0,
    youngs_modulus=30e9,
    poissons_ratio=0.20,
    thermal_conductivity=1.7,
    specific_heat=880.0,
)

rock = Material(
    name="rock",
    hydraulic_conductivity=1e-8,
    porosity=0.10,
    dry_density=2600.0,
    youngs_modulus=20e9,
    poissons_ratio=0.25,
    friction_angle=45.0,
    cohesion=1e6,
    thermal_conductivity=3.0,
    specific_heat=800.0,
)
