"""Physics: governing equations for geotechnical processes."""

from pygeotech.physics.base import PhysicsModule
from pygeotech.physics.darcy import Darcy
from pygeotech.physics.richards import Richards
from pygeotech.physics.transport import Transport
from pygeotech.physics.heat import HeatTransfer
from pygeotech.physics.mechanics import Mechanics
from pygeotech.physics.svat import SVAT

__all__ = [
    "PhysicsModule",
    "Darcy",
    "Richards",
    "Transport",
    "HeatTransfer",
    "Mechanics",
    "SVAT",
]
