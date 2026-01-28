"""FEniCS finite-element solver backend.

Requires DOLFINx (``pip install fenics-dolfinx``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pygeotech.solvers.base import Solver, Solution


class FEniCSBackend(Solver):
    """Finite-element solver using FEniCS/DOLFINx.

    This is a placeholder backend.  Full implementation requires
    DOLFINx and translates the pygeotech problem definition into
    FEniCS variational forms.
    """

    def solve(
        self,
        physics: Any,
        boundary_conditions: Any = None,
        time: Any | None = None,
        initial_condition: dict[str, float | np.ndarray] | None = None,
        **kwargs: Any,
    ) -> Solution:
        raise NotImplementedError(
            "FEniCSBackend requires fenics-dolfinx.  "
            "Install with: pip install fenics-dolfinx"
        )
