"""Comprehensive tests for the coupling module."""

import numpy as np
import pytest

from pygeotech.geometry.primitives import Rectangle
from pygeotech.materials.base import Material, assign
from pygeotech.physics.darcy import Darcy
from pygeotech.physics.richards import Richards
from pygeotech.physics.transport import Transport
from pygeotech.physics.heat import HeatTransfer
from pygeotech.physics.mechanics import Mechanics
from pygeotech.materials.constitutive import VanGenuchten
from pygeotech.coupling.sequential import Sequential
from pygeotech.coupling.iterative import Iterative
from pygeotech.boundaries.base import BoundaryConditions, Dirichlet
from pygeotech.boundaries.locators import left, right, top, bottom
from pygeotech.solvers.fipy_backend import FiPyBackend
from pygeotech.time.stepper import Stepper


def _make_flow_transport():
    """Create a coupled flow-transport problem."""
    domain = Rectangle(Lx=20, Ly=2)
    mesh = domain.generate_mesh(resolution=2.0)
    mat = Material(
        name="sand",
        hydraulic_conductivity=1e-4,
        porosity=0.35,
    )
    materials = assign(mesh, {"default": mat})
    flow = Darcy(mesh, materials)
    transport = Transport(mesh, materials)
    return flow, transport, mesh


def _make_flow_mechanics():
    """Create a coupled flow-mechanics problem."""
    domain = Rectangle(Lx=10, Ly=5)
    mesh = domain.generate_mesh(resolution=2.0)
    mat = Material(
        name="clay",
        hydraulic_conductivity=1e-8,
        porosity=0.40,
        youngs_modulus=50e6,
        poissons_ratio=0.3,
        dry_density=1600.0,
    )
    materials = assign(mesh, {"default": mat})
    flow = Darcy(mesh, materials)
    mech = Mechanics(mesh, materials)
    return flow, mech, mesh


class TestSequentialCoupling:
    def test_creation(self):
        flow, transport, _ = _make_flow_transport()
        seq = Sequential([flow, transport])
        assert seq.coupling_strategy == "sequential"
        assert len(seq.modules) == 2

    def test_step_flow_transport(self):
        """Sequential step should solve flow then transport."""
        flow, transport, mesh = _make_flow_transport()

        # Attach BCs
        flow_bc = BoundaryConditions()
        flow_bc.add(Dirichlet(field="H", value=10.0, where=left()))
        flow_bc.add(Dirichlet(field="H", value=5.0, where=right()))
        flow._boundary_conditions = flow_bc

        transport_bc = BoundaryConditions()
        transport_bc.add(Dirichlet(field="C", value=1.0, where=left()))
        transport_bc.add(Dirichlet(field="C", value=0.0, where=right()))
        transport._boundary_conditions = transport_bc

        seq = Sequential([flow, transport])
        solutions = {
            "darcy": np.full(mesh.n_nodes, 7.5),
            "transport": np.zeros(mesh.n_nodes),
        }

        updated = seq.step(solutions, dt=1000.0)
        assert "darcy" in updated
        assert "transport" in updated
        assert isinstance(updated["darcy"], np.ndarray)
        assert isinstance(updated["transport"], np.ndarray)

    def test_velocity_transfer(self):
        """Flow solution should transfer velocity to transport."""
        flow, transport, mesh = _make_flow_transport()

        flow_bc = BoundaryConditions()
        flow_bc.add(Dirichlet(field="H", value=10.0, where=left()))
        flow_bc.add(Dirichlet(field="H", value=5.0, where=right()))
        flow._boundary_conditions = flow_bc

        transport_bc = BoundaryConditions()
        transport_bc.add(Dirichlet(field="C", value=1.0, where=left()))
        transport_bc.add(Dirichlet(field="C", value=0.0, where=right()))
        transport._boundary_conditions = transport_bc

        seq = Sequential([flow, transport])
        solutions = {
            "darcy": np.full(mesh.n_nodes, 7.5),
            "transport": np.zeros(mesh.n_nodes),
        }

        updated = seq.step(solutions, dt=1000.0)
        # After stepping, transport should have had velocity set
        assert transport._velocity is not None

    def test_flow_mechanics_coupling(self):
        """Flow should transfer pore pressure to mechanics."""
        flow, mech, mesh = _make_flow_mechanics()

        flow_bc = BoundaryConditions()
        flow_bc.add(Dirichlet(field="H", value=10.0, where=top()))
        flow_bc.add(Dirichlet(field="H", value=5.0, where=bottom()))
        flow._boundary_conditions = flow_bc

        mech_bc = BoundaryConditions()
        mech_bc.add(Dirichlet(field="u", value=0.0, where=bottom()))
        mech_bc.add(Dirichlet(field="ux", value=0.0, where=left()))
        mech_bc.add(Dirichlet(field="ux", value=0.0, where=right()))
        mech._boundary_conditions = mech_bc

        seq = Sequential([flow, mech])
        solutions = {
            "darcy": np.full(mesh.n_nodes, 7.5),
            "mechanics": np.zeros(2 * mesh.n_nodes),
        }

        updated = seq.step(solutions)
        assert "darcy" in updated
        assert "mechanics" in updated
        assert isinstance(updated["mechanics"], np.ndarray)
        assert updated["mechanics"].shape == (2 * mesh.n_nodes,)


class TestIterativeCoupling:
    def test_creation(self):
        flow, transport, _ = _make_flow_transport()
        it = Iterative([flow, transport], max_iter=10, tol=1e-5)
        assert it.max_iter == 10
        assert it.tol == 1e-5
        assert it.coupling_strategy == "iterative"

    def test_convergence(self):
        """Iterative coupling should converge for simple problem."""
        flow, transport, mesh = _make_flow_transport()

        flow_bc = BoundaryConditions()
        flow_bc.add(Dirichlet(field="H", value=10.0, where=left()))
        flow_bc.add(Dirichlet(field="H", value=5.0, where=right()))
        flow._boundary_conditions = flow_bc

        transport_bc = BoundaryConditions()
        transport_bc.add(Dirichlet(field="C", value=1.0, where=left()))
        transport_bc.add(Dirichlet(field="C", value=0.0, where=right()))
        transport._boundary_conditions = transport_bc

        it = Iterative([flow, transport], max_iter=5, tol=1e-4)
        solutions = {
            "darcy": np.full(mesh.n_nodes, 7.5),
            "transport": np.zeros(mesh.n_nodes),
        }

        updated = it.step(solutions, dt=1000.0)
        assert "darcy" in updated
        assert "transport" in updated

    def test_converged_method(self):
        flow, transport, _ = _make_flow_transport()
        it = Iterative([flow, transport])
        old = {"darcy": np.array([1.0, 2.0, 3.0])}
        new = {"darcy": np.array([1.0, 2.0, 3.0])}
        assert it._converged(old, new) is True

        new2 = {"darcy": np.array([10.0, 20.0, 30.0])}
        assert it._converged(old, new2) is False


class TestCoupledSolver:
    def test_coupled_steady_state(self):
        """FiPyBackend should handle coupled steady-state."""
        flow, transport, mesh = _make_flow_transport()

        flow_bc = BoundaryConditions()
        flow_bc.add(Dirichlet(field="H", value=10.0, where=left()))
        flow_bc.add(Dirichlet(field="H", value=5.0, where=right()))

        transport_bc = BoundaryConditions()
        transport_bc.add(Dirichlet(field="C", value=1.0, where=left()))
        transport_bc.add(Dirichlet(field="C", value=0.0, where=right()))

        seq = Sequential([flow, transport])
        solver = FiPyBackend()
        sol = solver.solve(
            seq,
            boundary_conditions={"darcy": flow_bc, "transport": transport_bc},
        )
        assert "H" in sol.fields
        assert "C" in sol.fields

    def test_coupled_transient(self):
        """FiPyBackend should handle coupled transient."""
        flow, transport, mesh = _make_flow_transport()

        flow_bc = BoundaryConditions()
        flow_bc.add(Dirichlet(field="H", value=10.0, where=left()))
        flow_bc.add(Dirichlet(field="H", value=5.0, where=right()))

        transport_bc = BoundaryConditions()
        transport_bc.add(Dirichlet(field="C", value=1.0, where=left()))
        transport_bc.add(Dirichlet(field="C", value=0.0, where=right()))

        seq = Sequential([flow, transport])
        time = Stepper(t_end=2000.0, dt=1000.0)
        solver = FiPyBackend()
        sol = solver.solve(
            seq,
            boundary_conditions={"darcy": flow_bc, "transport": transport_bc},
            time=time,
            initial_condition={"H": 7.5, "C": 0.0},
        )
        assert "H" in sol.fields
        assert "C" in sol.fields
        assert sol.times is not None
