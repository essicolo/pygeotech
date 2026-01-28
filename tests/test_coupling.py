"""Tests for the coupling module."""

import pytest

from pygeotech.geometry.primitives import Rectangle
from pygeotech.materials.base import Material, assign
from pygeotech.physics.darcy import Darcy
from pygeotech.physics.transport import Transport
from pygeotech.coupling.sequential import Sequential
from pygeotech.coupling.iterative import Iterative
from pygeotech.coupling.monolithic import Monolithic


def _make_modules():
    domain = Rectangle(Lx=10, Ly=5)
    mesh = domain.generate_mesh(resolution=1.0)
    mat = Material(name="test", hydraulic_conductivity=1e-5, porosity=0.35)
    materials = assign(mesh, {"default": mat})
    flow = Darcy(mesh, materials)
    transport = Transport(mesh, materials)
    return flow, transport


class TestSequential:
    def test_creation(self):
        flow, transport = _make_modules()
        seq = Sequential([flow, transport])
        assert seq.coupling_strategy == "sequential"
        assert len(seq.modules) == 2
        assert seq.primary_fields == ["H", "C"]

    def test_is_transient(self):
        flow, transport = _make_modules()
        seq = Sequential([flow, transport])
        assert seq.is_transient  # transport is transient


class TestIterative:
    def test_creation(self):
        flow, transport = _make_modules()
        it = Iterative([flow, transport], max_iter=10, tol=1e-5)
        assert it.max_iter == 10
        assert it.tol == 1e-5


class TestMonolithic:
    def test_step_not_implemented(self):
        flow, transport = _make_modules()
        mono = Monolithic([flow, transport])
        with pytest.raises(NotImplementedError):
            mono.step({})
