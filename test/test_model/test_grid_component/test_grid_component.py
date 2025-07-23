import unittest

import pytest

from app.model.grid_component.grid_component import GridComponent


def test_default_phase_count_and_phase():
    component = GridComponent(id="G1")
    assert component.phase_count == 3
    assert component.phase is None


def test_single_phase_valid():
    component = GridComponent(id="G2", phase_count=1, phase="A")
    assert component.phase_count == 1
    assert component.phase == "A"


def test_three_phase_with_explicit_phase_raises():
    with pytest.raises(
        ValueError, match="Phase must not be set for three-phase buses."
    ):
        GridComponent(id="G3", phase_count=3, phase="A")


def test_single_phase_without_phase_raises():
    with pytest.raises(ValueError, match="Phase must be set for single-phase buses."):
        GridComponent(id="G4", phase_count=1)


def test_invalid_phase_count_raises():
    with pytest.raises(ValueError, match="Phase count must be 1 or 3."):
        GridComponent(id="G5", phase_count=2)


def test_override_defaults_explicitly():
    component = GridComponent(id="G6", phase_count=3)
    assert component.phase_count == 3
    assert component.phase is None


if __name__ == "__main__":
    unittest.main()
