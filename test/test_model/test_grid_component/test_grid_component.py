import unittest

import pytest

from app.model.grid_component.grid_component import GridComponent


def test_single_phase_valid():
    component = GridComponent(id="G2", phase="A")
    assert component.phase == "A"


if __name__ == "__main__":
    unittest.main()
