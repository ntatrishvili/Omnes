import unittest

from app.model.grid_component.line import Line


def test_line_initialization_with_defaults():
    line = Line(id="line1", from_bus="busA", to_bus="busB")
    assert line.id == "line1"
    assert line.from_bus == "busA"
    assert line.to_bus == "busB"
    assert "current" in line.quantities
    assert line.max_current.value is None


def test_line_with_custom_parameters():
    line = Line(
        id="line1",
        from_bus="busA",
        to_bus="busB",
        max_current=100,
        line_length=1.2,
        resistance=0.01,
        reactance=0.02,
    )
    assert line.max_current.value == 100
    assert line.line_length.value == 1.2
    assert line.resistance.value == 0.01
    assert line.reactance.value == 0.02


def test_line_str_representation():
    line = Line(id="lineX", from_bus="busA", to_bus="busB", line_length=1.0)
    assert "Line 'lineX'" in str(line)
    assert "busA--busB" in str(line)


if __name__ == "__main__":
    unittest.main()
