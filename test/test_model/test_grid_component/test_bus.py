import math
import unittest

from app.model.grid_component.bus import Bus, BusType


def test_bus_defaults():
    bus = Bus(id="bus1")
    assert bus.id == "bus1"
    assert bus.nominal_voltage.value is None
    assert bus.phase == 3
    assert bus.type.value == BusType.PQ


def test_bus_custom_nominal_voltage_and_type():
    bus = Bus(id="bus1", nominal_voltage=400.0, type="SLACK")
    assert math.isclose(bus.nominal_voltage.value, 400.0)
    assert bus.type.value == BusType.SLACK


def test_bus_str_representation():
    bus = Bus(id="bus1", nominal_voltage=230.0, type="I")
    assert "Bus 'bus1'" in str(bus)
    assert "nominal voltage=230.0" in str(bus)
    assert "type 'I'" in str(bus)


if __name__ == "__main__":
    unittest.main()
