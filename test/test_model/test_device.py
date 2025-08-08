import unittest

import pytest

from app.model.device import Device, Vector


def test_device_initialization_with_bus():
    device = Device(id="dev1", bus="bus1")
    assert device.id == "dev1"
    assert device.bus == "bus1"
    assert device.tags["vector"] == Vector.INVALID
    assert device.tags["contributes_to"] is None


def test_device_custom_tags():
    device = Device(id="dev1", bus="bus1", vector=Vector.HEAT, contributes_to="load1")
    assert device.tags["vector"] == Vector.HEAT
    assert device.tags["contributes_to"] == "load1"


def test_update_tags():
    device = Device(id="dev1", bus="bus1")
    device.update_tags(vector=Vector.ELECTRICITY)
    assert device.tags["vector"] == Vector.ELECTRICITY


if __name__ == "__main__":
    unittest.main()
