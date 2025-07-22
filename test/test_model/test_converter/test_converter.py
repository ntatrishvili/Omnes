import unittest

from app.infra.quantity import Parameter
from app.model.converter.converter import Converter, WaterHeater
from app.model.device import Device


def test_converter_basic_attributes():
    input_device = Device(id="dev1", bus="bus1")
    output_device = Device(id="dev2", bus="bus1")

    conv = Converter(
        id="conv1",
        input_device=input_device,
        output_device=output_device,
        conversion_efficiency=0.8,
    )

    assert conv.id == "conv1"
    assert conv.p_in.empty()
    assert conv.p_out.empty()
    assert isinstance(conv.conversion_efficiency, Parameter)
    assert conv.conversion_efficiency.value == 0.8
    assert conv.input_device == "dev1"
    assert conv.output_device == "dev2"
    assert conv.bus == "bus1"


def test_water_heater_initialization_defaults():
    heater = WaterHeater(id="wh1", charges="bat1", bus="bus1")

    assert heater.id == "wh1"
    assert heater.input_device == "wh1"
    assert heater.output_device == "bat1"
    assert heater.controllable is True
    assert heater.conversion_efficiency.empty()
    assert heater.p_out.empty()
    assert heater.p_in.empty()


if __name__ == "__main__":
    unittest.main()
