import math
import unittest

from app.model.grid_component.transformer import Transformer
from app.infra.quantity import Parameter


def test_transformer_defaults():
    tr = Transformer(id="tr1", from_bus="HV1", to_bus="LV1")
    assert tr.id == "tr1"
    # quantities should exist and be Parameter instances
    assert isinstance(tr.nominal_power, Parameter)
    assert isinstance(tr.nominal_voltage_hv_side, Parameter)
    assert isinstance(tr.nominal_voltage_lv_side, Parameter)
    assert isinstance(tr.type, Parameter)
    # default values from the class
    assert tr.nominal_power.value == Transformer.default_nominal_power
    assert (
        tr.nominal_voltage_hv_side.value == Transformer.default_nominal_voltage_hv_side
    )
    assert (
        tr.nominal_voltage_lv_side.value == Transformer.default_nominal_voltage_lv_side
    )
    assert tr.type.value == ""


def test_transformer_custom_values():
    tr = Transformer(
        id="trX",
        from_bus="HVX",
        to_bus="LVX",
        nominal_power=0.5,
        nominal_voltage_hv_side=20.0,
        nominal_voltage_lv_side=0.5,
        type="distribution",
    )
    assert math.isclose(tr.nominal_power.value, 0.5)
    assert math.isclose(tr.nominal_voltage_hv_side.value, 20.0)
    assert math.isclose(tr.nominal_voltage_lv_side.value, 0.5)
    assert tr.type.value == "distribution"


def test_transformer_str_includes_connector_info():
    tr = Transformer(id="trS", from_bus="fromB", to_bus="toB")
    s = str(tr)
    assert "Connector 'trS'" in s
    assert "fromB--toB" in s


if __name__ == "__main__":
    unittest.main()
