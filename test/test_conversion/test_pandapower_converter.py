import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from app.conversion.pandapower_converter import PandapowerConverter
from app.infra.quantity import Parameter


class DummyWithValue:
    """Helper that mimics a Quantity-like object with .empty() and .value(...)"""

    def __init__(self, value, empty=False):
        self._value = value
        self._empty = empty

    def empty(self):
        return self._empty

    def value(self, time_set=None, freq=None):
        # Mirror signature used by converter
        return self._value


class TestPandapowerConverter(unittest.TestCase):
    def setUp(self):
        self.conv = PandapowerConverter()
        # create empty net and ensure fresh mapping
        self.conv.net = self.conv.create_empty_net()
        self.conv.bus_map = {}

    def test_convert_quantity_empty_and_regular(self):
        # empty quantity -> None
        empty_q = Mock()
        empty_q.empty.return_value = True
        result = self.conv.convert_quantity(empty_q, "empty_test")
        self.assertIsNone(result)

        # non-Parameter quantity: should call .value(time_set=..., freq=...)
        q = DummyWithValue([1, 2, 3], empty=False)
        result2 = self.conv.convert_quantity(q, "regular", time_set=5, freq="1h")
        self.assertEqual(result2, [1, 2, 3])

        # Parameter branch: converter returns quantity.value (may be callable or attribute)
        mock_param = Mock(spec=Parameter)
        mock_param.empty.return_value = False

        # support both callable property and plain value by assigning .value to a callable
        def valfunc():
            return 99

        mock_param.value = valfunc
        result3 = self.conv.convert_quantity(mock_param, "param_test")
        # Should be the attribute stored on the object (callable in our mock)
        self.assertIs(result3, mock_param.value)

    def test_bus_creation_and_map(self):
        # Create a dummy bus with nominal_voltage.value used by _convert_bus
        bus = SimpleNamespace(
            id="bus_A",
            nominal_voltage=SimpleNamespace(value=400000),
            coordinates={"x": 11.1, "y": 42.1},
        )
        idx = self.conv._convert_bus(bus)
        # bus_map updated and bus exists in net
        self.assertIn("bus_A", self.conv.bus_map)
        net_idx = self.conv.bus_map["bus_A"]
        self.assertEqual(idx, net_idx)
        self.assertIn(net_idx, self.conv.net.bus.index)
        self.assertEqual(self.conv.net.bus.at[net_idx, "name"], "bus_A")

    def test_line_and_switch_creation(self):
        # prepare two buses
        b1 = SimpleNamespace(
            id="b1",
            nominal_voltage=SimpleNamespace(value=400000),
            coordinates={"x": 11.1, "y": 42.1},
        )
        b2 = SimpleNamespace(
            id="b2",
            nominal_voltage=SimpleNamespace(value=400000),
            coordinates={"x": 11.1, "y": 42.1},
        )
        self.conv._convert_bus(b1)
        self.conv._convert_bus(b2)

        # Create a line object with wrapped numeric attributes where needed
        class LineObj:
            def __init__(self, id, from_bus, to_bus, length, r, x, max_i, c):
                self.id = id
                self.from_bus = from_bus
                self.to_bus = to_bus
                # Wrap resistance and reactance with .value for _convert_line
                self.line_length = SimpleNamespace(value=length)
                self.resistance = SimpleNamespace(value=r)
                self.reactance = SimpleNamespace(value=x)
                self.max_current = SimpleNamespace(value=max_i)
                self.capacitance = SimpleNamespace(value=c)

        line = LineObj(
            "line_1", "b1", "b2", length=0.5, r=0.1, x=0.08, max_i=100.0, c=0.07
        )
        idx, kind = self.conv._convert_line(line)
        self.assertEqual(kind, "line")
        # check created line name present
        self.assertIn(idx, self.conv.net.line.index)
        self.assertEqual(self.conv.net.line.at[idx, "name"], "line_1")

        # Create a switch-like line (length 0 -> switch)
        sw = LineObj("switch_1", "b1", "b2", length=0, r=0, x=0, max_i=10.0, c=0)
        idx_sw, kind_sw = self.conv._convert_line(sw)
        self.assertEqual(kind_sw, "switch")
        # pandapower stores switches in net.switch
        self.assertIn(idx_sw, self.conv.net.switch.index)

    def test_slack_and_generators_and_load(self):
        # create bus and use it
        bus = SimpleNamespace(
            id="bus_gen",
            nominal_voltage=SimpleNamespace(value=400000),
            coordinates={"x": 11.1, "y": 42.1},
        )
        self.conv._convert_bus(bus)

        # Slack with bus name should create ext_grid
        slack = SimpleNamespace(id="slack_1", bus="bus_gen")
        slack_idx = self.conv._convert_slack(slack)
        self.assertIn(slack_idx, self.conv.net.ext_grid.index)
        self.assertEqual(self.conv.net.ext_grid.at[slack_idx, "name"], "slack_1")

        # PV-like static generator: object must have bus and peak_power.value
        pv = SimpleNamespace(
            id="pv_1", bus="bus_gen", peak_power=SimpleNamespace(value=1000.0)
        )
        pv_idx = self.conv._convert_pv(pv)
        self.assertIn(pv_idx, self.conv.net.sgen.index)
        self.assertEqual(self.conv.net.sgen.at[pv_idx, "name"], "pv_1")

        # Wind generator
        wind = SimpleNamespace(
            id="wind_1", bus="bus_gen", peak_power=SimpleNamespace(value=500.0)
        )
        wind_idx = self.conv._convert_wind(wind)
        self.assertIn(wind_idx, self.conv.net.sgen.index)
        self.assertEqual(self.conv.net.sgen.at[wind_idx, "name"], "wind_1")

        # Battery (sgen placeholder)
        batt = SimpleNamespace(
            id="bat_1", bus="bus_gen", max_charge_rate=SimpleNamespace(value=10)
        )
        batt_idx = self.conv._convert_battery(batt)
        self.assertIn(batt_idx, self.conv.net.sgen.index)
        self.assertEqual(self.conv.net.sgen.at[batt_idx, "name"], "bat_1")

        # Load creation uses tags for p_kw/q_kw - create a load-like object with tags
        load = SimpleNamespace(
            id="load_1",
            bus="bus_gen",
            nominal_power=SimpleNamespace(value=1000.0),
            tags={"p_kw": 2000.0, "q_kw": 500.0},
        )
        load_idx = self.conv._convert_load(load)
        self.assertIn(load_idx, self.conv.net.load.index)
        self.assertEqual(self.conv.net.load.at[load_idx, "name"], "load_1")
        # p_mw should be p_kw/1000
        self.assertAlmostEqual(self.conv.net.load.at[load_idx, "p_mw"], 1.0)

    def test_generic_trafo_type_and_transformer_creation(self):
        # Create Parameter-like objects that have .value attributes
        class ParamLike:
            def __init__(self, val):
                self.value = val

        quantities = {
            "id": ParamLike("TYP_001"),
            "sR": ParamLike(0.16),
            "vmHV": ParamLike(20.0),
            "vmLV": ParamLike(0.4),
            "vmImp": ParamLike(150),
            "pFe": ParamLike(2.35),
            "iNoLoad": ParamLike(0.46),
            "tapable": ParamLike(1),
            "tapside": ParamLike("HV"),
            "dVm": ParamLike(2.5),
            "dVa": ParamLike(0),
            "tapNeutr": ParamLike(-2),
            "tapMin": ParamLike(0),
            "tapMax": ParamLike(2),
            "nominal_power": ParamLike(0.16),
        }
        entity = SimpleNamespace(
            quantities=quantities, id="TYP_001_ent", nominal_power=0.16
        )
        std_name = self.conv._convert_generic_entity(entity)
        # std_name should be returned as a string
        self.assertIsInstance(std_name, str)

        # Now build two buses and create transformer referencing the std_type
        hv = SimpleNamespace(
            id="HV1",
            nominal_voltage=SimpleNamespace(value=20000),
            coordinates={"x": 11.1, "y": 42.1},
        )
        lv = SimpleNamespace(
            id="LV1",
            nominal_voltage=SimpleNamespace(value=400),
            coordinates={"x": 11.1, "y": 42.1},
        )
        self.conv._convert_bus(hv)
        self.conv._convert_bus(lv)

        # Create a Transformer-like object with type as an object with .value attribute
        trafo_obj = SimpleNamespace(
            id="trafo_1",
            from_bus="HV1",
            to_bus="LV1",
            type=SimpleNamespace(value=std_name),
            nominal_power=SimpleNamespace(value=0.16),
        )
        # Ensure std_types dict is properly initialized before calling _convert_trafo
        if (
            not hasattr(self.conv.net, "std_types")
            or "trafo" not in self.conv.net.std_types
        ):
            self.conv.net.std_types = {"trafo": {std_name: True}}
        trafo_idx = self.conv._convert_transformer(trafo_obj)
        # Should have created an element in net.trafo
        self.assertTrue(
            hasattr(self.conv.net, "trafo") and trafo_idx in self.conv.net.trafo.index
        )

    def test_convert_model_runs(self):
        # Ensure convert_model iterates entities and returns a net with time_set attribute
        # Create a very small model-like object with entities exposing convert()
        class SimpleEntity:
            def __init__(self, id):
                self.id = id
                self.quantities = {}
                self.sub_entities = {}
                self.relations = []

            def convert(self, time_set, freq, converter):
                # create a bus for demonstration by calling converter API
                b = SimpleNamespace(
                    id=self.id,
                    nominal_voltage=SimpleNamespace(value=400000),
                    coordinates={"x": 11.1, "y": 42.1},
                )
                converter._convert_bus(b)

        model_like = SimpleNamespace(
            entities={"m1": SimpleEntity("m1"), "m2": SimpleEntity("m2")},
            time_start="2020-01-01",
            time_end="2020-01-02",
            number_of_time_steps=8740,
            frequency="1h",
        )
        net = self.conv.convert_model(model_like)
        # Should be a pandapower net and have time_set attribute on it
        self.assertTrue(hasattr(net, "time_set"))

    # Additional basic sanity: create_empty_net structure
    def test_create_empty_net_profiles(self):
        net = self.conv.create_empty_net()
        self.assertTrue(net.__contains__("profiles"))
        self.assertIsInstance(net.profiles, dict)
        # expected profile tables exist
        self.assertIn("load", net.profiles)
        self.assertIsInstance(net.profiles["load"], pd.DataFrame)

    def test_convert_quantity_with_parameter_object(self):
        # Using the real Parameter class should return the stored scalar value
        p = Parameter(value=123.45)
        res = self.conv.convert_quantity(p, "param_scalar")
        self.assertEqual(res, 123.45)

        # Parameter with no value should yield None
        p_none = Parameter()
        res_none = self.conv.convert_quantity(p_none, "param_none")
        self.assertIsNone(res_none)


if __name__ == "__main__":
    unittest.main()
