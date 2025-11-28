from app.conversion.pulp_converter import PulpConverter
from app.infra.relation import Relation
from app.model.generator.pv import PV
from app.model.generator.wind_turbine import Wind
from app.model.grid_component.bus import Bus, BusType
from app.model.grid_component.line import Line
from app.model.load.load import Load
from app.model.model import Model
from app.model.slack import Slack
from app.model.storage.battery import Battery
from app.model.storage.hot_water_storage import HotWaterStorage
from app.model.transducer.transducer import Transducer
from app.operation.example_optimization import optimize_energy_system

Bus.default_nominal_voltage = 400
bus_mv = Bus(id="bus_MV", nominal_voltage=10000, type=BusType.SLACK, phase_count=3)
slack = Slack(id="slack", bus="bus_MV")

bus_lv1 = Bus(id="bus_LV1")
bus_lv2 = Bus(id="bus_LV2", phase="C", phase_count=1)
bus_lv3 = Bus(id="bus_LV3", phase_count=3)

# Lehetne a default egy külön struktúra?
# a line ne a buszok nevét vegye át, hanem a referenciát rájuk
Line.default_line_length = 0.1
Line.default_resistance = 0.05
Line.default_reactance = 0.05
line1 = Line(id="line1", from_bus="bus_MV", to_bus="bus_LV1")
line2 = Line(id="line2", from_bus="bus_LV1", to_bus="bus_LV2")
line3 = Line(id="line3", from_bus="bus_LV1", to_bus="bus_LV3")

# Instantiate devices
PV.default_efficiency = 0.9
# Instantiate PVs
# If 'col' not specified, default 'col' will be the ID of the element
pv1 = PV(
    id="pv1",
    bus="bus_LV1",
    peak_power=4,
    p_out={"input_path": "data/input.csv"},
    tags={"household": "HH1"},
)

pv2 = PV(
    id="pv2",
    bus="bus_LV2",
    peak_power=3,
    p_out={"input_path": "data/input2.csv"},
    tags={"household": "HH2"},
)

wind1 = Wind(
    id="wind1",
    bus="bus_LV3",
    peak_power=5,
    efficiency=0.95,
    p_out={"input_path": "data/input.csv", "col": "wind"},
)

# Instantiate Battery
#
relation2 = Relation(
    "if battery1.capacity < 6 then battery1.max_discharge_rate = 3",
    "Battery1.CapacityRelation",
)

battery1 = Battery(
    id="battery1",
    bus="bus_LV3",
    capacity=5,
    max_charge_rate=2,
    max_discharge_rate=2,
    charge_efficiency=0.95,
    discharge_efficiency=0.95,
    storage_efficiency=0.995,
    relations=[
        relation2,
    ],
)

hot_water_storage1 = HotWaterStorage(
    id="hot_water1",
    bus="bus_LV1",
    volume=120,
    set_temperature=60,
    p_out={"input_path": "data/input.csv", "col": "hot_water1"},
    tags={"household": "HH1"},
)

relation3 = Relation("heater1.power enabled from 10:00 to 16:00")
relation4 = Relation("heater1.min_on_duration = 2h")
water_heater1 = Transducer(
    id="heater1",
    controllable=True,
    charges="hot_water1",
    bus="bus_LV1",
    conversion_efficiency=0.95,
    tags={"household": "HH1"},
    relations=[relation3, relation4],
)

hot_water_storage2 = HotWaterStorage(
    id="hot_water2",
    bus="bus_LV2",
    volume=200,
    set_temperature=55,
    input={"input_path": "data/input.csv", "col": "hot_water2"},
    tags={"household": "HH2"},
)

relation5 = Relation("heater2.power enabled from 10:00 to 16:00")
relation6 = Relation("heater2.min_on_duration = 2h")
water_heater2 = Transducer(
    id="heater2",
    charges="hot_water2",
    bus="bus_LV2",
    conversion_efficiency=0.995,
    tags={"household": "HH2"},
    relations=[relation5, relation6],
)

# Instantiate Loads
load1 = Load(
    id="load1",
    bus="bus_LV1",
    input={"input_path": "data/input.csv"},
    tags={"household": "HH1"},
)
load2 = Load(
    id="load2",
    bus="bus_LV2",
    input={"input_path": "data/input2.csv"},
    tags={"household": "HH2"},
)

# relation1 = Relation("battery1.max_discharge_rate < 2 * pv1.peak_power")
# e = Entity(relations=[relation1,])

time_resolution = "1h"
model = Model(
    id="Energy_Community",
    time_start="2025-01-01 00:00",
    time_end="2025-01-02 00:00",
    resolution=time_resolution,
    entities=[
        bus_mv,
        bus_lv1,
        bus_lv2,
        bus_lv3,
        line1,
        line2,
        line3,
        pv1,
        pv2,
        wind1,
        battery1,
        slack,
        load1,
        load2,
        water_heater1,
        water_heater2,
        hot_water_storage1,
        hot_water_storage2,
    ],
)

number_of_time_steps = model.number_of_time_steps
problem = PulpConverter().convert_model(model)
optimize_energy_system(**problem)
