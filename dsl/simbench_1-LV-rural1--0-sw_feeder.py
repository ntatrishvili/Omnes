from pulp import LpMinimize, LpProblem

from app.conversion.pulp_converter import PulpConverter
from app.infra.relation import Relation
from app.model.converter.converter import Converter
from app.model.entity import Entity
from app.model.generator.pv import PV
from app.model.generator.wind_turbine import Wind
from app.model.grid_component.bus import Bus, BusType
from app.model.grid_component.line import Line
from app.model.load.load import Load
from app.model.model import Model
from app.model.slack import Slack
from app.model.storage.battery import Battery
from app.model.storage.hot_water_storage import HotWaterStorage
from app.operation.example_optimization import optimize_energy_system

# network base settings
Bus.default_nominal_voltage = 400  # LV line-to-line nominal
Line.default_line_length = 0.15    # km (example)
Line.default_resistance = 0.08     # ohm/km (example)
Line.default_reactance = 0.06      # ohm/km (example)
PV.default_efficiency = 0.9

# MV / grid connection
bus_mv = Bus(id="bus_MV", nominal_voltage=10000, type=BusType.SLACK, phase_count=3)
slack = Slack(id="slack", bus="bus_MV")

# LV feeder backbone: one LV feeder that splits into branches
bus_lv_head = Bus(id="bus_LV_head")            # feeder head (LV side of MV/LV transformer)
bus_lv_1 = Bus(id="bus_LV_1", phase="A", phase_count=1)   # household 1 (single-phase)
bus_lv_2 = Bus(id="bus_LV_2", phase="B", phase_count=1)   # household 2
bus_lv_3 = Bus(id="bus_LV_3", phase="C", phase_count=1)   # household 3
bus_lv_4 = Bus(id="bus_LV_4", phase_count=3)              # small multi-phase connection (e.g., small farm)
bus_lv_5 = Bus(id="bus_LV_5", phase="A", phase_count=1)   # household 4
bus_lv_6 = Bus(id="bus_LV_6", phase="B", phase_count=1)   # household 5

# Lines (simple radial layout, head -> branches)
line_hv_to_lv = Line(id="line_MV_LV", from_bus="bus_MV", to_bus="bus_LV_head")

line_1 = Line(id="line_1", from_bus="bus_LV_head", to_bus="bus_LV_1")
line_2 = Line(id="line_2", from_bus="bus_LV_head", to_bus="bus_LV_2")
line_3 = Line(id="line_3", from_bus="bus_LV_head", to_bus="bus_LV_3")
line_4 = Line(id="line_4", from_bus="bus_LV_head", to_bus="bus_LV_4")
line_5 = Line(id="line_5", from_bus="bus_LV_4", to_bus="bus_LV_5")
line_6 = Line(id="line_6", from_bus="bus_LV_4", to_bus="bus_LV_6")

# Generators: PVs at some households and one small WT at the multi-phase bus
pv_1 = PV(
    id="pv_1",
    bus="bus_LV_1",
    peak_power=3.5,  # kW
    input={"input_path": "data/simbench/rural1/pv_p_1.csv"},  # replace with real SimBench path
    tags={"household": "HH1"},
)

pv_2 = PV(
    id="pv_2",
    bus="bus_LV_2",
    peak_power=2.5,
    input={"input_path": "data/simbench/rural1/pv_p_2.csv"},
    tags={"household": "HH2"},
)

pv_3 = PV(
    id="pv_3",
    bus="bus_LV_5",
    peak_power=1.8,
    input={"input_path": "data/simbench/rural1/pv_p_5.csv"},
    tags={"household": "HH4"},
)

wind_1 = Wind(
    id="wind_1",
    bus="bus_LV_4",
    peak_power=4.0,
    efficiency=0.95,
    input={"input_path": "data/simbench/rural1/wind_p_4.csv", "col": "wind"},
    tags={"farm": "small_farm"},
)

# Shared battery at the end of feeder (e.g., community battery)
relation_batt = Relation("if battery_shared.capacity < 10 then battery_shared.max_discharge_rate = 2", "BatteryShared.CapacityRelation")
battery_shared = Battery(
    id="battery_shared",
    bus="bus_LV_6",
    capacity=8.0,               # kWh
    max_charge_rate=3.0,        # kW
    max_discharge_rate=3.0,     # kW
    charge_efficiency=0.95,
    discharge_efficiency=0.95,
    storage_efficiency=0.995,
    relations=[relation_batt],
    tags={"community": "shared_storage"},
)

# Hot water storages and converters (household-level flexible loads)
hot_water_1 = HotWaterStorage(
    id="hot_water_1",
    bus="bus_LV_1",
    volume=120,
    set_temperature=60,
    input={"input_path": "data/simbench/rural1/hw_1.csv", "col": "hw_1"},
    tags={"household": "HH1"},
)
rel_hw1_a = Relation("heater1.power enabled from 06:00 to 09:00")
rel_hw1_b = Relation("heater1.min_on_duration = 1h")
heater1 = Converter(
    id="heater1",
    controllable=True,
    charges="hot_water_1",
    bus="bus_LV_1",
    conversion_efficiency=0.95,
    tags={"household": "HH1"},
    relations=[rel_hw1_a, rel_hw1_b],
)

hot_water_2 = HotWaterStorage(
    id="hot_water_2",
    bus="bus_LV_2",
    volume=150,
    set_temperature=55,
    input={"input_path": "data/simbench/rural1/hw_2.csv", "col": "hw_2"},
    tags={"household": "HH2"},
)
rel_hw2_a = Relation("heater2.power enabled from 18:00 to 22:00")
rel_hw2_b = Relation("heater2.min_on_duration = 1h")
heater2 = Converter(
    id="heater2",
    controllable=True,
    charges="hot_water_2",
    bus="bus_LV_2",
    conversion_efficiency=0.95,
    tags={"household": "HH2"},
    relations=[rel_hw2_a, rel_hw2_b],
)

# Household electrical loads (map to SimBench profiles)
load_1 = Load(
    id="load_1",
    bus="bus_LV_1",
    input={"input_path": "data/simbench/rural1/load_1.csv"},
    tags={"household": "HH1"},
)

load_2 = Load(
    id="load_2",
    bus="bus_LV_2",
    input={"input_path": "data/simbench/rural1/load_2.csv"},
    tags={"household": "HH2"},
)

load_3 = Load(
    id="load_3",
    bus="bus_LV_3",
    input={"input_path": "data/simbench/rural1/load_3.csv"},
    tags={"household": "HH3"},
)

load_4 = Load(
    id="load_4",
    bus="bus_LV_5",
    input={"input_path": "data/simbench/rural1/load_5.csv"},
    tags={"household": "HH4"},
)

load_5 = Load(
    id="load_5",
    bus="bus_LV_6",
    input={"input_path": "data/simbench/rural1/load_6.csv"},
    tags={"household": "HH5"},
)

# Small additional device: local small battery in HH3 (example)
battery_hh3 = Battery(
    id="battery_hh3",
    bus="bus_LV_3",
    capacity=3.5,
    max_charge_rate=1.5,
    max_discharge_rate=1.5,
    charge_efficiency=0.95,
    discharge_efficiency=0.95,
    storage_efficiency=0.995,
    tags={"household": "HH3"},
)

# Build the model (1 day example at 1h resolution; adjust as needed)
time_resolution = "1h"
model = Model(
    id="Energy_Community_simbench_rural1",
    time_start="2025-01-01 00:00",
    time_end="2025-01-02 00:00",
    resolution=time_resolution,
    entities=[
        # buses & slack
        bus_mv,
        bus_lv_head,
        bus_lv_1,
        bus_lv_2,
        bus_lv_3,
        bus_lv_4,
        bus_lv_5,
        bus_lv_6,
        slack,
        # lines
        line_hv_to_lv,
        line_1,
        line_2,
        line_3,
        line_4,
        line_5,
        line_6,
        # generators
        pv_1,
        pv_2,
        pv_3,
        wind_1,
        # storages
        battery_shared,
        battery_hh3,
        hot_water_1,
        hot_water_2,
        # converters/heaters
        heater1,
        heater2,
        # loads
        load_1,
        load_2,
        load_3,
        load_4,
        load_5,
    ],
)

# convert to pulp (your existing pipeline) and run the example optimize
number_of_time_steps = model.number_of_time_steps
problem = PulpConverter().convert_model(model)
optimize_energy_system(**problem)
