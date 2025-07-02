from pulp import LpProblem, LpMinimize

from app.infra.relation import Relation
from app.infra.timeseries_object_factory import DefaultTimeseriesFactory
from app.model.generator.pv import PV
from app.model.generator.wind_turbine import Wind
from app.model.grid_component.bus import Bus, BusType
from app.model.grid_component.line import Line
from app.model.load import Load
from app.model.slack import Slack
from app.model.storage.battery import Battery

ts_factory = DefaultTimeseriesFactory()

nominal_voltage = 230
Bus.default_nominal_voltage = nominal_voltage

bus_mv = Bus(id="bus_MV", type=BusType.SLACK, phase_count=3)
slack = Slack(id="slack", ts_factory=ts_factory, bus="bus_MV")

bus_lv1 = Bus(id="bus_LV1",)
bus_lv2 = Bus(id="bus_LV2", phase="C")
bus_lv3 = Bus(id="bus_LV3", phase_count=3)

line_length = 0.1
resistance = 0.05
reactance = 0.05
Line.default_line_length = line_length
Line.default_resistance = resistance
Line.default_reactance = reactance

line1 = Line(id="line1", from_bus="bus_MV", to_bus="bus_LV1")
line2 = Line(id="line2", from_bus="bus_LV1", to_bus="bus_LV2")
line3 = Line(id="line3", from_bus="bus_LV1", to_bus="bus_LV3")

efficiency = 0.9
# Instantiate PVs
# If 'col' not specified, default 'col' will be the ID of the element
pv1 = PV(id="pv1", ts_factory=ts_factory, peak_power=4, efficiency=efficiency, input_path="data/input.csv", bus="bus_LV1", housedold="HH1")
pv2 = PV(id="pv2", ts_factory=ts_factory, peak_power=3, efficiency=efficiency, input_path="data/input2.csv", bus="bus_LV2", household="HH2")
wind1 = Wind(id="wind1", ts_factory=ts_factory, peak_power=5, efficiency=0.95, input_path="data/input.csv", col="wind", bus="bus_LV3")

# Instantiate Loads
load1 = Load(id="load1", ts_factory=ts_factory, input_path="data/input.csv")
load2 = Load(id="load2", ts_factory=ts_factory, input_path="data/input2.csv")

# Instantiate Battery
battery1 = Battery(
    id="battery1",
    capacity=5,
    max_power=2,  # Used as both max_charge_rate and max_discharge_rate
)


hot_water_storage = Hot()

# Example pulp problem
prob = LpProblem("EnergyCommunityModel", LpMinimize)

time_resolution = "1h"
number_of_time_steps = 24

slack_in = slack["p_slack_in"].to_pulp("slack_in", time_resolution, number_of_time_steps)
slack_out = slack["p_slack_out"].to_pulp("slack_out", time_resolution, number_of_time_steps)

# Assume you have pulp-compatible variables or values
battery1_max_discharge = battery1.quantities["max_power"].to_pulp(
    "battery1_max_discharge", time_resolution, number_of_time_steps
)
pv1_peak_power = pv1.quantities["p_pv"].to_pulp("pv1_power", time_resolution, number_of_time_steps)

# Example constraint from DSL:
# battery1.max_discharge_rate < 2 * pv1.peak_power
prob += battery1_max_discharge <= 2 * pv1_peak_power, "BatteryDischargeLimit"

# Example conditional (must be translated as logic):
# if battery1.capacity < 6 then battery1.max_discharge_rate < 3
if battery1.quantities["capacity"].value < 6:
    prob += battery1_max_discharge <= 3, "ConditionalBatteryLimit"

context = {
    "battery1.max_discharge_rate": battery1_max_discharge,
    "pv1.peak_power": pv1_peak_power,
}

relation1 = Relation("battery1.max_discharge_rate < 2 * pv1.peak_power")
relation2 = Relation("if battery1.capacity < 6 then battery1.max_discharge_rate < 3")

prob += relation1.to_pulp(context, time_set=number_of_time_steps), "R1_battery_limit"
prob += relation2.to_pulp(context, time_set=number_of_time_steps), "R2_conditional_discharge"

# === (Optional) Example manual constraints ===
if battery1.quantities["capacity"].value < 6:
    prob += context["battery1.max_discharge_rate"] <= 3, "Manual_Conditional_Limit"

relation = Relation("battery1.max_discharge_rate < 2 * pv1.peak_power")
prob += relation.to_pulp(context, time_set=number_of_time_steps)
