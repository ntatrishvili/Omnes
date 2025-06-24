from pulp import LpProblem, LpMinimize

from app.model.battery import Battery
from app.model.load import Load
from app.model.pv import PV
from app.model.relation import Relation
from app.model.slack import Slack
from app.model.timeseries_object_factory import TimeseriesFactory

ts_factory = TimeseriesFactory()

slack = Slack(id="slack", ts_factory=ts_factory, bus="bus_MV")

# Instantiate PVs
pv1 = PV(id="pv1", ts_factory=ts_factory, peak_power=4, power="input/pv1.csv")
pv2 = PV(id="pv2", ts_factory=ts_factory, peak_power=3, power="input/pv2.csv")

# Instantiate Loads
load1 = Load(id="load1", ts_factory=ts_factory, power="input/load1.csv")
load2 = Load(id="load2", ts_factory=ts_factory, power="input/load2.csv")

# Instantiate Battery
battery1 = Battery(
    id="battery1",
    ts_factory=ts_factory,
    capacity=5,
    max_power=2,  # Used as both max_charge_rate and max_discharge_rate
)

# Example pulp problem
prob = LpProblem("EnergyCommunityModel", LpMinimize)

slack_in = slack["p_slack_in"].to_pulp("slack_in", "1h", 24)
slack_out = slack["p_slack_out"].to_pulp("slack_out", "1h", 24)

# Assume you have pulp-compatible variables or values
battery1_max_discharge = battery1.quantities["max_power"].to_pulp(
    "battery1_max_discharge", "1h", 24
)
pv1_peak_power = pv1.quantities["p_pv"].to_pulp("pv1_power", "1h", 24)

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

relation = Relation("battery1.max_discharge_rate < 2 * pv1.peak_power")
prob += relation.to_pulp(context, time_set=24)
