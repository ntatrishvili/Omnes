from os.path import join

import pandas as pd

from app.conversion.pandapower_converter import PandapowerConverter
from app.conversion.pulp_converter import PulpConverter
from app.model.generator.pv import PV
from app.model.generator.wind_turbine import Wind
from app.model.generic_entity import GenericEntity
from app.model.grid_component.bus import Bus, BusType
from app.model.grid_component.line import Line
from app.model.grid_component.transformer import Transformer
from app.model.load.load import Load
from app.model.model import Model
from app.model.slack import Slack
from app.model.storage.battery import Battery
from app.operation.example_optimization import optimize_energy_system
from app.operation.example_simulation import simulate_energy_system
from app.infra.configuration import Config
from app.infra.logging_setup import get_logger, init_logging
from app.infra.visualize import elegant_draw_network


def build_model_from_simbench(**kwargs):
    datetime_properties = {
        "datetime_format": "%d.%m.%Y %H:%M",
        "datetime_column": "time",
        "tz": "Europe/Berlin",
    }

    config = Config()
    root = config.get("simbench", "simbench_input")
    nodes = read_data_file(root, "Node.csv")
    slack_units = read_data_file(root, "ExternalNet.csv")
    lines = read_data_file(root, "Line.csv")
    line_types = read_data_file(root, "LineType.csv")
    switches = read_data_file(root, "Switch.csv")
    # Try loading RES (Renewable Energy Sources)
    res_units = read_data_file(root, "RES.csv")
    load_units = read_data_file(root, "Load.csv")
    transformer_units = read_data_file(root, "Transformer.csv")
    transformer_types = read_data_file(root, "TransformerType.csv")
    coords_df = read_data_file(root, "Coordinates.csv").set_index("id")

    # -----------------------------
    # STEP B: CONVERT TO OMNES OBJECTS
    # -----------------------------
    # Slacks
    slacks = []
    for _, row in slack_units.iterrows():
        slacks.append(
            Slack(
                id=row["id"],
                bus=row["node"],
                volLvl=row["voltLvl"],
            )
        )

    slack_nodes = slack_units["node"].tolist()

    # Buses
    Bus.default_nominal_voltage = 400
    Bus.default_phase = "A"
    # Symmetric network
    Bus.default_phase_count = 1

    buses = []
    for _, row in nodes.iterrows():
        if row["type"] != "busbar":
            continue
        buses.append(
            Bus(
                id=row["id"],
                nominal_voltage=float(row["vmR"]) * 1000,
                type=BusType.PQ if row["id"] not in slack_nodes else BusType.SLACK,
                coordinates={
                    "x": coords_df.loc[row["coordID"], "x"],
                    "y": coords_df.loc[row["coordID"], "y"],
                },
            )
        )

    # Lines
    lines_omnes = []
    for _, row in lines.iterrows():
        lt = line_types[line_types["id"] == row["type"]]
        if not lt.empty:
            r_per_km = float(lt.iloc[0]["r"])
            x_per_km = float(lt.iloc[0]["x"])
            b_per_km = float(lt.iloc[0]["b"])
        else:
            r_per_km, x_per_km, b_per_km = 0.1, 0.08, 60

        line = Line(
            id=row["id"],
            from_bus=row["nodeA"],
            to_bus=row["nodeB"],
            line_length=float(row["length"]),
            resistance=r_per_km,
            reactance=x_per_km,
            capacitance=b_per_km,
            max_current=float(row["loadingMax"]),
        )
        lines_omnes.append(line)

    # Lines
    for _, row in switches.iterrows():
        switch = Line(
            id=row["id"],
            from_bus=row["nodeA"],
            to_bus=row["nodeB"],
            line_length=0,
            resistance=0,
            reactance=0,
            type=row["type"],
        )
        lines_omnes.append(switch)

    # -----------------------------
    # Parse RES: PVs and Wind Turbines
    # -----------------------------
    pvs, winds = [], []
    if not res_units.empty:
        for _, row in res_units.iterrows():
            tech = str(row.get("type", "")).lower()
            node = row["node"]
            p_peak_kw = float(row.get("pRES", 0.0)) * 1000  # SimBench uses MW

            if "pv" in tech or "solar" in tech:
                pvs.append(
                    PV(
                        id=row["id"],
                        bus=node,
                        peak_power=p_peak_kw,  # convert to kW for Omnes
                        p_out={
                            "input_path": join(root, "RESProfile.csv"),
                            "col": row["profile"],
                            "scale": p_peak_kw * kwargs.get("pv_scale", 1.0),
                            **datetime_properties,
                        },
                        tags={
                            "source": "simbench",
                            "sR": row["sR"],
                            "household": node,
                            "profile": row["profile"],
                        },
                    )
                )

            elif "wind" in tech:
                winds.append(
                    Wind(
                        id=row["id"],
                        bus=node,
                        peak_power=p_peak_kw,
                        efficiency=0.95,
                        p_out={
                            "input_path": join(root, "RESProfile.csv"),
                            "col": row["profile"],
                            "scale": p_peak_kw,
                            **datetime_properties,
                        },
                        tags={
                            "source": "simbench",
                            "sR": row["sR"],
                            "household": node,
                            "profile": row["profile"],
                        },
                    )
                )

    # -----------------------------
    # Static Loads
    # -----------------------------
    loads = []
    if not load_units.empty:
        for _, row in load_units.iterrows():
            node = row["node"]
            p_peak_kw = 1000 * float(row.get("pLoad", 0.0))  # SimBench uses MW
            q_peak_kw = 1000 * float(row.get("qLoad", 0.0))  # SimBench uses MW
            loads.append(
                Load(
                    id=f"{row['id']}",
                    bus=node,
                    nominal_power=row["sR"],
                    p_cons={
                        "input_path": join(root, "LoadProfile.csv"),
                        "col": f'{row["profile"]}_pload',
                        "scale": p_peak_kw,
                        **datetime_properties,
                    },
                    q_cons={
                        "input_path": join(root, "LoadProfile.csv"),
                        "col": f'{row["profile"]}_qload',
                        "scale": q_peak_kw,
                        **datetime_properties,
                    },
                    tags={
                        "source": "symbench",
                        "household": node,
                        "profile": row["profile"],
                    },
                )
            )

    # -----------------------------
    # Transformers
    # -----------------------------
    transformers = []
    if not transformer_units.empty:
        for _, row in transformer_units.iterrows():
            hv_bus = row["nodeHV"]
            lv_bus = row["nodeLV"]
            type = row["type"]
            transformers.append(
                Transformer(
                    id=f"tr_{row['id']}", from_bus=hv_bus, to_bus=lv_bus, type=type
                )
            )

    # -----------------------------
    # Transformers
    # -----------------------------
    transformer_type_entities = []
    if not transformer_types.empty:
        for _, row in transformer_types.iterrows():
            transformer_type_entities.append(GenericEntity(**row.to_dict()))

    # -----------------------------
    # The battery
    # -----------------------------
    battery = Battery(
        "battery",
        bus=buses[0].id,
        capacity=kwargs.get("bess_size", 500.0),  # in kWh
        max_charge_rate=kwargs.get("charge_rate", 10),
        max_discharge_rate=kwargs.get("charge_rate", 10),
    )
    # -----------------------------
    # Build model
    model = Model(
        id="SimBench_Rural1_Community",
        time_start="2016-01-01 00:00",
        time_end="2017-01-01 00:00",
        resolution="1h",
        time_kwargs={"inclusive": "left"},
        entities=buses
        + [battery]
        + transformer_type_entities
        + transformers
        + lines_omnes
        + slacks
        + pvs
        + winds
        + loads,
    )
    # -----------------------------
    return model


def read_data_file(root, filename, sep=";"):
    try:
        units = pd.read_csv(join(root, filename), sep=sep)
    except FileNotFoundError:
        get_logger(__name__).warning(f"No {filename} found – skipping...")
        units = pd.DataFrame()
    return units


if __name__ == "__main__":
    init_logging(
        level="INFO",
        log_dir="logs",
        log_file="app.log",
    )
    log = get_logger(__name__)

    pv_scale = 1.5  # scale PV sizes
    bess_size = 500.0  # in kWh
    log.info("Logging initialized")
    model = build_model_from_simbench(pv_scale=pv_scale, bess_size=bess_size)
    log.info("Model built successfully")

    problem = PulpConverter().convert_model(
        model, skip_entities=(Bus, Line, Transformer, GenericEntity)
    )
    log.info("Model converted to optimization problem successfully")
    log.info("Starting optimization")
    problem = optimize_energy_system(**problem)
    log.info("Optimization completed")

    # TODO: turn these into 'convert-back' functionalities inside the converters
    model.set({"battery.p_in": problem["battery.p_in"]})
    model.set({"battery.p_out": problem["battery.p_out"]})

    net = PandapowerConverter().convert_model(model)
    elegant_draw_network(net, output_path=Config().get("path", "output"))
    log.info("Model converted to pandapower net successfully")

    for battery_bus, scenario in zip(
        [-1, 4, 1, 13, 5, 3],
        [
            "No battery",
            "Feeder",
            "Branch 1 end",
            "Branch 2 end",
            "Branch 3 end",
            "Branch 4 end",
        ],
    ):
        if battery_bus == -1:
            net.sgen.loc[net.sgen.name.str.contains("battery"), "in_service"] = False
            print(net.sgen)
        else:
            net.sgen.loc[net.sgen.name.str.contains("battery"), "in_service"] = True
            net.sgen.loc[net.sgen.name.str.contains("battery"), "bus"] = net.bus.loc[
                net.bus.name == f"LV1.101 Bus {battery_bus}"
            ].index[0]
            print(net.sgen)
        log.info(f"Starting simulation for scenario {scenario}")
        simulate_energy_system(net, scenario=scenario)
        log.info("Simulation completed")
