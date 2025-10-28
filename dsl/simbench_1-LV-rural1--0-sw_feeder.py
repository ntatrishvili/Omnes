import configparser
from os.path import join

import pandas as pd

from app.conversion.pandapower_converter import PandapowerConverter
from app.model.generator.pv import PV
from app.model.generator.wind_turbine import Wind
from app.model.grid_component.bus import Bus, BusType
from app.model.grid_component.line import Line
from app.model.load.load import Load
from app.model.model import Model
from app.operation.example_simulation import simulate_energy_system
from utils.logging_setup import get_logger, init_logging


def build_model_from_simbench():
    config = configparser.ConfigParser(
        allow_no_value=True, interpolation=configparser.ExtendedInterpolation()
    )
    config.read("..\\config.ini")
    root = config.get("simbench", "simbench_input")
    nodes = pd.read_csv(join(root, "Node.csv"), sep=";")
    slack_units = pd.read_csv(join(root, "ExternalNet.csv"), sep=";")
    lines = pd.read_csv(join(root, "Line.csv"), sep=";")
    line_types = pd.read_csv(join(root, "LineType.csv"), sep=";")
    switches = pd.read_csv(join(root, "Switch.csv"), sep=";")
    datetime_properties = {
        "datetime_format": "%d.%m.%Y %H:%M",
        "datetime_column": "time",
        "tz": "Europe/Berlin",
    }

    # Try loading RES (Renewable Energy Sources)
    try:
        res_units = pd.read_csv(join(root, "RES.csv"), sep=";")
    except FileNotFoundError:
        print("No RES.csv found – skipping renewable generation.")
        res_units = pd.DataFrame()

    try:
        load_units = pd.read_csv(join(root, "Load.csv"), sep=";")
    except FileNotFoundError:
        print("No RES.csv found – skipping renewable generation.")
        load_units = pd.DataFrame()

    # -----------------------------
    # STEP B: CONVERT TO OMNES OBJECTS
    # -----------------------------
    Bus.default_nominal_voltage = 400
    Bus.default_phase = "A"
    # Symmetric network
    Bus.default_phase_count = 1

    buses = []
    for _, row in nodes.iterrows():
        bus = Bus(
            id=row["id"],
            nominal_voltage=float(row.get("vNom", 0.4)) * 1000,
            type=BusType.PQ,
        )
        buses.append(bus)

    # Lines
    lines_omnes = []
    for _, row in lines.iterrows():
        lt = line_types[line_types["id"] == row["type"]]
        if not lt.empty:
            r_per_km = float(lt.iloc[0]["r"])
            x_per_km = float(lt.iloc[0]["x"])
        else:
            r_per_km, x_per_km = 0.1, 0.08

        line = Line(
            id=row["id"],
            from_bus=row["nodeA"],
            to_bus=row["nodeB"],
            line_length=float(row.get("d", 100)),
            resistance=r_per_km,
            reactance=x_per_km,
            max_current=float(row.get("loading_max", 100))
        )
        lines_omnes.append(line)

    # Lines
    switches_omnes = []
    for _, row in switches.iterrows():
        switch = Line(
            id=row["id"],
            from_bus=row["nodeA"],
            to_bus=row["nodeB"],
            line_length=0,
            resistance=0,
            reactance=0,
        )
        lines_omnes.append(switch)

    # Slacks
    slacks = []
    for _, row in slack_units.iterrows():
        slacks.append(
            Bus(
                id=row["id"],
                bus=row["node"],
                type=BusType.SLACK,
                voltage=row["voltLvl"],
            )
        )

    # -----------------------------
    # Parse RES: PVs and Wind Turbines
    # -----------------------------
    pvs, winds = [], []
    if not res_units.empty:
        for _, row in res_units.iterrows():
            tech = str(row.get("type", "")).lower()
            node = row["node"]
            p_peak_kw = 1000 * float(row.get("pRES", 0.0))  # SimBench uses MW

            if "pv" in tech or "solar" in tech:
                pvs.append(
                    PV(
                        id=row["id"],
                        bus=node,
                        peak_power=p_peak_kw / 1000,  # convert to kW for Omnes
                        input={
                            "input_path": join(root, "RESProfile.csv"),
                            "col": row["profile"],
                            **datetime_properties,
                        },
                        tags={"source": "simbench", "sR": row["sR"], "household": node},
                    )
                )

            elif "wind" in tech:
                winds.append(
                    Wind(
                        id=row["id"],
                        bus=node,
                        peak_power=p_peak_kw / 1000,
                        efficiency=0.95,
                        input={
                            "input_path": join(root, "RESProfile.csv"),
                            "col": row["profile"],
                            **datetime_properties,
                        },
                        tags={"source": "simbench", "sR": row["sR"], "household": node},
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
            if "lv" in row["id"].lower():
                loads.append(
                    Load(
                        id=f"load_{row['id']}",
                        bus=node,
                        p_cons={
                            "input_path": join(root, "LoadProfile.csv"),
                            "col": f'{row["profile"]}_pload',
                            **datetime_properties,
                        },
                        q_cons={
                            "input_path": join(root, "LoadProfile.csv"),
                            "col": f'{row["profile"]}_qload',
                            **datetime_properties,
                        },
                        tags={
                            "source": "symbench",
                            "p_kw": p_peak_kw,
                            "q_kw": q_peak_kw,
                            "sR": row["sR"],
                            "household": node,
                        },
                    )
                )

    # -----------------------------
    # Build model
    # -----------------------------
    model = Model(
        id="SimBench_Rural1_Community",
        time_start="2016-01-01 00:00",
        time_end="2017-01-01 00:00",
        resolution="1h",
        entities=buses + lines_omnes + slacks + pvs + winds + loads,
    )
    return model


if __name__ == "__main__":
    init_logging(
        level="DEBUG",
        log_dir="logs",
        log_file="app.log",
    )
    log = get_logger(__name__)
    log.info("Logging initialized")
    model = build_model_from_simbench()
    log.info("Model built successfully")
    net = PandapowerConverter().convert_model(model)
    log.info("Model converted successfully")

    log.info("Starting simulation")
    simulate_energy_system(net)
    log.info("Simulation completed")
