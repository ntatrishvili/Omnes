"""Read and validate the KFKI input workbooks."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

REQUIRED = {
    "network": {
        "buses": ["id", "nominal_voltage_V", "level"],
        "lines": [
            "id",
            "from_bus",
            "to_bus",
            "length_km",
            "r_ohm_km",
            "x_ohm_km",
            "c_nF_km",
            "max_i_kA",
        ],
        "transformers": ["id", "from_bus", "to_bus", "type", "sn_kVA"],
        "trafo_types": [
            "id",
            "sn_kVA",
            "vn_hv_kV",
            "vn_lv_kV",
            "vk_percent",
            "vkr_percent",
            "pfe_kW",
            "i0_percent",
        ],
        "ext_grid": ["id", "bus", "vm_pu"],
    },
    "assets": {
        "loads": ["id", "bus", "rated_kVA", "csv_column", "power_factor"],
        "heat_pumps": ["id", "bus", "rated_kVA", "csv_column", "power_factor"],
        "pv": ["id", "bus", "base_kWp", "csv_column"],
        "battery": [
            "id",
            "bus",
            "power_kW",
            "capacity_kWh",
            "eff_charge",
            "eff_discharge",
            "soc_min",
            "soc_max",
            "soc_init",
        ],
        "pv_sites": ["id", "bus", "share", "csv_column"],
        "ev_sites": ["id", "bus", "charger_kW", "power_factor"],
        "ev_chargers": ["id", "csv_column"],
    },
    "scenarios": {
        "grid": ["added_pv_kWp", "battery_kWh", "n_chargers"],
        "settings": ["key", "value"],
    },
}


# Optional thermal columns of the heat_pumps sheet; needed only for heat-coupled scenarios.
HEAT_COLUMNS = [
    "cop",
    "heat_csv_column",
    "peak_heat_kW",
    "tes_volume_l",
    "tes_set_temp_c",
    "tes_return_temp_c",
    "tes_loss_per_h",
    "tes_soc_init",
]


@dataclass
class CampusData:
    """All input tables, one DataFrame per sheet."""

    buses: pd.DataFrame
    lines: pd.DataFrame
    transformers: pd.DataFrame
    trafo_types: pd.DataFrame
    ext_grid: pd.DataFrame
    loads: pd.DataFrame
    heat_pumps: pd.DataFrame
    pv: pd.DataFrame
    battery: pd.DataFrame
    pv_sites: pd.DataFrame
    ev_sites: pd.DataFrame  # where chargers can stand, in round-robin order
    ev_chargers: pd.DataFrame  # one demand profile (CSV column) per charger, in order
    grid: pd.DataFrame
    settings: dict

    @property
    def csv_columns(self) -> list[str]:
        """All CSV columns the model can read (excluding the timestamp)."""
        cols = []
        for df in (
            self.loads,
            self.heat_pumps,
            self.pv,
            self.pv_sites,
            self.ev_chargers,
        ):
            cols += df["csv_column"].tolist()
        if self.has_heat_side:
            cols += self.heat_pumps["heat_csv_column"].tolist()
        return list(dict.fromkeys(cols))

    @property
    def has_heat_side(self) -> bool:
        """True when the heat_pumps sheet carries the thermal columns (HEAT_COLUMNS)."""
        return all(c in self.heat_pumps.columns for c in HEAT_COLUMNS)


# Presentation only: loads that are not drawn on their own in the figures, mapped to the
# load whose entry they are added to (the model and all simulation results are unchanged).
PLOT_MERGE = {"kovacs_laszlo": "b21A"}


def display_names(d: CampusData) -> tuple[dict[str, str], dict[str, str]]:
    """English names for plots: (load id -> name, bus id -> name).

    Names come from the ``label`` column of the loads sheet (Hungarian originals are
    kept in ``label_hu``); the battery bus is named explicitly. Merged loads
    (``PLOT_MERGE``) carry the name of the load they are added to.
    """
    load_names = dict(zip(d.loads["id"], d.loads["label"]))
    bus_names = dict(zip(d.loads["bus"], d.loads["label"]))
    bus_of = dict(zip(d.loads["id"], d.loads["bus"]))
    for src, dst in PLOT_MERGE.items():
        if src in load_names and dst in load_names:
            load_names[src] = load_names[dst]
            bus_names[bus_of[src]] = load_names[dst]
    for bus in d.battery["bus"]:
        bus_names[bus] = "NaS battery"
    return load_names, bus_names


def merged_bus(d: CampusData, bus: str) -> str:
    """The bus a bus is drawn as (itself unless its load is merged into another one)."""
    owner = dict(zip(d.loads["bus"], d.loads["id"]))
    bus_of = dict(zip(d.loads["id"], d.loads["bus"]))
    src = owner.get(bus)
    return bus_of[PLOT_MERGE[src]] if src in PLOT_MERGE else bus


def bus_to_trafo(d: CampusData) -> dict[str, str]:
    """Map every LV bus to the id of the transformer that feeds it (radial network)."""
    trafo_of_lv = dict(zip(d.transformers["to_bus"], d.transformers["id"]))
    parent = dict(zip(d.lines["to_bus"], d.lines["from_bus"]))
    out = {}
    for bus in d.buses["id"]:
        node = bus
        while node in parent:
            node = parent[node]
        if node in trafo_of_lv:
            out[bus] = trafo_of_lv[node]
    return out


def _read(path: Path, spec: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Input workbook not found: {path}")
    sheets = pd.read_excel(path, sheet_name=list(spec))
    for sheet, cols in spec.items():
        missing = [c for c in cols if c not in sheets[sheet].columns]
        if missing:
            raise ValueError(f"{path.name}:{sheet} is missing columns {missing}")
        if sheets[sheet]["id" if "id" in cols else cols[0]].duplicated().any():
            raise ValueError(f"{path.name}:{sheet} has duplicate ids")
    return sheets


def load_campus_data(
    data_dir: Path = DATA_DIR, tan_phi_path: Path | None = None
) -> CampusData:
    """Read network.xlsx, assets.xlsx and scenarios.xlsx and validate them.

    ``tan_phi_path`` (optional, written by ``inputs.build_inputs``) adds a signed
    Q/P ``tan_phi`` column to the loads, measured from the meters' kVArh/kWh.
    """
    data_dir = Path(data_dir)
    net = _read(data_dir / "network.xlsx", REQUIRED["network"])
    assets = _read(data_dir / "assets.xlsx", REQUIRED["assets"])
    scen = _read(data_dir / "scenarios.xlsx", REQUIRED["scenarios"])
    settings = dict(zip(scen["settings"]["key"], scen["settings"]["value"]))
    data = CampusData(**net, **assets, grid=scen["grid"], settings=settings)
    _validate(data)
    if tan_phi_path is not None and Path(tan_phi_path).exists():
        tan = pd.read_csv(tan_phi_path).set_index("load_id")["tan_phi"]
        data.loads["tan_phi"] = data.loads["id"].map(tan)
    return data


def _validate(d: CampusData) -> None:
    """Referential integrity and basic sanity checks."""
    bus_ids = set(d.buses["id"])
    refs = {
        "lines.from_bus": d.lines["from_bus"],
        "lines.to_bus": d.lines["to_bus"],
        "transformers.from_bus": d.transformers["from_bus"],
        "transformers.to_bus": d.transformers["to_bus"],
        "ext_grid.bus": d.ext_grid["bus"],
        "loads.bus": d.loads["bus"],
        "heat_pumps.bus": d.heat_pumps["bus"],
        "pv.bus": d.pv["bus"],
        "battery.bus": d.battery["bus"],
        "pv_sites.bus": d.pv_sites["bus"],
        "ev_sites.bus": d.ev_sites["bus"],
    }
    for name, series in refs.items():
        unknown = set(series) - bus_ids
        if unknown:
            raise ValueError(f"{name} references unknown buses: {sorted(unknown)}")
    unknown_types = set(d.transformers["type"]) - set(d.trafo_types["id"])
    if unknown_types:
        raise ValueError(
            f"transformers reference unknown types: {sorted(unknown_types)}"
        )
    total = d.pv_sites["share"].sum()
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"pv_sites.share must sum to 1 (got {total})")
    max_chargers = int(d.grid["n_chargers"].max())
    if max_chargers > len(d.ev_chargers):
        raise ValueError(
            f"scenario grid asks for {max_chargers} chargers but only "
            f"{len(d.ev_chargers)} charger profiles are defined (ev_chargers sheet)"
        )
    if (d.lines[["length_km", "r_ohm_km", "x_ohm_km"]] <= 0).any().any():
        # the Pandapower converter turns zero-impedance lines into switches
        raise ValueError("lines must have strictly positive length, r and x")
