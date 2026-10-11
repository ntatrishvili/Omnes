"""Build the KFKI campus as an OMNES Model from the input workbooks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.case_studies.kfki_campus.components import EVCharger, HeatDemand, HeatPump
from app.case_studies.kfki_campus.excel_io import CampusData, load_campus_data
from app.infra.relation import Relation
from app.model.device import Vector
from app.model.generator.pv import PV
from app.model.generic_entity import GenericEntity
from app.model.grid_component.bus import Bus, BusType
from app.model.grid_component.line import Line
from app.model.grid_component.transformer import Transformer
from app.model.load.load import Load
from app.model.model import Model
from app.model.slack import Slack
from app.model.storage.battery import Battery
from app.model.storage.hot_water_storage import HotWaterStorage
from app.model.transducer.transducer import Transducer


@dataclass(frozen=True)
class Scenario:
    """One point of the scenario raster."""

    added_pv_kwp: float = 0.0
    battery_kwh: float = 1450.0
    n_chargers: int = 3  # today's number of chargers
    # Electricity-heat coupling: the heat pumps become converters (Transducer) that charge a
    # hot-water tank serving a heat demand, instead of plain electric loads.
    heat_coupling: bool = False
    tes_scale: float = 1.0  # tank volume relative to the heat_pumps sheet

    @property
    def id(self) -> str:
        sid = f"pv{self.added_pv_kwp:g}_bat{self.battery_kwh:g}_ev{self.n_chargers}"
        return sid + (f"_hp{self.tes_scale:g}" if self.heat_coupling else "")


def allocate(total: int, shares: list[float]) -> list[int]:
    """Split an integer total by shares (largest-remainder, sums exactly to total)."""
    raw = [total * s for s in shares]
    out = [int(r) for r in raw]
    order = sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)
    for i in order[: total - sum(out)]:
        out[i] += 1
    return out


def _ts(csv: Path, col: str, scale: float = 1.0) -> dict:
    return {
        "input_path": str(csv),
        "col": col,
        "scale": scale,
        "datetime_column": "timestamp",
    }


def build_kfki_model(
    scenario: Scenario,
    ts_csv: Path,
    data: Optional[CampusData] = None,
    time_start: str = "2025-01-01 00:00",
    time_end: str = "2026-01-01 00:00",
    resolution: Optional[str] = None,
) -> Model:
    """Construct the campus model for one scenario.

    Time series come from the single wide CSV ``ts_csv`` (see README for the
    column contract); ``scenario`` scales PV, battery and EV charger count.
    """
    d = data or load_campus_data()
    ts_csv = Path(ts_csv)
    resolution = resolution or d.settings.get("resolution", "1h")

    trafo_types = [
        GenericEntity(
            id=r.id,
            sR=r.sn_kVA / 1000.0,
            vmHV=r.vn_hv_kV,
            vmLV=r.vn_lv_kV,
            va0=r.shift_degree if "shift_degree" in d.trafo_types else 150,
            vmImp=r.vk_percent,
            vkr=r.vkr_percent,
            pFe=r.pfe_kW,
            iNoLoad=r.i0_percent,
            dVm=0.0,
            dVa=0.0,
            tapNeutr=0,
            tapMin=0,
            tapMax=0,
        )
        for r in d.trafo_types.itertuples()
    ]
    slack_bus = set(d.ext_grid["bus"])
    buses = [
        Bus(
            id=r.id,
            nominal_voltage=float(r.nominal_voltage_V),
            type=BusType.SLACK if r.id in slack_bus else BusType.PQ,
            coordinates={
                "x": float(getattr(r, "x", 0.0)),
                "y": float(getattr(r, "y", 0.0)),
            },
        )
        for r in d.buses.itertuples()
    ]
    trafos = [
        Transformer(
            id=r.id,
            from_bus=r.from_bus,
            to_bus=r.to_bus,
            type=r.type,
            nominal_power=float(r.sn_kVA),
            nominal_voltage_hv_side=_vn(d, r.from_bus),
            nominal_voltage_lv_side=_vn(d, r.to_bus),
        )
        for r in d.transformers.itertuples()
    ]
    lines = [
        Line(
            id=r.id,
            from_bus=r.from_bus,
            to_bus=r.to_bus,
            line_length=float(r.length_km),
            resistance=float(r.r_ohm_km),
            reactance=float(r.x_ohm_km),
            capacitance=float(r.c_nF_km),
            max_current=float(r.max_i_kA),
        )
        for r in d.lines.itertuples()
    ]
    slacks = [Slack(id=r.id, bus=r.bus) for r in d.ext_grid.itertuples()]

    loads = [
        Load(
            id=r.id,
            bus=r.bus,
            nominal_power=float(r.rated_kVA),
            p_cons=_ts(ts_csv, r.csv_column),
            tags={"power_factor": r.power_factor},
        )
        for r in d.loads.itertuples()
    ]
    heat_pumps = [
        HeatPump(
            id=r.id,
            bus=r.bus,
            nominal_power=float(r.rated_kVA),
            p_cons=_ts(ts_csv, r.csv_column),
            tags={"power_factor": r.power_factor},
        )
        for r in d.heat_pumps.itertuples()
    ]
    # Heat side. The heat pump Load above is the electrical interface (the only part the
    # power flow sees); the Transducer converts its input power into heat that charges the
    # hot-water tank, which together with the heat pump serves the building's heat demand.
    # Heat devices sit on the building's electric bus (there is no heat network here).
    heat_devices = []
    if scenario.heat_coupling:
        if not d.has_heat_side:
            raise ValueError("heat_pumps sheet has no thermal columns; rerun make_inputs")
        for hp, r in zip(heat_pumps, d.heat_pumps.itertuples()):
            tes_id = f"{r.id}_tes"
            heat_devices += [
                HotWaterStorage(
                    id=tes_id,
                    bus=r.bus,
                    volume=float(r.tes_volume_l) * scenario.tes_scale,
                    set_temperature=float(r.tes_set_temp_c),
                    water_in_temperature=float(r.tes_return_temp_c),
                    storage_efficiency=1.0 - float(r.tes_loss_per_h),
                    state_of_charge=float(r.tes_soc_init),
                ),
                Transducer(
                    id=f"{r.id}_conv",
                    input_device=hp,
                    charges=tes_id,
                    conversion_efficiency=float(r.cop),
                    tags={
                        "vector": Vector.ELECTRICITY,
                        "contributes_to": "electric_power_balance",
                    },
                    # the converter turns these into constraints (PulpConverter)
                    relations=[
                        Relation("$.p_out == $.p_in * $.conversion_efficiency"),
                        Relation(f"$.p_in <= {float(r.rated_kVA) * float(r.power_factor):.3f}"),
                    ],
                ),
                HeatDemand(
                    id=f"{r.id}_heat",
                    bus=r.bus,
                    p_cons=_ts(ts_csv, r.heat_csv_column),
                ),
            ]

    pvs = [
        PV(
            id=r.id,
            bus=r.bus,
            peak_power=float(r.base_kWp),
            p_out=_ts(ts_csv, r.csv_column, float(r.base_kWp)),
        )
        for r in d.pv.itertuples()
    ]
    if scenario.added_pv_kwp > 0:
        for r in d.pv_sites.itertuples():
            kwp = scenario.added_pv_kwp * float(r.share)
            pvs.append(
                PV(
                    id=r.id,
                    bus=r.bus,
                    peak_power=kwp,
                    p_out=_ts(ts_csv, r.csv_column, kwp),
                )
            )

    if scenario.n_chargers > len(d.ev_chargers):
        raise ValueError(
            f"{scenario.n_chargers} chargers requested, only {len(d.ev_chargers)} "
            "charger profiles available"
        )
    # Charger k uses profile k and stands at site k mod (number of sites): the sweep is
    # nested, each additional charger adds one at the next site.
    evs = []
    for k, c in enumerate(d.ev_chargers.head(scenario.n_chargers).itertuples()):
        site = d.ev_sites.iloc[k % len(d.ev_sites)]
        evs.append(
            EVCharger(
                id=c.id,
                bus=site.bus,
                n_chargers=1,
                charger_kw=float(site.charger_kW),
                p_cons=_ts(ts_csv, c.csv_column),
                tags={"power_factor": site.power_factor},
            )
        )

    batteries = []
    if scenario.battery_kwh > 0:
        b = d.battery.iloc[0]
        c_rate = float(b.power_kW) / float(b.capacity_kWh)
        power = scenario.battery_kwh * c_rate
        batteries.append(
            Battery(
                id=b.id,
                bus=b.bus,
                capacity=float(scenario.battery_kwh),
                max_charge_rate=power,
                max_discharge_rate=power,
                charge_efficiency=float(b.eff_charge),
                discharge_efficiency=float(b.eff_discharge),
                state_of_charge=float(b.soc_init),
            )
        )

    return Model(
        id=f"KFKI_{scenario.id}",
        time_start=time_start,
        time_end=time_end,
        resolution=resolution,
        time_kwargs={"inclusive": "left"},
        entities=trafo_types
        + buses
        + trafos
        + lines
        + slacks
        + loads
        + heat_pumps
        + heat_devices
        + pvs
        + evs
        + batteries,
    )


def _vn(d: CampusData, bus_id: str) -> float:
    """Nominal bus voltage in kV."""
    return float(d.buses.set_index("id").loc[bus_id, "nominal_voltage_V"]) / 1000.0
