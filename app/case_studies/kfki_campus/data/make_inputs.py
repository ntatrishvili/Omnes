"""Generate the KFKI input workbooks (network.xlsx, assets.xlsx, scenarios.xlsx).

Source tags used in the ``source`` columns:
  pdf-fig9   topology read from the PowerFactory single-line diagram (Fig. 9)
  pdf-fig10  load-flow feeder current in kA at 0.4 kV (Fig. 10), converted to kVA
  pdf-text   statement in the report text
  note       colleagues' comments on earlier modelling assumptions
  assumed    NOT in the documents - placeholder to be overwritten with real data

Run:  python -m app.case_studies.kfki_campus.data.make_inputs
"""

from math import sqrt
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent
KA_TO_KVA = sqrt(3) * 0.4 * 1000  # balanced 3-phase at 0.4 kV

# LV feeder cable: 4x240 mm2 Al (NAYY-type). Not a pandapower std type; r from the Al
# resistivity at 20 degC (as the std types), x as the std NAYY types, c and the in-ground
# rating are typical catalogue values - assumed, no data in the PDF
CABLE = dict(r_ohm_km=0.125, x_ohm_km=0.08, c_nF_km=300.0, max_i_kA=0.34)
CABLE_NAME = "NAYY 4x240"
DEFAULT_LENGTH_KM = 0.1  # assumed
# 10 kV cables (modern XLPE equivalent of the paper cables on the drawing), values of the
# pandapower std types "NA2XS2Y 1x240 / 1x150 RM/25 6/10 kV"
MV_CABLE_240 = dict(r_ohm_km=0.122, x_ohm_km=0.105, c_nF_km=456.0, max_i_kA=0.416)
MV_CABLE_150 = dict(r_ohm_km=0.206, x_ohm_km=0.110, c_nF_km=360.0, max_i_kA=0.315)
# Virányos infeed, 3x240 mm2 Al, 10 km (drawing "Egyvonalas kapcsolasi rajz, KFKI 10 kV").
# The Budaközép infeed (3x240 mm2 Al, ~9.1 km) is the open reserve and is not modelled.
VIRANYOS_LENGTH_KM = 10.0
# 10 kV cell of the receiving station feeding each transformer / the reactors, 3x150 mm2 Al,
# length read from the drawing (km). The cells carry the labels III/2 TRF (cell 7),
# IV/2 TRF (8), VI/1 TRF (9), reactor (4); the 20 m cell 10 ("Kémiai kút / felül. K.") is
# mapped to T by elimination - ASSUMED. Cell 7 length is faint on the photo (~530 m).
MV_FEED_KM = {"T": 0.020, "3T": 0.530, "4T": 0.710, "6T": 0.550, "10T": 0.320}
PF = 0.99  # Q/P ~ 10-15 % with PFC installed (pdf-text p.15)

# (transformer, bus_suffix, [(feeder_id, label, building, current_kA, significant)])
FEEDERS = {
    "T": [
        ("festo_muhely", "Festő műhely", "", 0.059, False),
        ("kovacs_laszlo", "Kovács László (bérlemény)", "", 0.059, False),
        ("sportpalya", "sportpálya", "", 0.059, False),
        ("b21A", "21/A ép.", "21/A", 0.074, True),
        ("b15", "15. ép.", "15", 0.059, False),
        ("b17", "17. ép.", "17", 0.119, True),
        ("b22", "22. ép.", "22", 0.059, False),
        ("fahaz_1_3", "Faház 1-3", "", 0.059, False),
    ],
    "3T": [
        ("egyeb_fogyasztok", "egyéb fogyasztók", "", 0.059, False),
        ("b31", "31. ép.", "31", 0.044, False),
        ("b2", "2. ép.", "2", 0.044, False),
        ("izotop_tarolo", "Izotóp tároló", "", 0.152, True),
        ("b14", "14. ép.", "14", 0.075, True),
        ("asztalos_muhely", "asztalos műhely", "", 0.044, False),
        ("villanyszerelo_muhely", "Villanyszerelő műhely", "", 0.044, False),
        ("kazanok", "kazánok", "", 0.044, False),
        ("daraboloi_vasraktar", "daraboló vasraktár", "", 0.044, False),
        ("b3", "3. ép.", "3", 0.075, True),
    ],
    "4T": [
        ("b4", "4. ép.", "4", 0.034, False),
        ("b19", "19. ép.", "19", 0.074, False),
        ("b26", "26. ép.", "26", 0.074, True),
        ("b23_5", "23/5 ép.", "23/5", 0.034, False),
    ],
    "6T": [
        ("b7", "7. ép.", "7", 0.090, False),
        ("b25", "25. ép.", "25", 0.090, False),
        ("b23", "23. ép.", "23", 0.090, False),
        ("b29", "29. ép.", "29", 0.089, True),
        ("mki_tiszta_labor", "MKI tiszta labor", "", 0.152, False),
        ("b6", "6. ép.", "6", 0.088, False),
        ("b1", "1. ép.", "1", 0.088, False),
    ],
    "10T": [
        ("reaktor_1", "Reaktor 1", "", 0.369, False),
        ("reaktor_2", "Reaktor 2", "", 0.391, False),
        ("tuziviz_1", "Tűzivíz 1 (label unclear)", "", 0.060, False),
        ("tuziviz_2", "Tűzivíz 2 (label unclear)", "", 0.060, False),
    ],
}
# English names of the consumers that are not numbered buildings (used on all plots);
# the Hungarian originals stay in the ``label_hu`` column of assets.xlsx
EN_NAMES = {
    "festo_muhely": "Paint workshop",
    "kovacs_laszlo": "Kovács László (tenant)",
    "sportpalya": "Sports ground",
    "fahaz_1_3": "Cabins 1-3",
    "egyeb_fogyasztok": "Other consumers",
    "izotop_tarolo": "Isotope store",
    "asztalos_muhely": "Carpenter's workshop",
    "villanyszerelo_muhely": "Electrician's workshop",
    "kazanok": "Boilers",
    "daraboloi_vasraktar": "Iron store (cutting shop)",
    "mki_tiszta_labor": "MKI clean lab",
    "reaktor_1": "Reactor 1",
    "reaktor_2": "Reactor 2",
    "tuziviz_1": "Fire water 1 (label unclear)",
    "tuziviz_2": "Fire water 2 (label unclear)",
}


def english_label(feeder_id: str, building: str) -> str:
    """'Bldg 17' for numbered buildings, the English name for the others.

    The diagram node '21. ep.' lumps buildings 21, 21/A and 21/B (one feeder, three
    meters), so it is simply 'Bldg 21'.
    """
    if feeder_id == "b21A":
        return "Bldg 21"
    return f"Bldg {building}" if building else EN_NAMES[feeder_id]


# Feeder (by feeder_id) -> trafo, for heat pumps / PV / battery sharing a building node
HEAT_PUMPS = [  # id, feeder, building, current_kA, source
    ("hp_21A", "b21A", "21/A", 0.074, "pdf-fig10"),
    ("hp_19", "b19", "19", 0.074, "pdf-fig10"),
    ("hp_26", "b26", "26", 0.074, "pdf-fig10"),
    ("hp_29", "b29", "29", 0.222, "pdf-fig10"),
]
# Buildings named in the report as technically suitable for additional PV (pdf-text p.11)
PV_SITES = [  # feeder, building, share of added PV capacity (assumed equal)
    ("b26", "26", 0.2),
    ("b4", "4", 0.2),
    ("b6", "6", 0.2),
    ("b17", "17", 0.2),
    ("reaktor_1", "reactor", 0.2),
]
# EV charger sites, one row per charger in the order the sweep adds them (charger k stands at
# row k; a tenth charger would wrap to row 0). 9 chargers: the 3 of today (assumed: large
# office/lab buildings), then 6 on the feeders whose PV roofs overload their line most
# (measured on the test days, uncontrolled charging: L_4T_b4 130 %, L_6T_b6 122 %,
# L_4T_b26 118 %), so that charging can absorb part of the PV export there - ASSUMED, not
# from the documents.
EV_SITES = [  # feeder, building, comment
    ("b26", "26", "today"),
    ("b29", "29", "today"),
    ("b17", "17", "today"),
    ("b4", "4", "4/T circuit, most overloaded line"),
    ("b4", "4", "4/T circuit, most overloaded line"),
    ("b6", "6", "6/T circuit, overloaded line"),
    ("b4", "4", "4/T circuit, most overloaded line"),
    ("b6", "6", "6/T circuit, overloaded line"),
    ("b26", "26", "4/T circuit, overloaded line"),
]
N_EV_PROFILES = 10  # simulated_profile_*.csv files = chargers available for the sweep


# Thermal side of the heat pumps (used only when a scenario enables heat coupling). The
# campus has no heat metering, so all of this is assumed: a constant COP, a peak heat load
# of 80 % of what the pump can deliver, and a hot-water tank holding 3 h of that peak.
HP_COP = 3.5
HP_PEAK_SHARE = 0.8  # peak thermal demand / (rated electric power x COP)
TES_HOURS = 3.0  # tank capacity in hours of peak heat demand
TES_SET_C, TES_RETURN_C = 60.0, 35.0
TES_LOSS_PER_H = 0.005  # standing loss, share of the stored heat per hour
C_WATER = 4.186 / 3600  # kWh per litre and kelvin


def _heat_side(p_el_kw: float, building: str) -> dict:
    """Assumed thermal data of one heat pump (see the constants above)."""
    peak = HP_PEAK_SHARE * p_el_kw * HP_COP
    volume = TES_HOURS * peak / ((TES_SET_C - TES_RETURN_C) * C_WATER)
    return dict(
        cop=HP_COP,
        heat_csv_column=f"heat_{building.replace('/', '')}",
        peak_heat_kW=round(peak, 1),
        tes_volume_l=round(volume, -2),
        tes_set_temp_c=TES_SET_C,
        tes_return_temp_c=TES_RETURN_C,
        tes_loss_per_h=TES_LOSS_PER_H,
        tes_soc_init=0.5,
        heat_source="assumed (no heat metering): COP, peak share, tank size",
    )


def _feeder_bus(trafo: str, feeder: str) -> str:
    return f"{trafo}_{feeder}"


def _cable(rated_kva: float) -> dict:
    """Assumed feeder cable: n parallel NAYY 4x240 so that 1.25 x rated current fits."""
    from math import ceil

    n = max(1, ceil(1.25 * rated_kva / KA_TO_KVA / CABLE["max_i_kA"]))
    return dict(
        r_ohm_km=CABLE["r_ohm_km"] / n,
        x_ohm_km=CABLE["x_ohm_km"] / n,
        c_nF_km=CABLE["c_nF_km"] * n,
        max_i_kA=CABLE["max_i_kA"] * n,
    )


def build_network() -> dict[str, pd.DataFrame]:
    buses = [
        dict(
            id=bid,
            nominal_voltage_V=10000,
            level="MV",
            building="",
            x=x,
            y=0.0,
            source="pdf-fig9",
        )
        for bid, x in (("MV_Viranyos", -10.0), ("MV_10kV", 0.0))
    ]
    lines = [
        dict(
            id="L_MV_Viranyos",
            from_bus="MV_Viranyos",
            to_bus="MV_10kV",
            length_km=VIRANYOS_LENGTH_KM,
            **MV_CABLE_240,
            source="drawing: Virányos infeed 3x240 mm2 Al, 10 km (XLPE std values)",
        )
    ]
    trafos = []
    for ti, (tr, feeders) in enumerate(FEEDERS.items()):
        x0 = ti * 10.0
        buses.append(
            dict(
                id=f"MV_{tr}",
                nominal_voltage_V=10000,
                level="MV",
                building="",
                x=x0,
                y=-0.5,
                source="drawing: 10 kV cell of the receiving station",
            )
        )
        lines.append(
            dict(
                id=f"L_MV_{tr}",
                from_bus="MV_10kV",
                to_bus=f"MV_{tr}",
                length_km=MV_FEED_KM[tr],
                **MV_CABLE_150,
                source="drawing: cell cable 3x150 mm2 Al (assumed size), length read "
                "from the drawing"
                + ("; cell-to-transformer mapping assumed" if tr == "T" else ""),
            )
        )
        buses.append(
            dict(
                id=f"LV_{tr}",
                nominal_voltage_V=400,
                level="LV",
                building="",
                x=x0,
                y=-1.0,
                source="pdf-fig9",
            )
        )
        trafos.append(
            dict(
                id=f"TR_{tr}",
                from_bus=f"MV_{tr}",
                to_bus=f"LV_{tr}",
                type="kfki_1MVA",
                sn_kVA=1000,
                source="pdf-text p.10",
            )
        )
        for fi, (fid, label, bld, ka, _) in enumerate(feeders):
            buses.append(
                dict(
                    id=_feeder_bus(tr, fid),
                    nominal_voltage_V=400,
                    level="LV",
                    building=bld,
                    x=x0 + fi,
                    y=-2.0,
                    source="pdf-fig9",
                )
            )
            lines.append(
                dict(
                    id=f"L_{tr}_{fid}",
                    from_bus=f"LV_{tr}",
                    to_bus=_feeder_bus(tr, fid),
                    length_km=DEFAULT_LENGTH_KM,
                    **_cable(ka * KA_TO_KVA),
                    source=f"assumed (parallel {CABLE_NAME} sized to 1.25 x Fig.10 current)",
                )
            )
    # battery feeder on 4/T bus (Line_NaS in Fig. 9)
    buses.append(
        dict(
            id="4T_nas",
            nominal_voltage_V=400,
            level="LV",
            building="5",
            x=30.0,
            y=-2.0,
            source="pdf-fig9",
        )
    )
    lines.append(
        dict(
            id="Line_NaS",
            from_bus="LV_4T",
            to_bus="4T_nas",
            length_km=DEFAULT_LENGTH_KM,
            **_cable(250.0),
            source=f"assumed (parallel {CABLE_NAME} sized to the 250 kW NaS rating)",
        )
    )
    trafo_types = [
        dict(
            id="kfki_1MVA",
            sn_kVA=1000,
            vn_hv_kV=10.0,
            vn_lv_kV=0.4,
            vk_percent=6.0,
            vkr_percent=1.0,
            pfe_kW=1.5,
            i0_percent=0.3,
            shift_degree=150,
            source="assumed",
        )
    ]
    return dict(
        buses=pd.DataFrame(buses),
        lines=pd.DataFrame(lines),
        transformers=pd.DataFrame(trafos),
        trafo_types=pd.DataFrame(trafo_types),
        ext_grid=pd.DataFrame(
            [
                dict(
                    id="grid",
                    bus="MV_Viranyos",
                    vm_pu=1.0,
                    source="pdf-fig9 (Virányos feeder in service, ideal source at its "
                    "far end; Budaközép reserve open and not modelled)",
                )
            ]
        ),
    )


def build_assets() -> dict[str, pd.DataFrame]:
    feeder_trafo = {fid: tr for tr, fs in FEEDERS.items() for fid, *_ in fs}
    fbus = lambda fid: _feeder_bus(feeder_trafo[fid], fid)  # noqa: E731
    loads = []
    for tr, feeders in FEEDERS.items():
        for fid, label, bld, ka, sig in feeders:
            loads.append(
                dict(
                    id=fid,
                    bus=_feeder_bus(tr, fid),
                    building=bld,
                    label=english_label(fid, bld),
                    label_hu=label,
                    rated_kVA=round(ka * KA_TO_KVA, 1),
                    csv_column=f"load_{fid}",
                    power_factor=PF,
                    significant=sig,
                    source="pdf-fig10",
                )
            )
    hps = [
        dict(
            id=i,
            bus=fbus(f),
            building=b,
            rated_kVA=round(ka * KA_TO_KVA, 1),
            csv_column=f"hp_{b.replace('/', '')}",
            power_factor=PF,
            source=s + "; 29: 47.6 kW measurement (context-free, pdf-text p.12)",
            **_heat_side(round(ka * KA_TO_KVA, 1) * PF, b),
        )
        for i, f, b, ka, s in HEAT_PUMPS
    ]
    pv = [
        dict(
            id="pv_29",
            bus=fbus("b29"),
            building="29",
            base_kWp=9.0,
            n_inverters=3,
            inverter_kW_ac=3.0,
            csv_column="pv_per_kwp",
            power_factor=1.0,
            source="note: 3 x SMA SB3000HF-30; kWp assumed = 9 kW AC; 7.41 kW reading is "
            "a sanity check only",
        )
    ]
    battery = [
        dict(
            id="nas_ess",
            bus="4T_nas",
            building="5",
            power_kW=250.0,
            capacity_kWh=1450.0,
            eff_charge=0.9,
            eff_discharge=0.9,
            soc_min=0.0,
            soc_max=1.0,
            soc_init=0.5,
            dc_v_nom=192,
            dc_v_min=140,
            dc_v_max=230,
            source="note (power, capacity, DC band); efficiencies assumed; "
            "DC side not simulated",
        )
    ]
    pv_sites = [
        dict(
            id=f"pvx_{f}",
            bus=fbus(f),
            building=b,
            share=s,
            csv_column="pv_per_kwp",
            source="pdf-text p.11 (sites); share assumed",
        )
        for f, b, s in PV_SITES
    ]
    # One row per charger in the order of EV_SITES: charger k sits at row k mod 9, so the
    # sweep is nested (an additional charger only adds one at the next row).
    seen: dict[str, int] = {}
    ev_sites = []
    for f, b, note in EV_SITES:
        seen[f] = seen.get(f, 0) + 1
        ev_sites.append(
            dict(
                id=f"ev_{f}" if seen[f] == 1 else f"ev_{f}_{seen[f]}",
                bus=fbus(f),
                building=b,
                charger_kW=22.0,  # outlet rating, inferred from the 22 kW steps in the profiles
                power_factor=1.0,
                source=f"{note}; sites assumed; 22 kW from the profile data",
            )
        )
    # one demand profile (kW) per charger, from data/input/simulated_profile_*.csv
    ev_chargers = [
        dict(
            id=f"ev_c{k:02d}",
            csv_column=f"ev_charger_{k:02d}",
            source=f"simulated_profile_*_seed{41 + k}.csv",
        )
        for k in range(1, N_EV_PROFILES + 1)
    ]
    return dict(
        loads=pd.DataFrame(loads),
        heat_pumps=pd.DataFrame(hps),
        pv=pd.DataFrame(pv),
        battery=pd.DataFrame(battery),
        pv_sites=pd.DataFrame(pv_sites),
        ev_sites=pd.DataFrame(ev_sites),
        ev_chargers=pd.DataFrame(ev_chargers),
    )


def build_scenarios() -> dict[str, pd.DataFrame]:
    # axes may have different lengths (shorter ones padded with blanks). Campus demand is
    # ~0.8 MW on average (6.5-8 GWh/yr), so PV must reach MWp scale to create surplus.
    grid = pd.DataFrame(
        {
            "added_pv_kWp": pd.Series([0, 100, 250, 500, 750, 1000, 1500, 2000]),
            "battery_kWh": pd.Series([0, 725, 1450, 2900, 5800]),
            "n_chargers": pd.Series([3, 4, 5, 6, 7, 8, 9]),  # 3 = today's chargers
        }
    )
    settings = pd.DataFrame(
        [
            ("resolution", "1h", "simulation / optimisation time resolution"),
            (
                "battery_c_rate_per_h",
                250 / 1450,
                "battery kW per kWh when capacity is scaled",
            ),
            ("v_min_pu", 0.9, "voltage limit (EN 50160 -10 %)"),
            ("v_max_pu", 1.1, "voltage limit (EN 50160 +10 %)"),
            ("v_max_strict_pu", 1.05, "stricter reporting band"),
            ("horizon_h", 24, "rolling-horizon commit length"),
            ("lookahead_h", 24, "rolling-horizon extra look-ahead"),
            (
                "trafo_limit_kVA",
                1000,
                "per-transformer apparent power limit in dispatch",
            ),
        ],
        columns=["key", "value", "note"],
    )
    return dict(grid=grid, settings=settings)


def build_meter_map() -> dict[str, pd.DataFrame]:
    """LiveView main meters (serials in the UtilityBook file names) -> model loads.

    Several meters can feed one model load (their power is summed). Sources: the meter
    register (KFKI_merok_kapcsolatai) for building/transformer, the diagram for the nodes.
    """
    rows = [
        ("36708390", "b17", "17. ep."),
        ("36708404", "b21A", "21. ep. - diagram node '21. ep.' lumps 21, 21/A, 21/B"),
        ("36708426", "b21A", "21/A ep."),
        ("36708393", "b21A", "21/B ep."),
        ("36708392", "b2", "2. ep. (register: 3-as trafohaz)"),
        ("36708419", "b3", "3. ep."),
        ("36708442", "b19", "19. ep. (heat-pump meter 27335127 not in LiveView)"),
        ("36708424", "b26", "26. ep. 1st feeder"),
        ("36708389", "b26", "26. ep. 2nd feeder"),
        ("36708394", "b6", "6. ep. new switching area (swings vs the old one)"),
        ("36708425", "b6", "6. ep. old switching area"),
        ("36708411", "b1", "1. ep. (register lists the serial as 36708711)"),
        ("36708443", "b23", "23. ep. (register: 3-as trafo, diagram: 6/T)"),
    ]
    ignored = [
        (
            "36708395",
            "4-es trafo NAS segeduzem: battery auxiliary supply, ~4 kWh/day "
            "(~0.15 kW) and 54 missing days; negligible",
        ),
    ]
    return dict(
        map=pd.DataFrame(rows, columns=["meter_id", "load_id", "note"]),
        ignored=pd.DataFrame(ignored, columns=["meter_id", "note"]),
    )


def write_all(out: Path = OUT) -> None:
    for name, sheets in (
        ("network", build_network()),
        ("assets", build_assets()),
        ("scenarios", build_scenarios()),
        ("meter_map", build_meter_map()),
    ):
        with pd.ExcelWriter(out / f"{name}.xlsx", engine="openpyxl") as xw:
            for sheet, df in sheets.items():
                df.to_excel(xw, sheet_name=sheet, index=False)


if __name__ == "__main__":
    write_all()
    print(f"written to {OUT}")
