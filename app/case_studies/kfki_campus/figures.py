"""All extra figures (network structure, input time series, scenario weeks) in one call."""

from pathlib import Path
from typing import Optional

import pandas as pd

from app.case_studies.kfki_campus.excel_io import (
    DATA_DIR,
    display_names,
    load_campus_data,
)
from app.case_studies.kfki_campus.plots_heat import heat_comparison, heat_week
from app.case_studies.kfki_campus.plots_network import (
    network_topology,
    network_voltage_maps,
)
from app.case_studies.kfki_campus.plots_timeseries import (
    input_carpets,
    input_composition_weeks,
    input_daily_profiles,
    input_energy_by_load,
    input_ev_profiles,
    input_measured_buildings,
    input_overview,
    scenario_week,
    split_loads,
    voltage_duration,
)

SUMMER_WEEK = "2025-06-16"
WINTER_WEEK = "2026-01-12"
WEEK_SCENARIOS = [
    ("pv0_bat1450_ev3", SUMMER_WEEK),
    ("pv1000_bat1450_ev9", SUMMER_WEEK),
    ("pv1000_bat1450_ev9", WINTER_WEEK),
    ("pv2000_bat2900_ev9", SUMMER_WEEK),
]
VOLTAGE_PANELS = [
    ("pv0_bat1450_ev3", "min", "Baseline (9 kWp PV, 1.45 MWh NaS, 3 chargers)"),
    ("pv2000_bat0_ev3", "max", "2000 kWp added PV, no battery, 3 chargers"),
    ("pv2000_bat5800_ev3", "max", "2000 kWp added PV, 5.8 MWh battery, 3 chargers"),
    ("pv2000_bat5800_ev9", "min", "2000 kWp added PV, 5.8 MWh battery, 9 chargers"),
]


def make_extra_figures(
    out_dir: Path,
    csv_path: Path,
    extra_weeks: Optional[list[tuple[str, str]]] = None,
    baseline_dir: Optional[Path] = None,
) -> list[Path]:
    """Write the extra figures to ``out_dir/figures``; return their paths.

    ``extra_weeks`` = [(scenario id, week start date), ...] adds more scenario weeks.
    ``baseline_dir`` = the electricity-only sweep; when ``out_dir`` is a heat-coupled
    sweep (``--heat``) it adds the with / without heat coupling comparison.
    """
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_path)
    data = load_campus_data()
    csv = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    mapped = set(
        pd.read_excel(DATA_DIR / "meter_map.xlsx", sheet_name="map")["load_id"]
    )
    weather_path = csv_path.with_name("weather_hourly.csv")
    weather = (
        pd.read_csv(weather_path, index_col=0, parse_dates=True)
        if weather_path.exists()
        else None
    )
    loads = split_loads(csv, data, mapped)
    names = display_names(data)[0]  # English load names for the plots
    made = [
        network_topology(data, csv, mapped, fig_dir / "network_topology.png"),
        input_overview(csv, weather, loads, fig_dir / "ts_overview.png"),
        input_daily_profiles(csv, loads, fig_dir / "ts_daily_profiles.png"),
        input_carpets(csv, loads, fig_dir / "ts_carpets.png"),
        input_composition_weeks(
            loads, mapped, fig_dir / "ts_composition_weeks.png", names=names
        ),
        input_measured_buildings(
            loads, mapped, fig_dir / "ts_measured_buildings.png", names=names
        ),
        input_energy_by_load(
            loads, mapped, fig_dir / "ts_energy_by_load.png", names=names
        ),
        input_ev_profiles(csv, fig_dir / "ts_ev_profiles.png"),
    ]
    details = out_dir / "details"
    kpi_path = out_dir / "kpis.csv"
    if details.exists() and kpi_path.exists():
        kpis = pd.read_csv(kpi_path)
        made.append(
            network_voltage_maps(
                data, csv, details, VOLTAGE_PANELS, fig_dir / "network_voltage_maps.png"
            )
        )
        made.append(voltage_duration(kpis, details, fig_dir / "voltage_duration.png"))
        for sid, start in WEEK_SCENARIOS + list(extra_weeks or []):
            made.append(
                scenario_week(details, sid, start, fig_dir / f"week_{sid}_{start}.png")
            )
            # heat-coupled sweep (--heat): same scenario with the tank suffix, plus the
            # electric / heat / tank view of any scenario that has heat pumps
            for hid in (sid, f"{sid}_hp1"):
                made.append(
                    heat_week(details, hid, start, fig_dir / f"heat_{hid}_{start}.png")
                )
                if hid != sid:
                    made.append(
                        scenario_week(
                            details, hid, start, fig_dir / f"week_{hid}_{start}.png"
                        )
                    )
        if baseline_dir is not None and (Path(baseline_dir) / "kpis.csv").exists():
            made.append(
                heat_comparison(
                    pd.read_csv(Path(baseline_dir) / "kpis.csv"),
                    kpis,
                    fig_dir / "heat_comparison.png",
                )
            )
    return [p for p in made if p is not None]
