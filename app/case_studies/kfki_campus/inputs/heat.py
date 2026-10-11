"""Synthetic heat demand of the heat-pump buildings (thermal kW, hourly).

NOT measured: the campus has no heat metering. The demand is a degree-hour model driven
by the outdoor temperature of ``weather_hourly.csv``, scaled to the ``peak_heat_kW`` of
each heat pump (heat_pumps sheet):

  space heating  linear in (T_base - T_out), 0 above T_base, 1 at T_design, with a night
                 and weekend setback typical of office buildings
  hot water      a small flat baseline on working days

The columns ``heat_<building>`` are added to the wide CSV of the README contract.

  python -m app.case_studies.kfki_campus.inputs.heat --csv <wide csv> [--weather <csv>]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.excel_io import CampusData, load_campus_data

T_BASE_C = 15.0  # heating limit temperature
T_DESIGN_C = -10.0  # outdoor design temperature (full space-heating load)
SETBACK = 0.6  # night / weekend space-heating level relative to occupied hours
DHW_SHARE = 0.08  # hot-water share of the peak load


def heat_profile(temp_c: pd.Series) -> pd.Series:
    """Normalised heat demand (peak <= 1) from the hourly outdoor temperature."""
    degree = ((T_BASE_C - temp_c) / (T_BASE_C - T_DESIGN_C)).clip(0.0, 1.0)
    hour, weekday = temp_c.index.hour, temp_c.index.dayofweek < 5
    occupied = weekday & (hour >= 6) & (hour < 18)
    level = np.where(occupied, 1.0, SETBACK)
    dhw = np.where(weekday & (hour >= 6) & (hour < 20), 1.0, 0.3)
    prof = (1 - DHW_SHARE) * degree * level + DHW_SHARE * dhw
    return pd.Series(prof, index=temp_c.index).clip(upper=1.0)


def heat_columns(temp_c: pd.Series, data: CampusData) -> pd.DataFrame:
    """One column ``heat_csv_column`` (kW thermal) per heat pump."""
    prof = heat_profile(temp_c.interpolate(limit_direction="both"))
    return pd.DataFrame(
        {
            r.heat_csv_column: prof * float(r.peak_heat_kW)
            for r in data.heat_pumps.itertuples()
        }
    )


def add_heat_columns(
    csv_path: Path, weather_csv: Path | None = None, data: CampusData | None = None
) -> Path:
    """Add (or refresh) the heat columns of an existing wide CSV, in place."""
    csv_path = Path(csv_path)
    data = data or load_campus_data()
    weather_csv = Path(weather_csv or csv_path.with_name("weather_hourly.csv"))
    ts = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    temp = pd.read_csv(weather_csv, index_col=0, parse_dates=True)["temp_c"]
    heat = heat_columns(temp.reindex(ts.index), data)
    ts = ts.drop(columns=[c for c in heat.columns if c in ts.columns]).join(heat)
    ts.rename_axis("timestamp").to_csv(csv_path, date_format="%Y-%m-%d %H:%M:%S")
    return csv_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--weather", type=Path, default=None)
    a = p.parse_args(argv)
    print("written:", add_heat_columns(a.csv, a.weather))


if __name__ == "__main__":
    main()
