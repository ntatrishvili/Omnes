"""CLI: run the scenario sweep and/or create the figures.

Examples (from the repo root, venv active):
  python -m app.case_studies.kfki_campus.inputs.build_inputs --start 2025-05-28 --end 2026-03-01
  python -m app.case_studies.kfki_campus.run_case_study --csv app/case_studies/kfki_campus/data/processed/kfki_timeseries_2025-05-28_2026-03-01.csv --start 2025-05-28 --end 2026-03-01 --out results --workers 6
  python -m app.case_studies.kfki_campus.run_case_study --synthetic --start 2025-01-01 --end 2025-01-03 --limit 2 --out results_smoke
  python -m app.case_studies.kfki_campus.run_case_study --out results --figures-only

The CSV clock is fixed UTC+1; --start/--end are the first and last day (inclusive).
"""

import argparse
from pathlib import Path

import pandas as pd

from app.case_studies.kfki_campus.excel_io import load_campus_data
from app.case_studies.kfki_campus.figures import make_extra_figures
from app.case_studies.kfki_campus.inputs.build_inputs import PROCESSED_DIR
from app.case_studies.kfki_campus.plots import make_figures
from app.case_studies.kfki_campus.scenarios import run_sweep, scenario_grid
from app.case_studies.kfki_campus.synthetic import make_synthetic_csv
from app.infra.logging_setup import init_logging


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--csv", type=Path, help="wide time-series CSV (see README for the contract)"
    )
    p.add_argument(
        "--synthetic", action="store_true", help="use generated placeholder data"
    )
    p.add_argument("--out", type=Path, default=Path("results_kfki"))
    p.add_argument("--start", default="2025-05-28", help="first day (inclusive)")
    p.add_argument("--end", default="2026-03-01", help="last day (inclusive)")
    p.add_argument(
        "--tan-phi",
        type=Path,
        default=None,
        help="load_tan_phi.csv (default: the one next to --csv, if present)",
    )
    p.add_argument(
        "--resolution", default=None, help="override settings resolution, e.g. 1h"
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--limit", type=int, default=None, help="run at most N new scenarios"
    )
    p.add_argument(
        "--no-details", action="store_true", help="do not store per-scenario arrays"
    )
    p.add_argument(
        "--heat",
        action="store_true",
        help="electricity-heat coupling: heat pumps feed hot-water tanks (use its own --out)",
    )
    p.add_argument(
        "--tes-scale",
        type=float,
        default=1.0,
        help="with --heat: hot-water tank volume relative to the heat_pumps sheet",
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="electricity-only results dir: adds the heat-coupling comparison figure",
    )
    p.add_argument("--figures-only", action="store_true")
    p.add_argument(
        "--week",
        action="append",
        default=[],
        metavar="SCENARIO:DATE",
        help="extra scenario-week figure, e.g. pv1500_bat1450_ev0:2025-07-07 (repeatable)",
    )
    p.add_argument(
        "--no-extra-figures",
        action="store_true",
        help="skip network / time-series figures",
    )
    a = p.parse_args(argv)

    init_logging(level="INFO", log_dir=str(a.out / "logs"), log_file="run.log")
    if not a.figures_only:
        start = pd.Timestamp(a.start)
        end = pd.Timestamp(a.end) + pd.Timedelta(days=1)  # exclusive
        csv = a.csv
        if a.synthetic:
            csv = make_synthetic_csv(
                a.out / "synthetic_timeseries.csv",
                a.start,
                (end - start).days,
            )
        if csv is None:
            p.error("give --csv or --synthetic")
        tan_path = a.tan_phi or Path(csv).with_name("load_tan_phi.csv")
        data = load_campus_data(tan_phi_path=tan_path)
        run_sweep(
            csv,
            a.out,
            scenarios=(
                scenario_grid(data, True, a.tes_scale) if a.heat else None
            ),
            workers=a.workers,
            limit=a.limit,
            keep_details=not a.no_details,
            data=data,
            time_start=str(start),
            time_end=str(end),
            resolution=a.resolution,
        )
    for f in make_figures(a.out):
        print("figure:", f)
    if not a.no_extra_figures:
        csv = a.csv or (
            PROCESSED_DIR / f"kfki_timeseries_{a.start}_{a.end}.csv"
            if not a.synthetic
            else a.out / "synthetic_timeseries.csv"
        )
        if Path(csv).exists():
            weeks = [tuple(w.split(":", 1)) for w in a.week]
            for f in make_extra_figures(a.out, csv, weeks, a.baseline):
                print("figure:", f)
        else:
            print(f"skipping network / time-series figures: {csv} not found")


if __name__ == "__main__":
    main()
