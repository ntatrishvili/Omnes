"""CLI of the RL workflow: train, evaluate, figures, sweep, generalization, all.

Examples (repo root, venv active; ``pip install -r requirements-rl.txt`` for training):
  python -m app.case_studies.kfki_campus.rl.data.make_rl_inputs
  python -m app.case_studies.kfki_campus.rl.run_rl all --csv <csv> --start 2025-05-28 --end 2026-03-01 --session-file sessions.csv --out results_rl
  python -m app.case_studies.kfki_campus.rl.run_rl all --synthetic --start 2025-01-01 --end 2025-01-14 --timesteps 3000 --out results_rl_smoke
  python -m app.case_studies.kfki_campus.rl.run_rl evaluate --csv <csv> ... --out results_rl      (re-evaluate saved models)
  python -m app.case_studies.kfki_campus.rl.run_rl figures --out results_rl
  python -m app.case_studies.kfki_campus.rl.run_rl sweep --sweep unmet_kwh:1,2,5,10 ...

The CSV clock is fixed UTC+1; --start / --end are the first and last day (inclusive).
Problem settings come from ``--problem`` (workbook, default ``rl/data/rl.xlsx``); a few
common ones can be overridden on the command line.
"""

import argparse
from pathlib import Path

import pandas as pd

from app.case_studies.kfki_campus.excel_io import load_campus_data
from app.case_studies.kfki_campus.rl.plots_rl import make_rl_figures
from app.case_studies.kfki_campus.rl.problem_io import DATA_DIR, load_problem
from app.case_studies.kfki_campus.rl.train import (
    Setup,
    evaluate_all,
    generalization,
    load_events,
    reference_agents,
    summarize,
    sweep_weight,
    train_agent,
)
from app.case_studies.kfki_campus.synthetic import make_synthetic_csv
from app.infra.logging_setup import init_logging
from app.operation.rl.agent import BasicPPOAgent


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "command",
        choices=["train", "evaluate", "figures", "sweep", "generalization", "all"],
    )
    p.add_argument(
        "--csv", type=Path, help="wide time-series CSV (case-study README contract)"
    )
    p.add_argument(
        "--synthetic", action="store_true", help="generated placeholder data"
    )
    p.add_argument("--start", default="2025-05-28")
    p.add_argument("--end", default="2026-03-01")
    p.add_argument("--out", type=Path, default=Path("results_rl"))
    p.add_argument(
        "--problem", type=Path, default=DATA_DIR / "rl.xlsx", help="RL problem workbook"
    )
    p.add_argument(
        "--session-file",
        type=Path,
        default=None,
        help="charging-session CSV (see sessions.py)",
    )
    p.add_argument(
        "--events",
        choices=["file", "generator", "mixed"],
        default=None,
        help="override events.source",
    )
    p.add_argument(
        "--timesteps", type=int, default=None, help="override agent.total_timesteps"
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="training seeds")
    p.add_argument(
        "--power-flow",
        choices=["step", "post"],
        default=None,
        help="override power_flow_mode",
    )
    p.add_argument(
        "--pv-kwp",
        type=float,
        default=None,
        help="override scenario.added_pv_kwp (2000 = 2009 kWp built in)",
    )
    p.add_argument(
        "--chargers", type=int, default=None, help="override scenario.n_chargers"
    )
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument(
        "--eval-days", type=int, default=None, help="evaluate on at most N test days"
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=5000,
        help="learning-curve evaluation interval (steps)",
    )
    p.add_argument("--no-oracle", action="store_true")
    p.add_argument("--no-full-period", action="store_true")
    p.add_argument(
        "--sweep", default=None, metavar="METRIC:V1,V2,...", help="reward-weight sweep"
    )
    return p


def _setup(a, problem) -> Setup:
    start = pd.Timestamp(a.start)
    end = pd.Timestamp(a.end) + pd.Timedelta(days=1)  # exclusive
    csv = a.csv
    if a.synthetic:
        csv = make_synthetic_csv(
            a.out / "synthetic_timeseries.csv", a.start, (end - start).days
        )
    if csv is None:
        raise SystemExit("give --csv or --synthetic")
    tan = Path(csv).with_name("load_tan_phi.csv")
    data = load_campus_data(tan_phi_path=tan)
    setup = Setup(Path(csv), data, str(start), str(end))
    if a.session_file is not None:
        setup.events = load_events(a.session_file, problem, setup)
    return setup


def main(argv=None) -> None:
    a = _parser().parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    init_logging(level="INFO", log_dir=str(a.out / "logs"), log_file="rl.log")
    # results go relative to the working directory: say where that is
    print("results directory:", a.out.resolve())
    if a.command == "figures":
        for f in make_rl_figures(a.out):
            print("figure:", f)
        return
    problem = load_problem(a.problem)
    if a.events:
        problem.events.source = a.events
    if a.power_flow:
        problem.power_flow_mode = a.power_flow
    if a.pv_kwp is not None:
        problem.scenario["added_pv_kwp"] = a.pv_kwp
    if a.chargers is not None:
        problem.scenario["n_chargers"] = a.chargers
    if a.n_envs:
        problem.agent.n_envs = a.n_envs
    problem.validate()
    setup = _setup(a, problem)
    if problem.events.source in ("file", "mixed") and setup.events is None:
        raise SystemExit(
            f"events.source '{problem.events.source}' needs --session-file"
        )

    agents = {}
    if a.command in ("train", "all"):
        for s in a.seeds:
            agent, _ = train_agent(problem, setup, a.out, s, a.timesteps, a.eval_every)
            agents[f"{problem.agent.algo.lower()}_seed{s}"] = agent
    if a.command == "evaluate":
        for s in a.seeds:
            name = f"{problem.agent.algo.lower()}_seed{s}"
            path = a.out / "models" / f"{name}.zip"
            if path.exists():
                agents[name] = BasicPPOAgent.load(path, problem)
    if a.command in ("evaluate", "all"):
        agents.update(reference_agents(problem, oracle=not a.no_oracle))
        days = None
        if a.eval_days:
            from app.case_studies.kfki_campus.rl.train import make_env

            td = make_env(problem, setup, "test").test_days
            days = td[:: max(1, len(td) // a.eval_days)][: a.eval_days]
        ev = evaluate_all(problem, setup, agents, a.out, days, not a.no_full_period)
        summary = summarize(ev.kpis)
        summary.to_csv(a.out / "rl_summary.csv")
        print(summary.round(3).to_string())
    if a.command == "sweep":
        if not a.sweep:
            raise SystemExit("--sweep METRIC:V1,V2,...")
        metric, vals = a.sweep.split(":")
        sweep_weight(
            problem,
            setup,
            a.out,
            metric,
            [float(v) for v in vals.split(",")],
            a.timesteps,
        )
    if a.command == "generalization":
        generalization(problem, setup, a.out, a.timesteps)
    if a.command in ("evaluate", "all", "sweep", "generalization"):
        for f in make_rl_figures(a.out):
            print("figure:", f)


if __name__ == "__main__":
    main()
