"""Run the RL pipeline with a Python event generator (script instead of the CLI).

``run_rl`` builds the problem from the workbook, which can only hold text; a Python object (a
fitted distribution, a generator subclass) has to be handed over from a script. This one does
that end to end on placeholder data:

    python -m app.case_studies.kfki_campus.rl.examples.run_custom_events --generator kde --out results_custom

It evaluates the reference policies on the test days (and trains a PPO agent if
``--train-steps`` > 0 and Stable-Baselines3 is installed) and writes ``rl_kpis.csv`` and the figures.
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from app.case_studies.kfki_campus.excel_io import load_campus_data
from app.case_studies.kfki_campus.rl.examples import (
    ex_bootstrap,
    ex_gmm_files,
    ex_kde_joint,
    ex_scipy_marginals,
    ex_weekday_generator,
)
from app.case_studies.kfki_campus.rl.plots_rl import make_rl_figures
from app.case_studies.kfki_campus.rl.train import (
    Setup,
    evaluate_all,
    load_events,
    make_env,
    reference_agents,
    summarize,
    train_agent,
    with_events,
)
from app.case_studies.kfki_campus.synthetic import make_synthetic_csv
from app.operation.rl.events import GENERATOR_REGISTRY
from app.operation.rl.problem import EventSpec, RLProblem

EXAMPLES = {
    "marginals": ex_scipy_marginals.make_generator,
    "kde": ex_kde_joint.make_generator,
    "bootstrap": ex_bootstrap.make_generator,
    "weekday": ex_weekday_generator.make_generator,
}


# [start:wire]
def make_problem(generator, n_chargers: int = 3) -> RLProblem:
    """Hand a generator OBJECT to the problem: register a factory under a name, use the name."""
    GENERATOR_REGISTRY["custom"] = (
        lambda **_: generator
    )  # the env calls it with the params
    return RLProblem(
        scenario=dict(n_chargers=n_chargers),
        power_flow_mode="post",  # fast; use "step" for voltage / loading terms
        events=EventSpec(source="generator", generator="custom"),
    )


# [end:wire]


def main(argv=None) -> pd.DataFrame:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--generator", choices=sorted(EXAMPLES) + ["gmm"], default="kde")
    p.add_argument(
        "--gmm-workday", type=Path, help="workday .gmm (with --generator gmm)"
    )
    p.add_argument(
        "--gmm-weekend", type=Path, help="weekend .gmm (with --generator gmm)"
    )
    p.add_argument(
        "--workday-rate",
        type=float,
        default=1.5,
        help="plug-ins per charger and workday",
    )
    p.add_argument(
        "--weekend-rate",
        type=float,
        default=0.6,
        help="plug-ins per charger and weekend day",
    )
    p.add_argument(
        "--csv", type=Path, help="real time-series CSV (default: placeholder data)"
    )
    p.add_argument("--start", default="2025-06-02", help="first day (inclusive)")
    p.add_argument("--end", default=None, help="last day (inclusive; with --csv)")
    p.add_argument("--days", type=int, default=6, help="length of the placeholder data")
    p.add_argument("--session-file", type=Path, help="fixed charging sessions (CSV)")
    p.add_argument(
        "--events",
        choices=["generator", "mixed", "file"],
        default=None,
        help="training events (default: generator; mixed / file need --session-file)",
    )
    p.add_argument(
        "--eval-events",
        choices=["generator", "mixed", "file"],
        default=None,
        help="events of the evaluation (default: as --events)",
    )
    p.add_argument("--chargers", type=int, default=3)
    p.add_argument("--eval-days", type=int, default=3)
    p.add_argument(
        "--train-steps", type=int, default=0, help="0 = reference policies only"
    )
    p.add_argument("--out", type=Path, default=Path("results_custom_events"))
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    print("results directory:", a.out.resolve())

    if a.generator == "gmm":
        if not (a.gmm_workday and a.gmm_weekend):
            p.error("--generator gmm needs --gmm-workday and --gmm-weekend")
        generator = ex_gmm_files.make_generator(
            a.gmm_workday, a.gmm_weekend, a.workday_rate, a.weekend_rate
        )
    else:
        generator = EXAMPLES[a.generator]()
    events = a.events or ("mixed" if a.session_file else "generator")
    eval_events = a.eval_events or events
    if "file" in (events, eval_events) or events == "mixed" or eval_events == "mixed":
        if a.session_file is None:
            p.error("--events / --eval-events 'mixed' or 'file' need --session-file")

    if a.csv:
        if a.end is None:
            p.error("--csv needs --end")
        csv = a.csv
        data = load_campus_data(tan_phi_path=Path(csv).with_name("load_tan_phi.csv"))
        end = pd.Timestamp(a.end) + pd.Timedelta(days=1)  # exclusive
    else:
        data = load_campus_data()
        csv = make_synthetic_csv(
            a.out / "synthetic_timeseries.csv", a.start, a.days, data=data
        )
        end = pd.Timestamp(a.start) + pd.Timedelta(days=a.days)
    setup = Setup(csv, data, str(pd.Timestamp(a.start)), str(end))
    problem = make_problem(generator, a.chargers)
    problem.events.source = events
    if a.session_file:
        setup.events = load_events(a.session_file, problem, setup)
    problem.validate()

    agents = reference_agents(problem, oracle=False)
    if a.train_steps > 0:
        agent, _ = train_agent(
            problem,
            setup,
            a.out,
            0,
            a.train_steps,
            eval_every=a.train_steps,
            eval_days=2,
        )
        agents = {"ppo_seed0": agent, **agents}
    eval_problem = with_events(
        problem, source=eval_events
    )  # e.g. train on GMM, test on file
    test_days = make_env(eval_problem, setup, "test").test_days[: a.eval_days]
    ev = evaluate_all(eval_problem, setup, agents, a.out, test_days, full_period=False)
    print(summarize(ev.kpis).round(3).to_string())
    make_rl_figures(a.out)
    return ev.kpis


if __name__ == "__main__":
    main()
