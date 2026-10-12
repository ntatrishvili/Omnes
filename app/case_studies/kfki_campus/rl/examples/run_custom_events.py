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
    ex_kde_joint,
    ex_scipy_marginals,
    ex_weekday_generator,
)
from app.case_studies.kfki_campus.rl.plots_rl import make_rl_figures
from app.case_studies.kfki_campus.rl.train import (
    Setup,
    evaluate_all,
    make_env,
    reference_agents,
    summarize,
    train_agent,
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
    p.add_argument("--generator", choices=sorted(EXAMPLES), default="kde")
    p.add_argument("--start", default="2025-06-02")
    p.add_argument("--days", type=int, default=6)
    p.add_argument("--chargers", type=int, default=3)
    p.add_argument("--eval-days", type=int, default=3)
    p.add_argument(
        "--train-steps", type=int, default=0, help="0 = reference policies only"
    )
    p.add_argument("--out", type=Path, default=Path("results_custom_events"))
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    data = load_campus_data()
    csv = make_synthetic_csv(
        a.out / "synthetic_timeseries.csv", a.start, a.days, data=data
    )
    end = pd.Timestamp(a.start) + pd.Timedelta(days=a.days)
    setup = Setup(csv, data, str(pd.Timestamp(a.start)), str(end))
    problem = make_problem(EXAMPLES[a.generator](), a.chargers)

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
    test_days = make_env(problem, setup, "test").test_days[: a.eval_days]
    ev = evaluate_all(problem, setup, agents, a.out, test_days, full_period=False)
    print(summarize(ev.kpis).round(3).to_string())
    make_rl_figures(a.out)
    return ev.kpis


if __name__ == "__main__":
    main()
