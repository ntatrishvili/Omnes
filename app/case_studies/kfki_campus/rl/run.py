"""RL case study of the KFKI campus: charging control of the EV chargers, one command.

    python -m app.case_studies.kfki_campus.rl.run

All parameters have defaults: the campus CSV of the case study, the trained models in
``results_rl_final`` (if present; missing ones are trained), the output in ``results_rl`` and
as many worker processes as the machine has (at most 8).

Stages (see ``rl/README.md`` for the plain-language description):

1. **GMM sessions**   sample charging sessions from the campus mixtures (no training needed);
2. **agents**         PPO agents of the profiles ``pv_max`` / ``grid_safe`` / ``balanced`` in the
                      ``standard`` and ``hard`` environments, 3 seeds; models found in
                      ``--models`` (or in ``--out``) are reused, missing ones are trained;
3. **evaluation**     the agents, the rules and the oracle on the same test days;
4. **sensitivity**    the trained ``standard`` agents re-evaluated (not retrained) with other
                      numbers of sessions, stays and PV sizes;
5. **figures**        ``figures/*.png|pdf`` and ``tablazat.csv|tex``.

``--figures-only`` redraws the figures from the saved tables; ``--quick`` is a smoke test.
"""

import argparse
import os
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.excel_io import load_campus_data
from app.case_studies.kfki_campus.rl.baselines import BASELINES
from app.case_studies.kfki_campus.rl.evaluate import evaluate_policies
from app.case_studies.kfki_campus.rl.gmm_events import (
    WEEKEND_FILE,
    WORKDAY_FILE,
    GmmBundleGenerator,
    load_bundle,
)
from app.case_studies.kfki_campus.rl import figures_hu as F
from app.case_studies.kfki_campus.rl.profiles import (
    ENVIRONMENTS,
    default_problem,
    make_problem,
)
from app.case_studies.kfki_campus.rl.train import (
    Setup,
    eval_day_list,
    make_env,
    reference_agents,
    train_agent,
)
from app.infra.logging_setup import get_logger, init_logging
from app.operation.rl.agent import BasicPPOAgent

log = get_logger(__name__)

PROFILES = ("pv_max", "grid_safe", "balanced")
ENVS = ("standard", "hard")  # the two environments of this case study
RATED_KW = 22.0

# defaults of the command line
DEFAULT_CSV = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "kfki_timeseries_2025-05-28_2026-03-01.csv"
)
DEFAULT_MODELS = Path("results_rl_final")
DEFAULT_OUT = Path("results_rl")

# sensitivity: (axis, label, values, how it changes the problem)
SENS_AXES = {
    "n": ("workday_rate", (1.5, 2.5, 4.0, 8.0)),  # plug-in attempts per charger and day
    "flex": ("stay_scale", (0.4, 0.7, 1.0, 1.5)),  # multiplies every stay
    "pv": ("added_pv_kwp", (1000.0, 1500.0, 2000.0, 3000.0)),  # installed PV (no overload below)
}


# ------------------------------------------------------------------------------ setup
def make_setup(csv: Path, start: str, end: str) -> Setup:
    data = load_campus_data(tan_phi_path=Path(csv).with_name("load_tan_phi.csv"))
    end_excl = pd.Timestamp(end) + pd.Timedelta(days=1)
    return Setup(Path(csv), data, str(pd.Timestamp(start)), str(end_excl))


def model_name(profile: str, seed: int) -> str:
    return f"ppo_{profile}_s{seed}"


def find_model(env: str, name: str, dirs: list[Path]):
    for d in dirs:
        p = Path(d) / env / "models" / f"{name}.zip"
        if p.exists():
            return p
    return None


# ------------------------------------------------------------------------------ 2 agents
def _train_job(job: dict) -> str:
    """Train one agent in a worker process (warm start, as in the scenario runs)."""
    init_logging(level="WARNING")
    setup = make_setup(job["csv"], job["start"], job["end"])
    p = make_problem(default_problem(), job["env"], job["profile"])
    train_agent(
        p, setup, Path(job["out"]) / job["env"], job["seed"], job["timesteps"],
        eval_every=max(job["timesteps"] // 4, 1000), eval_days=4,
        name=model_name(job["profile"], job["seed"]), warm_start=True,
    )
    return f"{job['env']}/{model_name(job['profile'], job['seed'])}"


def ensure_models(a, seeds: list[int]) -> None:
    """Copy the models found in ``--models`` into ``--out``; train the missing ones."""
    out = Path(a.out)
    missing = []
    for env in ENVS:
        for prof in PROFILES:
            for s in seeds:
                name = model_name(prof, s)
                if find_model(env, name, [out]):
                    continue
                src = find_model(env, name, [Path(a.models)] if a.models else [])
                if src:
                    (out / env / "models").mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, out / env / "models" / src.name)
                    log.info(f"reusing {src}")
                else:
                    missing.append(dict(csv=str(a.csv), start=a.start, end=a.end, out=str(out), env=env, profile=prof, seed=s, timesteps=a.timesteps))
    if not missing:
        return
    log.info(f"training {len(missing)} missing agents ({a.workers} in parallel)")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for done in ex.map(_train_job, missing):
            log.info(f"trained {done}")


def load_agents(env: str, problem, seeds: list[int], out: Path) -> dict:
    agents = {}
    for prof in PROFILES:
        for s in seeds:
            name = model_name(prof, s)
            agents[name] = BasicPPOAgent.load(find_model(env, name, [out]), problem)
    return agents


# ------------------------------------------------------------------------------ 3 evaluation
def _eval_job(job: dict):
    """Evaluate one environment in a worker process."""
    init_logging(level="WARNING")
    setup = make_setup(job["csv"], job["start"], job["end"])
    return job["env"], evaluate_environment(job["out"], job["env"], setup, job["seeds"], job["days"])


def evaluate_environment(out, env: str, setup: Setup, seeds: list[int], n_days: int):
    """KPI table (per seed) and episodes of all policies of one environment."""
    out = Path(out)
    p = make_problem(default_problem(), env, "balanced")
    e = make_env(p, setup, "test")
    days = eval_day_list(e, n_days)
    agents = reference_agents(p, oracle=True)
    agents = {k: v for k, v in agents.items() if k not in ("random", "least_laxity")}
    agents.update(load_agents(env, p, seeds, out))
    ev = evaluate_policies(e, agents, days, seed=0, keep_episodes=True)
    for ep in ev.episodes.values():  # keep what the figures need, drop the rest
        ep.pop("vm_pu", None)
        ep.pop("sessions", None)
        ep.pop("action", None)
        ep.pop("connected", None)
    trafo_names = list(e.net.trafo.name)
    line_names = list(e.net.line.name)
    return ev.kpis, ev.episodes, dict(trafo_names=trafo_names, line_names=line_names, days=days)


# ------------------------------------------------------------------------------ 1 GMM samples
def gmm_samples(n_work: int = 400, n_weekend: int = 200, seed: int = 11):
    """Sessions and aggregate uncontrolled load of 9 chargers sampled from the campus GMMs."""
    gen = GmmBundleGenerator(workday_rate=4.0, weekend_rate=1.6)
    chargers = {f"c{i}": RATED_KW for i in range(9)}
    rng = np.random.default_rng(seed)
    rows, curves = [], {}
    for kind, n, first in (("workday", n_work, "2025-06-02"), ("weekend", n_weekend, "2025-06-07")):
        mats = []
        for i in range(n):
            # Monday / Saturday of a random week: the day type is fixed, the draws are new
            day = pd.Timestamp(first) + pd.Timedelta(weeks=int(rng.integers(0, 40)))
            ev = gen.sample(rng, day, day + pd.Timedelta(days=1), chargers)
            load = np.zeros(24)
            for e in ev:
                t0 = (e.plug_in - day) / pd.Timedelta("1h")
                need = min(e.energy_kwh / RATED_KW, e.connected_h)  # hours at full power
                for h in range(24):  # overlap of [t0, t0 + need] with hour h
                    ov = max(0.0, min(h + 1, t0 + need) - max(h, t0))
                    load[h] += ov * RATED_KW
                if e.plug_in >= day:
                    rows.append(dict(day_type=kind, start_hour=t0, energy_kwh=e.energy_kwh, stay_h=e.connected_h))
            mats.append(load)
        curves[kind] = np.array(mats)
    sess = pd.DataFrame(rows)
    info = dict(
        n_fit_workday=int(load_bundle(WORKDAY_FILE).get("sample_count", 0)),
        n_fit_weekend=int(load_bundle(WEEKEND_FILE).get("sample_count", 0)),
    )
    return sess, curves, info


# ------------------------------------------------------------------------------ 4 sensitivity
def _sens_job(job: dict) -> pd.DataFrame:
    """One condition: uncontrolled, rule, oracle and the trained agents on the same days."""
    init_logging(level="WARNING")
    setup = make_setup(job["csv"], job["start"], job["end"])
    p = make_problem(default_problem(), "standard", "grid_safe")
    axis, value = job["axis"], job["value"]
    if axis == "n":
        p.events.generator_params.update(workday_rate=value, weekend_rate=0.4 * value)
    elif axis == "flex":
        p.events.generator_params.update(stay_scale=value)
    else:
        p.scenario["added_pv_kwp"] = value
    env = make_env(p, setup, "test")
    days = eval_day_list(env, job["days"])
    agents = {k: BASELINES[k](p) for k in ("uncontrolled", "line_following", "oracle")}
    for s in job["seeds"]:
        name = model_name("grid_safe", s)
        agents[name] = BasicPPOAgent.load(find_model("standard", name, [Path(job["out"])]), p)
    k = evaluate_policies(env, agents, days, seed=0, keep_episodes=False).kpis
    k.insert(0, "axis", axis)
    k.insert(1, "value", value)
    return k[["axis", "value", "policy", "day", "ev_sessions", "ev_pv_share", "ev_line_overload_kvah"]]


def run_sensitivity(a, seeds: list[int], n_days: int) -> pd.DataFrame:
    jobs = []
    for axis, (_, values) in SENS_AXES.items():
        for v in values:
            jobs.append(dict(csv=str(a.csv), start=a.start, end=a.end, out=str(a.out), axis=axis, value=v, days=n_days, seeds=seeds))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        frames = list(ex.map(_sens_job, jobs))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------------------ figures
def draw_figures(out: Path) -> list[Path]:
    data = out / "data"
    fig_dir = out / "figures"
    paths = []
    paths += F.fig_network(load_campus_data(), fig_dir)
    with open(data / "gmm.pkl", "rb") as f:
        sess, curves, info = pickle.load(f)
    paths += F.fig_gmm_sessions(sess, info, fig_dir)
    paths += F.fig_gmm_load(curves, fig_dir)
    kpis, eps, meta = {}, {}, {}
    for env in ENVS:
        kpis[env] = pd.read_csv(data / f"kpis_{env}.csv")
        with open(data / f"episodes_{env}.pkl", "rb") as f:
            eps[env], meta[env] = pickle.load(f)
    limit = ENVIRONMENTS["hard"]["trafo_limits"]["TR_4T"]
    paths += F.fig_campus(eps["hard"], meta["hard"]["line_names"], limit, out=fig_dir, trafo_idx=meta["hard"]["trafo_names"].index("TR_4T"))
    paths += F.fig_rl_profiles(kpis["standard"], fig_dir)
    paths += F.fig_example_day(eps["standard"], meta["standard"]["line_names"], fig_dir)
    paths += F.fig_battery_trafo(kpis["hard"], eps["hard"], limit, fig_dir, trafo_idx=meta["hard"]["trafo_names"].index("TR_4T"))
    if (data / "sensitivity.csv").exists():
        paths += F.fig_sensitivity(pd.read_csv(data / "sensitivity.csv"), fig_dir)
    paths.append(F.result_table(kpis, out))
    return paths


# ------------------------------------------------------------------------------ main
def _parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="campus time-series CSV")
    p.add_argument("--start", default="2025-05-28")
    p.add_argument("--end", default="2026-03-01")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--models", type=Path, default=DEFAULT_MODELS, help="folder with trained models (<env>/models/*.zip) to reuse; missing models are trained")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-days", type=int, default=20)
    p.add_argument("--sens-days", type=int, default=12)
    p.add_argument("--timesteps", type=int, default=150_000, help="training steps of agents that have to be trained")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="parallel processes (training, evaluation, sensitivity)")
    p.add_argument("--no-sensitivity", action="store_true")
    p.add_argument("--figures-only", action="store_true", help="redraw from the saved tables")
    p.add_argument("--quick", action="store_true", help="smoke test: 1 seed, few days, short training")
    return p


def main(argv=None) -> None:
    a = _parser().parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    init_logging(level="INFO", log_dir=str(a.out / "logs"), log_file="rl.log")
    print("results directory:", a.out.resolve())
    if a.figures_only:
        for p in draw_figures(a.out):
            print("figure:", p)
        return
    if not Path(a.csv).exists():
        raise SystemExit(f"campus CSV not found: {a.csv} (give --csv)")
    seeds, n_days, n_sens, steps = list(a.seeds), a.eval_days, a.sens_days, a.timesteps
    if a.quick:
        seeds, n_days, n_sens, steps = seeds[:1], 4, 3, 4000
        a.timesteps = steps
    (a.out / "data").mkdir(exist_ok=True)

    log.info("1/5 GMM sessions")
    with open(a.out / "data" / "gmm.pkl", "wb") as f:
        pickle.dump(gmm_samples(100 if a.quick else 400, 50 if a.quick else 200), f)

    log.info("2/5 agents")
    ensure_models(a, seeds)

    log.info("3/5 evaluation")
    jobs = [dict(csv=str(a.csv), start=a.start, end=a.end, out=str(a.out), env=env, seeds=seeds, days=n_days) for env in ENVS]
    with ProcessPoolExecutor(max_workers=min(a.workers, len(jobs))) as ex:
        for env, (kpis, eps, meta) in ex.map(_eval_job, jobs):
            kpis.to_csv(a.out / "data" / f"kpis_{env}.csv", index=False)
            with open(a.out / "data" / f"episodes_{env}.pkl", "wb") as f:
                pickle.dump((eps, meta), f)

    if not a.no_sensitivity:
        log.info("4/5 sensitivity")
        run_sensitivity(a, seeds, n_sens).to_csv(a.out / "data" / "sensitivity.csv", index=False)

    log.info("5/5 figures")
    for p in draw_figures(a.out):
        print("figure:", p)


if __name__ == "__main__":
    main()
