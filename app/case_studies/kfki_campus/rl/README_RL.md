# Reinforcement learning of EV-charger control (KFKI campus)

> **New to reinforcement learning?** Start with the learning path right below, then follow steps 0-10.
> Charging-event scenarios (fixed sessions from a file, random sessions, your own distributions) are in step 4.

## Before you start - learning reinforcement learning

You do not need to know RL to run this project, but you need it to *understand* what you are
doing. This is a beginner path; plan **about four weeks at a few hours per week**. Read in this
order and stop when you have what you need - the project only uses the basics (an environment,
a reward, the PPO algorithm).

### 1. Get the intuition (1-2 hours)

| Source | What for |
|---|---|
| Lilian Weng, ["A (Long) Peek into Reinforcement Learning"](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) (blog, 2018) | One readable overview: the vocabulary, Bellman equations, and the main families of methods. Skim the maths on a first read. |
| Hugging Face, [Deep Reinforcement Learning Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) (free, hands-on) | Follow the first unit(s): an agent is trained in a ready-made environment, which makes the loop observation -> action -> reward concrete. |

### 2. Learn the foundations (core, 2-4 weeks)

| Source | What for |
|---|---|
| Sutton & Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html), 2nd ed. (free online from the authors' page; the site's certificate sometimes triggers a browser warning) | The standard textbook. For this project **chapters 1 (introduction), 3 (finite MDPs), 6 (temporal-difference learning) and 13 (policy gradient methods)** are enough; do the small exercises. |
| David Silver, [RL lectures](https://davidstarsilver.wordpress.com/teaching/) (slides and videos) | Lecture 1 (introduction), 2 (Markov decision processes) and 7 (policy gradient methods) - the same ideas as the book, in lecture form. |
| OpenAI, [Spinning Up: Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) | A short, precise glossary: states and observations, action spaces, policies, reward and return, value functions. Keep it open while you read this guide. |

### 3. Get your hands dirty (alongside the reading)

| Source | What for |
|---|---|
| [Gymnasium documentation](https://gymnasium.farama.org/introduction/basic_usage/) | `reset` / `step`, observation and action spaces. Then its tutorial *Create a custom environment*: that is exactly what `OmnesEnv` is. |
| [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) docs, esp. [RL Tips and Tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) | The library that trains the PPO agent here; the tips page explains evaluation, normalisation and why one run proves nothing. |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | PPO in one readable file - read it once to see what "training" does. |

### 4. Go deeper (optional)

| Source | What for |
|---|---|
| Sergey Levine, [CS 285 Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) (UC Berkeley) | A full university course (policy gradients, actor-critic, value-based, model-based). |
| Schulman et al., ["Proximal Policy Optimization Algorithms"](https://arxiv.org/abs/1707.06347) (2017) | The paper behind the algorithm we train with. |

### 5. RL for energy systems

| Source | What for |
|---|---|
| Vázquez-Canteli & Nagy, "Reinforcement learning for demand response: A review of algorithms and modeling techniques", *Applied Energy* 235 (2019), 1072-1089, [doi:10.1016/j.apenergy.2018.11.002](https://doi.org/10.1016/j.apenergy.2018.11.002) | The standard review: what has been done, and the reproducibility problems of the field. |
| [CityLearn](https://github.com/intelligent-environments-lab/CityLearn) | An open-source Gymnasium environment for building and community energy coordination: compare its design with ours. |
| [Grid2Op](https://github.com/Grid2op/grid2op) and the L2RPN competitions | RL for power-grid operation; the grid-side counterpart of what we do with Pandapower. |

### RL vocabulary and where it lives in this project

| RL term | Meaning here | Where |
|---|---|---|
| environment | the campus: loads, PV, battery, chargers, network | `KfkiChargingEnv` (`rl/env.py`), base `OmnesEnv` |
| step | one time step of the model (1 h) | `OmnesEnv.step` |
| episode | one day (`episode_days`) | `reset(options={"day": ...})` |
| state / observation | what the agent sees: time, PV, load, battery, charger status | `ObsState`, observation terms (`observations.py`) |
| action | the fraction (0..1) of rated power of each charger | `action_space = Box(0, 1, (n_chargers,))` |
| reward | the negative cost of a step (SS, SC, penalties) | `RewardFunction` (`rewards.py`) |
| return | the sum of rewards of an episode | `EpisodeResult.reward` |
| policy | the rule action = f(observation) | `RLAgent.predict` |
| value function, advantage | the agent's estimate of future reward (inside the neural network) | Stable-Baselines3 (PPO) |
| discount factor | how much future reward counts (`gamma`) | `AgentSpec.gamma` |
| exploration | random actions while learning | PPO's stochastic policy; `deterministic=True` when evaluating |
| on-policy | PPO learns from data of its own current policy | PPO |
| training / test | days the agent learns on / days it never saw | `EpisodeSpec.train_fraction`, `split_seed` |
| baseline | a simple policy to beat | `uncontrolled`, `pv_following`, `oracle` (`rl/baselines.py`) |

### A four-week study path with exercises in this repo

| Week | Read / watch | Do |
|---|---|---|
| 1 | Weng's overview, Sutton & Barto ch. 1 and 3, Spinning Up *Key Concepts* | Run the smoke test of step 8 and open `rl_example_days.png` and `rl_kpi_comparison.png`: what do `uncontrolled` and `pv_following` do differently? |
| 2 | Sutton & Barto ch. 6, Gymnasium *Basic usage* | Read `OmnesEnv.step`; change **one** reward weight (step 6) and compare the `rl_reward_decomposition.png` before and after. |
| 3 | Sutton & Barto ch. 13, SB3 *RL Tips and Tricks* | Train PPO with two seeds (`--seeds 0 1`) and read the learning curve and `rl_seed_variability.png`. |
| 4 | Gymnasium *Create a custom environment*, Silver lecture 7 | Plug in an event distribution of your own (step 4.2, "B2") and compare the policies on it. |

### Common beginner mistakes

* **Reward is not the KPI.** The agent maximises the reward you wrote, not SC or SS. If the two disagree, the reward is wrong (step 6).
* **More training steps do not always help**, and a rising curve on the training days says nothing: look at the test days.
* **One run proves nothing.** Neural-network training is random; use at least three seeds before you claim a difference.
* **Test days must be unseen.** Never tune on the days you report.
* **Compare with a simple baseline** before celebrating: if `pv_following` is as good as the agent, the agent has learned nothing useful yet.
* **Do not trust a result on `--synthetic` data**: it only tests the tooling.

## What this guide does

A step-by-step guide to define, train and evaluate an RL controller for the EV chargers of the
KFKI campus REC. The agent decides **how much power each charger delivers in every time step**
(V1G smart charging, no discharging to the grid). It is rewarded for **self-sufficiency (SS)**
and **self-consumption (SC)** first, and for keeping voltages, transformers, lines and the
battery healthy second.

```
              +-------------------- OMNES Model (build_kfki_model) --------------------+
 sessions --> | loads, PV (2009 kWp), NaS battery, EV chargers  --PandapowerConverter--> net
 (file or     +-------------------------------------------------------------------------+
  generator)                |                                            |
                            v                                            v
   RLProblem -----> KfkiChargingEnv (Gymnasium) -- per step --> PowerFlowStepper (voltages, loadings, losses)
 (what to learn)            |   action -> feasible EV power, battery rule, campus balance
                            v
                  RLAgent (PPO) / baselines  -->  dispatch + SimResult  -->  compute_kpis  -->  figures
```

The RL environment is the third operation mode next to the optimiser: it produces the same
dispatch table and power-flow result as `EnergySystemOptimizer` + `simulate`, so SC, SS, voltage
violations and loadings are computed by the **same** `compute_kpis` and plotted with the same style.

| Module | Role |
|---|---|
| `app/operation/rl/problem.py` | `RLProblem` and its parts (the "DSL" of the RL problem; plain dataclasses) |
| `app/operation/rl/env.py` | `OmnesEnv`: generic Gymnasium environment (day split, windows, events, reward, spaces) |
| `app/operation/rl/rewards.py` | metric registry and `RewardFunction` (the payoff) |
| `app/operation/rl/observations.py` | observation-term registry |
| `app/operation/rl/events.py` | `ChargingEvent`, generators (`dummy`, `joint` stub, file) |
| `app/operation/rl/agent.py` | `RLAgent`, `RandomAgent`, `BasicPPOAgent` (Stable-Baselines3) |
| `rl/env.py` | `KfkiChargingEnv`: the campus physics |
| `rl/sessions.py` | charging-session file: read, validate, write |
| `rl/baselines.py` | uncontrolled, PV-following and oracle policies |
| `rl/train.py`, `rl/run_rl.py` | training, evaluation, experiments, CLI |
| `rl/plots_rl.py` | figures |
| `rl/data/make_rl_inputs.py` | writes `rl/data/rl.xlsx` and an **example** session file |
| `rl/examples/` | runnable examples of event distributions (scipy marginals, KDE, bootstrap, weekday generator, `.gmm` files), `SafeSampler`, sampler checks, `run_custom_events.py` |
| `dsl/model_example.txt` (`RL:` block) | syntax for the future DSL compiler |

---

## Step 0 - Install

```bash
pip install -r requirements.txt        # includes gymnasium (environment, baselines, evaluation)
pip install -r requirements-rl.txt     # adds Stable-Baselines3 + PyTorch (only for training PPO)
python -m app.case_studies.kfki_campus.rl.data.make_rl_inputs    # writes rl/data/rl.xlsx and an example session file
```

## Step 1 - Prepare the data

1. **Time-series CSV** - the case study's contract (`../README.md`): hourly, fixed UTC+1, columns
   `load_*`, `hp_*`, `pv_per_kwp`, `ev_charger_NN`. The `ev_charger_NN` columns are still
   required by the model builder but are **not used** by the RL environment (the chargers are
   driven by the sessions below).
2. **Charging sessions** (one CSV, long format, one row per plug-in). Timestamps are naive on the
   same fixed UTC+1 clock as the time-series CSV; `.` is the decimal separator.

   | column | required | meaning |
   |---|---|---|
   | `session_id` | yes | unique id |
   | `charger_id` | yes | `ev_c01` ... (`assets.xlsx:ev_chargers`); only the first `n_chargers` are used |
   | `plug_in` | yes | ISO 8601 connection time, e.g. `2025-06-02 08:15:00` |
   | `connected_h` | yes, or `plug_out` | how long the vehicle stays connected (h) |
   | `plug_out` | alternative | disconnection time; if both are given they must agree (1 min) |
   | `energy_kwh` | yes | energy to deliver **at the charger (AC side)**, > 0 |
   | `max_power_kw` | no | vehicle limit; default = charger rating |

   ```csv
   session_id,charger_id,plug_in,connected_h,energy_kwh,max_power_kw
   s0001,ev_c01,2025-06-02 08:15:00,9.5,24.0,
   s0002,ev_c02,2025-06-02 17:40:00,13.0,11.5,7.4
   ```

   `read_sessions` validates the file: missing columns, unknown charger, duplicate ids,
   non-positive values and overlapping sessions on one charger raise a `ValueError`. A session
   asking for more than rated power x connected time is **clipped** to what is deliverable and
   listed in the returned report (also logged); sessions outside the study period are dropped.
   `sessions_to_profile` turns the sessions into the old *uncontrolled* `ev_charger_NN` kW
   profile, so the optimisation pipeline can use the same file.
3. Without a session file the random generator is used. `rl/data/charging_sessions_example.csv`
   is generated by the dummy generator and only shows the format - it is **not** measured data.

## Step 2 - Set the constants

Everything below is a field of `RLProblem` (workbook `rl.xlsx`, Python, or the `RL:` block of the DSL)
and has a default. The **bold** ones almost always need a conscious choice.

| Constant | Where | Default | Notes |
|---|---|---|---|
| **built-in PV** | `scenario.added_pv_kwp` | 2000 | + 9 kWp existing = **2009 kWp**; split equally over the 5 roofs of `assets.xlsx:pv_sites` |
| battery | `scenario.battery_kwh` | 1450 | NaS; power scales with capacity (C-rate of the workbook) |
| **number of chargers** | `scenario.n_chargers` | 6 | max 10; charger k stands at site k mod 4 (`assets.xlsx:ev_sites`) |
| **charger power** | `action.charger_kw` | site value (22) | kW per charger / outlet; one `charger_id` = one outlet |
| time step | model `resolution` (`settings` sheet, `--resolution`) | `1h` | the environment step = one model step; 15 min needs a 15-min CSV |
| network limits | `settings` sheet | v 0.9 / 1.1 pu, strict 1.05 pu | KPIs and reward band |
| battery behaviour | `battery_policy` | `self_consumption_rule` | charge on PV surplus, discharge on deficit; `none` = battery idle |
| forecasts | `forecast_mode` | `perfect` | `persistence` = value of the previous day |
| power flow | `power_flow_mode` | `step` | `post` = no power flow during training (about 10x faster; voltage / loading terms are then 0) |

## Step 3 - Define the RL problem

### 3a. In Python

```python
from app.operation.rl.problem import RLProblem, RewardTerm, EpisodeSpec, EventSpec

problem = RLProblem(
    id="kfki_ev_v1g",
    scenario=dict(added_pv_kwp=2000.0, battery_kwh=1450.0, n_chargers=6),
    events=EventSpec(source="mixed", session_file="sessions.csv", p_generator=0.5),
    episode=EpisodeSpec(episode_days=1, train_fraction=0.7),
    reward=[RewardTerm("import_kwh", 1.0), RewardTerm("export_kwh", 1.0),
            RewardTerm("unmet_kwh", 5.0)],          # replaces the default payoff
)
```

### 3b. In the workbook (`rl/data/rl.xlsx`)

* `problem`: `key | value | source`, dotted keys (`scenario.n_chargers`, `episode.episode_days`,
  `events.source`, `agent.gamma`, ...); lists are JSON (`[128, 128]`), empty = `None`.
* `reward`: `metric | weight | transform | threshold | params | source` (`params` is JSON).
* `observation`: `name | params | source`.

```python
from app.case_studies.kfki_campus.rl.problem_io import load_problem
problem = load_problem("app/case_studies/kfki_campus/rl/data/rl.xlsx")
```

### 3c. In the DSL

`dsl/model_example.txt` shows the `RL:` block (Episode, Events, Action, Observe, Reward, Agent
groups). The compiler is under construction; the block uses the same keys as the workbook.

### The descriptor groups

| Group | Key | Default | Meaning |
|---|---|---|---|
| problem | `power_flow_mode`, `battery_policy`, `forecast_mode` | see step 2 | |
| episode | `episode_days` | 1 | window length; vehicles connected at the window start enter with pro-rata remaining energy, sessions leaving after the end are cut pro rata |
| | `mode` | `random_day` | `random_day` (train days at random), `sequential`, `full_period` |
| | `train_fraction`, `split_seed` | 0.7, 0 | seeded split of the days; `train_days` / `test_days` set them explicitly |
| events | `source` | `generator` | `file`, `generator`, `mixed` (`p_generator` = share of generated episodes) |
| | `session_file`, `generator`, `generator_params` | - | generator name from `GENERATOR_REGISTRY` (`dummy`, `joint`, or your own); the scenarios are explained in **step 4** (4.1 file, 4.2 random / own distribution, 4.3 mixed) |
| action | `charger_kw`, `safety_layer` | site, `none` | `laxity` forces the minimum power that still delivers the energy in time |
| observation | list of terms | see step 5 | |
| reward | list of terms, `reward_baseline`, `reward_scale` | see step 6 | |
| agent | `algo`, `policy`, `total_timesteps`, `learning_rate`, `gamma`, `n_envs`, `seed`, `net_arch`, `n_steps`, `batch_size`, `ent_coef` | PPO, MlpPolicy, 200000, 3e-4, 0.995, 4, 0, [128,128], 512, 128, 0 | read by `BasicPPOAgent` |

Unknown names, negative weights and impossible values raise a `ValueError` when the problem is built.

## Step 4 - Charging-event scenarios

The agent can only learn from the vehicles it sees. A **charging event** (session) says: *when
a vehicle plugs in, how long it stays, how much energy it needs*. Where the events come from is
the most important modelling choice after the reward.

### 4.0 Overview

First, what the agent controls. The **action space** is `Box(0, 1, (n_chargers,))`: the fraction
of the rated power of each charger. The environment projects the request on what is physically
possible: no vehicle -> 0; at most the rated / vehicle power; at most the remaining energy of the
session. The request removed by this projection is the metric `clipped_kw`.

| Scenario | `events.source` | Events | Same every episode of a day? | Typical use | Needs |
|---|---|---|---|---|---|
| **A. Fixed sessions** | `file` | exactly the rows of your session file | yes | evaluation; training when behaviour is known | a session file |
| **B1. Dummy generator** | `generator` (`dummy`) | random, independent marginals | no (seeded) | first experiments, tests | nothing |
| **B2. Own distribution** | `generator` (`joint`, or your own class) | random, from your distribution | no (seeded) | training on realistic variety | a fitted distribution / sampler |
| **C. Mixed** | `mixed` | file with probability `1 - p_generator`, else generated | partly | robustness: learn from variety, check on reality | both |

Your own files (`.gmm` mixtures, session CSVs) have a short recipe in **4.4**.

What each source can teach, and its danger:

* **fixed** - the agent can *memorise* the days of the file; keep test days unseen (step 4.1, point 6);
* **random** - the agent learns a policy that works for *many* plausible days, but only as realistic as your distribution (step 4.2).

### 4.1 Scenario A - fixed charging sessions from a file

**1. What you need.** One row per plug-in, with the columns of step 1 (`session_id`, `charger_id`,
`plug_in`, `connected_h` or `plug_out`, `energy_kwh`, optional `max_power_kw`). Charger back-ends
usually export start time, end time and energy per session; turn that export into the file with
pandas (adapt the column names to your export - this snippet is illustrative):

```python
import pandas as pd

raw = pd.read_csv("export.csv")                       # one row per charging session
ids = {"Charger A": "ev_c01", "Charger B": "ev_c02"}  # your names -> ids of assets.xlsx:ev_chargers
sessions = pd.DataFrame(
    {
        "session_id": raw["Session ID"],
        "charger_id": raw["Charger"].map(ids),
        "plug_in": pd.to_datetime(raw["Start"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
        "plug_out": pd.to_datetime(raw["End"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
        "energy_kwh": raw["Energy (kWh)"],             # at the charger, AC side
    }
)
sessions.to_csv("sessions.csv", index=False)
```

Two things go wrong most often: the **clock** (the file must be on the fixed UTC+1 clock of the
time-series CSV, not on local time with daylight saving) and the **units** (kWh and hours, not Wh
and minutes).

**2. Validate it.** `read_sessions` checks the file and returns the sessions plus a report of
what it had to change:

```python
import pandas as pd
from app.case_studies.kfki_campus.excel_io import load_campus_data
from app.case_studies.kfki_campus.rl.env import charger_ratings
from app.case_studies.kfki_campus.rl.sessions import read_sessions

chargers = charger_ratings(load_campus_data(), 10)        # rated kW of every charger id
period = (pd.Timestamp("2025-05-28"), pd.Timestamp("2026-03-02"))
events, report = read_sessions("sessions.csv", chargers, period)
print(len(events), "sessions;", len(report), "adjusted or dropped")
print(report)        # session_id | issue | energy_kwh | energy_kept_kwh
```

A `ValueError` means the file is wrong (missing column, unknown charger, duplicate id, negative
value, overlapping sessions on one charger). A row in the report means the session was *kept but
adjusted* ("clipped to power x time": it asked for more energy than the charger can deliver while
the car stays) or *dropped* ("outside period"). Many clipped rows usually mean a wrong unit or
a wrong charger rating.

**3. Look at the data before you train.** The key quantity is the **laxity**: how many hours the
vehicle stays *beyond* the time it needs at full power. No laxity = no flexibility = nothing for
the agent to gain.

```python
import matplotlib.pyplot as plt

df = pd.DataFrame(
    [
        {
            "day": e.plug_in.date(),
            "energy_kwh": e.energy_kwh,
            "connected_h": e.connected_h,
            "laxity_h": e.connected_h - e.energy_kwh / chargers[e.charger_id],
        }
        for e in events
    ]
)
print(df.groupby("day").size().describe())                 # sessions per day
print(df[["energy_kwh", "connected_h", "laxity_h"]].describe())
df[["energy_kwh", "connected_h", "laxity_h"]].hist(bins=30)
plt.show()
```

**4. Use it.** The events enter through the `Setup`; the command line does it with
`--session-file`:

```bash
python -m app.case_studies.kfki_campus.rl.run_rl all --csv <csv> --start 2025-05-28 --end 2026-03-01 \
    --session-file sessions.csv --events file --out results_rl
```

or from Python:

```python
from app.operation.rl.problem import RLProblem, EventSpec
from app.case_studies.kfki_campus.rl.train import Setup, load_events, make_env

problem = RLProblem(scenario=dict(n_chargers=6), events=EventSpec(source="file"))
setup = Setup(csv, load_campus_data(), "2025-05-28 00:00", "2026-03-02 00:00")
setup.events = load_events("sessions.csv", problem, setup)   # validated like step 2
env = make_env(problem, setup, split="test")
```

(`EventSpec.session_file` / the workbook cell only document the path; the file is read where
`--session-file` or `load_events` says so.)

**5. What happens inside.** For an episode window the environment takes the sessions that
overlap it; the same day therefore always has the same sessions and the seed has no effect on
the events. A vehicle already connected when the window starts enters with the energy left
pro rata; a vehicle leaving after the end is cut pro rata. Only the first `n_chargers`
charger ids of the file are used (the others are ignored, not an error). On the time grid a
session occupies the steps from `floor(plug_in)` to `ceil(plug_out)` and its energy is clipped
to what the rated power can deliver in them - with 1 h steps a 20-minute stop looks like an
hour: use a finer resolution if that matters.

**6. Consequences for training and evaluation.** (a) The agent can learn the specific days:
keep the day split (`EpisodeSpec.train_fraction`, `split_seed`) and report **test days only**.
(b) A day without sessions teaches nothing but still costs training time. (c) The more distinct days
the file covers, the better; with only a few weeks of data use `mixed` (4.3) so the agent also
sees other days.

**7. Common problems.**

| Symptom | Cause | Fix |
|---|---|---|
| `unknown charger_id [...]` | id not in `assets.xlsx:ev_chargers` | map your names (point 1) |
| `overlapping sessions A / B on ev_c01` | two vehicles on one charger at once | one `charger_id` is one outlet: give the second vehicle its own id |
| `connected_h must be positive` | `plug_out` before `plug_in` (clock, date format) | check the time columns |
| report: "clipped to power x time" | energy in Wh, or time in minutes, or too low a charger rating (`action.charger_kw`) | check units and rating |
| report: "outside period" | the file covers more than `--start`..`--end` | normal; or widen the period |
| all policies identical | no laxity (cars stay only as long as they need) | look at `laxity_h` (point 3) |
| evaluation much better than a fresh test | the agent saw those days | evaluate on test days only |

**8. Exercise.** Compute the laxity distribution of your file. Predict which policy gains most
(`pv_following` wins when laxity is large *and* the sessions overlap the sun hours). Then run
`evaluate` and check your prediction in `rl_kpis.csv` (`ev_pv_share`, `ev_unmet_kwh`).

### 4.2 Scenario B - randomly sampled charging sessions

**1. How it works.** Every episode asks a *generator* for events:

```
 rng (seeded by reset(seed=...)) --> generator.sample(rng, start, end, chargers) --> [ChargingEvent, ...] --> environment
```

`sample` receives the environment's random-number generator, so a seed reproduces an episode
exactly, and different seeds give different days. The agent therefore never sees the same day
twice, which forces it to learn a *rule* instead of memorising days. Rule for everything below:
**use only the `rng` you are given** (never `np.random` or a generator created inside), otherwise
runs cannot be reproduced.

**B1. The dummy generator** (`events.generator = dummy`) draws, for every charger and day, a
Poisson number of arrivals, an hour of arrival from an hour-of-day profile, a lognormal energy,
and a connected time = the time at full power x `flex_factor` + a lognormal extra dwell time.
Its parameters go into `events.generator_params`:

| Parameter | Default | Effect |
|---|---|---|
| `arrivals_per_day` | 1.5 | mean plug-ins per charger on a weekday; raise = busier chargers |
| `weekend_factor` | 0.4 | weekend rate relative to weekdays |
| `energy_median_kwh`, `energy_sigma` | 14, 0.6 | typical energy and its spread (lognormal) |
| `energy_max_kwh` | 80 | upper cut |
| `flex_factor` | 1.3 | connected time / charging time: **the flexibility knob** (1.0 = no slack) |
| `dwell_median_h`, `dwell_sigma` | 2.0, 0.7 | extra time the car stays after it could have finished |
| `hour_weights_weekday`, `hour_weights_weekend` | peaks 8-9 h and 17 h | 24 weights for the arrival hour |

Start, energy and connected time are **independent** here: a big energy does not imply a long
stay (except through `flex_factor`). It is a placeholder for experiments - not for conclusions.

**B2. Plug in an existing distribution** - the main tutorial. You have a distribution of
(start, energy, connected time), maybe fitted by a colleague, and you want the environment to
draw its sessions from it. The tool is `JointDistributionGenerator(sampler, arrivals_per_day)`,
and everything you have to provide is the **sampler**.

*Step 1 - What form does your distribution have?*

| You have | Recipe |
|---|---|
| a fitted `scipy.stats` distribution for each variable (independent) | Recipe A: `MarginalsSampler` |
| a fitted *joint* object that can draw samples (`scipy.stats.gaussian_kde`, scikit-learn `GaussianMixture` / `KernelDensity`, a copula) | Recipe B: wrap its sampling call |
| a table of measured sessions | Recipe C: bootstrap, or fit a KDE to it (Recipe B) |
| only a function / formulas in numpy | Recipe D: write the `sample` method yourself |
| distributions that depend on the day type or the charger | Step 6: a generator subclass |

*Step 2 - The contract.* A sampler is any Python object with one method:

```python
sampler.sample(n, rng) -> (start_hour, energy_kwh, connected_h)
```

* `n` is how many sessions to draw; return **three arrays of length n**;
* `start_hour`: hour of the day, `0 <= x < 24` (8.5 = 08:30);
* `energy_kwh`: energy to deliver, **kWh, > 0**;
* `connected_h`: how long the vehicle stays, **hours, > 0**;
* use **only `rng`** (a `numpy.random.Generator`); be vectorised; keep no state between calls.

What the generator does with it, and what it does *not* do: for every charger and every day (starting
one day before the window, so that vehicles connected at midnight exist) it draws a Poisson
number of arrivals with mean `arrivals_per_day` and asks the sampler for exactly that many
sessions. It does **not** validate the sampler's output (a negative energy would become a
negative-energy session), the sampler does not know the day, the charger or its rated power, and
the arrival count does not come from your distribution - if any of that matters, use Step 6.
Sessions that would overlap on one charger are dropped (the earlier wins), so a too-high
`arrivals_per_day` loses events.

Because of the missing validation, wrap your sampler in `SafeSampler`
(`rl/examples/_guards.py`): it checks the shape, **redraws** rows with non-positive or non-finite
energy / time, and wraps the hour into `[0, 24)`.

*Step 3 - Write the sampler.* Four recipes; each is a runnable file in
`rl/examples/` (try it with `run_custom_events`, step 7).

**Recipe A - independent `scipy.stats` marginals** (`ex_scipy_marginals.py`):

```python
class MarginalsSampler:
    """Independent start hour, energy and connected time.

    :ivar np.ndarray hour_weights: 24 weights, arrival probability per hour of the day
    :ivar energy: Frozen scipy distribution of the energy (kWh)
    :ivar dwell: Frozen scipy distribution of the connected time (h)
    """

    def __init__(self, hour_weights, energy, dwell):
        w = np.asarray(hour_weights, dtype=float)
        self.hour_p = w / w.sum()
        self.energy = energy
        self.dwell = dwell

    def sample(self, n, rng):
        hour = rng.choice(24, size=n, p=self.hour_p) + rng.random(n)  # hour + minutes
        energy = self.energy.rvs(size=n, random_state=rng)  # always pass the given rng
        dwell = self.dwell.rvs(size=n, random_state=rng)
        return hour, energy, dwell
```

The parameters of the frozen distributions come from your fit, e.g.
`stats.lognorm(s=0.6, scale=12.0)` (median 12 kWh) and `stats.gamma(a=3.0, scale=1.5)`
(mean 4.5 h). Pass `random_state=rng` to every `.rvs` call.

**Recipe B - a fitted joint object** (`ex_kde_joint.py`): a kernel-density estimate of
(start hour, log energy, log connected time) keeps the correlations of your data:

```python
class KdeSampler:
    """Joint density of (start hour, log energy, log connected time).

    Logs keep energy and time positive after the transformation back; ``SafeSampler``
    wraps the start hour into [0, 24) and redraws anything that is still invalid.

    :param pd.DataFrame sessions: Columns ``plug_in``, ``connected_h``, ``energy_kwh``
    """

    def __init__(self, sessions: pd.DataFrame):
        t = pd.to_datetime(sessions["plug_in"])
        hour = t.dt.hour + t.dt.minute / 60.0
        data = np.vstack(
            [hour, np.log(sessions["energy_kwh"]), np.log(sessions["connected_h"])]
        )
        self.kde = stats.gaussian_kde(data)  # data shape: (variables, observations)

    def sample(self, n, rng):
        hour, log_energy, log_dwell = self.kde.resample(n, seed=rng)  # pass the rng
        return hour, np.exp(log_energy), np.exp(log_dwell)
```

Any fitted object that can draw samples fits the same pattern: call its drawing method inside
`sample`. Two traps: (1) it must use *your* `rng` - scikit-learn's `GaussianMixture.sample` does
not accept one, and a fixed `random_state` would return the same numbers at every call (see the
commented variant in `ex_kde_joint.py`); (2) a normal or KDE model can give negative values -
modelling the **logarithm** (as above) or using `SafeSampler` avoids that.

**Recipe C - bootstrap from a table of sessions** (`ex_bootstrap.py`): draw whole rows, so every
correlation in the data survives:

```python
class BootstrapSampler:
    """Resample rows of a sessions table, with a little noise so that days differ.

    :param pd.DataFrame sessions: Columns ``plug_in``, ``connected_h``, ``energy_kwh``
    :param float jitter: Relative noise of energy and connected time (0 = exact copies)
    :param float hour_jitter: Noise of the start hour in hours
    """

    def __init__(
        self, sessions: pd.DataFrame, jitter: float = 0.1, hour_jitter: float = 0.25
    ):
        t = pd.to_datetime(sessions["plug_in"])
        self.hour = (t.dt.hour + t.dt.minute / 60.0).to_numpy()
        self.energy = sessions["energy_kwh"].to_numpy(dtype=float)
        self.dwell = sessions["connected_h"].to_numpy(dtype=float)
        self.jitter, self.hour_jitter = jitter, hour_jitter

    def sample(self, n, rng):
        i = rng.integers(0, len(self.hour), size=n)  # whole rows keep the correlations
        noise = lambda: rng.lognormal(0.0, self.jitter, size=n)  # noqa: E731
        hour = self.hour[i] + rng.normal(0.0, self.hour_jitter, size=n)
        return hour, self.energy[i] * noise(), self.dwell[i] * noise()
```

Use it with your own file: `BootstrapSampler(pd.read_csv("sessions.csv"))`.

**Recipe D - your own function.** Anything with the right `sample` method works, for example a
sampler from formulas:

```python
class FormulaSampler:
    def sample(self, n, rng):
        hour = rng.normal(8.5, 1.5, n)                       # arrivals around 08:30
        energy = rng.gamma(shape=3.0, scale=5.0, size=n)     # kWh, mean 15
        dwell = 8.0 + rng.normal(0.0, 1.0, n)                # about a working day
        return hour, energy, dwell                           # SafeSampler fixes hour/sign problems
```

*Step 4 - Check the sampler before you use it.* Never train on a sampler you have not looked at:

```python
from app.case_studies.kfki_campus.rl.examples._guards import SafeSampler, check_sampler, plot_sampler

sampler = SafeSampler(FormulaSampler())
summary, corr = check_sampler(sampler, n=10_000)
print(summary)    # min / 5% / median / 95% / max per variable, share_non_positive must be 0
print(corr)       # do start, energy and connected time correlate as in your data?
plot_sampler(sampler, "sampler_check.png")
```

Check: hours in [0, 24); energies of a few to a few tens of kWh (not 15 000: Wh!); connected times of
minutes to a day *in hours* (not 480: minutes!); the histograms look like your data.

*Step 5 - Wrap it and look at the events.* Estimate `arrivals_per_day` from your data (sessions per
charger per day: `len(sessions) / (n_chargers * n_days)`), then:

```python
import numpy as np
import pandas as pd
from app.operation.rl.events import JointDistributionGenerator

generator = JointDistributionGenerator(sampler, arrivals_per_day=1.5)
events = generator.sample(
    np.random.default_rng(0),
    pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-09"),     # one week
    {"ev_c01": 22.0, "ev_c02": 22.0},                           # charger id -> rated kW
)
print(len(events), "events;", sum(e.energy_kwh for e in events), "kWh")
```

Run it twice with the same seed: you must get the same events; with another seed, other ones.

*Step 6 - When a sampler is not enough: write a generator.* If the distribution depends on the
weekday, the charger, or if the number of arrivals itself should follow your data, subclass
`ChargingEventGenerator` (`ex_weekday_generator.py`): it owns the loop over days and chargers.

```python
class WeekdayGenerator(ChargingEventGenerator):
    """Weekday / weekend samplers and a plug-in rate per charger.

    :ivar weekday: Sampler used Monday-Friday
    :ivar weekend: Sampler used Saturday-Sunday
    :ivar dict rates: Mean plug-ins per day for each charger id (``default_rate`` otherwise)
    :ivar float weekend_factor: Weekend rate relative to the weekday rate
    """

    def __init__(
        self,
        weekday,
        weekend,
        rates: Optional[dict] = None,
        default_rate: float = 1.5,
        weekend_factor: float = 0.4,
    ):
        self.weekday, self.weekend = weekday, weekend
        self.rates = rates or {}
        self.default_rate = default_rate
        self.weekend_factor = weekend_factor

    def sample(self, rng, start, end, chargers):
        # start one day early: vehicles connected at ``start`` came the day before
        days = pd.date_range(start.floor("D") - pd.Timedelta(days=1), end, freq="D")
        events = []
        for cid in chargers:
            for day in days:
                weekend = day.dayofweek >= 5
                rate = self.rates.get(cid, self.default_rate)
                n = rng.poisson(rate * (self.weekend_factor if weekend else 1.0))
                if n == 0:
                    continue
                hour, energy, dwell = (
                    self.weekend if weekend else self.weekday
                ).sample(n, rng)
                for k in range(n):
                    events.append(
                        ChargingEvent(
                            f"{cid}_{day:%Y%m%d}_{k}",
                            cid,
                            day + pd.Timedelta(hours=float(hour[k])),
                            float(dwell[k]),
                            float(energy[k]),
                        )
                    )
        events = drop_overlaps(events)  # one vehicle per charger at a time
        return [e for e in events if e.plug_out > start and e.plug_in < end]
```

Register it once, then it can be named in `events.generator`:

```python
GENERATOR_REGISTRY["weekday"] = (
    WeekdayGenerator  # now usable as ``events.generator: weekday``
)
```

*Step 7 - Run training and evaluation with it.* The command line builds the problem from the
workbook, and a workbook holds text only - it can name a registered generator (`events.generator:
weekday`) but cannot carry a *fitted Python object*. So hand the object over from a short script.
The trick is to register a factory under a name and use that name:

```python
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
```

`rl/examples/run_custom_events.py` does this end to end (synthetic data, the chosen example
generator, the reference policies, optional PPO training, figures):

```bash
python -m app.case_studies.kfki_campus.rl.examples.run_custom_events --generator kde --out results_custom
python -m app.case_studies.kfki_campus.rl.examples.run_custom_events --generator weekday --train-steps 5000 --out results_custom
```

Copy the script, replace the example generator by yours and the synthetic CSV by your time series
(`Setup(csv, data, start, end)`); the functions `train_agent`, `evaluate_all`, `sweep_weight` and
`generalization` of `rl/train.py` accept the same `problem` / `setup`.

*Step 8 - Check the result.*

* Events per day and charger vs your data (Step 5); the share dropped by overlaps (compare the number of sessions the generator returns with `arrivals_per_day * days * chargers`);
* `uncontrolled` should have **~0 unmet energy** in `rl_kpis.csv`; if not, the sessions are infeasible (energy too large for the stay) - check the sampler;
* `ev_pv_share` and the example-days figure: do the sessions overlap the sun hours at all?
* the reward decomposition: is `unmet_kwh` or a grid term dominating for a reason you understand?
* the generalisation test (4.3).

*Troubleshooting.*

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: sampler returned too many invalid rows` | sampler gives mostly non-positive energy / time (units, wrong sign) | `check_sampler`; model logs; fix the units |
| `ValueError: sampler must return 3 arrays of length n` | wrong return shape or order | return `(start_hour, energy_kwh, connected_h)` as 1-D arrays |
| far fewer events than expected | arrivals overlap on one charger (connected times too long for the rate) | lower `arrivals_per_day` or the dwell times |
| all sessions start at the same hour | `start_hour` in the wrong unit (minutes, or a fraction of the day) | hours, 0..24 |
| `uncontrolled` has unmet energy | energy larger than rated power x stay | smaller energies / longer stays / check units |
| two runs with one seed differ | sampler uses `np.random` or its own generator | use only the given `rng` (also in scikit-learn models: reseed from it) |
| every day looks identical | a fixed `random_state` inside a fitted object | reseed it from `rng` at each call |
| pickling error (only if you switch training to `SubprocVecEnv`) | the generator object cannot be pickled | define sampler and generator classes in a module, not interactively |

### 4.3 Scenario C - mixed sources and how to design the experiment

`events.source = mixed` takes the file with probability `1 - p_generator` and a generated
episode otherwise (this needs a session file *and* a generator). It is the standard choice when
you have real sessions for only some days.

A protocol that holds up:

| Question | Train on | Test on | How |
|---|---|---|---|
| Does smart charging help on real behaviour? | file (train days) | file (test days) | `run_rl all --events file` |
| Does the agent cope with variety? | generator | generator (other seeds) | `--events generator` |
| Does it transfer from random to real? | generator or mixed | **file** (test days) | `run_rl generalization --session-file ...` |
| How much does the distribution matter? | generator A, generator B | the same file | repeat with two generators (step 7 script) |

`run_rl generalization` trains one agent per source (`file`, `generator`, `mixed`) and tests each on
`file` and on `generator` events; `rl_generalization.png` shows reward, SC, SS and unmet energy as
a train-source x test-source matrix. How to read it: the **diagonal** shows how well an agent does
on what it learned; the **off-diagonal** cells show transfer - an agent that is good on its own
source and bad on the other has learned the source, not charging. With a custom generator call
`generalization(problem, setup, out_dir)` from your script (step 7).

Use at least three seeds (`--seeds 0 1 2`), report test days only, and always include
`uncontrolled` and `pv_following` in the comparison.

### 4.4 Plug in your own files: `.gmm` mixtures and `session.csv` fixed events

A short, step-by-step recipe for the two kinds of files of this project. Background on the
sampler contract and the traps is in 4.2; the file format in step 1 and 4.1.

| You have | Used for | Route |
|---|---|---|
| `workday.gmm` and `weekend.gmm` - joblib dumps of a scikit-learn `GaussianMixture` over **(start hour, energy kWh, connected hours)**, raw values, the same for all chargers | random events (training and/or testing) | Recipe 1 |
| one or more session CSV files (fixed events) | evaluation on "reality"; training when behaviour is known | Recipe 2 |
| both | **train on GMM events, test on the fixed sessions** (recommended), or train on `mixed` | Recipe 3 |

**Prerequisite:** `pip install -r requirements-rl.txt` (it contains scikit-learn). Install the
**same scikit-learn version that saved the `.gmm` files** - pickles are version-sensitive - and open
only files you trust (pickle can run code).

#### Recipe 1 - GMM files for random events

**1. Look inside the file.** `load_gmm` loads it and checks that it is a fitted mixture over three
dimensions:

```python
def load_gmm(path):
    """Load a ``.gmm`` file and check that it is a mixture over (start hour, energy, time).

    Only open files you trust: pickle can execute code.
    """
    import joblib  # installed with scikit-learn

    gmm = joblib.load(path)
    means = getattr(gmm, "means_", None)
    if means is None or means.shape[1] != 3:
        raise ValueError(
            f"{path}: expected a fitted GaussianMixture over 3 dimensions "
            f"(start hour, energy kWh, connected h), got {type(gmm).__name__}"
            + ("" if means is None else f" with {means.shape[1]} dimensions")
        )
    return gmm
```

```python
from app.case_studies.kfki_campus.rl.examples.ex_gmm_files import load_gmm

gmm = load_gmm("workday.gmm")
print(gmm.weights_.round(2))   # component weights
print(gmm.means_.round(1))     # one row per component: start hour, energy kWh, connected h
```

The means must look like *start hour, kWh, hours* (for example `8.5, 12.0, 4.0`). Energy means of
about `2.5` or connected means of `1.4` would mean the mixture was fitted on **logarithms**; means in
the hundreds would mean Wh or minutes; a different column order shows up here too. In those cases
do not continue - this recipe expects raw values in the order (start hour, energy, connected time);
ask the author of the mixture or adapt `GmmSampler` (exp / unit conversion / reorder columns).

**2. Check the raw mixture.** A Gaussian mixture over raw values can produce negative energies and
times and hours outside 0..24. Look at how often:

```python
from app.case_studies.kfki_campus.rl.examples._guards import check_sampler
from app.case_studies.kfki_campus.rl.examples.ex_gmm_files import GmmSampler

summary, corr = check_sampler(GmmSampler(gmm), n=10_000)
print(summary[["min", "5%", "50%", "95%", "max", "share_non_positive"]])
x, _ = gmm.sample(10_000)
print("hours outside 0..24:", ((x[:, 0] < 0) | (x[:, 0] >= 24)).mean())
```

Few percent are normal (the generator redraws non-positive energy / time - the same as truncating
the mixture - and wraps hours into 0..24). **More than about 10-20 %** means the mixture does not
describe this variable set well: stop and check the units (step 1).

**3. Decide the arrival rates.** A mixture says *what* a session looks like, not *how many* sessions
there are. The generator draws, per charger and day, a Poisson number of arrivals with the mean
`workday_rate` (Mon-Fri) or `weekend_rate` (Sat-Sun). If you have a session table, take the rates
from it:

```python
def arrival_rates(sessions: pd.DataFrame) -> tuple[float, float]:
    """Mean plug-ins per charger and day on workdays and on weekends, from a session table.

    :param pd.DataFrame sessions: Columns ``charger_id`` and ``plug_in`` (the session file)
    :returns: ``(workday_rate, weekend_rate)``
    """
    t = pd.to_datetime(sessions["plug_in"])
    days = pd.date_range(t.min().normalize(), t.max().normalize(), freq="D")
    per_day = t.dt.normalize().value_counts().reindex(days, fill_value=0)
    n_chargers = sessions["charger_id"].nunique()
    workday = per_day[days.dayofweek < 5].mean() / n_chargers
    weekend = per_day[days.dayofweek >= 5].mean() / n_chargers
    return float(workday), float(weekend)
```

```python
import pandas as pd
from app.case_studies.kfki_campus.rl.examples.ex_gmm_files import arrival_rates

workday_rate, weekend_rate = arrival_rates(pd.read_csv("sessions.csv"))
print(workday_rate, weekend_rate)   # plug-ins per charger and day
```

Without data, assume a value, say so, and test the sensitivity (for example 0.5x, 1x, 2x).

**4. Build the generator.** `GmmFilesGenerator` takes the two files and the two rates; it picks the
workday or the weekend mixture by the day of the week and uses it for **every** charger:

```python
class GmmSampler:
    """Draw (start hour, energy kWh, connected h) from a fitted ``GaussianMixture``.

    The mixture is over the raw values, so it can return negative energies or times and hours
    outside 0..24: wrap it in ``SafeSampler`` (it redraws those rows).
    """

    def __init__(self, gmm):
        self.gmm = gmm

    def sample(self, n, rng):
        # GaussianMixture.sample cannot take our rng: reseed it from it at every call,
        # otherwise every call (every day) would return the same numbers.
        self.gmm.random_state = int(rng.integers(2**31))
        x, _ = self.gmm.sample(n)
        return x[:, 0], x[:, 1], x[:, 2]
```

```python
class GmmFilesGenerator(WeekdayGenerator):
    """Events from a workday and a weekend ``.gmm`` file (the same for all chargers).

    :param workday: Path of the workday ``.gmm`` file
    :param weekend: Path of the weekend ``.gmm`` file
    :param float workday_rate: Mean plug-ins per charger on a workday
    :param float weekend_rate: Mean plug-ins per charger on a weekend day
    """

    def __init__(self, workday, weekend, workday_rate=1.5, weekend_rate=0.6):
        super().__init__(
            SafeSampler(GmmSampler(load_gmm(workday))),
            SafeSampler(GmmSampler(load_gmm(weekend))),
            default_rate=workday_rate,
            weekend_factor=weekend_rate / workday_rate,
        )


GENERATOR_REGISTRY["gmm_files"] = GmmFilesGenerator  # usable as ``events.generator``
```

```python
import numpy as np
import pandas as pd
from app.case_studies.kfki_campus.rl.examples.ex_gmm_files import make_generator

generator = make_generator("workday.gmm", "weekend.gmm", workday_rate, weekend_rate)
events = generator.sample(
    np.random.default_rng(0),
    pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-16"),   # two weeks, a Monday start
    {"ev_c01": 22.0, "ev_c02": 22.0},                         # charger id -> rated kW
)
print(len(events), "events;", round(sum(e.energy_kwh for e in events)), "kWh")
```

The sampler reseeds the mixture from the environment's random-number generator at every call (a
fixed `random_state` would repeat the same numbers every day); the same seed gives the same events.

**5. Compare with your data.** Count events per day type and compare with the session table, and
compare the energy / connected-time distributions (`check_sampler`, `plot_sampler` from 4.2).
`uncontrolled` should show ~0 unmet energy later: if not, the sessions ask for more energy than
the stay allows - check the units.

**6. Run it.** The command line cannot carry the objects, so use the script. With your time series:

```bash
python -m app.case_studies.kfki_campus.rl.examples.run_custom_events \
    --generator gmm --gmm-workday workday.gmm --gmm-weekend weekend.gmm \
    --workday-rate 1.8 --weekend-rate 0.7 \
    --csv <csv> --start 2025-05-28 --end 2026-03-01 --chargers 6 \
    --train-steps 200000 --out results_gmm
```

Without `--csv` the script uses placeholder data (good for a first try, `--days 6`). It evaluates
`uncontrolled`, `pv_following` and `random` (and the trained agent with `--train-steps`) on the
test days and writes `rl_kpis.csv` and the figures into `--out`.

**7. (Optional) Name it in the workbook.** `GmmFilesGenerator` is registered as `gmm_files`, so the
problem can say which files to use - paths are plain text, the workbook can hold them. In
`rl.xlsx`, sheet `problem`: `events.generator` = `gmm_files` and `events.generator_params` =
`{"workday": "workday.gmm", "weekend": "weekend.gmm", "workday_rate": 1.8, "weekend_rate": 0.7}`.
The module must be imported for the name to exist (`run_rl` does not import it; do it in your own
script with `import app.case_studies.kfki_campus.rl.examples.ex_gmm_files`).

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: ... expected a fitted GaussianMixture over 3 dimensions` | wrong object or a mixture with other variables | check what was saved (step 1) |
| `InconsistentVersionWarning` or an error while loading | scikit-learn version differs from the one that saved the file | install that version, or re-save the mixture in a version-independent way (parameters as arrays) |
| `RuntimeError: sampler returned too many invalid rows` | most samples are non-positive (log scale, Wh / minutes) | step 1 and 2 |
| all days look alike | the mixture was loaded with a fixed seed somewhere else | use `GmmSampler` (reseeds from our rng) |
| far fewer events than expected | rates too low, or sessions overlap on a charger and are dropped | check the rates (step 3); long connected times with high rates lose events |
| weekend and workday identical | the files are swapped or the day-of-week is not what you think | check the means of both files |

#### Recipe 2 - `session.csv` files with fixed events

1. **Check the columns**: `session_id`, `charger_id`, `plug_in`, `connected_h` (or `plug_out`),
   `energy_kwh`, optional `max_power_kw`; times on the fixed UTC+1 clock, kWh and hours (step 1).
   If your files have other column names, rename them first (4.1, point 1).
2. **One file for the pipeline.** The reader takes one CSV. If you have several (per month or per
   charger), merge them and make sure the ids stay unique:

   ```python
   import glob
   import pandas as pd

   parts = [pd.read_csv(f) for f in sorted(glob.glob("sessions_*.csv"))]
   sessions = pd.concat(parts, ignore_index=True)
   assert not sessions["session_id"].duplicated().any(), "duplicate session ids across files"
   sessions.to_csv("sessions_all.csv", index=False)
   ```

3. **Validate and look at it** with `read_sessions` and the laxity snippet of 4.1 (points 2 and 3):
   a `ValueError` is a problem of the file; rows in the report were clipped or dropped.
4. **Run on the fixed sessions only** with the command line:

   ```bash
   python -m app.case_studies.kfki_campus.rl.run_rl all --csv <csv> --start 2025-05-28 --end 2026-03-01 \
       --session-file sessions_all.csv --events file --chargers 6 --out results_fixed
   ```

   Only the first `--chargers` charger ids are used; the others in the file are ignored.
5. Remember (4.1, points 5-6): the days of the file are the same in every episode - evaluate on
   test days only.

#### Recipe 3 - train on the GMM events, test on the fixed sessions

This answers "does what the agent learned on random events work on real days?":

```bash
python -m app.case_studies.kfki_campus.rl.examples.run_custom_events \
    --generator gmm --gmm-workday workday.gmm --gmm-weekend weekend.gmm \
    --workday-rate 1.8 --weekend-rate 0.7 \
    --csv <csv> --start 2025-05-28 --end 2026-03-01 --chargers 6 \
    --session-file sessions_all.csv --events generator --eval-events file \
    --train-steps 200000 --out results_gmm_vs_file
```

* `--events` decides the training events (`generator` = GMM, `mixed` = GMM and file, `file`);
* `--eval-events` decides the evaluation events (`file` = the fixed sessions); the reference policies
  run on exactly the same days and sessions;
* use `--events mixed` when the file covers only part of the period;
* repeat with different `--workday-rate` / `--weekend-rate` to see how much the result depends on
  the rates, and use several seeds (step 8) before you claim a difference.

## Step 5 - Observation

The observation is the concatenation of the listed terms (all roughly in [-1, 1]); the order
is the order of the list. Terms (`OBSERVATION_REGISTRY`):

`time_sin_cos`, `weekday`, `pv_now`, `pv_forecast{k}`, `base_load_now`, `base_load_forecast{k}`,
`net_residual` (base load - PV), `battery_soc`, `trafo_loading_prev`, `v_max_prev`, and per
charger: `charger_connected`, `charger_energy_remaining`, `charger_time_remaining`,
`charger_laxity` (time left - time needed at full power), `charger_p_prev`.

Add a term with `@register_observation("name", size_fn)` in `observations.py`. The agent
cannot see what is not here: no PV forecast = it cannot wait for the sun.

## Step 6 - The payoff (reward)

`r_t = -( C_t - C_ref,t ) / E_ref`, with `C_t = sum_k  w_k * transform_k( metric_k )`.
All costs are in **kWh-equivalents** and `E_ref` (default: rated energy of all chargers per step)
makes the reward O(1).

| Term (`metric`) | per-step quantity | default w | Purpose |
|---|---|---|---|
| `import_kwh` | grid import energy | **1.0** | **SS**: summed over an episode it equals (1 - SS) x demand energy |
| `export_kwh` | grid export energy | **1.0** | **SC**: summed it equals (1 - SC) x PV energy (large with 2009 kWp at noon) |
| `unmet_kwh` | requested energy not delivered at departure | **5.0** | must stay **above** the import weight, otherwise not charging looks like an SS gain |
| `voltage_excess_pu` | sum over buses of the voltage beyond `v_hi`=1.05 / `v_lo`=0.95 pu | 200 | EN 50160 band (hard limits 0.9 / 1.1 stay in the KPIs) |
| `trafo_overload_kvah` | transformer energy above the rating | 3.0 | avoid transformer overloading |
| `line_overload_kvah` | line energy above the thermal limit | 3.0 | avoid line overloading |
| `battery_throughput_kwh` | battery charge + discharge energy | 0.1 | cycling / ageing |
| `losses_kwh` | line + transformer losses | 0.5 | |
| `ramp_kw` | sum of changes of charger power | 0.005 | no chattering |
| `clipped_kw` | request removed by the projection | 0 (off) | teaches feasibility |

* `transform`: `linear` (default), `square`, `hinge` (`max(m - threshold, 0)`).
* `reward_baseline = exogenous` adds back `C_ref`, the import / export cost of the campus **without
  EV and battery**. It depends only on data and time, not on the policy, so the optimal policy is
  unchanged, but the large load/PV swing of the campus (MW) disappears from the signal that has to
  be learned from a few tens of kW of EV flexibility.
* Because the dense import / export terms add up to the KPIs, no extra sparse "final SC/SS" bonus is needed.
* Voltage, transformer, line and loss terms only exist with `power_flow_mode = step`.
* **Tuning rules**: (1) `unmet_kwh` > `import_kwh`; (2) grid-protection weights large enough that
  one overload step costs more than the energy it could save; (3) battery / ramp terms small;
  (4) change one weight at a time (`run_rl sweep`).
* New metric: `@register_metric("name")` in `rewards.py`, taking `(ctx, params)`; the context keys
  are listed in the module docstring.

## Step 7 - The agent

`RLAgent` has `bind`, `train`, `predict`, `save`, `load`. `BasicPPOAgent` wraps Stable-Baselines3
(`AgentSpec`: algorithm, policy, learning rate, gamma, ...). To specialise: subclass it and override
`build(env)` (algorithm, `policy_kwargs`, custom extractor) or register your own class in
`AGENT_REGISTRY` and set `agent.algo`. The comparison policies use the same interface:

| Policy | Behaviour |
|---|---|
| `uncontrolled` | full power from plug-in (what the legacy profiles do) |
| `pv_following` | charge with the PV surplus over the rest of the campus, but at least the power that still finishes the session in time |
| `oracle` | perfect-foresight LP of the EV power for the episode (minimises import + export, no network terms, battery by the same rule); a benchmark, not a strict bound |
| `random` | uniform setpoints |

## Step 8 - Train, evaluate, draw

```bash
# everything: train 3 seeds, evaluate on the test days against the baselines, figures
python -m app.case_studies.kfki_campus.rl.run_rl all \
    --csv app/case_studies/kfki_campus/data/processed/kfki_timeseries_2025-05-28_2026-03-01.csv \
    --start 2025-05-28 --end 2026-03-01 --session-file my_sessions.csv \
    --events mixed --seeds 0 1 2 --timesteps 200000 --out results_rl

python -m app.case_studies.kfki_campus.rl.run_rl evaluate ...      # re-evaluate saved models
python -m app.case_studies.kfki_campus.rl.run_rl figures --out results_rl
python -m app.case_studies.kfki_campus.rl.run_rl sweep --sweep unmet_kwh:1,2,5,10 ...        # reward-weight sweep
python -m app.case_studies.kfki_campus.rl.run_rl generalization --session-file my_sessions.csv ...  # train/test event source
# smoke test with placeholder data
python -m app.case_studies.kfki_campus.rl.run_rl all --synthetic --start 2025-06-01 --end 2025-06-30 --timesteps 4000 --n-envs 2 --chargers 4 --out results_rl_smoke
```

Outputs in `--out`: `rl_kpis.csv` (policy x test day: SC, SS, import, export, EV unmet / satisfaction /
PV share, voltages, loadings, losses, battery cycles, reward by term), `rl_summary.csv`,
`rl_full_period.csv` and `details/<policy>.npz` (one continuous run per policy, same keys as the
scenario sweep plus `ev_power`), `models/*.zip` and `*_curve.csv`, `episodes.pkl`, `figures/`, `logs/`.

**Where are my files?** A relative `--out` (default `results_rl`) is relative to the *working
directory* of the process (in an IDE: the run configuration's working directory), not to the
script. `run_rl` prints `results directory: <absolute path>` first and the figures with
absolute paths. The figures are in `<out>/figures/`. Note that the repository's `.gitignore`
excludes `*.png`, `*.csv`, `*.npz` and `*.pdf`: tools that honour it (the Git/commit view of an
IDE, some file trees and searches) do **not show** these results although they are on disk, and
they are never committed.

| Figure | What it shows |
|---|---|
| `rl_learning_curve` | test reward, SC, SS, unmet energy vs training steps (band = seeds) |
| `rl_kpi_comparison` | KPI distributions over the test days, all policies |
| `rl_reward_decomposition` | which term costs each policy how much reward |
| `rl_example_days` | EV power vs PV surplus on a high-, median- and low-PV day |
| `rl_ev_carpet` | EV power by hour of day x test day, per policy |
| `rl_session_gantt` | sessions of one day (connected window, power) per charger |
| `rl_session_satisfaction` | delivered / requested per session, unmet-energy CDF |
| `rl_grid_duration` | duration curves of the highest voltage and transformer loading |
| `rl_policy_diagnostics` | power used vs PV surplus, per policy (what the agent learned) |
| `rl_seed_variability` | spread between seeds |
| `rl_sweep_pareto` | SS / SC vs unmet energy / violations along a reward weight |
| `rl_generalization` | train source x test source (file / generator / mixed) |

Fair comparison: all policies run the same test days with the same event seeds. The test days are a seeded
random sample of the data period; report them together with the results.

## Step 9 - What influences the outcome

| What | Effect | Symptom if wrong | Remedy |
|---|---|---|---|
| EV share of the campus load | The agent can only move 6 x 22 kW against a campus of ~1-2 MW: SC / SS change in the 3rd decimal; the clearest differences are in `ev_pv_share`, peak EV power and unmet energy | "no difference between policies" in SC / SS | read the EV-specific KPIs and the reward decomposition; increase chargers or flexibility to see larger effects |
| Flexibility of the sessions | `connected_h` relative to the charging time decides how far charging can be shifted | all policies equal | check the distribution of laxity in the sessions |
| Time resolution | 1 h hides sub-hourly sessions and ramps | sessions look longer than they are | use a finer CSV and `resolution` |
| PV size (2009 kWp) | determines the surplus the EVs can absorb and the grid-side risk (overvoltage) | no surplus = nothing to learn for SC | `--pv-kwp`; check `export_kwh` of the uncontrolled policy |
| Battery policy | the rule already absorbs surplus; the EV benefit shrinks with a large battery | small differences | `battery_policy=none` or smaller `battery_kwh` to isolate the EV effect |
| `unmet_kwh` weight | too low: the agent skips charging to "improve" SS; too high: it charges as soon as possible | high unmet energy / no PV shift | keep above import weight; sweep it |
| Grid weights | too low: voltage / loading never matters; too high: dominates, the agent learns the safety terms only | violations or flat policy | check `viol_frac`, sweep |
| `power_flow_mode` | `post` is 10x faster but blind to voltages and loadings | agent ignores the grid | `step` for the final runs |
| Reward baseline / scale | `exogenous` lowers variance; the scale sets the gradient size | noisy, slow learning | keep defaults; consider `VecNormalize` |
| Event source / seed | a policy trained on one source may not transfer | good on file, bad on generator | use `mixed`, run `generalization` (step 4.3) |
| Train / test split | the data has 278 days, no spring | optimistic estimates | evaluate on test days only; vary `split_seed` |
| Training length, `gamma`, network | PPO needs 1e5 - 1e6 steps for a day-long problem | learning curve still rising | more steps, more seeds, `net_arch` |
| Observation | missing information cannot be learned | agent no better than random | add forecast, laxity |

## Step 10 - Extension points and known limits

* The battery follows a rule inside the environment; the agent does not control it (planned hook:
  an `Action` group listing more controllable devices).
* No V2G, no tariff / price (placeholders: add a metric and a weight), no per-phase effects.
* `JointDistributionGenerator` needs a sampler of your distribution (step 4.2, "B2"); it does not
  validate it (use `SafeSampler`), draws the number of arrivals itself and knows neither the day
  type nor the charger (write a generator subclass for that). A Python object cannot come from
  `rl.xlsx`: use a script (`rl/examples/run_custom_events.py`).
* `EventSpec.session_file` only documents the path; the file is read where `--session-file` /
  `load_events` says so.
* The scikit-learn variant in `rl/examples/ex_kde_joint.py` is untested (scikit-learn is not a
  dependency of this project).
* The Pandapower model is the case study's (balanced, constant-PQ loads, assumed cable data); all
  statements about line loading inherit that.
* Results on `--synthetic` data are placeholders for testing, **not** findings about the campus.
* Tests: `test/test_operation/test_rl`, `test/test_case_studies/test_kfki_campus/test_rl_*.py`.
