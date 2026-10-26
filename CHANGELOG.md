# CHANGELOG

## Unreleased
* Added a reinforcement-learning workflow: generic Gymnasium environment, payoff / observation / event / agent registries in `app/operation/rl` (feeder and transformer overload terms, connection limit per bus, agent-controlled battery, warm start of PPO from a rule), `simulate.py` got a reusable per-step `PowerFlowStepper`, `kpis.py` an EV KPI function, the `RL:` syntax was added to `dsl/model_example.txt`. New dependencies `gymnasium` and (training) `requirements-rl.txt` (Stable-Baselines3, scikit-learn).
* KFKI case study, RL part (`app/case_studies/kfki_campus/rl`, one command `python -m app.case_studies.kfki_campus.rl.run`, description in `rl/README.md`): charging sessions sampled from the measured Gaussian mixtures (`gmm_events.py`); the `standard` and `hard` cases (agent-controlled battery and a 250 kVA limit of transformer 4T) with three reward profiles (`profiles.py`); agents compared with uncontrolled charging, simple rules and a perfect-foresight optimum (`baselines.py`); sensitivity study and Hungarian figures with the network schematic (`figures_hu.py`).
* Added the KFKI Campus Energy Community case study (`app/case_studies/kfki_campus`): Excel-driven network, PuLP rolling-horizon dispatch, Pandapower year simulation, scenario sweep (PV x battery x EV chargers) with SC/SS/voltage-violation KPIs and figures.
* KFKI case study: input pipeline (`inputs/`) building the hourly CSV from LiveView daily meters, E.ON 15-min campus data and the weather logger (pvlib PV model), with a data quality report; the case study now runs on the period where all sources overlap (2025-05-28 .. 2026-03-01); measured Q/P (tan phi) used in the power flow; CLI takes `--start/--end`.
* PandapowerConverter: bus ids containing `_` are resolved by exact match first; transformer `vkr_percent` is read from the type entity (`vkr`, default 0.8).

## 0.12.0 (2025-12-08)
* Added pipeline for docs generation.
* Added SonarCloud support.

## 0.11.2 (2025-12-04)
* Added a module for visualization utilities. 

## 0.11.1 (2025-11-28)
* Restructure architecture
* Extend unit tests

## 0.11.0 (2025-10-25)
* Support conversion into Pandapower.
* Add simbench benchmark operation model.

## 0.10.0 (2025-10-06)
* Add logging.  

## 0.9.0 (2025-10-06)
* Extend DSL handling - parse initial relations.  

## 0.8.0 (2025-09-16)
* Added XArray wrapper around Timeseries objects.  

## 0.7.0 (2025-08-08)
* Added changelog, updated documentation.

## 0.6.0 (2025-07-16)
* Modified quantities accessor with . operator instead of dictionary access.

## 0.5.0 (2025-07-06)
* Added unit tests.
* Added pipelines for unit test coverage.

## 0.4.0 (2025-06-11)
* Added a sample DSL.

## 0.3.0 (2025-05-25)
* Use Timeseries objects instead of Pandas Dataframes.

## 0.2.0 (2025-04-16)
* Added more test cases.

## 0.1.0 (2025-04-10)
* Created an initial MVP converting a JSON description to a simple PuLP problem and displying outputs.