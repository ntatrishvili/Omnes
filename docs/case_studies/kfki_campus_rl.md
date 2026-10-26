# KFKI campus - reinforcement learning of EV-charger control

An RL agent decides how much power each EV charger of the KFKI campus delivers in every hour
(V1G smart charging) and, in the harder case, also dispatches the campus battery, so that the PV
surplus is used and the overloaded feeder cables and the limited 4T transformer are relieved. It
is the third operation mode of OMNES next to the PuLP optimiser and the Pandapower simulator and is
evaluated with the same power flow and KPIs.

* The charging sessions are sampled from Gaussian mixtures fitted to measured events
  (`rl/gmm_events.py`, files in `kfki_campus/data/input`).
* Two cases (`rl/profiles.py`): **standard** (no battery, every vehicle can be fully charged) and
  **hard** (the agent also controls the battery under a 250 kVA limit of transformer 4T). Three
  reward profiles (PV use, grid relief, balanced); the agents are compared with uncontrolled
  charging, simple rules and a perfect-foresight optimum.
* One command runs everything (training of missing agents, evaluation, sensitivity study) and
  draws the Hungarian figures: `python -m app.case_studies.kfki_campus.rl.run`; the description of
  the steps, assumptions and figures is in `app/case_studies/kfki_campus/rl/README.md`.
* Generic code: `app/operation/rl` (environment, payoff, observations, events, agents)
* Case study: `app/case_studies/kfki_campus/rl`
* DSL syntax of the problem description: the `RL:` block of `dsl/model_example.txt`
