# KFKI campus - reinforcement learning of EV-charger control

An RL agent learns how much power each EV charger of the KFKI campus delivers in every time step
(V1G smart charging) so that self-sufficiency and self-consumption are maximised while voltages,
transformer and line loadings and battery cycling stay within limits. It is the third operation
mode of OMNES next to the PuLP optimiser and the Pandapower simulator and is evaluated with the same
power flow and KPIs.

* Step-by-step guide (data, constants, descriptors, payoff, agent, evaluation, what influences the
  outcome): `app/case_studies/kfki_campus/rl/README_RL.md`
* The guide starts with a learning path for beginners (reading list, RL vocabulary mapped to this
  project, four-week plan); step 4 explains the charging-event scenarios (fixed session file,
  random sessions, plugging in an existing distribution as a Python object, mixed sources) with
  runnable examples in `app/case_studies/kfki_campus/rl/examples`
* Generic code: `app/operation/rl` (environment, payoff, observations, events, agents)
* Case study: `app/case_studies/kfki_campus/rl`
* DSL syntax of the problem description: the `RL:` block of `dsl/model_example.txt`
