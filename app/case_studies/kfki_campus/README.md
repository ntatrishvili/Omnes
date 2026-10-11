# KFKI Campus Energy Community - OMNES case study

Year-long optimisation + simulation of the KFKI campus network, wrapped in a
sweep over **added PV capacity x battery capacity x number of EV chargers**.
KPIs per scenario: voltage violations, self-consumption (SC), self-sufficiency (SS).

## Pipeline

```
data/*.xlsx --excel_io--> build_model (OMNES Model) --PandapowerConverter--> pandapower net
                                   |                                             |
                      time series CSV (user)                          simulate.py (power flow / step)
                                   |                                             |
                          example_optimization.py (PulpConverter + PuLP, rolling) --run view of the Model--+--> kpis.py --> kpis.csv --> plots.py
```

| Module | Role |
|---|---|
| `data/make_inputs.py` | writes `network.xlsx`, `assets.xlsx`, `scenarios.xlsx` (every value carries a `source` tag) |
| `excel_io.py` | reads and validates the workbooks |
| `components.py` | `HeatPump`, `EVCharger` (both `Load` subclasses, so existing converters handle them), `HeatDemand` (heat-vector `Device`, outside the power flow) |
| `build_model.py` | `build_kfki_model(Scenario, csv)` -> OMNES `Model` |
| `app/operation/example_optimization.py` | `EnergySystemOptimizer`: converts the model once with `PulpConverter` (variables + the entities' `Relation`s), adds the electricity / heat balances and storage dynamics, solves LP/MILP in a rolling horizon (24 h committed + 24 h look-ahead) and writes the dispatch back into the model (`run(run_id).result` of the entities' `p_in`/`p_out`/`e_stor`, the heat-pump `p_cons`, the slack exchange); `opt.run_view()` reads it |
| `dispatch.py` | the earlier hand-written PuLP dispatch; no longer used by the sweep, kept as a cross-check in the tests |
| `simulate.py` | Pandapower Newton-Raphson per step (recycled network matrices); `scenarios.run_scenario` takes the battery and heat-pump powers from the model's run view |
| `kpis.py` | SC, SS, voltage violations, loadings, losses |
| `scenarios.py` | scenario grid, per-scenario run, resumable (parallel) sweep |
| `plots.py` | KPI figures (heatmaps, trade-off, voltage envelope) |
| `plots_network.py`, `plots_timeseries.py`, `figures.py` | network structure, input time series and scenario-week figures |
| `plots_heat.py` | electricity-heat figures (electric / heat / tank week, with-vs-without comparison) |
| `run_case_study.py` | CLI |
| `synthetic.py` | placeholder time series for tests and smoke runs (**not real data**) |
| `inputs/` | raw files -> cleaned hourly CSV (`readers`, `register`, `disaggregate`, `residual`, `pv`, `heat`, `build_inputs`) |

## Running

```bash
python -m app.case_studies.kfki_campus.data.make_inputs        # (re)generate the workbooks
python -m app.case_studies.kfki_campus.inputs.build_inputs --start 2025-05-28 --end 2026-03-01   # raw files -> hourly CSV
python -m app.case_studies.kfki_campus.run_case_study --csv app/case_studies/kfki_campus/data/processed/kfki_timeseries_2025-05-28_2026-03-01.csv --start 2025-05-28 --end 2026-03-01 --out results --workers 6
python -m app.case_studies.kfki_campus.run_case_study --out results --figures-only
python -m app.case_studies.kfki_campus.run_case_study --synthetic --start 2025-01-01 --end 2025-01-03 --limit 2 --out results_smoke   # smoke test
```

`--start/--end` are the first and last day (inclusive) on the CSV's clock (fixed UTC+1).
The model reads the CSV from its first row on (positional), so `--start` must be the CSV's first day.

## Multi-energy: electricity-heat coupling

With `--heat` every scenario of the grid is run as an electricity-heat system instead of an
electricity-only one (use a separate `--out`; the plain results are unchanged). The four
heat pumps stop being plain electric loads and become converters between two energy
vectors, built from OMNES entities (`build_model.py`, `Scenario(heat_coupling=True)`):

```
 electric network --hp_<id> (Load, electric interface)--> hp_<id>_conv (Transducer, COP)
                                                     --> hp_<id>_tes  (HotWaterStorage) --> hp_<id>_heat (HeatDemand)
```

* `Transducer`: electric input power x COP = heat output; `HotWaterStorage`: tank (volume,
  set / return temperature -> capacity; standing losses); `HeatDemand`: building heat demand
  (`Vector.HEAT`, `heat_balance`).
* `PulpConverter` turns the Transducer's `Relation`s into constraints (`p_out == p_in * COP`,
  `p_in <= rated power`); `EnergySystemOptimizer` adds the heat balance (`heat pump output +
  tank discharge = heat demand + tank charge`) to the electric one. The heat pump electricity is part of the campus demand, so the
  optimiser moves it into PV-surplus hours and the tank carries the heat over to the demand.
* The Pandapower converter ignores heat-vector devices and Transducers; the dispatched heat
  pump power enters the power flow as a load at the building bus, so voltages, transformer
  loading and losses include the heat pumps.
* Extra KPIs: `hp_kwh`, `heat_demand_kwh`, `heat_supply_kwh`, `heat_unmet_kwh`,
  `hp_cop_seasonal`, `tes_cycles` (`heat_unmet_kwh` is 0 by construction). `--tes-scale 0` runs heat pumps without a tank,
  `--tes-scale 2` doubles it.

```bash
python -m app.case_studies.kfki_campus.run_case_study --csv <csv> --start 2025-05-28 --end 2026-03-01 --heat --out results_heat --workers 6
python -m app.case_studies.kfki_campus.run_case_study --csv <csv> --start 2025-05-28 --end 2026-03-01 --heat --out results_heat --figures-only --baseline results --week pv2000_bat1450_ev3_hp1:2025-10-13
```

Figures: `heat_<scenario>_<date>` (electric, heat and tank state for one week) and
`heat_comparison` (SC / SS with and without heat coupling, needs `--baseline`).

**Assumptions (no heat is metered on campus):** COP 3.5 (constant), peak heat demand 80 % of
what the pump can deliver (`peak_heat_kW`), tank = 3 h of peak heat (60/35 degC, 0.5 %/h
losses, half full at start). The `heat_<bldg>` demand is a synthetic degree-hour profile of
the measured outdoor temperature (`inputs/heat.py`; heating limit 15 degC, design -10 degC,
night / weekend setback). The heat pump electricity is **added** to the building meters (the
`hp_*` CSV columns stay 0 and are replaced by the dispatched power); read the scenario as
electrifying the heat supply of these four buildings. All values are in the `heat_pumps` sheet.

Figures (`results/figures/`): besides the KPI heatmaps, `run_case_study` draws the
network structure (`network_topology`, `network_voltage_maps`), the input time series
(`ts_*`: overview, daily profiles, carpets, weekly composition, measured buildings,
energy per load) and scenario weeks (`week_<scenario>_<date>`, `voltage_duration`).
Any other week: `--figures-only --week pv1500_bat725_ev0:2025-07-07` (repeatable);
`--no-extra-figures` skips this group.

The sweep appends one row per scenario to `results/kpis.csv` and is resumable
(finished scenario ids are skipped). `results/details/<scenario>.npz` hold
per-step arrays used by the figures.

## Real input data (`inputs/`, raw files in `data/input/`)

`inputs.build_inputs` turns the raw files into the CSV below plus a quality report
(`data/processed/data_quality_report.md`, `imputation_flags.csv`, `qc_*.png`).

| Raw file | Used for | Cleaning |
|---|---|---|
| `UtilityBook_CSV_*` (LiveView; **daily** kWh/kVArh, or **30-min** interval files) and `Multiview_CSV_*` (several meters side by side, active energy only) | 13 metered points -> 9 model loads (`data/meter_map.xlsx`) | meter id from the title line (the `21b` file name carries the wrong serial); a meter keeps its daily history and any interval data (measured hours win); signed kVArh/kWh gives `load_tan_phi.csv`; 36708395 (NAS auxiliary, 4 kWh/day) ignored |
| `EON adatok kimutatas *.xlsx` (campus supply, 15-min kWh) | hourly campus total and the intra-day shape | two timestamp formats and shifted columns in appended blocks, duplicates removed, **zeros = "no data"** treated as missing, local DST clock converted to fixed UTC+1, gaps filled from same-weekday profiles |
| `meteorological_data_*.csv` (1-min logger, fixed UTC+1) | `pv_per_kwp` via pvlib from GHI, air temperature, wind | unit rows/duplicates dropped, temperature/humidity sentinels removed, measured diffuse/direct not used (inconsistent) |
| `KFKI_merok_kapcsolatai_*.xlsx` (meter register) | meter -> building -> transformer check; `meter_download_priority.csv` (which unmetered meters to download next) | hierarchy parsed from sheet `Elektromos` |

Building profiles: where a meter has 30-min data it is summed to hours directly
(currently all 13 mapped meters, whole period). Otherwise each
LiveView day's measured kWh is spread over the day with that day's normalised E.ON
campus shape (daily totals exact; the intra-day shape is the campus', not the
building's). The unmetered rest (E.ON - metered - 1 % losses, ~55 % of the energy,
includes the reactors) is split over the other loads by rated kVA. New interval or
Multiview exports dropped into `data/input/` are picked up automatically.

LiveView clock: interval stamps are interval *ends* on a **fixed clock without DST**
(48 intervals every day, also on 2025-10-26). Its offset is not documented, so it was
estimated: summing the 13 mapped meters (all with 30-min data, 43 % of the campus) and
subtracting them from the E.ON total, the residual is smoothest when the LiveView stamps
are shifted back by exactly 1 h in summer, autumn and winter, with no step at the DST
change. That means **fixed UTC+2** (`--liveview-offset`, default 2).
It is still worth confirming in the LiveView settings. The daily export is the sum of
the interval export over the same calendar day (checked exactly on 271 days). An hour
with a missing half-hour is scaled up from the half-hour present and short gaps are
interpolated (flag `meter_gap_scaled`).

**Period: only where all sources overlap, 2025-05-28 .. 2026-03-01 (278 days)** - no
spring, so KPIs are period KPIs and must not be read as annual figures.
EV chargers: `data/input/simulated_profile_*.csv` (one file per charger, 15-min kW in
`total_power_kw`, 10 files = up to 10 chargers) are averaged to hours; the timestamps
are read as local wall-clock time (`--ev-clock fixed` if they are already UTC+1). If no
file exists, synthetic placeholders are used and the quality report says so. Charger
`k` uses profile `k` (files in seed order) and stands at site `k mod 4` of
`assets.xlsx:ev_sites` (buildings 26, 29, 17, 3 - **assumed**), so the sweep is nested;
today's 3 chargers are the baseline. Open points: each profile peaks at 66-85 kW
(22 and 44 kW are the most common steps), i.e. more than one 22 kW outlet per profile.
Other placeholders: `hp_*` columns are 0 (inside the meters/residual); PV array
tilt/azimuth (30 deg / south) are assumed.

## Time-series CSV contract

One wide CSV for the simulated period, first column `timestamp` (naive, **fixed
UTC+1**, regular step, no gaps or NaN, covering `--start` .. `--end`). Columns
(all in **kW**, except the PV column):

| Column | Meaning |
|---|---|
| `load_<id>` | active power of each building / consumer in `assets.xlsx:loads` (e.g. `load_b29`, `load_izotop_tarolo`, `load_reaktor_1`) |
| `hp_<bldg>` | heat-pump electrical power: `hp_21A`, `hp_19`, `hp_26`, `hp_29`. If the building meter already includes the heat pump, set the `hp_*` column to 0 to avoid double counting |
| `heat_<bldg>` | thermal heat demand of the heat-pump buildings (kW thermal), only read by heat-coupled scenarios (`--heat`); written by `inputs.heat` / `build_inputs` |
| `pv_per_kwp` | PV output **per installed kWp** (kW/kWp, 0..~1) so that PV capacity can be scaled; used for the existing 9 kWp system and for all added PV |
| `ev_charger_01` ... `ev_charger_10` | demand of each EV charger (kW), one column per charger (`assets.xlsx:ev_chargers`) |

Column names come from the `csv_column` fields of the workbooks and can be
renamed there. Reactive power: the measured signed Q/P (`tan_phi` from
`load_tan_phi.csv`, picked up automatically next to the CSV) is used when available;
otherwise it is derived from `power_factor` (default 0.99).

## Data provenance

Source tags: `pdf-fig9` single-line diagram (Fig. 9), `pdf-fig10` load-flow
feeder currents (Fig. 10, 0.4 kV, converted to kVA), `pdf-text`, `note`
(colleagues' comments), `assumed` (not available - **replace with real data**).

| Item | Value | Source |
|---|---|---|
| Topology | 10 kV bus -> 5 transformers -> LV busbars -> radial feeders (33 consumers, 4 heat pumps, PV, NaS) | pdf-fig9 |
| Transformers | **5** x 1 MVA, 10/0.4 kV (text says six; the diagram shows five - the diagram is followed) | pdf-fig9 / pdf-text |
| Feeder rated power | Fig. 10 current x sqrt(3) x 0.4 kV | pdf-fig10 |
| Existing PV | bldg 29, 3 x SMA SB 3000HF-30 = 9 kW AC (kWp assumed = 9); 7.41 kW reading not used as profile | note |
| NaS battery | 250 kW / 1.45 MWh at 4/T bus, 192 V DC (140-230 V); modelled on the AC side, DC band not simulated; round-trip efficiency 0.9 x 0.9 assumed | note + assumed |
| Heat pumps | bldgs 19, 26, 29, 21/A; the 47.6 kW measurement of bldg 29 is context-free and not used | note / pdf-text |
| External grid | Virányos feeder in service, ideal source at 10 kV, 1.0 pu; Budaközép reserve feeder open, not modelled | pdf-fig9/10 |
| **LV cables** | 0.1 km per feeder, parallel NAYY 4x150 sized to 1.25 x Fig. 10 current | **assumed** |
| **Transformer data** | vk 6 %, vkr 1 %, Pfe 1.5 kW, i0 0.3 %, Dyn5 | **assumed** |
| Candidate PV roofs | 26, 4, 6, 17, reactor; added PV split equally | sites pdf-text, split assumed |
| EV chargers | 3 today (sweep 3-9); sites 26, 29, 17, 3 in round-robin order, 22 kW outlets; profiles from `simulated_profile_*.csv` | sites **assumed**, profiles user data |
| Peak demand | 2.055 MW (Dec 2023) - useful to check the demand CSV | pdf-text |

## Method notes and simplifications

* **Dispatch**: campus copper plate; minimise grid import + export; battery
  charge/discharge mutually exclusive (MILP); battery transformer limited to 1 MVA;
  perfect foresight inside each 48 h window; the battery ends each window no lower
  than min(start state, initial state).
* **Simulation**: balanced single-phase equivalent; loads are constant-PQ (the
  converter's 1 % constant-Z share is overridden for speed); PV and battery at
  unity power factor; no tap changer; one power flow per step.
* **SC** = 1 - export/PV energy. **SS** = 1 - import/demand energy (demand
  excludes battery charging). **Voltage violations** = share of (bus, step)
  pairs outside 0.9-1.1 pu (EN 50160 +/-10 %); also reported above 1.05 pu.
* EV smart charging (V1G) by reinforcement learning: see `rl/README_RL.md` (the sweep above uses the uncontrolled profiles).
* Not modelled: DC side of the NaS, EV V2G, heat-pump flexibility in the electricity-only runs (see the multi-energy section for `--heat`),
  unbalance, protection, the six-transformer discrepancy.

