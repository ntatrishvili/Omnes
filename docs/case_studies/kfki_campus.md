# KFKI Campus Energy Community - case study report

> **Status: full results on real data, including real EV charging profiles.** The sweep
> covers **2025-05-28 .. 2026-03-01 (278 days, no spring)**, so all KPIs are period KPIs,
> not annual ones. The LV cable and transformer impedances and the charger locations are
> **assumed**: read the line-loading and EV-location findings accordingly (see
> "Limitations").

## 1. Goal

Quantify how additional PV capacity, battery capacity and EV chargers change
(i) voltage quality, (ii) self-consumption (SC) and (iii) self-sufficiency (SS)
of the KFKI campus network, using the same OMNES model for the network
power-flow (Pandapower) and the dispatch optimisation (PuLP) over the simulated
period (278 days of real measurements).

## 2. Network and data

Source: Á. Weidinger's energy-community report (PowerFactory model, Fig. 9-11)
and the colleagues' notes. The model reproduces the single-line diagram:

* one 10 kV busbar fed by the Virányos line (the Budaközép reserve line is open),
* **five** 1 MVA 10/0.4 kV transformers (T, 3/T, 4/T, 6/T, 10/T) feeding 33
  consumers (buildings, the isotope store, two reactors, workshops, ...),
* four heat pumps (buildings 19, 26, 29, 21/A),
* the 9 kW PV system of building 29 (3 x SMA SB 3000HF-30) on 6/T,
* the NaS battery (250 kW / 1.45 MWh) on 4/T.

| Quantity | Value | Origin |
|---|---|---|
| Transformers | 5 x 1 MVA (report text says 6) | diagram followed |
| Feeder ratings | Fig. 10 load-flow currents | report |
| LV cables, transformer impedance | placeholders | **assumed** |
| EV chargers | 3 today (sweep 3-9), profiles from `simulated_profile_*.csv` | user data |
| EV charger locations (buildings 26, 29, 17, 3, round-robin), split of added PV | equal split over named sites | **assumed** |

Full provenance: `app/case_studies/kfki_campus/README.md`. Reference check for the
demand series: annual 6.5-8 GWh, peak 2.055 MW (Dec 2023).

## 3. Method

1. **Inputs** are three Excel workbooks (`network`, `assets`, `scenarios`) read by
   `excel_io.py`; a single wide hourly CSV (fixed UTC+1) holds the time series, built by
   `inputs.build_inputs` from LiveView meters (13 points = 43 % of the campus energy, all
   measured at 30-min resolution), the E.ON 15-min campus total (shape and the
   unmetered 55 %) and the weather logger (PV via pvlib). See the package README for cleaning steps and the quality report.
2. **Model**: `build_kfki_model(scenario, csv)` creates the OMNES entities
   (Bus, Line, Transformer, Slack, Load/HeatPump/EVCharger, PV, Battery).
3. **Dispatch (PuLP)**: rolling horizon, 24 h committed + 24 h look-ahead,
   perfect foresight. Objective: minimise grid import + export; battery charge and
   discharge are mutually exclusive; the battery's transformer (4/T) is limited to
   1 MVA as a *soft* constraint (strongly penalised), because with several MWp of PV on
   that transformer the base flow alone exceeds it; the overload shows up in the
   power-flow results instead.
4. **Simulation (Pandapower)**: balanced single-phase equivalent, one Newton-Raphson
   power flow per step with the dispatched battery power and the input profiles.
5. **KPIs** per scenario: SC = 1 - export/PV; SS = 1 - import/demand; voltage
   violations = share of (bus, step) outside 0.9-1.1 pu (also > 1.05 pu), plus
   transformer/line loading, losses and battery cycles.
6. **Sweep**: Cartesian grid from `scenarios.xlsx`
   (default 8 PV x 5 battery x 7 charger levels (3-9) = 280 scenarios).

### Multi-energy extension: electricity and heat

With `--heat` the four heat pumps are modelled as energy-vector converters: an OMNES
`Transducer` (electricity in, heat out, COP 3.5) charges a `HotWaterStorage` that, together
with the pump, serves a `HeatDemand` of the building. The dispatch balances electricity and
heat jointly, so the heat pumps run when PV is in surplus and the tank bridges to the heat
demand; the dispatched heat pump power is then fed into the power flow like any load. See
the package README for the entities, KPIs, commands and figures (`heat_*`, `heat_comparison`).
Heat demand and heat pump sizing are **assumed** (no heat metering): a degree-hour profile
of the measured temperature and a tank holding 3 h of peak demand.

## 4. Limitations

* Cable lengths/types and transformer impedances are assumptions; voltage and
  line-loading results scale directly with them. Lines of weakly loaded feeders
  (sized to today's current) are overloaded by any sizeable added PV, so the line
  loading KPI is only indicative until real cable data are entered.
* The EV profiles are user simulations (read as local wall-clock time), one per charger,
  with fixed locations (assumed). Each peaks at 66-85 kW although the common power steps
  are 22 and 44 kW, so one profile covers more than one 22 kW outlet; if each file is
  meant to be a single outlet, the EV load in this study is overstated by 3-4 times.
  Charging is unmanaged; there is no smart charging or V2G.
* The period has no spring (Mar-May), so the PV/battery KPIs are not annual figures.
* The unmetered 55 % of the energy (including the reactors) has no data of its own: its
  profile is the E.ON campus total minus the metered buildings, split over the other
  loads by rated power.
* No EV smart charging / V2G, no heat-pump flexibility (except in the `--heat` extension), no tap changer, balanced
  network only, DC side of the NaS ignored. PV array tilt/azimuth assumed (30 deg, south).
* Dispatch has perfect foresight inside its window and a campus-level copper-plate
  balance (network constraints are only checked afterwards, apart from the
  battery's transformer limit).

## 5. Results

280 scenarios (added PV 0-2000 kWp in 8 steps x battery 0-5800 kWh x 3-9 EV chargers),
278 days each; every power flow converged. All 13 mapped meters are measured at 30-min
resolution and the EV demand comes from the 10 simulated charger profiles. Baseline =
existing 9 kWp PV, the 1.45 MWh NaS and today's 3 chargers. Campus demand in the period:
5.68 GWh (mean 0.85 MW), of which the 3 chargers are 32 MWh (0.6 %).

**Voltage.** No scenario leaves the EN 50160 band (0.9-1.1 pu): the highest voltage in the
sweep is 1.053 pu and the lowest 0.920 pu. Baseline voltages stay within 0.963-1.000 pu.
PV alone raises the maximum only modestly: 1.005 pu at 250 kWp, 1.011 at 500, 1.023 at
1000 and 1.047 at 2000 kWp (all without a large battery), always at the end of the 4/T
feeders. Only 14 scenarios exceed 1.05 pu, **all with the largest battery (5.8 MWh, i.e.
1 MW)** and 1500-2000 kWp, in every charger variant. That battery's own bus (NaS battery) is then
the extreme in both directions: up to 1.053 pu when it discharges and down to 0.920 pu
when it charges (0.940 at 1500 kWp). Battery size, not PV, stresses the voltage most.

![voltage envelope](img/kfki_voltage_envelope.png)
![highest bus voltage](img/kfki_heatmap_vmax.png)

**Self-consumption (SC) and self-sufficiency (SS)** (3 chargers; the charger count hardly
matters, see below).
* With 9 kWp the campus is so large relative to its PV that SS is 0.1 % and all PV is used.
* Up to 500 kWp (SS 8 %) everything is consumed on site (SC = 1.00) and a battery adds
  nothing; at 750 kWp SC is still 0.997.
* SC first drops below 0.99 at 1000 kWp (0.977; SS 16 %) and to 0.91 at 1500 kWp and 0.80
  at 2000 kWp (SS 22 % and 25 %). The first 725 kWh of battery recovers +0.02 (1000 kWp),
  +0.04 (1500) and +0.05 (2000); returns diminish with size. With 5.8 MWh SC is at least
  0.999 everywhere.
* SS grows almost linearly with PV (1.7, 4.1, 8.0, 11.9, 15.5, 21.5, 25.3 % for 100 to
  2000 kWp without battery); even the largest battery adds at most ~5 points (2000 kWp:
  25 -> 30 %).

![SC](img/kfki_heatmap_SC.png)
![SS](img/kfki_heatmap_SS.png)
![dispatch week](img/kfki_dispatch_week.png)

**EV chargers (3 -> 9).** Going from 3 to 9 chargers adds 65 MWh (+1.1 % of the campus
demand; the hourly charging peak rises from 88 to 205 kW against a campus peak of
1.77 MW). Consequently the effect is small everywhere:
* SS falls by 0.0-0.25 percentage points (largest at 1500 kWp), SC changes by at most
  +0.7 points (only at 2000 kWp without a large battery, where the extra daytime demand
  absorbs a little surplus).
* The highest voltage and the transformer loading do not change (at most 2 points).
* The lowest voltage drops by up to 0.003 pu (0.963 -> 0.960 pu), as a step at the 7th
  charger, the second one placed at building 17's feeder (the largest load; chargers go to
  buildings 26, 29, 17, 3 in turn); the loading of that feeder's (assumed) cable rises
  from 102 % to 113 %.
* None of this creates a voltage violation.

![EV effect](img/kfki_ev_effect.png)

**Transformers.** Transformer 4/T (battery plus roofs of buildings 4 and 26) is the most
loaded: 50 % in the baseline and no more than 54 % up to 1000 kWp. At 2000 kWp it reaches
57 % without a battery and up to 92 % with a 2.9 MWh battery, because the battery's power
adds to the PV export through the same transformer.

![transformer duration](img/kfki_trafo_duration.png)

**Line loading** is 98-102 % for up to 1000 kWp of PV with at most a 1.45 MWh battery and
3 chargers (113 % with 7-9 chargers), and reaches 290 % at 2000 kWp with the largest battery. These values depend directly on the assumed cables (sized to the diagram's
feeder currents, which are below the measured loads) and are indicative only.

![SC vs SS](img/kfki_sc_vs_ss.png)

**Take-aways.** (1) Up to ~750 kWp of additional PV the campus absorbs everything and a
battery brings nothing. (2) From ~1000 kWp surplus appears and a battery of 1-3 MWh
recovers most of it, with diminishing returns; SS is driven by PV size, not by storage.
(3) PV alone keeps the voltage below 1.05 pu even at 2000 kWp. The limits that appear are
the transformer 4/T loading and the voltage swing on the battery feeder when a 1 MW
battery is installed there. (4) Growing from 3 to 9 chargers is not a network or
self-consumption issue at this scale (+1 % energy, morning peak around 08:00, little PV
at that time of day); its only visible effect is local, at the feeder of the building
where the chargers are placed. (5) Real cable data and the real charger locations and
outlet ratings are the inputs that would change these conclusions most.

### How to reproduce

```bash
python -m app.case_studies.kfki_campus.inputs.build_inputs --start 2025-05-28 --end 2026-03-01
python -m app.case_studies.kfki_campus.run_case_study --csv app/case_studies/kfki_campus/data/processed/kfki_timeseries_2025-05-28_2026-03-01.csv --start 2025-05-28 --end 2026-03-01 --out results --workers 6
```

Runtime was 1-2.5 min per scenario (mean 98 s) for the 278 days at hourly
resolution, i.e. about 50 min for the 160 scenarios with 6 workers on an 8-core
machine. The sweep is resumable: re-running the command skips finished scenarios.

Figures produced in `results/figures/`:

| Figure | Content |
|---|---|
| `heatmap_SC.png`, `heatmap_SS.png` | SC and SS over PV x battery, one panel per charger count |
| `heatmap_violations.png`, `heatmap_vmax.png` | voltage violation share and highest bus voltage |
| `sc_vs_ss.png` | trade-off between SC and SS across all scenarios |
| `voltage_envelope.png` | daily max/min bus voltage, baseline vs highest-voltage scenario |
| `trafo_duration.png` | loading duration curve of the most loaded transformer |
| `dispatch_week.png` | summer week of the highest-voltage scenario |

Questions the results should answer:

1. At which added-PV level do voltage violations first appear, and at which bus?
2. How much SC does a battery of a given size recover, and where does it saturate?
3. Do EV chargers (as an unmanaged load) shift the PV limit or improve SS?
4. Which transformer becomes the bottleneck first?

## 6. Network and time-series figures

### Network structure

![network topology](img/kfki_network_topology.png)

The schematic follows the PowerFactory diagram: the 10 kV bus feeds five 1 MVA
transformers, each with an LV busbar and radial feeders. Blue nodes are buildings
with their own meter, grey nodes are estimated from the E.ON residual; node area is
the mean load. Average load per transformer over the period: T 0.20 MW, 3/T 0.13 MW,
4/T 0.10 MW, 6/T 0.23 MW, 10/T 0.19 MW. The markers show where the assets sit: the
existing 9 kWp PV on building 29, the five candidate roofs, the four assumed EV sites,
the NaS battery on the 4/T bus and the heat-pump nodes (no data of their own).

![voltage maps](img/kfki_network_voltage_maps.png)

Bus voltages on the same layout. Baseline: the lowest voltage (0.963 pu) is at the
largest load, building 17. With 2000 kWp of added PV and no battery, the highest
voltages are at the ends of the 4/T feeders that carry the new roofs (building 4:
1.047 pu; building 26 also elevated). With the 5.8 MWh battery the extreme moves to the
battery's own bus: 1.053 pu when it discharges and 0.921 pu when it charges.

### Input time series

![overview](img/kfki_ts_overview.png)

Daily campus demand split into the 13 measured buildings and the unmetered rest,
PV yield per kWp from the measured irradiance, and air temperature. Monthly mean demand
rises from 0.66-0.77 MW (Jun-Sep) to 0.9-1.1 MW (Oct-Feb), with daily means up to 1.4 MW.
The unmetered share, which contains the
reactors, the isotope store and the other loads without meters, shows multi-day
excursions from October on.

![daily profiles](img/kfki_ts_daily_profiles.png)

Mean daily profiles by season (band = quartiles) for weekdays and weekends, and the
mean PV profile per kWp. The weekday peak is about 1.0 MW in summer, 1.2 MW in autumn
and 1.35 MW in winter; weekend demand stays flat at 0.55, 0.65 and 0.83 MW. The mean
PV peak falls from 0.62 kW/kWp in summer to 0.26 in winter, and the weekday demand
peak (09-13 h) is well aligned with the PV peak, which is why up to ~750 kWp is
absorbed completely.

![carpets](img/kfki_ts_carpets.png)

Hour-by-day maps of campus demand and PV output. The weekday/weekend rhythm is clear,
and so are the dark all-day stripes in the demand map (from October), which look like
operating periods of a large unmetered consumer such as the reactors. This is an
interpretation; no reactor data are available.

![composition weeks](img/kfki_ts_composition_weeks.png)
![measured buildings](img/kfki_ts_measured_buildings.png)

A summer and a winter week stacked by consumer group, and the measured buildings
individually (daily and 7-day mean). Seasonality differs per building: buildings 17,
26 and 19 are 40-100 % higher in winter (heating), building 1 is roughly flat and
building 23 is lower in winter. Buildings 23, 1, 21 and 6 dip around the New Year
closure, and building 2 is tiny (4 kW mean) although the diagram's rating assumes 30 kVA.

![energy by load](img/kfki_ts_energy_by_load.png)

Energy per load. The unmetered loads (grey) are allocated proportionally to their rated
power from the diagram, so e.g. each reactor appears with ~0.55 GWh. This is an
allocation of the residual, **not a measurement**; real reactor and isotope-store
meters (see `meter_download_priority.csv`) would replace it. Building 21 stands for the
diagram node that lumps buildings 21, 21/A and 21/B; the tenant's feeder (Kovács László)
is drawn as part of it in all figures: in this chart its 84 MWh is the grey part of the
bar (267 MWh measured + 84 MWh estimated). This is a presentation choice only; the model
and the simulation results are unchanged.

![EV profiles](img/kfki_ts_ev_profiles.png)

The ten simulated charger profiles (15-min kW averaged to hours): one winter week of
chargers 1-3 (today) and 4-9 (the sweep), and the mean weekday profile for 3 and 9
chargers. Charging is concentrated on weekday mornings (peak around 08:00-09:00, mean
~1.6 kW per charger) and almost absent at weekends.

### Scenario time series and voltage statistics

![summer week](img/kfki_week_pv2000_bat2900_ev9_2025-06-16.png)

One week of a scenario with 2000 kWp added PV, a 2.9 MWh battery and 50 chargers: power
flows, battery power and stored energy, loading of the five transformers and the highest
and lowest bus voltage. The battery charges from the midday surplus on the weekend
(Sat-Sun) and discharges into the evening; export only occurs when it is full.

![winter week](img/kfki_week_pv1000_bat1450_ev9_2026-01-12.png)

The same view in winter for a smaller scenario (1000 kWp, 1.45 MWh, 9 chargers): PV never exceeds demand, so the battery
stays idle and the voltage stays near the baseline (the battery panels are hidden when
it is idle).

![voltage duration](img/kfki_voltage_duration.png)

Duration curves of the highest and lowest bus voltage for increasing added PV (battery
1.45 MWh, 3 chargers). The highest voltage exceeds 1.00 pu only about 40-45 % of the
time (daylight), reaching 1.046 pu for a few hours at 2000 kWp; the lowest voltage is
nearly independent of PV (0.964-0.994 pu).

Any other scenario and week can be rendered with
`run_case_study --out results/kfki --figures-only --week pv1500_bat725_ev0:2025-07-07`.

