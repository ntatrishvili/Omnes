# Input data quality report (2025-05-28 -> 2026-03-01)

* 6672 hourly rows (278 days), clock = fixed UTC+1, power in kW
* campus peak (hourly): **1774 kW** (reported worst hour Dec 2023: 2055 kW); mean 863 kW
* metered points: 13 meters -> 9 model loads, 43.3% of campus energy; unmetered residual 54.9%

## Imputation / flags (share of hours)

| flag | hours | share |
|---|---|---|
| campus_filled | 45 | 0.67% |
| campus_estimated_by_eon | 0 | 0.00% |
| residual_clipped | 0 | 0.00% |
| meter_gap_scaled | 1 | 0.01% |
| edge_filled | 0 | 0.00% |
| weather_interpolated | 0 | 0.00% |

Metered energy is measured (no imputation). Measured 30-min hours in the window per meter: 36708389: 6672/6672, 36708390: 6672/6672, 36708392: 6672/6672, 36708393: 6672/6672, 36708394: 6672/6672, 36708404: 6672/6672, 36708411: 6672/6672, 36708419: 6672/6672, 36708424: 6672/6672, 36708425: 6672/6672, 36708426: 6672/6672, 36708442: 6672/6672, 36708443: 6672/6672. Hours without interval data use the daily energy and the campus shape.
LiveView clock assumed fixed UTC+2: the residual E.ON - metered is smoothest when the LiveView stamps are shifted back by 1 h (see README); confirm in the LiveView time-zone setting.
EV: 10 charger profiles, user data (simulated_profile_*.csv, read as local wall-clock time); per charger mean 1.60 kW, hourly peak up to 69 kW, 10.7 MWh each over the period.
Heat-pump columns are 0 (already inside building meters / residual).

## Monthly energy (MWh)

| month | campus_MWh | metered_MWh | residual_MWh |
|---|---|---|---|
| 2025-05 | 69 | 32 | 36 |
| 2025-06 | 551 | 248 | 292 |
| 2025-07 | 516 | 253 | 252 |
| 2025-08 | 488 | 226 | 251 |
| 2025-09 | 504 | 230 | 263 |
| 2025-10 | 723 | 277 | 433 |
| 2025-11 | 696 | 291 | 393 |
| 2025-12 | 676 | 288 | 376 |
| 2026-01 | 781 | 347 | 420 |
| 2026-02 | 736 | 291 | 432 |
| 2026-03 | 17 | 8 | 9 |

## PV model

* site 47.490003 N, 18.955120 E, tilt 30 deg, azimuth 180 deg (tilt/azimuth assumed)
* yield over the period: 895 kWh/kWp (no annual extrapolation: the period has no spring); peak 0.91 kW/kWp
* the 9 kWp system would peak at 8.2 kW (measured reading in the report: 7.41 kW)

## Signed tan(phi) per load (Q/P, capacitive < 0)

asztalos_muhely: +0.24, b1: -0.27, b14: +0.24, b15: +0.24, b17: +0.25, b19: -0.26, b2: -0.83, b21A: +0.32, b22: +0.24, b23: +0.30, b23_5: +0.24, b25: +0.24, b26: +0.08, b29: +0.24, b3: -0.24, b31: +0.24, b4: +0.24, b6: -0.14, b7: +0.24, daraboloi_vasraktar: +0.24, egyeb_fogyasztok: +0.24, fahaz_1_3: +0.24, festo_muhely: +0.24, izotop_tarolo: +0.24, kazanok: +0.24, kovacs_laszlo: +0.24, mki_tiszta_labor: +0.24, reaktor_1: +0.24, reaktor_2: +0.24, sportpalya: +0.24, tuziviz_1: +0.24, tuziviz_2: +0.24, villanyszerelo_muhely: +0.24
