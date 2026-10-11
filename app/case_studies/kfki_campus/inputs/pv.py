"""PV output per installed kWp from the measured global horizontal irradiance (pvlib).

Only GHI, air temperature, wind and pressure of the logger are used. The measured
diffuse / direct channels are inconsistent (diffuse > global in ~5 % of samples), so
DNI/DHI are derived from GHI with the Erbs decomposition.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pvlib

LATITUDE = 47.490003
LONGITUDE = 18.955120
ALTITUDE_M = 220.0


@dataclass(frozen=True)
class PVConfig:
    latitude: float = LATITUDE
    longitude: float = LONGITUDE
    altitude: float = ALTITUDE_M
    tilt: float = 30.0  # deg, assumed (not documented for the building-29 array)
    azimuth: float = 180.0  # deg, south
    albedo: float = 0.2
    gamma_pdc: float = -0.004  # 1/K
    system_losses: float = 0.14  # soiling, wiring, mismatch, ...
    inverter_efficiency: float = 0.963  # SMA SB 3000HF-30 maximum efficiency
    dc_ac_ratio: float = 1.1  # AC clipping at 1/ratio kW per kWp


def pv_per_kwp(weather: pd.DataFrame, cfg: PVConfig = PVConfig()) -> pd.Series:
    """AC output in kW per kWp for each hour (index = hour start, fixed UTC+1).

    ``weather`` needs columns ghi (W/m2), temp_c, wind_ms and pressure_pa.
    """
    idx = weather.index.tz_localize("Etc/GMT-1")
    mid = idx + pd.Timedelta(minutes=30)  # hourly means are centred half an hour later
    loc = pvlib.location.Location(cfg.latitude, cfg.longitude, altitude=cfg.altitude)
    sol = loc.get_solarposition(
        mid,
        pressure=weather["pressure_pa"].ffill().bfill().to_numpy(),
        temperature=weather["temp_c"].ffill().bfill().to_numpy(),
    )
    ghi = weather["ghi"].fillna(0.0).to_numpy()
    ghi = pd.Series(ghi, index=mid)
    dni_extra = pvlib.irradiance.get_extra_radiation(mid)
    split = pvlib.irradiance.erbs(ghi, sol["zenith"], mid)
    poa = pvlib.irradiance.get_total_irradiance(
        cfg.tilt,
        cfg.azimuth,
        sol["apparent_zenith"],
        sol["azimuth"],
        split["dni"].fillna(0.0),
        ghi,
        split["dhi"].fillna(0.0),
        dni_extra=dni_extra,
        albedo=cfg.albedo,
        model="haydavies",
    )["poa_global"].fillna(0.0)
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
        "open_rack_glass_glass"
    ]
    t_cell = pvlib.temperature.sapm_cell(
        poa,
        weather["temp_c"].ffill().bfill().to_numpy(),
        weather["wind_ms"].ffill().bfill().to_numpy(),
        **params,
    )
    dc = pvlib.pvsystem.pvwatts_dc(poa, t_cell, pdc0=1.0, gamma_pdc=cfg.gamma_pdc)
    ac = dc * (1 - cfg.system_losses) * cfg.inverter_efficiency
    ac = ac.clip(lower=0, upper=1.0 / cfg.dc_ac_ratio)
    ac[sol["apparent_zenith"].to_numpy() > 90] = 0.0
    return pd.Series(ac.to_numpy(), index=weather.index, name="pv_per_kwp")
