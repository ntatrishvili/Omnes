# from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import xarray as xr
#
#
# class BaseProfileGenerator(ABC):
#     def __init__(self):
#         pass
#
#     @abstractmethod
#     def _base_profile(self, time_index: pd.DatetimeIndex) -> np.ndarray:
#         pass
#
#     def add_seasonality(self, mod_func, amplitude=0.1):
#         self._seasonalities.append(
#             {"mod_func": mod_func, "amplitude": amplitude})
#
#     def add_noise(self, noise_func):
#         self._noises.append(noise_func)
#
#     def apply_seasonality(self, profile, time_index, mod_func=np.sim, amplitude=):
#         modulation = seasonality["mod_func"](time_index)
#         profile *= 1 + seasonality["amplitude"] * modulation
#         return profile
#
#     def apply_noises(self, profile):
#         for noise_func in self._noises:
#             noise = noise_func(profile.shape)
#             profile += profile * noise
#         return profile
#
#     def generate_profile(self, time_index: pd.DatetimeIndex) -> xr.DataArray:
#         profile = self.base_profile(time_index)
#         profile = self.apply_seasonalities(profile, time_index)
#         profile = self.apply_noises(profile)
#         return xr.DataArray(profile, coords={"time": time_index},
#                             dims=["time"])
#
#
# class HouseholdLoadGenerator(BaseProfileGenerator):
#     def base_profile(self, time_index: pd.DatetimeIndex) -> np.ndarray:
#         hours = time_index.hour + time_index.minute / 60
#         profile = (
#                 0.3 * np.sin((hours - 7) * np.pi / 12) ** 2 +
#                 0.5 * np.sin((hours - 18) * np.pi / 8) ** 2
#         )
#         return profile.clip(0, None)
#
#
# class CommercialLoadGenerator(BaseProfileGenerator):
#     def base_profile(self, time_index: pd.DatetimeIndex) -> np.ndarray:
#         hours = time_index.hour + time_index.minute / 60
#         weekday = time_index.dayofweek < 5
#         profile = (
#                           0.6 * np.sin((hours - 9) * np.pi / 10) ** 2
#                   ) * weekday
#         return profile.clip(0, None)
#
#
# class SyntheticPVGenerator(BaseProfileGenerator):
#     def __init__(self, latitude=45.0, tilt=30.0, azimuth=180.0):
#         super().__init__()
#         self.latitude = np.radians(latitude)
#         self.tilt = np.radians(tilt)
#         self.azimuth = np.radians(azimuth)
#
#     def base_profile(self, time_index: pd.DatetimeIndex) -> np.ndarray:
#         doy = time_index.dayofyear.values
#         hod = time_index.hour.values + time_index.minute.values / 60
#         days = np.arange(1, 367)
#         decl = 23.44 * np.pi / 180 * np.sin(2 * np.pi * (days - 81) / 365)
#         decl_interp = np.interp(doy, days, decl)
#
#         solar_noon = 12
#         cosH0 = -np.tan(self.latitude) * np.tan(decl_interp)
#         cosH0 = np.clip(cosH0, -1, 1)
#         H0 = np.arccos(cosH0)
#         daylight_duration = 2 * H0 * 180 / np.pi / 15
#         sunrise = solar_noon - daylight_duration / 2
#         sunset = solar_noon + daylight_duration / 2
#
#         in_daylight = (hod >= sunrise) & (hod <= sunset)
#         norm_time = (hod - sunrise) / (sunset - sunrise)
#         base_profile = np.where(in_daylight, np.sin(np.pi * norm_time), 0.0)
#
#         sun_elevation = np.clip(
#             np.sin(self.latitude) * np.sin(decl_interp)
#             + np.cos(self.latitude) * np.cos(decl_interp), 0, 1
#         )
#         sun_azimuth = np.where(
#             hod < solar_noon,
#             180 - 90 * (1 - norm_time),
#             180 + 90 * (norm_time - 1),
#         )
#         sun_azimuth_rad = np.radians(sun_azimuth)
#
#         orientation_factor = (
#                 np.cos(self.tilt)
#                 * sun_elevation
#                 + np.sin(self.tilt)
#                 * np.cos(sun_azimuth_rad - self.azimuth)
#                 * np.sqrt(1 - sun_elevation ** 2)
#         )
#         orientation_factor = np.clip(orientation_factor, 0, 1)
#
#         return base_profile * orientation_factor
