"""Observation terms of the RL problem (registry, name -> size + function).

Each term maps the environment state (``ObsState``) to a small, normalised vector; the
observation of the agent is the concatenation of the terms listed in the problem. Add a term
with ``register_observation``.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class ObsState:
    """What the observation terms can see (filled by the environment every step).

    :ivar int t: Absolute step index in the data
    :ivar pd.DatetimeIndex index: Time stamps of the data
    :ivar dict series: Exogenous kW series (``pv_kw``, ``base_load_kw``)
    :ivar dict scale: Normalisation constant per series
    :ivar np.ndarray connected: Per charger, a vehicle is connected
    :ivar np.ndarray remaining_kwh: Per charger, energy still to deliver
    :ivar np.ndarray steps_left: Per charger, steps until departure
    :ivar np.ndarray rated_kw: Per charger, rated power
    :ivar np.ndarray p_prev: Per charger, power of the previous step
    """

    t: int
    index: pd.DatetimeIndex
    series: dict
    scale: dict
    steps_per_day: int
    dt_h: float
    forecast_mode: str
    soc: float
    trafo_loading_prev: float
    v_max_prev: float
    connected: np.ndarray
    remaining_kwh: np.ndarray
    steps_left: np.ndarray
    rated_kw: np.ndarray
    p_prev: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def n_chargers(self) -> int:
        return len(self.rated_kw)


ObsFn = Callable[[ObsState, dict], np.ndarray]
OBSERVATION_REGISTRY: dict[str, tuple[Callable[[int, dict], int], ObsFn]] = {}


def register_observation(name: str, size: Callable[[int, dict], int]):
    """Decorator registering an observation term; ``size(n_chargers, params)`` is its length."""

    def deco(fn: ObsFn) -> ObsFn:
        OBSERVATION_REGISTRY[name] = (size, fn)
        return fn

    return deco


def observation_size(terms, n_chargers: int) -> int:
    """Length of the observation vector for the given terms."""
    return sum(OBSERVATION_REGISTRY[t.name][0](n_chargers, t.params) for t in terms)


def build_observation(terms, state: ObsState) -> np.ndarray:
    """Concatenate all terms into one float32 vector."""
    parts = [OBSERVATION_REGISTRY[t.name][1](state, t.params) for t in terms]
    return np.concatenate(parts).astype(np.float32)


def _future(s: ObsState, key: str, k: int) -> np.ndarray:
    """Next ``k`` values of a series (perfect foresight or the value one day earlier)."""
    x = s.series[key]
    T = len(x)
    ahead = np.arange(1, k + 1)
    if s.forecast_mode == "perfect":
        idx = np.minimum(s.t + ahead, T - 1)
    else:
        idx = np.clip(s.t + ahead - s.steps_per_day, 0, T - 1)
    return x[idx] / s.scale[key]


@register_observation("time_sin_cos", lambda n, p: 2)
def time_sin_cos(s, params):
    ts = s.index[s.t]
    h = (ts.hour + ts.minute / 60.0) / 24.0
    return np.array([np.sin(2 * np.pi * h), np.cos(2 * np.pi * h)])


@register_observation("weekday", lambda n, p: 1)
def weekday(s, params):
    return np.array([float(s.index[s.t].dayofweek >= 5)])


@register_observation("pv_now", lambda n, p: 1)
def pv_now(s, params):
    return np.array([s.series["pv_kw"][s.t] / s.scale["pv_kw"]])


@register_observation("pv_forecast", lambda n, p: int(p.get("k", 3)))
def pv_forecast(s, params):
    return _future(s, "pv_kw", int(params.get("k", 3)))


@register_observation("base_load_now", lambda n, p: 1)
def base_load_now(s, params):
    return np.array([s.series["base_load_kw"][s.t] / s.scale["base_load_kw"]])


@register_observation("base_load_forecast", lambda n, p: int(p.get("k", 3)))
def base_load_forecast(s, params):
    return _future(s, "base_load_kw", int(params.get("k", 3)))


@register_observation("net_residual", lambda n, p: 1)
def net_residual(s, params):
    """(Base load - PV) / base-load scale: positive = deficit, negative = surplus."""
    net = s.series["base_load_kw"][s.t] - s.series["pv_kw"][s.t]
    return np.array([net / s.scale["base_load_kw"]])


@register_observation("battery_soc", lambda n, p: 1)
def battery_soc(s, params):
    return np.array([s.soc])


@register_observation("trafo_loading_prev", lambda n, p: 1)
def trafo_loading_prev(s, params):
    return np.array([min(s.trafo_loading_prev / 100.0, 2.0)])


@register_observation("v_max_prev", lambda n, p: 1)
def v_max_prev(s, params):
    return np.array([(s.v_max_prev - 1.0) * 10.0])


@register_observation("charger_connected", lambda n, p: n)
def charger_connected(s, params):
    return s.connected.astype(float)


@register_observation("charger_energy_remaining", lambda n, p: n)
def charger_energy_remaining(s, params):
    """Remaining energy / (rated power x one day)."""
    return np.where(s.connected, s.remaining_kwh / (s.rated_kw * 24.0), 0.0)


@register_observation("charger_time_remaining", lambda n, p: n)
def charger_time_remaining(s, params):
    """Remaining connected time / one day."""
    return np.where(s.connected, s.steps_left * s.dt_h / 24.0, 0.0)


@register_observation("charger_laxity", lambda n, p: n)
def charger_laxity(s, params):
    """(Time left - time needed at full power) / one day; < 0 = cannot finish any more."""
    need_h = s.remaining_kwh / s.rated_kw
    return np.where(s.connected, (s.steps_left * s.dt_h - need_h) / 24.0, 0.0)


@register_observation("charger_p_prev", lambda n, p: n)
def charger_p_prev(s, params):
    return s.p_prev / s.rated_kw
