"""Observation terms of the RL problem (registry, name -> size + function).

Each term maps the environment state (``ObsState``) to a small, normalised vector; the
observation of the agent is the concatenation of the terms listed in the problem. Add a term
with ``register_observation``.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

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
    :ivar np.ndarray site_injection_kw: Per charger, PV minus load (kW) on its bus, without
        the chargers (positive = the feeder exports); ``None`` if the case study has none
    :ivar np.ndarray line_sn_kva: Per charger, rating of the line feeding its bus
    :ivar np.ndarray line_loading_prev: Per charger, loading (%) of that line in the previous step
    :ivar np.ndarray site_id: Per charger, id of its bus (chargers with the same id share a
        connection); ``None`` = every charger is alone
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
    site_injection_kw: Optional[np.ndarray] = None
    line_sn_kva: Optional[np.ndarray] = None
    line_loading_prev: Optional[np.ndarray] = None
    site_id: Optional[np.ndarray] = None

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
    """Remaining energy / (rated power x ``horizon_h``, default one day)."""
    horizon = float(params.get("horizon_h", 24.0))
    return np.where(s.connected, s.remaining_kwh / (s.rated_kw * horizon), 0.0)


@register_observation("charger_time_remaining", lambda n, p: n)
def charger_time_remaining(s, params):
    """Remaining connected time / ``horizon_h`` (default one day)."""
    horizon = float(params.get("horizon_h", 24.0))
    return np.where(s.connected, s.steps_left * s.dt_h / horizon, 0.0)


@register_observation("charger_laxity", lambda n, p: n)
def charger_laxity(s, params):
    """(Time left - time needed at full power) / ``horizon_h`` (default one day); < 0 = cannot finish."""
    horizon = float(params.get("horizon_h", 24.0))
    need_h = s.remaining_kwh / s.rated_kw
    return np.where(s.connected, (s.steps_left * s.dt_h - need_h) / horizon, 0.0)


@register_observation("charger_p_prev", lambda n, p: n)
def charger_p_prev(s, params):
    return s.p_prev / s.rated_kw


@register_observation("charger_line_loading_prev", lambda n, p: n)
def charger_line_loading_prev(s, params):
    """Loading of the line feeding each charger's bus in the previous step (1 = rating)."""
    if s.line_loading_prev is None:
        return np.zeros(s.n_chargers)
    return np.clip(np.asarray(s.line_loading_prev, float) / 100.0, 0.0, 2.0)


@register_observation("charger_site_injection", lambda n, p: n)
def charger_site_injection(s, params):
    """PV minus load on each charger's bus / line rating: > 1 = the feeder would overload."""
    if s.site_injection_kw is None or s.line_sn_kva is None:
        return np.zeros(s.n_chargers)
    return np.clip(s.site_injection_kw / np.maximum(s.line_sn_kva, 1.0), -2.0, 2.0)


def _site_ids(s: ObsState) -> np.ndarray:
    return np.arange(s.n_chargers) if s.site_id is None else np.asarray(s.site_id)


@register_observation("charger_site_rank", lambda n, p: n)
def charger_site_rank(s, params):
    """Urgency rank of each connected vehicle among the connected ones on its bus.

    0 = least laxity (serve first), 1 = most laxity; a vehicle alone on its bus and an empty
    charger give 0 and 1. Needed to decide who gets the power under a connection limit.
    """
    ids = _site_ids(s)
    lax = s.steps_left * s.dt_h - s.remaining_kwh / s.rated_kw
    out = np.ones(s.n_chargers)
    for g in np.unique(ids):
        on = np.flatnonzero((ids == g) & s.connected)
        if len(on) == 0:
            continue
        rank = np.argsort(np.argsort(lax[on]))
        out[on] = rank / max(len(on) - 1, 1)
    return out


@register_observation("charger_site_count", lambda n, p: n)
def charger_site_count(s, params):
    """Number of connected vehicles on the bus of each charger / chargers on that bus."""
    ids = _site_ids(s)
    out = np.zeros(s.n_chargers)
    for g in np.unique(ids):
        members = ids == g
        out[members] = (members & s.connected).sum() / members.sum()
    return out
