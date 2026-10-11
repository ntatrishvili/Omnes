"""Payoff of the RL problem: metrics of one step -> weighted cost -> reward.

Every metric is a function ``fn(ctx, params) -> float`` of the step context ``ctx`` (a dict
filled by the environment, see ``OmnesEnv``). New metrics are added with
``register_metric``; the DSL refers to them by name, like converters are looked up by
entity class.

Context keys (units kW / kWh / pu / %): ``dt_h``, ``import_kwh``, ``export_kwh``,
``unmet_kwh``, ``p_ev`` and ``p_ev_prev`` (per charger), ``p_bat_ch``, ``p_bat_dis``,
``clipped_kw``, and, when the power flow ran, ``vm_pu``, ``trafo_loading_pct``,
``line_loading_pct``, ``trafo_sn_kva``, ``line_sn_kva``, ``losses_kw``.
"""

from typing import Callable

import numpy as np

MetricFn = Callable[[dict, dict], float]
METRIC_REGISTRY: dict[str, MetricFn] = {}


def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Decorator registering a metric under ``name``."""

    def deco(fn: MetricFn) -> MetricFn:
        METRIC_REGISTRY[name] = fn
        return fn

    return deco


def _pf(ctx: dict, key: str):
    """Power-flow quantity of the step, or ``None`` if there is none (post mode / diverged)."""
    value = ctx.get(key)
    return (
        None
        if value is None or np.isnan(np.asarray(value, dtype=float)).all()
        else value
    )


@register_metric("import_kwh")
def import_kwh(ctx, params):
    """Energy imported from the grid (self-sufficiency)."""
    return float(ctx["import_kwh"])


@register_metric("export_kwh")
def export_kwh(ctx, params):
    """Energy exported to the grid (self-consumption)."""
    return float(ctx["export_kwh"])


@register_metric("unmet_kwh")
def unmet_kwh(ctx, params):
    """Requested EV energy not delivered at departure."""
    return float(ctx["unmet_kwh"])


@register_metric("voltage_excess_pu")
def voltage_excess_pu(ctx, params):
    """Sum over buses of the voltage outside ``[v_lo, v_hi]`` (pu)."""
    vm = _pf(ctx, "vm_pu")
    if vm is None:
        return 0.0
    vm = np.asarray(vm, dtype=float)
    hi, lo = params.get("v_hi", 1.05), params.get("v_lo", 0.95)
    return float(np.nansum(np.maximum(vm - hi, 0.0) + np.maximum(lo - vm, 0.0)))


@register_metric("trafo_overload_kvah")
def trafo_overload_kvah(ctx, params):
    """Transformer energy above the rating (kVAh)."""
    load = _pf(ctx, "trafo_loading_pct")
    if load is None:
        return 0.0
    over = np.maximum(np.asarray(load, dtype=float) - 100.0, 0.0) / 100.0
    return float(np.nansum(over * np.asarray(ctx["trafo_sn_kva"])) * ctx["dt_h"])


@register_metric("line_overload_kvah")
def line_overload_kvah(ctx, params):
    """Line energy above the thermal rating (kVAh)."""
    load = _pf(ctx, "line_loading_pct")
    if load is None:
        return 0.0
    over = np.maximum(np.asarray(load, dtype=float) - 100.0, 0.0) / 100.0
    return float(np.nansum(over * np.asarray(ctx["line_sn_kva"])) * ctx["dt_h"])


@register_metric("battery_throughput_kwh")
def battery_throughput_kwh(ctx, params):
    """Battery charge plus discharge energy (cycling)."""
    return float((ctx["p_bat_ch"] + ctx["p_bat_dis"]) * ctx["dt_h"])


@register_metric("losses_kwh")
def losses_kwh(ctx, params):
    """Line and transformer losses."""
    loss = _pf(ctx, "losses_kw")
    return 0.0 if loss is None else float(loss) * ctx["dt_h"]


@register_metric("ramp_kw")
def ramp_kw(ctx, params):
    """Sum of the power changes of the chargers (anti-chatter)."""
    return float(np.abs(np.asarray(ctx["p_ev"]) - np.asarray(ctx["p_ev_prev"])).sum())


@register_metric("clipped_kw")
def clipped_kw(ctx, params):
    """Requested charger power removed by the feasibility projection."""
    return float(ctx["clipped_kw"])


TRANSFORMS: dict[str, Callable[[float, float], float]] = {
    "linear": lambda m, thr: m,
    "square": lambda m, thr: m * m,
    "hinge": lambda m, thr: max(m - thr, 0.0),
}


class RewardFunction:
    """Reward ``r = -(C - C_ref) / E_ref`` with ``C = sum_k w_k * transform_k(metric_k)``.

    ``C_ref`` (``baseline == "exogenous"``) is the import / export cost of the campus without
    EV and battery; it does not depend on the policy and only lowers the variance.

    :ivar float e_ref: Reference energy (kWh) the cost is divided by
    """

    def __init__(self, terms, e_ref: float, baseline: str = "exogenous"):
        self.terms = list(terms)
        self.e_ref = float(e_ref)
        self.baseline = baseline
        self.weights = {}
        for t in self.terms:
            self.weights[t.metric] = self.weights.get(t.metric, 0.0) + t.weight

    def metrics(self, ctx: dict) -> dict[str, float]:
        """Raw (untransformed) value of every metric of the problem."""
        return {t.metric: METRIC_REGISTRY[t.metric](ctx, t.params) for t in self.terms}

    def __call__(self, ctx: dict, ref: tuple[float, float] = (0.0, 0.0)):
        """Return ``(reward, breakdown)``; ``ref`` = (import, export) kWh of the reference.

        ``breakdown`` maps each term to its (negative) contribution; ``"baseline"`` is the
        reference added back.
        """
        breakdown: dict[str, float] = {}
        for t in self.terms:
            m = METRIC_REGISTRY[t.metric](ctx, t.params)
            cost = t.weight * TRANSFORMS[t.transform](m, t.threshold)
            breakdown[t.metric] = breakdown.get(t.metric, 0.0) - cost / self.e_ref
        base = 0.0
        if self.baseline == "exogenous":
            base = (
                self.weights.get("import_kwh", 0.0) * ref[0]
                + self.weights.get("export_kwh", 0.0) * ref[1]
            ) / self.e_ref
        breakdown["baseline"] = base
        return float(sum(breakdown.values())), breakdown
