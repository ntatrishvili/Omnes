"""Pandapower time-stepping simulation of the campus network.

The net is built once by ``PandapowerConverter``; per step only the ``p_mw`` /
``q_mvar`` columns of loads and sgens are overwritten, then a Newton-Raphson
power flow is run (warm-started from the previous step).
"""

from dataclasses import dataclass

import numpy as np
import pandapower as pp
import pandas as pd

from app.infra.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class SimResult:
    """Per-step network results (arrays are time x element)."""

    index: pd.DatetimeIndex
    bus_names: list[str]
    vm_pu: np.ndarray
    trafo_names: list[str]
    trafo_loading_pct: np.ndarray
    line_loading_pct: np.ndarray
    losses_kw: np.ndarray
    ext_grid_p_kw: np.ndarray  # + = import from the external grid
    converged: np.ndarray


@dataclass
class StepResult:
    """Network results of one power flow (``converged`` False -> arrays are NaN)."""

    vm_pu: np.ndarray
    trafo_loading_pct: np.ndarray
    line_loading_pct: np.ndarray
    losses_kw: float
    ext_grid_p_kw: float
    converged: bool


class PowerFlowStepper:
    """Per-step Newton-Raphson power flow on a net built once by ``PandapowerConverter``.

    Used by ``simulate`` (whole horizon) and by step-wise operation such as the RL
    environment. ``load_kw`` / ``sgen_kw`` of ``step`` follow ``net.load.name`` /
    ``net.sgen.name`` order.

    :ivar list load_names: Load names in the net's order
    :ivar list sgen_names: Static-generator names in the net's order
    """

    def __init__(
        self,
        net: pp.pandapowerNet,
        load_pf: dict[str, float],
        load_tan_phi: dict[str, float] | None = None,
    ):
        self.net = net
        self.load_names = list(net.load.name)
        self.sgen_names = list(net.sgen.name)
        # signed Q/P measured from the meters (negative = capacitive) wins over the power factor
        measured = load_tan_phi or {}
        self.tan_phi = np.array(
            [
                measured.get(n, np.tan(np.arccos(load_pf.get(n, 1.0))))
                for n in self.load_names
            ]
        )
        # Constant-power loads (the converter's 1 % constant-Z share forces pandapower onto
        # a ~5x slower voltage-dependent-load path).
        net.load["const_z_p_percent"] = 0.0
        net.load["const_z_q_percent"] = 0.0
        # 'recycle' reuses the built network matrices and only refreshes bus P/Q: ~10x
        # faster per step and numerically identical (checked to 1e-11 pu).
        self._recycle = dict(trafo=False, bus_pq=True, gen=False)
        self._have_ppc = False

    def reset(self) -> None:
        """Forget the warm start (call before a non-consecutive step)."""
        self._have_ppc = False

    def step(self, load_kw: np.ndarray, sgen_kw: np.ndarray) -> StepResult:
        """Run one power flow for the given load and generation powers (kW)."""
        net = self.net
        net.load["p_mw"] = load_kw / 1000.0
        net.load["q_mvar"] = load_kw * self.tan_phi / 1000.0
        net.sgen["p_mw"] = sgen_kw / 1000.0
        try:
            if self._have_ppc:
                pp.runpp(net, algorithm="nr", init="results", recycle=self._recycle)
            else:
                pp.runpp(net, algorithm="nr", init="auto")
                self._have_ppc = True
        except pp.LoadflowNotConverged:
            self._have_ppc = False
            try:
                pp.runpp(net, algorithm="nr", init="flat", max_iteration=50)
                self._have_ppc = True
            except pp.LoadflowNotConverged:
                return StepResult(
                    np.full(len(net.bus), np.nan, dtype=np.float32),
                    np.full(len(net.trafo), np.nan, dtype=np.float32),
                    np.full(len(net.line), np.nan, dtype=np.float32),
                    np.nan,
                    np.nan,
                    False,
                )
        return StepResult(
            net.res_bus.vm_pu.to_numpy().copy(),
            net.res_trafo.loading_percent.to_numpy().copy(),
            net.res_line.loading_percent.to_numpy().copy(),
            (net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()) * 1000.0,
            net.res_ext_grid.p_mw.sum() * 1000.0,
            True,
        )


def simulate(
    net: pp.pandapowerNet,
    index: pd.DatetimeIndex,
    load_kw: pd.DataFrame,
    load_pf: dict[str, float],
    sgen_kw: pd.DataFrame,
    load_tan_phi: dict[str, float] | None = None,
) -> SimResult:
    """Run one power flow per step.

    ``load_kw`` has one column per ``net.load.name``; ``sgen_kw`` one column per
    ``net.sgen.name`` (generation positive, so a charging battery is negative).
    ``load_tan_phi`` (optional, signed Q/P per load id) overrides ``load_pf``.
    """
    T = len(index)
    stepper = PowerFlowStepper(net, load_pf, load_tan_phi)
    load_p = load_kw[stepper.load_names].to_numpy(dtype=float)
    sgen_p = sgen_kw[stepper.sgen_names].to_numpy(dtype=float)

    vm = np.full((T, len(net.bus)), np.nan, dtype=np.float32)
    tl = np.full((T, len(net.trafo)), np.nan, dtype=np.float32)
    ll = np.full((T, len(net.line)), np.nan, dtype=np.float32)
    loss = np.full(T, np.nan)
    grid = np.full(T, np.nan)
    ok = np.zeros(T, dtype=bool)

    for t in range(T):
        r = stepper.step(load_p[t], sgen_p[t])
        if not r.converged:
            logger.warning(f"power flow did not converge at {index[t]}")
            continue
        ok[t] = True
        vm[t] = r.vm_pu
        tl[t] = r.trafo_loading_pct
        ll[t] = r.line_loading_pct
        loss[t] = r.losses_kw
        grid[t] = r.ext_grid_p_kw

    return SimResult(
        index=index,
        bus_names=list(net.bus.name),
        vm_pu=vm,
        trafo_names=list(net.trafo.name),
        trafo_loading_pct=tl,
        line_loading_pct=ll,
        losses_kw=loss,
        ext_grid_p_kw=grid,
        converged=ok,
    )
