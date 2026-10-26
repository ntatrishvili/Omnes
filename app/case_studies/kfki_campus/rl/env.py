"""KFKI campus EV-charging environment (V1G setpoints) on the OMNES model.

The campus is built exactly like in ``scenarios.run_scenario`` (``build_kfki_model`` ->
``PandapowerConverter``); the EV chargers of the model are then controlled by the agent while
everything else (loads, PV, heat pumps) is replayed from the time-series CSV. Per step:

1. the action (fraction of the rated power per charger) is projected on what is feasible
   (a vehicle is connected, the requested energy is not exceeded, the rated / vehicle power);
2. the NaS battery follows a self-consumption rule (``battery_policy``);
3. the campus balance is the one of ``EnergySystemOptimizer`` (``balance = demand + p_ch -
   pv - p_dis``; import / export are its positive / negative part), so SC and SS are the
   same quantities as in the optimisation runs;
4. one Pandapower power flow (``PowerFlowStepper``) gives voltages, loadings and losses.

After an episode ``dispatch()`` / ``sim_result()`` return the same objects the optimisation
pipeline produces, so ``compute_kpis`` and the existing figures apply unchanged.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.build_model import Scenario, build_kfki_model
from app.case_studies.kfki_campus.excel_io import CampusData, load_campus_data
from app.case_studies.kfki_campus.rl import gmm_events  # noqa: F401  (registers 'gmm_bundle')
from app.case_studies.kfki_campus.simulate import (
    PowerFlowStepper,
    SimResult,
    simulate,
)
from app.conversion.pandapower_converter import PandapowerConverter
from app.infra.logging_setup import get_logger
from app.model.storage.battery import Battery
from app.operation.rl.env import OmnesEnv
from app.operation.rl.events import ChargingEvent, SessionFileGenerator
from app.operation.rl.observations import ObsState
from app.operation.rl.problem import RLProblem

log = get_logger(__name__)

# voltage (pu) of the feeders while they export PV: the closed-form feeder loading divides the
# apparent power by it (the power flow's current is S / (sqrt(3) V))
FEEDER_V_PU = 1.03

# 2000 kWp added to the existing 9 kWp = 2009 kWp built-in PV
DEFAULT_SCENARIO = dict(added_pv_kwp=2000.0, battery_kwh=1450.0, n_chargers=9)


def charger_ratings(
    d: CampusData, n_chargers: int, charger_kw: Optional[float] = None
) -> dict[str, float]:
    """Rated power (kW) per charger id; charger k stands at site k mod n_sites."""
    out = {}
    for k, cid in enumerate(d.ev_chargers["id"].head(n_chargers)):
        site = d.ev_sites.iloc[k % len(d.ev_sites)]
        out[cid] = float(charger_kw or site.charger_kW)
    return out


class KfkiChargingEnv(OmnesEnv):
    """Gymnasium environment of the KFKI campus with controllable EV chargers.

    :param RLProblem problem: Problem description (``problem.scenario`` sets PV / battery /
        chargers; defaults: 2000 kWp added, 1450 kWh, 9 chargers)
    :param Path ts_csv: Time-series CSV (README contract)
    :param CampusData data: Input workbooks (default: the case study's)
    :param str time_start: First time stamp of the period (CSV clock)
    :param str time_end: End of the period (exclusive)
    :param list session_events: Deterministic charging events (session file)
    :param str split: ``train`` or ``test`` days
    """

    def __init__(
        self,
        problem: RLProblem,
        ts_csv: Path,
        data: Optional[CampusData] = None,
        time_start: str = "2025-01-01 00:00",
        time_end: str = "2026-01-01 00:00",
        session_events: Optional[list[ChargingEvent]] = None,
        split: str = "train",
        resolution: Optional[str] = None,
    ):
        self._cfg = (ts_csv, data, time_start, time_end, session_events, resolution)
        super().__init__(problem, split)

    # ------------------------------------------------------------------ setup
    def _setup(self) -> None:
        ts_csv, data, time_start, time_end, events, resolution = self._cfg
        d = data or load_campus_data()
        self.data = d
        sc = Scenario(**{**DEFAULT_SCENARIO, **self.problem.scenario})
        self.scenario = sc
        self.model = build_kfki_model(
            sc, Path(ts_csv), d, time_start, time_end, resolution
        )
        self.net = PandapowerConverter().convert_model(self.model)
        self.index = pd.DatetimeIndex(self.model.time_set.time_points)
        self.dt_h = pd.Timedelta(self.index.freq or (self.index[1] - self.index[0])) / (
            pd.Timedelta("1h")
        )
        self.steps_per_day = int(round(24 / self.dt_h))
        self.n_steps_total = len(self.index)

        # chargers: charger k stands at site k mod n_sites (as in build_kfki_model)
        self.chargers = charger_ratings(
            d, sc.n_chargers, self.problem.action.charger_kw
        )
        ids = list(self.chargers)
        self._rated = np.array(list(self.chargers.values()))

        load_kw = self.net.profiles["load"].rename(
            columns=lambda c: c.removesuffix("_p_cons")
        )
        pv_kw = self.net.profiles["renewables"].rename(
            columns=lambda c: c.removesuffix("_p_out")
        )
        load_kw.index = pv_kw.index = self.index
        self.stepper = PowerFlowStepper(self.net, self._load_pf(d), self._tan_phi(d))
        names = self.stepper.load_names
        self._ev_idx = np.array([names.index(c) for c in ids])
        load = load_kw[names].to_numpy(dtype=float)
        load[:, self._ev_idx] = 0.0
        self._load_base = load
        sgen_names = self.stepper.sgen_names
        self._sgen_base = pv_kw.reindex(columns=sgen_names, fill_value=0.0).to_numpy(
            dtype=float
        )
        bat = next(
            (e for e in self.model.entities.values() if isinstance(e, Battery)), None
        )
        self._bat = None
        self._bat_idx = None
        if bat is not None:
            self._bat = dict(
                id=bat.id,
                cap=float(bat.capacity.value),
                rc=float(bat.max_charge_rate.value),
                rd=float(bat.max_discharge_rate.value),
                ce=float(bat.charge_efficiency.value),
                de=float(bat.discharge_efficiency.value),
                keep=float(bat.storage_efficiency.value) ** self.dt_h,
                soc0=float(bat.state_of_charge.value),
            )
            self._bat_idx = sgen_names.index(bat.id)
            pv_cols = [i for i in range(len(sgen_names)) if i != self._bat_idx]
        else:
            pv_cols = list(range(len(sgen_names)))
        self._pv_total = self._sgen_base[:, pv_cols].sum(axis=1)
        self._setup_sites(pv_cols)
        self._base_total = load.sum(axis=1)
        self.series = {"pv_kw": self._pv_total, "base_load_kw": self._base_total}
        self.scale = {
            "pv_kw": max(float(self._pv_total.max()), 1.0),
            "base_load_kw": max(float(self._base_total.max()), 1.0),
        }
        net = self.net
        self._trafo_sn = net.trafo.sn_mva.to_numpy() * 1000.0
        vn = net.bus.vn_kv.loc[net.line.from_bus].to_numpy()
        self._line_sn = (
            np.sqrt(3)
            * vn
            * net.line.max_i_ka.to_numpy()
            * net.line.parallel.to_numpy()
            * net.line.df.to_numpy()
            * 1000.0
        )
        self._ev_line_sn = np.where(
            self._ev_line >= 0, self._line_sn[np.maximum(self._ev_line, 0)], np.inf
        )
        self._setup_trafos()
        s = d.settings
        self.v_limits = (
            float(s["v_min_pu"]),
            float(s["v_max_pu"]),
            float(s["v_max_strict_pu"]),
        )
        if events is not None:
            self._file_generator = SessionFileGenerator(events)

    def _setup_sites(self, pv_cols: list[int]) -> None:
        """Bus, feeding line and local PV-minus-load of every charger (sites share a bus)."""
        net = self.net
        load_bus = net.load.bus.to_numpy()
        sgen_bus = net.sgen.bus.to_numpy()
        self._ev_bus = load_bus[self._ev_idx]
        self._groups = [
            np.flatnonzero(self._ev_bus == b) for b in dict.fromkeys(self._ev_bus)
        ]
        line_of = {}
        for j, (fb, tb) in enumerate(zip(net.line.from_bus, net.line.to_bus)):
            line_of.setdefault(tb, j)  # a feeder bus is the end of its line
        self._ev_line = np.array([line_of.get(b, -1) for b in self._ev_bus])
        inj = np.zeros((self.n_steps_total, len(self._ev_idx)))
        for i, b in enumerate(self._ev_bus):
            gen = [c for c in pv_cols if sgen_bus[c] == b]
            use = np.flatnonzero(load_bus == b)
            inj[:, i] = self._sgen_base[:, gen].sum(axis=1) - self._load_base[
                :, use
            ].sum(axis=1)
        self._site_inj = inj
        q = np.zeros_like(inj)
        tan = self.stepper.tan_phi
        for i, b in enumerate(self._ev_bus):
            use = np.flatnonzero(load_bus == b)
            q[:, i] = (self._load_base[:, use] * tan[use]).sum(axis=1)
        self._site_q = q

    def _setup_trafos(self) -> None:
        """Operating limits of transformers and the base (no EV, no battery) flow through each."""
        net = self.net
        names = list(net.trafo.name)
        limits = self.problem.trafo_limits_kva or {}
        unknown = set(limits) - set(names)
        if unknown:
            raise ValueError(f"trafo_limits_kva: unknown transformers {sorted(unknown)}")
        self._trafo_eff_sn = self._trafo_sn.copy()
        for n, v in limits.items():
            self._trafo_eff_sn[names.index(n)] = float(v)
        self._trafo_limited = np.array([n in limits for n in names])
        # bus -> transformer (feeder buses are named <trafo>_<building>, LV buses LV_<trafo>)
        by_key = {n.removeprefix("TR_"): i for i, n in enumerate(names)}

        def of_bus(bus_name: str) -> int:
            key = bus_name.removeprefix("LV_") if bus_name.startswith("LV_") else bus_name.split("_")[0]
            return by_key.get(key, -1)

        bus_name = net.bus.name
        load_tr = np.array([of_bus(bus_name[b]) for b in net.load.bus])
        sgen_tr = np.array([of_bus(bus_name[b]) for b in net.sgen.bus])
        ntr = len(names)
        tan = self.stepper.tan_phi
        base_p = np.zeros((self.n_steps_total, ntr))
        base_q = np.zeros_like(base_p)
        for j in range(ntr):
            use = np.flatnonzero(load_tr == j)
            base_p[:, j] += self._load_base[:, use].sum(axis=1)
            base_q[:, j] += (self._load_base[:, use] * tan[use]).sum(axis=1)
            gen = [c for c in range(len(sgen_tr)) if sgen_tr[c] == j and c != self._bat_idx]
            base_p[:, j] -= self._sgen_base[:, gen].sum(axis=1)
        self._trafo_base_p, self._trafo_base_q = base_p, base_q
        self._ev_trafo = load_tr[self._ev_idx]
        self._bat_trafo = int(sgen_tr[self._bat_idx]) if self._bat_idx is not None else -1

    def _trafo_loading(self, t: int, p: np.ndarray, pch: float, pdis: float) -> np.ndarray:
        """Loading (% of the operating limit) of the limited transformers (others NaN)."""
        ntr = len(self._trafo_sn)
        out = np.full(ntr, np.nan)
        for j in np.flatnonzero(self._trafo_limited):
            pk = self._trafo_base_p[t, j] + float(p[self._ev_trafo == j].sum())
            if j == self._bat_trafo:
                pk += pch - pdis
            out[j] = 100.0 * np.hypot(pk, self._trafo_base_q[t, j]) / self._trafo_eff_sn[j]
        return out

    def _load_pf(self, d: CampusData) -> dict:
        return {
            **dict(zip(d.loads["id"], d.loads["power_factor"])),
            **dict(zip(d.heat_pumps["id"], d.heat_pumps["power_factor"])),
            **dict.fromkeys(d.ev_chargers["id"], 1.0),
        }

    def _tan_phi(self, d: CampusData) -> Optional[dict]:
        if "tan_phi" in d.loads:
            return dict(zip(d.loads["id"], d.loads["tan_phi"].astype(float)))
        return None

    # ------------------------------------------------------------------ episode
    def _start_episode(self, t0: int, T: int, events: list[ChargingEvent]) -> None:
        n = len(self.chargers)
        ids = list(self.chargers)
        start = self.index[t0]
        dt = pd.Timedelta(hours=self.dt_h)
        self._queue: dict[int, list] = {i: [] for i in range(n)}
        clipped_events = 0
        for e in events:
            i = ids.index(e.charger_id)
            a = int(np.floor((e.plug_in - start) / dt + 1e-9))
            d = int(np.ceil((e.plug_out - start) / dt - 1e-9))
            if d <= 0 or a >= T:
                continue
            energy = (
                e.energy_kwh * (min(d, T) - max(a, 0)) / (d - a)
            )  # pro-rata at the edges
            a, d = max(a, 0), min(d, T)
            pmax = min(self._rated[i], e.max_power_kw or np.inf)
            if (
                self.problem.events.clip_to_deliverable
                and energy > pmax * (d - a) * self.dt_h
            ):
                energy = pmax * (d - a) * self.dt_h
                clipped_events += 1
            self._queue[i].append([a, d, energy, pmax, e.event_id])
        for i in range(n):  # sessions of one charger cannot overlap on the grid
            q = sorted(self._queue[i])
            fixed = []
            for s in q:
                if fixed:
                    s[0] = max(s[0], fixed[-1][1])
                if s[0] < s[1]:
                    fixed.append(s)
            self._queue[i] = fixed
        self._ptr = [0] * n
        self.connected = np.zeros(n, dtype=bool)
        self.remaining = np.zeros(n)
        self._requested = np.zeros(n)
        self._depart = np.zeros(n, dtype=int)
        self._pmax = self._rated.copy()
        self._sid = [""] * n
        self._arrive = np.zeros(n, dtype=int)
        self.p_prev = np.zeros(n)
        self.sessions_log: list[dict] = []
        self.n_events = sum(len(q) for q in self._queue.values())
        b = self._bat
        self.e_kwh = b["soc0"] * b["cap"] if b else 0.0
        self._last_trafo, self._last_vmax = 0.0, 1.0
        self._bat_action = 0.5  # effective battery setpoint of the last step (see battery_policy)
        self._last_line = np.zeros(len(self.net.line))
        if self.problem.power_flow_mode == "step":
            self.stepper.reset()
        nb, ntr, nl = len(self.net.bus), len(self.net.trafo), len(self.net.line)
        self._h = dict(
            demand=np.zeros(T),
            pv=np.zeros(T),
            p_charge=np.zeros(T),
            p_discharge=np.zeros(T),
            soc_kwh=np.zeros(T),
            p_import=np.zeros(T),
            p_export=np.zeros(T),
            p_ev=np.zeros((T, n)),
            action=np.zeros((T, n)),
            connected=np.zeros((T, n), dtype=bool),
            vm=np.full((T, nb), np.nan, dtype=np.float32),
            trafo=np.full((T, ntr), np.nan, dtype=np.float32),
            line=np.full((T, nl), np.nan, dtype=np.float32),
            losses=np.full(T, np.nan),
            grid=np.full(T, np.nan),
            ok=np.zeros(T, dtype=bool),
        )
        self._activate(0)

    def _activate(self, k: int) -> None:
        """Connect the vehicles whose session starts at step ``k``."""
        for i, q in self._queue.items():
            if self.connected[i] or self._ptr[i] >= len(q):
                continue
            a, d, energy, pmax, sid = q[self._ptr[i]]
            if a <= k:
                self.connected[i] = True
                self.remaining[i] = self._requested[i] = energy
                self._depart[i] = d
                self._pmax[i] = pmax
                self._sid[i] = sid
                self._arrive[i] = a
                self._ptr[i] += 1

    # ------------------------------------------------------------------ one step
    def _physics_step(self, action: np.ndarray):
        k, t, dt = self.k, self.t0 + self.k, self.dt_h
        n_ch = len(self.chargers)
        bat_action = action[n_ch] if len(action) > n_ch else None
        action = action[:n_ch]
        con = (
            self.connected.copy()
        )  # state during this step (departures change it below)
        steps_left = np.where(con, self._depart - k, 1)
        limit = np.where(
            con,
            np.minimum(np.minimum(self._rated, self._pmax), self.remaining / dt),
            0.0,
        )
        p_req = action * self._rated
        p = np.minimum(np.where(con, p_req, 0.0), limit)
        if self.problem.action.safety_layer == "laxity":
            p_min = np.minimum(self.remaining / (np.maximum(steps_left, 1) * dt), limit)
            p = np.where(con, np.maximum(p, p_min), 0.0)
        cap = self.problem.action.site_power_cap_kw
        if cap is not None:  # connection limit of the chargers sharing a bus
            for g in self._groups:
                total = float(p[g].sum())
                if total > cap:
                    p[g] *= cap / total
        clipped = float(np.where(con, np.maximum(p_req - limit, 0.0), 0.0).sum())
        self.remaining = np.where(con, self.remaining - p * dt, 0.0)
        ev = float(p.sum())

        base, pv = self._base_total[t], self._pv_total[t]
        pch = pdis = 0.0
        b = self._bat
        policy = self.problem.battery_policy
        if b is not None and policy in ("self_consumption_rule", "agent"):
            net_kw = base + ev - pv
            if policy == "agent" and bat_action is not None:
                s = 2.0 * float(bat_action) - 1.0  # -1 full discharge .. +1 full charge
                if s > 0:
                    pch = min(s * b["rc"], (b["cap"] - self.e_kwh) / (b["ce"] * dt))
                else:
                    pdis = min(-s * b["rd"], self.e_kwh * b["de"] / dt)
            elif net_kw < 0:
                pch = min(-net_kw, b["rc"], (b["cap"] - self.e_kwh) / (b["ce"] * dt))
            else:
                pdis = min(net_kw, b["rd"], self.e_kwh * b["de"] / dt)
            pch, pdis = max(pch, 0.0), max(pdis, 0.0)
            self._bat_action = 0.5 + 0.5 * (pch / b["rc"] - pdis / b["rd"])
            self.e_kwh = float(
                np.clip(
                    b["keep"] * self.e_kwh + (b["ce"] * pch - pdis / b["de"]) * dt,
                    0.0,
                    b["cap"],
                )
            )
        balance = base + ev + pch - pv - pdis
        imp, exp = max(balance, 0.0) * dt, max(-balance, 0.0) * dt

        ctx = dict(
            dt_h=dt,
            import_kwh=imp,
            export_kwh=exp,
            p_ev=p,
            p_ev_prev=self.p_prev,
            p_bat_ch=pch,
            p_bat_dis=pdis,
            clipped_kw=clipped,
            trafo_sn_kva=self._trafo_sn,
            line_sn_kva=self._line_sn,
        )
        h = self._h
        if self.problem.power_flow_mode == "step":
            load_row = self._load_base[t].copy()
            load_row[self._ev_idx] = p
            sgen_row = self._sgen_base[t].copy()
            if self._bat_idx is not None:
                sgen_row[self._bat_idx] = pdis - pch
            r = self.stepper.step(load_row, sgen_row)
            if r.converged:
                ctx.update(
                    vm_pu=r.vm_pu,
                    trafo_loading_pct=r.trafo_loading_pct,
                    line_loading_pct=r.line_loading_pct,
                    losses_kw=r.losses_kw,
                )
                if self._trafo_limited.any():  # percent of the operating limit
                    scale = self._trafo_sn / self._trafo_eff_sn
                    ctx["trafo_loading_pct"] = np.where(
                        self._trafo_limited, r.trafo_loading_pct * scale, np.nan
                    )
                    ctx["trafo_sn_kva"] = self._trafo_eff_sn
                self._last_trafo = float(np.nanmax(ctx["trafo_loading_pct"]))
                self._last_vmax = float(np.nanmax(r.vm_pu))
                self._last_line = np.nan_to_num(np.asarray(r.line_loading_pct, float))
                h["vm"][k], h["trafo"][k], h["line"][k] = (
                    r.vm_pu,
                    r.trafo_loading_pct,
                    r.line_loading_pct,
                )
                h["losses"][k], h["grid"][k], h["ok"][k] = (
                    r.losses_kw,
                    r.ext_grid_p_kw,
                    True,
                )

        elif self.problem.power_flow_mode == "feeder":
            ctx["line_loading_pct"] = h["line"][k] = self._feeder_loading(t, p)
            if self._trafo_limited.any():
                ctx["trafo_loading_pct"] = h["trafo"][k] = self._trafo_loading(t, p, pch, pdis)
                ctx["trafo_sn_kva"] = self._trafo_eff_sn
                self._last_trafo = float(np.nanmax(ctx["trafo_loading_pct"]))

        # departures at the end of this step: undelivered energy is lost
        unmet = 0.0
        for i in np.flatnonzero(con):
            if self._depart[i] == k + 1:
                short = max(float(self.remaining[i]), 0.0)
                short = 0.0 if short < 1e-6 else short
                unmet += short
                self.sessions_log.append(
                    dict(
                        session_id=self._sid[i],
                        charger_id=list(self.chargers)[i],
                        requested_kwh=float(self._requested[i]),
                        delivered_kwh=float(self._requested[i]) - short,
                        arrival=self.index[self.t0 + self._arrive[i]],
                        departure=self.index[t] + pd.Timedelta(hours=dt),
                    )
                )
                self.connected[i] = False
                self.remaining[i] = 0.0
        ctx["unmet_kwh"] = unmet

        h["demand"][k], h["pv"][k] = base + ev, pv
        h["p_charge"][k], h["p_discharge"][k] = pch, pdis
        h["soc_kwh"][k] = self.e_kwh
        h["p_import"][k], h["p_export"][k] = imp / dt, exp / dt
        h["p_ev"][k] = p
        h["action"][k] = action
        h["connected"][k] = con
        self.p_prev = p.copy()
        self._activate(k + 1)
        ref = (max(base - pv, 0.0) * dt, max(pv - base, 0.0) * dt)
        return ctx, ref

    def _feeder_loading(self, t: int, p: np.ndarray) -> np.ndarray:
        """Loading (%) of the lines that feed charger buses (others NaN).

        A charger bus hangs on a radial feeder, so the line carries exactly the load, the PV
        and the chargers of that bus: S = |P + jQ|, current = S / (sqrt(3) V) with ``FEEDER_V_PU``.
        """
        out = np.full(len(self.net.line), np.nan)
        for g in self._groups:
            j = self._ev_line[g[0]]
            if j < 0:
                continue
            pk = -self._site_inj[t, g[0]] + float(p[g].sum())
            out[j] = (
                100.0
                * np.hypot(pk, self._site_q[t, g[0]])
                / (self._line_sn[j] * FEEDER_V_PU)
            )
        self._last_line = np.where(np.isnan(out), self._last_line, out)
        return out

    def _observe_state(self) -> ObsState:
        t = min(self.t0 + self.k, self.n_steps_total - 1)
        return ObsState(
            t=t,
            index=self.index,
            series=self.series,
            scale=self.scale,
            steps_per_day=self.steps_per_day,
            dt_h=self.dt_h,
            forecast_mode=self.problem.forecast_mode,
            soc=self.e_kwh / self._bat["cap"] if self._bat else 0.0,
            trafo_loading_prev=self._last_trafo,
            v_max_prev=self._last_vmax,
            connected=self.connected.copy(),
            remaining_kwh=self.remaining.copy(),
            steps_left=np.where(self.connected, self._depart - self.k, 0),
            rated_kw=self._rated,
            p_prev=self.p_prev,
            site_injection_kw=self._site_inj[t],
            line_sn_kva=self._ev_line_sn,
            site_id=self._ev_bus,
            line_loading_prev=np.where(
                self._ev_line >= 0, self._last_line[np.maximum(self._ev_line, 0)], 0.0
            ),
        )

    # ------------------------------------------------------------------ results of an episode
    @property
    def window_index(self) -> pd.DatetimeIndex:
        return self.index[self.t0 : self.t0 + self.k]

    def dispatch(self) -> pd.DataFrame:
        """Dispatch of the episode with the columns of ``EnergySystemOptimizer.rolling``."""
        h, k = self._h, self.k
        out = pd.DataFrame(
            {
                c: h[c][:k]
                for c in (
                    "demand",
                    "pv",
                    "p_charge",
                    "p_discharge",
                    "soc_kwh",
                    "p_import",
                    "p_export",
                )
            },
            index=self.window_index,
        )
        out.attrs["battery_kwh"] = self._bat["cap"] if self._bat else 0.0
        return out

    def ev_power(self) -> pd.DataFrame:
        """Charger power (kW) per charger over the episode."""
        return pd.DataFrame(
            self._h["p_ev"][: self.k],
            index=self.window_index,
            columns=list(self.chargers),
        )

    def history(self, key: str) -> np.ndarray:
        """Recorded per-step array of the episode (``action``, ``connected``, ``p_ev`` ...)."""
        return self._h[key][: self.k]

    def sessions(self) -> pd.DataFrame:
        """Requested / delivered energy of the sessions that ended in the episode."""
        return pd.DataFrame(self.sessions_log)

    def sim_result(self) -> SimResult:
        """Network results of the episode (from the per-step power flows, else post hoc)."""
        h, k = self._h, self.k
        if self.problem.power_flow_mode == "step" and h["ok"][:k].any():
            return SimResult(
                index=self.window_index,
                bus_names=list(self.net.bus.name),
                vm_pu=h["vm"][:k],
                trafo_names=list(self.net.trafo.name),
                trafo_loading_pct=h["trafo"][:k],
                line_loading_pct=h["line"][:k],
                losses_kw=h["losses"][:k],
                ext_grid_p_kw=h["grid"][:k],
                converged=h["ok"][:k],
            )
        sl = slice(self.t0, self.t0 + k)
        load = self._load_base[sl].copy()
        load[:, self._ev_idx] = h["p_ev"][:k]
        sgen = self._sgen_base[sl].copy()
        if self._bat_idx is not None:
            sgen[:, self._bat_idx] = h["p_discharge"][:k] - h["p_charge"][:k]
        return simulate(
            self.net,
            self.window_index,
            pd.DataFrame(
                load, index=self.window_index, columns=self.stepper.load_names
            ),
            self._load_pf(self.data),
            pd.DataFrame(
                sgen, index=self.window_index, columns=self.stepper.sgen_names
            ),
            self._tan_phi(self.data),
        )

    def write_back(self, run_id: str) -> None:
        """Store the episode's EV power in the model's ``run(run_id)`` (full-period episodes).

        Afterwards ``RunView(model, run_id)`` returns the controlled charger power as
        ``p_cons``, like the dispatch of the optimiser.
        """
        if self.k != self.n_steps_total:
            raise ValueError("write_back needs a full-period episode")
        ev = self.ev_power()
        for cid in self.chargers:
            q = self.model.entities[cid].quantities["p_cons"]
            q.run(run_id).result = [float(v) for v in ev[cid]]
