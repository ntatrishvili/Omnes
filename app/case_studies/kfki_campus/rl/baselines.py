"""Reference policies with the ``RLAgent`` interface (compared with the learned agent).

* ``UncontrolledAgent``  charge at full power from plug-in (what the legacy profiles do)
* ``PVFollowingAgent``   charge with the PV surplus, but at least the power that still
  finishes the session in time (laxity rule)
* ``LeastLaxityAgent``    full power, but under a connection limit the vehicles with the least
  slack (time left minus time needed) are served first (what a smart charging station does)
* ``LineFollowingAgent``  charge where the feeder of the charger would exceed its line rating
  (the chargers of that bus absorb the excess), otherwise only what the deadline requires
* ``OracleAgent``        perfect-foresight LP of the EV power for the episode (PuLP) with the
  weights of the payoff, feeder / transformer overload and the connection limit; an upper
  bound for the V1G benefit
"""

import numpy as np
import pulp

from app.operation.rl.agent import RLAgent


class _EnvAgent(RLAgent):
    """Rule-based agent that reads the environment instead of the observation vector."""

    def bind(self, env) -> None:
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env


class UncontrolledAgent(_EnvAgent):
    """Full power whenever a vehicle is connected."""

    name = "uncontrolled"

    def predict(self, obs, deterministic=True):
        return np.ones(len(self.env.chargers))


class PVFollowingAgent(_EnvAgent):
    """PV surplus first, deadline-driven minimum power otherwise."""

    name = "pv_following"

    def predict(self, obs, deterministic=True):
        e = self.env
        t = e.t0 + e.k
        surplus = e._pv_total[t] - e._base_total[t]
        con = e.connected
        steps_left = np.maximum(e._depart - e.k, 1)
        frac_min = e.remaining / (steps_left * e.dt_h) / e._rated
        share = np.clip(surplus / max(float((e._rated * con).sum()), 1e-9), 0.0, 1.0)
        return np.where(con, np.maximum(share, frac_min), 0.0)


class LeastLaxityAgent(_EnvAgent):
    """Serve the most urgent vehicles first within the connection limit of each bus."""

    name = "least_laxity"

    def predict(self, obs, deterministic=True):
        e = self.env
        cap = e.problem.action.site_power_cap_kw
        laxity = (e._depart - e.k) * e.dt_h - e.remaining / np.minimum(
            e._rated, e._pmax
        )
        action = np.zeros(len(e._rated))
        for g in e._groups:
            budget = np.inf if cap is None else cap
            for i in sorted(g[e.connected[g]], key=lambda i: laxity[i]):
                p = min(e._rated[i], e._pmax[i], e.remaining[i] / e.dt_h, budget)
                action[i] = p / e._rated[i]
                budget -= p
        return action


class LineFollowingAgent(_EnvAgent):
    """Relieve overloaded feeders: absorb the PV export above ``threshold`` x line rating.

    The chargers of one bus share the excess of that bus; a vehicle that would otherwise miss
    its energy gets at least the laxity power. Sees the local PV minus load (a forecast the
    learned agent also has), not the campus balance.
    """

    name = "line_following"

    def __init__(self, problem=None, threshold: float = 0.9):
        super().__init__(problem)
        self.threshold = threshold

    def predict(self, obs, deterministic=True):
        e = self.env
        t = e.t0 + e.k
        con = e.connected
        excess = np.maximum(e._site_inj[t] - self.threshold * e._ev_line_sn, 0.0)
        share = np.zeros(len(con))
        for g in e._groups:
            cap = max(float((e._rated[g] * con[g]).sum()), 1e-9)
            share[g] = np.clip(excess[g] / cap, 0.0, 1.0)
        steps_left = np.maximum(e._depart - e.k, 1)
        frac_min = e.remaining / (steps_left * e.dt_h) / e._rated
        return np.where(con, np.maximum(share, frac_min), 0.0)


class GridGuardAgent(LineFollowingAgent):
    """``line_following`` for the chargers plus a reactive battery guard for the transformers.

    When a transformer with an operating limit carries more than ``threshold`` x the limit
    (apparent power now, chargers included), the battery on it takes the excess (discharges
    when the transformer imports, charges when it exports); otherwise the battery follows the
    self-consumption rule. It only sees the present step: the benchmark for what a careful
    hand-written rule can do against the limit, without foresight.
    """

    name = "grid_guard"

    def __init__(self, problem=None, threshold: float = 0.95, **kw):
        super().__init__(problem, **kw)
        self.guard = threshold

    def _battery_action(self, ev_kw: float) -> float:
        e = self.env
        b = e._bat
        t, dt = e.t0 + e.k, e.dt_h
        j = e._bat_trafo
        if j >= 0 and e._trafo_limited[j]:
            on_j = e._ev_trafo == j
            p_ev = getattr(self, "_ev_now", np.zeros(len(on_j)))
            pk = e._trafo_base_p[t, j] + float(p_ev[on_j].sum())
            s_kva = float(np.hypot(pk, e._trafo_base_q[t, j]))
            excess = s_kva - self.guard * e._trafo_eff_sn[j]
            if excess > 0:
                if pk > 0:  # importing: discharge
                    pdis = min(excess, b["rd"], e.e_kwh * b["de"] / dt)
                    return 0.5 - 0.5 * pdis / b["rd"]
                pch = min(excess, b["rc"], (b["cap"] - e.e_kwh) / (b["ce"] * dt))
                return 0.5 + 0.5 * pch / b["rc"]
        net = e._base_total[t] + ev_kw - e._pv_total[t]  # self-consumption rule
        if net < 0:
            pch = min(-net, b["rc"], (b["cap"] - e.e_kwh) / (b["ce"] * dt))
            return 0.5 + 0.5 * pch / b["rc"]
        pdis = min(net, b["rd"], e.e_kwh * b["de"] / dt)
        return 0.5 - 0.5 * pdis / b["rd"]

    def predict(self, obs, deterministic=True):
        e = self.env
        action = super().predict(obs, deterministic)
        self._ev_now = np.minimum(
            action * e._rated, np.where(e.connected, np.minimum(e._rated, e._pmax), 0.0)
        )
        if e._bat is None or e.problem.battery_policy != "agent":
            return action
        return np.append(action, self._battery_action(float(self._ev_now.sum())))


_ANGLES = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)


class OracleAgent(_EnvAgent):
    """Perfect-foresight benchmark: the plan that minimises the payoff of the episode.

    A linear programme over the sessions of the episode (PuLP) with the weights of the
    problem's reward: import and export of the campus, unmet energy (a slack per session;
    a delivery guarantee of the laxity layer makes it very expensive), the overload of the
    chargers' feeder lines and of the transformers with an operating limit (apparent power
    ``|P + jQ|`` linearised with 16 tangents, within 2 %), and the connection limit per bus.

    With a battery on the self-consumption rule the plan depends on the battery's reaction,
    which the LP cannot model: the rule is simulated for the plan and the LP is solved again
    with that battery trajectory, ``rounds`` times (fixed point). The oracle is then the best
    *EV* plan against the rule battery, a tight but not exact upper bound there; without a
    battery it is exact up to the linearisation and the nominal feeder voltage. If the agent
    also controls the battery (``battery_policy="agent"``) the battery is a variable of the
    LP (a MILP: charging and discharging are exclusive in every step) and the oracle returns
    its setpoint as the last action entry.
    """

    name = "oracle"

    def __init__(self, problem=None, rounds: int = 3):
        super().__init__(problem)
        self.rounds = rounds
        self._plan = None
        self._plan_key = None

    # -- battery rule of the environment, applied to a plan --------------------------------
    def _rule_battery(self, ev_kw: np.ndarray) -> np.ndarray:
        e = self.env
        b, dt = e._bat, e.dt_h
        T = e.T
        out = np.zeros(T)
        if b is None or e.problem.battery_policy != "self_consumption_rule":
            return out
        sl = slice(e.t0, e.t0 + T)
        net = e._base_total[sl] + ev_kw - e._pv_total[sl]
        soc = b["soc0"] * b["cap"]
        for k in range(T):
            pch = pdis = 0.0
            if net[k] < 0:
                pch = min(-net[k], b["rc"], (b["cap"] - soc) / (b["ce"] * dt))
            else:
                pdis = min(net[k], b["rd"], soc * b["de"] / dt)
            pch, pdis = max(pch, 0.0), max(pdis, 0.0)
            soc = float(
                np.clip(
                    b["keep"] * soc + (b["ce"] * pch - pdis / b["de"]) * dt, 0.0, b["cap"]
                )
            )
            out[k] = pch - pdis
        return out

    def _solve_once(self, bat_kw: np.ndarray, joint: bool = False):
        """Plan (T x n fractions), charger power (kW per step) and battery setpoints.

        The battery is fixed to ``bat_kw`` (kW, + charging) or, with ``joint``, optimised.
        """
        from app.case_studies.kfki_campus.rl.env import FEEDER_V_PU

        e = self.env
        T, n, dt = e.T, len(e.chargers), e.dt_h
        w = e.problem.reward_weights()
        w_imp, w_exp = w.get("import_kwh", 0.0), w.get("export_kwh", 0.0)
        w_unmet, w_line = w.get("unmet_kwh", 0.0), w.get("line_overload_kvah", 0.0)
        w_trafo = w.get("trafo_overload_kvah", 0.0)
        if e.problem.action.safety_layer == "laxity":
            w_unmet = max(w_unmet, 1000.0)  # the layer delivers the energy
        cap = e.problem.action.site_power_cap_kw
        sl = slice(e.t0, e.t0 + T)
        base, pv = e._base_total[sl], e._pv_total[sl]
        prob = pulp.LpProblem("ev_oracle", pulp.LpMinimize)
        imp = [pulp.LpVariable(f"i{k}", 0) for k in range(T)]
        exp = [pulp.LpVariable(f"x{k}", 0) for k in range(T)]
        cost = []
        power: dict = {}
        for i, q in e._queue.items():
            for j, (a, d, energy, pmax, _) in enumerate(q):
                vs = [pulp.LpVariable(f"p{i}_{j}_{k}", 0, pmax) for k in range(a, d)]
                short = pulp.LpVariable(f"u{i}_{j}", 0)
                prob += pulp.lpSum(vs) * dt + short == energy
                cost.append(w_unmet * short)
                for k, v in zip(range(a, d), vs):
                    power[(i, k)] = power.get((i, k), 0) + v
        ev = [pulp.lpSum(power.get((i, k), 0) for i in range(n)) for k in range(T)]
        bat = [float(x) for x in bat_kw]
        if joint:  # battery as decision variables: E_{k+1} = keep E_k + (ce ch - dis / de) dt
            b = e._bat
            ch = [pulp.LpVariable(f"bc{k}", 0, b["rc"]) for k in range(T)]
            dis = [pulp.LpVariable(f"bd{k}", 0, b["rd"]) for k in range(T)]
            mode = [pulp.LpVariable(f"bz{k}", cat="Binary") for k in range(T)]
            soc = [pulp.LpVariable(f"be{k}", 0, b["cap"]) for k in range(T + 1)]
            prob += soc[0] == b["soc0"] * b["cap"]
            for k in range(T):
                prob += ch[k] <= b["rc"] * mode[k]
                prob += dis[k] <= b["rd"] * (1 - mode[k])
                prob += soc[k + 1] == b["keep"] * soc[k] + (b["ce"] * ch[k] - dis[k] * (1.0 / b["de"])) * dt
            bat = [ch[k] - dis[k] for k in range(T)]
        for k in range(T):
            prob += imp[k] - exp[k] == base[k] + ev[k] + bat[k] - pv[k]
            cost += [w_imp * imp[k] * dt, w_exp * exp[k] * dt]
        if cap is not None:
            for g in e._groups:
                for k in range(T):
                    prob += pulp.lpSum(power.get((i, k), 0) for i in g) <= cap
        if w_line > 0:
            for g in e._groups:
                j = e._ev_line[g[0]]
                if j < 0:
                    continue
                for k in range(T):
                    t = e.t0 + k
                    P = -e._site_inj[t, g[0]] + pulp.lpSum(power.get((i, k), 0) for i in g)
                    Q = e._site_q[t, g[0]]
                    o = pulp.LpVariable(f"ol{j}_{k}", 0)
                    for th in _ANGLES:
                        prob += o >= (np.cos(th) * P + np.sin(th) * Q) / FEEDER_V_PU - e._line_sn[j]
                    cost.append(w_line * o * dt)
        if w_trafo > 0:
            for j in np.flatnonzero(e._trafo_limited):
                mine = [i for i in range(n) if e._ev_trafo[i] == j]
                for k in range(T):
                    t = e.t0 + k
                    P = e._trafo_base_p[t, j] + pulp.lpSum(power.get((i, k), 0) for i in mine)
                    if j == e._bat_trafo:
                        P = P + bat[k]
                    Q = e._trafo_base_q[t, j]
                    o = pulp.LpVariable(f"ot{j}_{k}", 0)
                    for th in _ANGLES:
                        prob += o >= np.cos(th) * P + np.sin(th) * Q - e._trafo_eff_sn[j]
                    cost.append(w_trafo * o * dt)
        prob += pulp.lpSum(cost)
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))
        plan = np.zeros((T, n))
        for (i, k), expr in power.items():
            plan[k, i] = pulp.value(expr) or 0.0
        bat_act = np.full(T, 0.5)
        if joint:
            b = e._bat
            for k in range(T):
                bat_act[k] = 0.5 + 0.5 * (
                    (pulp.value(ch[k]) or 0.0) / b["rc"] - (pulp.value(dis[k]) or 0.0) / b["rd"]
                )
        return plan / e._rated, plan.sum(axis=1), bat_act

    def _solve(self) -> np.ndarray:
        """Plan (T x n) of the chargers; with an agent-controlled battery one more column."""
        e = self.env
        if e._bat is not None and e.problem.battery_policy == "agent":
            plan, _, bat_act = self._solve_once(np.zeros(e.T), joint=True)
            return np.column_stack([plan, bat_act])
        bat = np.zeros(e.T)
        plan = None
        for r in range(self.rounds if e._bat is not None else 1):
            plan, ev_kw, _ = self._solve_once(bat)
            if e._bat is None:
                break
            bat = self._rule_battery(ev_kw)
        return plan

    def predict(self, obs, deterministic=True):
        e = self.env
        key = (e.t0, e.T, e.n_events, id(e._queue))
        if self._plan is None or self._plan_key != key:
            self._plan, self._plan_key = self._solve(), key
        return np.clip(self._plan[min(e.k, e.T - 1)], 0.0, 1.0)


BASELINES = {
    "uncontrolled": UncontrolledAgent,
    "pv_following": PVFollowingAgent,
    "line_following": LineFollowingAgent,
    "least_laxity": LeastLaxityAgent,
    "grid_guard": GridGuardAgent,
    "oracle": OracleAgent,
}
