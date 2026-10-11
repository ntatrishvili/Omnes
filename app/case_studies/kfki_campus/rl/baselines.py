"""Reference policies with the ``RLAgent`` interface (compared with the learned agent).

* ``UncontrolledAgent``  charge at full power from plug-in (what the legacy profiles do)
* ``PVFollowingAgent``   charge with the PV surplus, but at least the power that still
  finishes the session in time (laxity rule)
* ``OracleAgent``        perfect-foresight LP of the EV power for the episode (PuLP); an upper
  bound for the V1G benefit under the same battery rule and without network terms
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


class OracleAgent(_EnvAgent):
    """Perfect-foresight benchmark: minimise import + export of the episode over the sessions."""

    name = "oracle"

    def __init__(self, problem=None):
        super().__init__(problem)
        self._plan = None
        self._plan_key = None

    def _solve(self) -> np.ndarray:
        e = self.env
        T, n, dt = e.T, len(e.chargers), e.dt_h
        sl = slice(e.t0, e.t0 + T)
        base, pv = e._base_total[sl], e._pv_total[sl]
        prob = pulp.LpProblem("ev_oracle", pulp.LpMinimize)
        imp = [pulp.LpVariable(f"i{k}", 0) for k in range(T)]
        exp = [pulp.LpVariable(f"x{k}", 0) for k in range(T)]
        ev = [[] for _ in range(T)]
        power = {}
        for i, q in e._queue.items():
            for j, (a, d, energy, pmax, _) in enumerate(q):
                vs = [pulp.LpVariable(f"p{i}_{j}_{k}", 0, pmax) for k in range(a, d)]
                prob += pulp.lpSum(vs) * dt == energy
                for k, v in zip(range(a, d), vs):
                    ev[k].append(v)
                    power[(i, k)] = power.get((i, k), 0) + v
        for k in range(T):
            prob += imp[k] - exp[k] == base[k] + pulp.lpSum(ev[k]) - pv[k]
        prob += pulp.lpSum(imp) + pulp.lpSum(exp)
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=20))
        plan = np.zeros((T, n))
        for (i, k), expr in power.items():
            plan[k, i] = pulp.value(expr) or 0.0
        return plan / e._rated

    def predict(self, obs, deterministic=True):
        e = self.env
        key = (e.t0, e.T, e.n_events, id(e._queue))
        if self._plan is None or self._plan_key != key:
            self._plan, self._plan_key = self._solve(), key
        return np.clip(self._plan[min(e.k, e.T - 1)], 0.0, 1.0)


BASELINES = {
    "uncontrolled": UncontrolledAgent,
    "pv_following": PVFollowingAgent,
    "oracle": OracleAgent,
}
