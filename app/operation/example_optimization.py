"""Energy system optimisation on top of ``PulpConverter``.

``PulpConverter`` turns the OMNES model into PuLP variables (free ones for every
quantity without data, constants for quantities with data) and into constraints for every
``Relation`` attached to an entity. This module adds what the converter does not provide:
the energy balances, the storage dynamics, the objective and the solve.

Balances (selected by entity type and ``tags["vector"]``):
  electricity (one campus balance)
      PV + grid import + battery out == loads + converter input + battery in + grid export
  heat (one balance per bus carrying heat devices)
      converter output (Transducer charging a HEAT device) + storage out - storage in
      == HEAT demand
Storages (``Battery``, ``HotWaterStorage``): energy balance with charge / discharge /
standing efficiency, power and capacity limits from the entity parameters.

``EnergySystemOptimizer`` converts the model once and solves any window of the horizon
(rolling horizon); ``optimize_energy_system_pulp`` solves the whole horizon at once.
"""

import uuid
from dataclasses import dataclass, field
from typing import Generator, Optional, Sequence

import numpy as np
import pandas as pd
import pulp

from app.conversion.pulp_converter import PulpConverter
from app.infra.logging_setup import get_logger
from app.infra.timeseries_object import TimeseriesObject
from app.model.device import Device, Vector
from app.model.entity import Entity
from app.model.generator.pv import PV
from app.model.generic_entity import GenericEntity
from app.model.grid_component.bus import Bus
from app.model.grid_component.line import Line
from app.model.grid_component.transformer import Transformer
from app.model.load.load import Load
from app.model.model import Model
from app.model.run_view import RunView
from app.model.slack import Slack
from app.model.storage.battery import Battery
from app.model.storage.storage import Storage
from app.model.transducer.transducer import Transducer

log = get_logger(__name__)

THROUGHPUT_PENALTY = 1e-3  # kW-equivalent cost per kW of battery throughput
TRAFO_PENALTY = 10.0  # cost per kW above the transformer limit (>> import/export cost)
HEAT_STORE_PENALTY = 1.0  # cost per kWh a heat store ends below its initial level


def all_timeseries(model: Model) -> Generator[TimeseriesObject, None, None]:
    """Recursively yield all TimeSeriesObjects in model.

    :param Model model: The model to traverse
    :yields TimeseriesObject: Each time series in the model
    """
    for entity in model.entities.values():
        for qty in entity.quantities.values():
            if isinstance(qty, TimeseriesObject):
                yield qty
        yield from _all_timeseries_from_entity(entity)


def _all_timeseries_from_entity(
    entity: Entity,
) -> Generator[TimeseriesObject, None, None]:
    """Recurse into sub-entities to find all TimeSeriesObjects.

    :param Entity entity: The entity to traverse
    :yields TimeseriesObject: Each time series found
    """
    for sub in entity.sub_entities.values():
        for qty in sub.quantities.values():
            if isinstance(qty, TimeseriesObject):
                yield qty
        yield from _all_timeseries_from_entity(sub)


@dataclass(frozen=True)
class TrafoLimit:
    """Soft apparent-power limit of the transformer a battery is connected to.

    ``base_kw`` is the net demand through the transformer without battery and converters
    (full horizon), ``converter_ids`` the converters (e.g. heat pumps) on its LV side.
    """

    base_kw: np.ndarray
    limit_kw: float
    converter_ids: Sequence[str] = ()


@dataclass
class WindowSolution:
    """Solved window; all arrays have one entry per step of the window."""

    problem: pulp.LpProblem
    grid_import: np.ndarray
    grid_export: np.ndarray
    storage_in: dict[str, np.ndarray] = field(default_factory=dict)
    storage_out: dict[str, np.ndarray] = field(default_factory=dict)
    storage_e: dict[str, np.ndarray] = field(default_factory=dict)  # end of each step
    converter_in: dict[str, np.ndarray] = field(default_factory=dict)
    converter_out: dict[str, np.ndarray] = field(default_factory=dict)


class EnergySystemOptimizer:
    """Converts the model once with ``PulpConverter`` and solves windows of its horizon."""

    # no optimisation variables or constraints from the electrical network itself
    NOT_OPTIMISED = {Bus, Line, Transformer, GenericEntity}

    def __init__(self, model: Model, freq: Optional[str] = None):
        self.model = model
        self.time_set = model.time_set
        self.n = self.time_set.number_of_time_steps
        self.freq = freq or self.time_set.freq
        self.dt_h = pd.Timedelta(pd.tseries.frequencies.to_offset(self.freq)) / (
            pd.Timedelta("1h")
        )
        self.run_id = f"opt_{uuid.uuid4().hex}"
        for ts_obj in all_timeseries(model):
            if not ts_obj.empty():
                ts_obj.run(self.run_id).aligned = ts_obj.value(
                    time_set=self.n, freq=self.freq
                )
        self.converter = PulpConverter()
        self.vars = self.converter.convert_model(
            model, run_id=self.run_id, skip_entities=self.NOT_OPTIMISED
        )
        self.relations = self.converter.relation_constraints(self.vars)

        ents = list(model.entities.values())
        of = lambda cls: [e for e in ents if isinstance(e, cls)]  # noqa: E731
        self.pvs, self.loads, self.slacks = of(PV), of(Load), of(Slack)
        self.batteries = of(Battery)
        self.tanks = [s for s in of(Storage) if s.tags.get("vector") == Vector.HEAT]
        self.converters = of(Transducer)
        self.heat_loads = [
            e
            for e in ents
            if isinstance(e, Device)
            and e.tags.get("vector") == Vector.HEAT
            and "p_cons" in e.quantities
        ]
        for c in self.converters:
            out = model.entities.get(c.output_device)
            if out is None or out.tags.get("vector") != Vector.HEAT:
                raise NotImplementedError(
                    f"Transducer '{c.id}': only converters charging a HEAT device are supported"
                )
        self.demand = self._sum_constants(self.loads, "p_cons")
        self.pv_out = self._sum_constants(self.pvs, "p_out")
        self.heat_demand: dict[str, np.ndarray] = {}
        for e in self.heat_loads:
            self.heat_demand[e.bus] = self.heat_demand.get(
                e.bus, np.zeros(self.n)
            ) + np.asarray(self.vars[f"{e.id}.p_cons"], dtype=float)

    # ------------------------------------------------------------------ helpers
    def _sum_constants(self, entities, qty) -> np.ndarray:
        total = np.zeros(self.n)
        for e in entities:
            total = total + np.asarray(self.vars[f"{e.id}.{qty}"], dtype=float)
        return total

    @property
    def storages(self) -> list:
        return self.batteries + self.tanks

    def initial_state(self) -> dict[str, float]:
        """Stored energy (kWh) of every storage at the start: state_of_charge x capacity."""
        return {
            s.id: float(s.state_of_charge.value) * float(s.capacity.value)
            for s in self.storages
        }

    def release(self) -> None:
        """Drop the solver variables held by the time series (the results stay)."""
        for ts_obj in all_timeseries(self.model):
            ts_obj.run(self.run_id)._vars = {}

    # ------------------------------------------------------------------ solve
    def solve(
        self,
        start: int = 0,
        steps: Optional[int] = None,
        soc0: Optional[dict[str, float]] = None,
        trafo: Optional[TrafoLimit] = None,
        solver=None,
    ) -> WindowSolution:
        """Solve steps ``start .. start+steps`` from the given stored energies (kWh).

        The plain LP is solved first. The min(import + export) objective rewards burning PV
        surplus through the battery's round-trip losses (discharging while the campus
        exports, charging while it imports, or both at once), which a real operation would
        not do. If the LP does any of that, the MILP is solved instead, in which the battery
        may only discharge in steps without export and only charge in steps without import.
        Most windows have no surplus, so this is much faster.
        """
        steps = self.n - start if steps is None else min(steps, self.n - start)
        soc0 = {**self.initial_state(), **(soc0 or {})}
        sol = self._solve(start, steps, soc0, trafo, solver, exclusive=False)
        if self.batteries and self._fights_the_grid(sol):
            sol = self._solve(start, steps, soc0, trafo, solver, exclusive=True)
        return sol

    def _fights_the_grid(self, sol: WindowSolution) -> bool:
        """True if the battery charges while the campus imports, discharges while it
        exports, or charges and discharges in the same step."""
        ch = sum(sol.storage_in[b.id] for b in self.batteries)
        dis = sum(sol.storage_out[b.id] for b in self.batteries)
        simultaneous = any(
            np.minimum(sol.storage_in[b.id], sol.storage_out[b.id]).max() > 1e-6
            for b in self.batteries
        )
        return bool(
            simultaneous
            or ((ch > 1e-3) & (sol.grid_import > 1e-3)).any()
            or ((dis > 1e-3) & (sol.grid_export > 1e-3)).any()
        )

    def _solve(self, start, n, soc0, trafo, solver, exclusive) -> WindowSolution:
        sl = slice(start, start + n)
        dt = self.dt_h
        cache: dict = {}

        def V(e, qty):
            key = (e.id, qty)
            if key not in cache:
                cache[key] = list(self.vars[f"{e.id}.{qty}"][sl])
            return cache[key]

        def bound(vs, lo=0.0, hi=None):
            for v in vs:
                if isinstance(v, pulp.LpVariable):
                    v.lowBound, v.upBound = lo, hi

        prob = pulp.LpProblem("energy_system", pulp.LpMinimize)
        for s in self.slacks:
            bound(V(s, "p_in")), bound(V(s, "p_out"))
        imp = [pulp.lpSum(V(s, "p_out")[t] for s in self.slacks) for t in range(n)]
        exp = [pulp.lpSum(V(s, "p_in")[t] for s in self.slacks) for t in range(n)]

        # converters: electric input >= 0 (upper limits come from their Relations)
        for c in self.converters:
            bound(V(c, "p_in")), bound(V(c, "p_out"))

        # storages
        penalties = []
        short = []
        for s in self.storages:
            cap = float(s.capacity.value)
            rc, rd = s.max_charge_rate.value, s.max_discharge_rate.value
            ce, de = float(s.charge_efficiency.value), float(s.discharge_efficiency.value)
            keep = float(s.storage_efficiency.value) ** dt
            p_in, p_out, e = V(s, "p_in"), V(s, "p_out"), V(s, "e_stor")
            bound(p_in, 0.0, None if rc is None else float(rc))
            bound(p_out, 0.0, None if rd is None else float(rd))
            bound(e, 0.0, cap)
            for t in range(n):
                prev = soc0[s.id] if t == 0 else e[t - 1]
                prob += e[t] == keep * prev + (ce * p_in[t] - p_out[t] * (1.0 / de)) * dt
            init = float(s.state_of_charge.value) * cap
            if isinstance(s, Battery):
                # no energy mining across the window edge
                prob += e[n - 1] >= min(soc0[s.id], init)
                penalties.append(
                    THROUGHPUT_PENALTY
                    * pulp.lpSum(p_in[t] + p_out[t] for t in range(n))
                    * dt
                )
                if exclusive:
                    # direction of the campus' grid exchange in each step: 1 = import
                    # (no export), 0 = export (no import). The battery may discharge only
                    # while importing and charge only while exporting, so it can neither
                    # charge and discharge at once nor burn surplus through its losses.
                    if "grid_mode" not in cache:
                        cache["grid_mode"] = [
                            pulp.LpVariable(f"grid_import_{t}", cat=pulp.LpBinary)
                            for t in range(n)
                        ]
                        rates = sum(
                            float(b.max_charge_rate.value) + float(b.max_discharge_rate.value)
                            for b in self.batteries
                        )
                        # converter (heat pump) input has no fixed cap here; allow up to
                        # three times the campus demand, which is far above any rating
                        conv = 3.0 * float(np.max(self.demand[sl])) if self.converters else 0.0
                        for t in range(n):
                            # tight per-step big-M (numerically safer than one global M)
                            m_imp = float(self.demand[start + t]) + conv + rates + 1.0
                            m_exp = float(self.pv_out[start + t]) + rates + 1.0
                            prob += imp[t] <= m_imp * cache["grid_mode"][t]
                            prob += exp[t] <= m_exp * (1 - cache["grid_mode"][t])
                    g = cache["grid_mode"]
                    for t in range(n):
                        prob += p_in[t] <= float(rc) * (1 - g[t])
                        prob += p_out[t] <= float(rd) * g[t]
            else:
                # heat store: end at least at the initial level (soft, so that a store
                # that starts a window below it is refilled where it is cheapest)
                sh = pulp.LpVariable(f"short_{s.id}", 0)
                prob += e[n - 1] + sh >= init
                short.append(sh)

        # electricity balance of the campus (copper plate)
        conv_in = [pulp.lpSum(V(c, "p_in")[t] for c in self.converters) for t in range(n)]
        for t in range(n):
            prob += (
                self.pv_out[start + t]
                + imp[t]
                + pulp.lpSum(V(b, "p_out")[t] for b in self.batteries)
                == self.demand[start + t]
                + conv_in[t]
                + pulp.lpSum(V(b, "p_in")[t] for b in self.batteries)
                + exp[t]
            )

        # heat balance, one per bus that carries heat devices
        buses = set(self.heat_demand) | {s.bus for s in self.tanks}
        buses |= {c.bus for c in self.converters}
        for bus in sorted(buses):
            convs = [c for c in self.converters if c.bus == bus]
            tanks = [s for s in self.tanks if s.bus == bus]
            dem = self.heat_demand.get(bus, np.zeros(self.n))[sl]
            for t in range(n):
                prob += (
                    pulp.lpSum(V(c, "p_out")[t] for c in convs)
                    + pulp.lpSum(V(s, "p_out")[t] - V(s, "p_in")[t] for s in tanks)
                    == dem[t]
                )

        # the model's own relations (e.g. conversion efficiency, rated power of a Transducer)
        for i, (name, cons) in enumerate(self.relations.items()):
            if len(cons) != self.n:
                raise ValueError(
                    f"Relation '{name}' has {len(cons)} constraints for {self.n} steps; "
                    "window-wise optimisation needs one constraint per step"
                )
            for t, con in enumerate(cons[sl]):
                prob.addConstraint(con, name=f"rel{i}_{t}")

        # soft transformer limit of the battery's transformer
        over = []
        if trafo is not None:
            ids = set(trafo.converter_ids)
            for t in range(n):
                net = (
                    trafo.base_kw[start + t]
                    + pulp.lpSum(V(c, "p_in")[t] for c in self.converters if c.id in ids)
                    + pulp.lpSum(V(b, "p_in")[t] - V(b, "p_out")[t] for b in self.batteries)
                )  # + = demand drawn through the trafo
                # Soft: with a lot of added PV on this transformer the base flow alone can
                # exceed it; the overload is then reported by the power-flow simulation.
                up, dn = pulp.LpVariable(f"tu_{t}", 0), pulp.LpVariable(f"td_{t}", 0)
                prob += net <= trafo.limit_kw + up
                prob += net >= -trafo.limit_kw - dn
                over += [up, dn]

        prob.setObjective(
            pulp.lpSum((imp[t] + exp[t]) * dt for t in range(n))
            + pulp.lpSum(penalties)
            + TRAFO_PENALTY * pulp.lpSum(over) * dt
            + HEAT_STORE_PENALTY * pulp.lpSum(short)
        )
        status = prob.solve(solver)
        if exclusive and pulp.LpStatus[status] == "Infeasible":
            # CBC's pre-processing sometimes declares the (feasible) grid-direction MILP
            # infeasible; the same problem solves with pre-processing off
            status = prob.solve(
                pulp.PULP_CBC_CMD(
                    msg=False,
                    timeLimit=getattr(solver, "timeLimit", None) or 5,
                    gapRel=getattr(solver, "gapRel", None),
                    options=["preprocess off"],
                )
            )
        # With a time limit CBC may stop on a feasible (near-optimal) integer solution.
        have_solution = all(
            pulp.value(v) is not None for vs in cache.values() for v in vs
            if isinstance(v, pulp.LpVariable)
        )
        if pulp.LpStatus[status] != "Optimal" and not (
            pulp.LpStatus[status] == "Not Solved" and have_solution
        ):
            raise RuntimeError(
                f"optimisation window infeasible/unsolved: {pulp.LpStatus[status]}"
            )
        val = lambda vs: np.array([pulp.value(v) or 0.0 for v in vs], dtype=float)  # noqa: E731
        return WindowSolution(
            problem=prob,
            grid_import=val(imp),
            grid_export=val(exp),
            storage_in={s.id: val(V(s, "p_in")) for s in self.storages},
            storage_out={s.id: val(V(s, "p_out")) for s in self.storages},
            storage_e={s.id: val(V(s, "e_stor")) for s in self.storages},
            converter_in={c.id: val(V(c, "p_in")) for c in self.converters},
            converter_out={c.id: val(V(c, "p_out")) for c in self.converters},
        )

    # ------------------------------------------------------------------ results
    def run_view(self) -> RunView:
        """Read-only view of the model with the dispatch of this optimiser's run."""
        return RunView(self.model, self.run_id, self.time_set)

    def _store(self, entity, qty: str, values) -> None:
        """Write one result series into the entity's time series for this run."""
        quantity = entity.quantities.get(qty)
        if isinstance(quantity, TimeseriesObject):
            quantity.run(self.run_id).result = [float(v) for v in values]

    def _write_back(self, s_in, s_out, s_e, c_in, c_out, out: pd.DataFrame) -> None:
        """Store the dispatch in the model (``run(run_id).result`` of each quantity).

        Storages get ``p_in``, ``p_out`` and ``e_stor``; converters ``p_in`` and ``p_out``;
        the device feeding a converter gets the dispatched electric power as ``p_cons``; a
        single slack gets the grid exchange (``p_out`` = import, ``p_in`` = export).
        Read them back with :meth:`run_view`.
        """
        for s in self.storages:
            self._store(s, "p_in", s_in[s.id])
            self._store(s, "p_out", s_out[s.id])
            self._store(s, "e_stor", s_e[s.id])
        for c in self.converters:
            self._store(c, "p_in", c_in[c.id])
            self._store(c, "p_out", c_out[c.id])
            feeder = self.model.entities.get(c.input_device)
            if feeder is not None:
                self._store(feeder, "p_cons", c_in[c.id])
        if len(self.slacks) == 1:
            self._store(self.slacks[0], "p_out", out["p_import"])
            self._store(self.slacks[0], "p_in", out["p_export"])

    # ------------------------------------------------------------------ rolling horizon
    def rolling(
        self,
        index: pd.DatetimeIndex,
        horizon_steps: int,
        lookahead_steps: int,
        trafo: Optional[TrafoLimit] = None,
        solver=None,
    ) -> pd.DataFrame:
        """Solve the whole horizon in committed blocks of ``horizon_steps``.

        Each window also sees ``lookahead_steps`` more steps (perfect foresight) and starts
        from the stored energies at the end of the previous committed block.

        Returns a DataFrame (kW, kWh) with columns demand (incl. converter input), pv,
        p_charge, p_discharge, soc_kwh, p_import, p_export and, with converters,
        ``hp_<input device id>`` (electric kW), p_hp, heat_demand, heat_supply,
        tes_soc_kwh, tes_charge, tes_discharge (campus totals).
        """
        T = self.n
        zeros = lambda: {s.id: np.zeros(T) for s in self.storages}  # noqa: E731
        s_in, s_out, s_e = zeros(), zeros(), zeros()
        c_in = {c.id: np.zeros(T) for c in self.converters}
        c_out = {c.id: np.zeros(T) for c in self.converters}
        if self.storages or self.converters:
            # time/gap limits: a near-optimal feasible dispatch is enough, and the exclusive
            # charge/discharge MILP can otherwise run for minutes on degenerate windows
            solver = solver or pulp.PULP_CBC_CMD(msg=False, timeLimit=5, gapRel=0.02)
            soc = self.initial_state()
            for s in range(0, T, horizon_steps):
                e = min(T, s + horizon_steps + lookahead_steps)
                c = min(horizon_steps, T - s)
                r = self.solve(s, e - s, soc, trafo, solver)
                for dst, src in (
                    (s_in, r.storage_in),
                    (s_out, r.storage_out),
                    (s_e, r.storage_e),
                    (c_in, r.converter_in),
                    (c_out, r.converter_out),
                ):
                    for k, arr in src.items():
                        dst[k][s : s + c] = arr[:c]
                soc = {k: arr[c - 1] for k, arr in r.storage_e.items()}
        total = lambda d, ids: sum((d[i] for i in ids), np.zeros(T))  # noqa: E731
        bat = [b.id for b in self.batteries]
        tank = [s.id for s in self.tanks]
        conv = [c.id for c in self.converters]
        hp_total = total(c_in, conv)
        total_demand = self.demand + hp_total
        p_ch, p_dis = total(s_in, bat), total(s_out, bat)
        balance = total_demand + p_ch - self.pv_out - p_dis  # >0 import, <0 export
        out = pd.DataFrame(
            {
                "demand": total_demand,
                "pv": self.pv_out,
                "p_charge": p_ch,
                "p_discharge": p_dis,
                "soc_kwh": total(s_e, bat),
                "p_import": np.clip(balance, 0, None),
                "p_export": np.clip(-balance, 0, None),
            },
            index=index,
        )
        self._write_back(s_in, s_out, s_e, c_in, c_out, out)
        if conv:
            for c in self.converters:
                out[f"hp_{c.input_device or c.id}"] = c_in[c.id]
            out["p_hp"] = hp_total
            out["heat_demand"] = sum(self.heat_demand.values(), np.zeros(T))
            out["heat_supply"] = total(c_out, conv)
            out["tes_soc_kwh"] = total(s_e, tank)
            out["tes_charge"] = total(s_in, tank)
            out["tes_discharge"] = total(s_out, tank)
            out.attrs["tes_kwh"] = float(sum(s.capacity.value for s in self.tanks))
        return out


def optimize_energy_system_pulp(model, timeset=None, freq=None, **kwargs):
    """Optimise the whole horizon of the model at once.

    Converts the model with ``PulpConverter``, adds the electricity / heat balances and the
    storage dynamics (``EnergySystemOptimizer``), solves, and writes the results back to
    the model's time series.
    :param Model model: The model to optimize
    :param timeset: TimeSet whose frequency is used for resampling (optional)
    :param freq: Target frequency string (optional)
    :param kwargs: ``soc0``, ``trafo`` and ``solver`` of ``EnergySystemOptimizer.solve``
    :returns: RunView of optimized model
    """
    freq = freq or (timeset.freq if timeset is not None else None)
    log.info(f"Starting optimization of the {model.id} model")
    opt = EnergySystemOptimizer(model, freq)
    sol = opt.solve(**kwargs)
    log.info(f"Objective value: {pulp.value(sol.problem.objective)}")
    opt.converter.convert_back(
        model,
        problem=sol.problem,
        pulp_variables=opt.vars,
        run_id=opt.run_id,
        time_set=opt.time_set,
    )
    opt.release()
    return RunView(model, opt.run_id, opt.time_set)
