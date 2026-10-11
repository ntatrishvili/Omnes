"""Scenario evaluation and the outer sweep over (PV x battery x chargers)."""

import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.build_model import Scenario, build_kfki_model
from app.case_studies.kfki_campus.excel_io import (
    CampusData,
    bus_to_trafo,
    load_campus_data,
)
from app.case_studies.kfki_campus.kpis import compute_kpis
from app.case_studies.kfki_campus.simulate import simulate
from app.conversion.pandapower_converter import PandapowerConverter
from app.infra.logging_setup import get_logger
from app.model.storage.battery import Battery
from app.operation.example_optimization import EnergySystemOptimizer, TrafoLimit

logger = get_logger(__name__)


def scenario_grid(
    data: CampusData, heat_coupling: bool = False, tes_scale: float = 1.0
) -> list[Scenario]:
    """Cartesian product of the (non-empty) axis values in scenarios.xlsx:grid.

    ``heat_coupling`` switches every scenario to the electricity-heat model (heat pumps
    with hot-water tanks, tank volume scaled by ``tes_scale``).
    """
    g = data.grid
    axis = lambda c: g[c].dropna().tolist()  # noqa: E731
    return [
        Scenario(float(pv), float(bat), int(ev), heat_coupling, tes_scale)
        for pv, bat, ev in itertools.product(
            axis("added_pv_kWp"), axis("battery_kWh"), axis("n_chargers")
        )
    ]


def run_scenario(
    scenario: Scenario,
    ts_csv: Path,
    data: Optional[CampusData] = None,
    time_start: str = "2025-01-01 00:00",
    time_end: str = "2026-01-01 00:00",
    resolution: Optional[str] = None,
    solver=None,
    details_dir: Optional[Path] = None,
) -> dict:
    """Build -> convert -> optimise -> simulate -> KPIs for one scenario."""
    t0 = time.time()
    d = data or load_campus_data()
    model = build_kfki_model(scenario, ts_csv, d, time_start, time_end, resolution)
    net = PandapowerConverter().convert_model(model)  # Pandapower side of the model
    index = pd.DatetimeIndex(model.time_set.time_points)
    dt_h = pd.Timedelta(index.freq or (index[1] - index[0])) / pd.Timedelta("1h")

    load_kw = net.profiles["load"].rename(columns=lambda c: c.removesuffix("_p_cons"))
    pv_kw = net.profiles["renewables"].rename(
        columns=lambda c: c.removesuffix("_p_out")
    )
    load_kw.index = pv_kw.index = index
    bat_ent = next((e for e in model.entities.values() if isinstance(e, Battery)), None)
    opt = EnergySystemOptimizer(model)  # PulpConverter: variables + the model's Relations
    trafo = None
    if bat_ent is not None:
        # net demand through the battery's transformer, without battery and heat pumps
        b2t = bus_to_trafo(d)
        tid = b2t[bat_ent.bus]
        in_group = lambda bus_idx: b2t.get(net.bus.name[bus_idx]) == tid  # noqa: E731
        l_cols = [n for n, bi in zip(net.load.name, net.load.bus) if in_group(bi)]
        p_cols = [
            n
            for n, bi in zip(net.sgen.name, net.sgen.bus)
            if in_group(bi) and n in pv_kw
        ]
        trafo = TrafoLimit(
            base_kw=load_kw[l_cols].sum(axis=1).to_numpy()
            - pv_kw[p_cols].sum(axis=1).to_numpy(),
            limit_kw=float(d.settings.get("trafo_limit_kVA", 1000)),
            converter_ids=[c.id for c in opt.converters if b2t.get(c.bus) == tid],
        )

    res_h = lambda key: int(round(float(d.settings[key]) / dt_h))  # noqa: E731
    disp = opt.rolling(
        index, res_h("horizon_h"), res_h("lookahead_h"), trafo, solver
    )
    opt.release()
    disp.attrs["battery_kwh"] = bat_ent.capacity.value if bat_ent else 0.0
    heat = "heat_demand" in disp
    # The optimiser stored its dispatch in the model; the power flow reads it from there.
    run = opt.run_view()
    dispatched = lambda ent_id, qty: np.asarray(  # noqa: E731
        getattr(run[ent_id], qty).value, dtype=float
    )
    for c in opt.converters:  # the dispatched heat pump power enters the power flow as a load
        load_kw[c.input_device] = dispatched(c.input_device, "p_cons")
    sgen_kw = pv_kw.reindex(columns=list(net.sgen.name), fill_value=0.0)
    if bat_ent is not None:
        sgen_kw[bat_ent.id] = dispatched(bat_ent.id, "p_out") - dispatched(
            bat_ent.id, "p_in"
        )
    pf = {
        **dict(zip(d.loads["id"], d.loads["power_factor"])),
        **dict(zip(d.heat_pumps["id"], d.heat_pumps["power_factor"])),
        **dict.fromkeys(d.ev_chargers["id"], 1.0),  # unity power factor at the sites
    }
    tan = (
        dict(zip(d.loads["id"], d.loads["tan_phi"].astype(float)))
        if "tan_phi" in d.loads
        else None
    )
    sim = simulate(net, index, load_kw, pf, sgen_kw, tan)

    kpis = compute_kpis(
        disp,
        sim,
        dt_h,
        v_min=float(d.settings["v_min_pu"]),
        v_max=float(d.settings["v_max_pu"]),
        v_max_strict=float(d.settings["v_max_strict_pu"]),
    )
    heat_arrays = {}
    if heat:
        heat_arrays = {
            c: disp[c].to_numpy()
            for c in (
                "p_hp",
                "heat_demand",
                "heat_supply",
                "tes_soc_kwh",
                "tes_charge",
                "tes_discharge",
            )
        }
    if details_dir is not None:
        details_dir = Path(details_dir)
        details_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            details_dir / f"{scenario.id}.npz",
            index=index.values.astype("datetime64[s]").astype(np.int64),
            vm_max=np.nanmax(sim.vm_pu, axis=1),
            vm_min=np.nanmin(sim.vm_pu, axis=1),
            vm_pu=sim.vm_pu,
            bus_names=np.array(sim.bus_names),
            trafo_loading=sim.trafo_loading_pct,
            trafo_names=np.array(sim.trafo_names),
            demand=disp["demand"].to_numpy(),
            pv=disp["pv"].to_numpy(),
            p_charge=disp["p_charge"].to_numpy(),
            p_discharge=disp["p_discharge"].to_numpy(),
            soc_kwh=disp["soc_kwh"].to_numpy(),
            p_import=disp["p_import"].to_numpy(),
            p_export=disp["p_export"].to_numpy(),
            **heat_arrays,
        )
    # the heat fields only appear in heat-coupled sweeps (keeps old kpis.csv files valid)
    fields = {
        k: v
        for k, v in asdict(scenario).items()
        if scenario.heat_coupling or k not in ("heat_coupling", "tes_scale")
    }
    return {
        "scenario": scenario.id,
        **fields,
        **kpis,
        "runtime_s": round(time.time() - t0, 1),
    }


def _run_one(args):
    scenario, kwargs = args
    return run_scenario(scenario, **kwargs)


def run_sweep(
    ts_csv: Path,
    out_dir: Path,
    scenarios: Optional[list[Scenario]] = None,
    workers: int = 1,
    limit: Optional[int] = None,
    keep_details: bool = True,
    **run_kwargs,
) -> pd.DataFrame:
    """Evaluate all scenarios, appending each result row to ``out_dir/kpis.csv``.

    Resumable: scenarios already present in kpis.csv are skipped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kpi_path = out_dir / "kpis.csv"
    data = run_kwargs.pop("data", None) or load_campus_data()
    scenarios = scenarios or scenario_grid(data)
    done = set(pd.read_csv(kpi_path)["scenario"]) if kpi_path.exists() else set()
    todo = [s for s in scenarios if s.id not in done][:limit]
    logger.info(f"{len(todo)} scenarios to run ({len(done)} already done)")
    kwargs = dict(ts_csv=Path(ts_csv), data=data, **run_kwargs)
    if keep_details:
        kwargs["details_dir"] = out_dir / "details"

    def _append(row):
        pd.DataFrame([row]).to_csv(
            kpi_path, mode="a", header=not kpi_path.exists(), index=False
        )
        logger.info(
            f"done {row['scenario']} in {row['runtime_s']} s: SC={row['SC']:.3f} "
            f"SS={row['SS']:.3f} viol={row['viol_frac']:.2e}"
        )

    if workers <= 1:
        for s in todo:
            _append(_run_one((s, kwargs)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for fut in as_completed([pool.submit(_run_one, (s, kwargs)) for s in todo]):
                _append(fut.result())
    return pd.read_csv(kpi_path) if kpi_path.exists() else pd.DataFrame()
