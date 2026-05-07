import hashlib
from typing import Generator

import numpy as np
import pulp

from app.conversion.pulp_converter import PulpConverter
from app.infra.configuration import Config
from app.infra.logging_setup import get_logger
from app.infra.timeseries_object import TimeseriesObject
from app.model.entity import Entity
from app.model.model import Model
from app.infra.visualize import plot_energy_flows
from app.model.run_view import RunView

log = get_logger(__name__)


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


def _all_timeseries_from_entity(entity: Entity) -> Generator[TimeseriesObject, None, None]:
    """Recurse into sub-entities to find all TimeSeriesObjects.
    
    :param Entity entity: The entity to traverse
    :yields TimeseriesObject: Each time series found
    """
    for sub in entity.sub_entities.values():
        for qty in sub.quantities.values():
            if isinstance(qty, TimeseriesObject):
                yield qty
        yield from _all_timeseries_from_entity(sub)


def optimize_energy_system_pulp(model, timeset=None, freq=None, **kwargs):
    """Optimize energy system without duplicating model.
    
    Flow:
    1. Generate run_id
    2. Project (resample once per TimeSeriesObject)
    3. Convert to solver variables
    4. Solve
    5. Extract results
    6. Cleanup solver variables
    7. Return RunView
    
    :param Model model: The model to optimize
    :param timeset: Target TimeSet for resampling (optional)
    :param freq: Target frequency string (optional)
    :param kwargs: Additional parameters
    :returns: RunView of optimized model
    """
    # 1. Generate run_id (unique identifier for this run)
    # Use model's time_set if timeset not provided
    effective_timeset = timeset if timeset is not None else model.time_set
    effective_freq = freq if freq is not None else (model.time_set.freq if hasattr(model.time_set, 'freq') else None)
    
    run_id = f"{id(effective_timeset)}_{effective_freq or 'auto'}"
    # run_id = hashlib.sha256(run_id_base.encode()).hexdigest()[:16]
    
    log.info(f"Starting optimization run: {run_id} (timeset: {id(effective_timeset)}, freq: {effective_freq or 'auto'})")
    
    # 2. PROJECTION: Resample all time series once
    log.info("Resampling time series...")
    ts_count = 0
    for ts_obj in all_timeseries(model):
        run = ts_obj.run(run_id)
        if run.aligned is None:
            # Resample to target timeset/freq (ONCE)
            # time_set parameter is number of time steps, not TimeSet object
            num_steps = effective_timeset.number_of_time_steps if hasattr(effective_timeset, 'number_of_time_steps') else -1
            if effective_freq is not None:
                run.aligned = ts_obj.value(time_set=num_steps, freq=effective_freq)
            else:
                run.aligned = ts_obj.value(time_set=num_steps)  # Use raw data
            ts_count += 1
    log.info(f"Resampled {ts_count} time series to {effective_freq or 'raw'}")
    
    # 3. CONVERSION: Build solver variables from aligned data
    log.info("Converting to solver variables...")
    converter = PulpConverter()
    pulp_variables = converter.convert_model(model, run_id=run_id, timeset=effective_timeset, freq=effective_freq, **kwargs)
    time_set = pulp_variables.get("time_set", model.time_set)

    # Automatically detect all components
    # TODO: improve understanding of component types, we need more automatic indicators
    pv_names = model.find_all_of_type_in_subentities("PV")
    load_names = model.find_all_of_type_in_subentities("Load")
    bess_names = model.find_all_of_type_in_subentities("Battery")
    slack_names = model.find_all_of_type_in_subentities("Slack")
    log.info(f"Detected PV entities: {pv_names}")
    log.info(f"Detected load entities: {load_names}")
    log.info(f"Detected batteries: {bess_names}")
    log.info(f"Detected slack entities: {slack_names}")

    # 4. SOLVE
    log.info("Solving optimization problem...")
    prob = pulp.LpProblem("CSCopt", pulp.LpMinimize)

    # Global energy balance
    for t in range(time_set.number_of_time_steps):
        total_pv = np.sum(pulp_variables[f"{pv}.p_out"][t] for pv in pv_names)
        total_load = np.sum(pulp_variables[f"{load}.p_cons"][t] for load in load_names)
        total_bess_in = pulp.lpSum(pulp_variables[f"{b}.p_in"][t] for b in bess_names)
        total_bess_out = pulp.lpSum(pulp_variables[f"{b}.p_out"][t] for b in bess_names)
        total_slack_in = pulp.lpSum(pulp_variables[f"{s}.p_in"][t] for s in slack_names)
        total_slack_out = pulp.lpSum(
            pulp_variables[f"{s}.p_out"][t] for s in slack_names
        )

        # Energy balance constraint
        # We need to know which one is in and which one is out
        # Generation + Discharge + Import = Consumption + Charge + Export
        prob += (
            total_pv + total_slack_out + total_bess_out
            == total_load + total_slack_in + total_bess_in
        )

        # Battery constraints
        for b in bess_names:
            p_in = pulp_variables[f"{b}.p_in"]
            p_out = pulp_variables[f"{b}.p_out"]
            e_stor = pulp_variables[f"{b}.e_stor"]
            cap = pulp_variables[f"{b}.capacity"]
            max_p = pulp_variables[f"{b}.max_charge_rate"]

            # maximum input and output power and mutual exclusivity
            k = (t + 1) % time_set.number_of_time_steps
            prob += e_stor[k] - e_stor[t] == p_in[t] - p_out[t]
            prob += e_stor[t] <= cap
            prob += p_in[t] <= min(
                max_p, (total_pv - total_load if total_pv > total_load else 0)
            )
            prob += p_out[t] <= min(
                max_p, (total_load - total_pv if total_load > total_pv else 0)
            )

    # Objective: minimize total exchange with the grid
    prob += pulp.lpSum(
        pulp_variables[f"{s}.p_in"][t] + pulp_variables[f"{s}.p_out"][t]
        for s in slack_names
        for t in range(time_set.number_of_time_steps)
    )   

    status = prob.solve()
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Optimization failed: {pulp.LpStatus[status]}")

    log.info(f"Optimization successful: {pulp.LpStatus[status]}")
    log.info(f"Objective value: {pulp.value(prob.objective)}")

    # 5. EXTRACT RESULTS
    log.info("Extracting results...")
    converter.extract_results_to_runs(model, problem=prob, pulp_variables=pulp_variables, run_id=run_id, time_set=time_set, **kwargs)

    # 6. CLEANUP: Remove solver variables
    log.info("Cleaning up solver variables...")
    for ts_obj in all_timeseries(model):
        run = ts_obj.run(run_id)
        run._vars = None  # Explicitly clear (garbage collection)

    # 7. RETURN RunView (not modified model)
    log.info(f"Optimization complete: {pulp.LpStatus[status]}")
    return RunView(model, run_id)
