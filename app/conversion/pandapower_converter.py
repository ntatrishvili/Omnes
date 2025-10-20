"""
app/conversion/pandapower_converter.py

A converter that maps a minimal Omnes Model to a pandapower.Net and provides helpers
to run a single timestep (set injections/loads from model timeseries, run runpp,
and write results back to the Omnes model).

Limitations & notes:
 - This converter currently models everything as *balanced* (single-phase equivalent).
   For full three-phase / unbalanced studies you should use pandapower's three-phase
   extension (or pandapower 3-phase API) and extend this converter to handle per-phase
   injections and line geometries.
 - Batteries and flexible devices are modeled as sgen/load elements whose p_mw are set
   externally each timestep. If you want integrated control inside pandapower, implement
   controllers (e.g. using pandapower's controller framework).
 - We assume device time series provide values in kW. Conversion to MW is handled here.
"""

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
from typing import Tuple, Dict, Any

from app.model.grid_component.line import Line


class PandapowerConverter:
    def __init__(self, model):
        self.model = model
        self.net = pp.create_empty_network()
        # keep maps for entity id -> pandapower element index
        self.bus_map = {}
        self.load_map = {}
        self.sgen_map = {}
        self.line_map = {}

    def _to_kv(self, nominal_voltage_v: float) -> float:
        # Convert given nominal voltage (V) to kV for pandapower
        return float(nominal_voltage_v) / 1000.0

    def build_buses_and_slack(self):
        # create buses
        for ent in self.model.entities:
            # we only create Bus and Slack here
            from app.model.grid_component.bus import Bus as OmnesBus
            from app.model.slack import Slack as OmnesSlack
            if isinstance(ent, OmnesBus):
                vn_kv = self._to_kv(getattr(ent, "nominal_voltage", 400))
                # pandapower bus name stored as 'name'
                idx = pp.create_bus(self.net, vn_kv=vn_kv, name=ent.id)
                self.bus_map[ent.id] = idx

        # create slack / external grid
        for ent in self.model.entities:
            from app.model.slack import Slack as OmnesSlack
            if isinstance(ent, OmnesSlack):
                # assume slack refers to a bus id
                slack_bus_idx = self.bus_map.get(ent.bus)
                if slack_bus_idx is None:
                    raise ValueError(f"Slack refers to unknown bus {ent.bus}")
                # create an external grid on that bus
                pp.create_ext_grid(self.net, bus=slack_bus_idx, vm_pu=1.0, name=ent.id)

    def build_lines(self):
        # create simple lines (single segment radial lines)
        from app.model.grid_component.line import Line as OmnesLine
        for ent in self.model.entities:
            if isinstance(ent, OmnesLine):
                from_idx = self.bus_map.get(ent.from_bus)
                to_idx = self.bus_map.get(ent.to_bus)
                if from_idx is None or to_idx is None:
                    raise ValueError(f"Line {ent.id} references unknown bus")
                # use defaults or per-line values if available
                length_km = getattr(ent, "length", getattr(Line, "default_line_length", 0.1))
                # default line type selection: use 'underground_cable' or 'overhead_line' as available
                # We create line with simple standard type using pandapower.create_line_from_parameters
                r_ohm_per_km = getattr(ent, "r", getattr(Line, "default_resistance", 0.05))
                x_ohm_per_km = getattr(ent, "x", getattr(Line, "default_reactance", 0.05))
                c_nf_per_km = getattr(ent, "c", 0)  # optional
                pp.create_line_from_parameters(
                    self.net,
                    from_idx,
                    to_idx,
                    length_km=length_km,
                    r_ohm_per_km=r_ohm_per_km,
                    x_ohm_per_km=x_ohm_per_km,
                    c_nf_per_km=c_nf_per_km,
                    max_i_ka=0.2,
                    name=ent.id,
                )
                self.line_map[ent.id] = (from_idx, to_idx)

    def build_loads_and_gens(self):
        # Loads
        from app.model.load.load import Load as OmnesLoad
        from app.model.generator.pv import PV as OmnesPV
        from app.model.generator.wind_turbine import Wind as OmnesWind
        from app.model.storage.battery import Battery as OmnesBattery
        for ent in self.model.entities:
            if isinstance(ent, OmnesLoad):
                bus_idx = self.bus_map.get(ent.bus)
                if bus_idx is None:
                    raise ValueError(f"Load {ent.id} references unknown bus {ent.bus}")
                # create pandapower load with p_mw = 0 initially; we set time series later
                idx = pp.create_load(self.net, bus=bus_idx, p_mw=0.0, q_mvar=0.0, name=ent.id)
                self.load_map[ent.id] = idx
            elif isinstance(ent, (OmnesPV, OmnesWind)):
                bus_idx = self.bus_map.get(ent.bus)
                if bus_idx is None:
                    raise ValueError(f"Generator {ent.id} references unknown bus {ent.bus}")
                # create static generator (sgen) element - set p_mw = 0 initially
                # We will set sgen injection as negative p_mw (injection) when running timesteps.
                idx = pp.create_sgen(self.net, bus=bus_idx, p_mw=0.0, q_mvar=0.0, name=ent.id)
                self.sgen_map[ent.id] = idx
            elif isinstance(ent, OmnesBattery):
                # for now create an sgen for battery (we control sign externally)
                bus_idx = self.bus_map.get(ent.bus)
                if bus_idx is None:
                    raise ValueError(f"Battery {ent.id} references unknown bus {ent.bus}")
                idx = pp.create_sgen(self.net, bus=bus_idx, p_mw=0.0, q_mvar=0.0, name=ent.id)
                self.sgen_map[ent.id] = idx

    def model_to_pp_net(self) -> pp.networks.Net:
        """
        Build a pandapower network from the Omnes model. After this call self.net is populated.
        """
        # create buses & slack first
        self.build_buses_and_slack()
        # lines
        self.build_lines()
        # loads, gens, storages (as sgens)
        self.build_loads_and_gens()

        # done
        return self.net

    def _get_entity_profile_value(self, entity, timestep_idx: int) -> float:
        """
        Given an Omnes entity that contains input={"input_path": "..."} or
        input={"input_path": "...", "col": "..."} this helper loads the CSV and returns
        the value at timestep_idx. Returns value in kW (as expected in Omnes).
        """
        import pandas as pd
        ip = getattr(entity, "input", None)
        if not ip:
            return 0.0
        path = ip.get("input_path")
        if not path:
            return 0.0
        df = pd.read_csv(path)
        col = ip.get("col", None)
        if col:
            values = df[col].values
        else:
            # if timestamp present, skip timestamp column
            if "timestamp" in df.columns:
                values = df["value"].values if "value" in df.columns else df.iloc[:, 1].values
            else:
                # single column
                if df.shape[1] == 1:
                    values = df.iloc[:, 0].values
                else:
                    # choose the last column as values (fallback)
                    values = df.iloc[:, -1].values
        # guard index
        if timestep_idx < 0 or timestep_idx >= len(values):
            raise IndexError(f"Requested timestep {timestep_idx} out of range for file {path}")
        return float(values[timestep_idx])

    def set_timestep_values(self, timestep_idx: int):
        """
        For each load/sgens set p_mw according to the model time-series values (kW -> MW).
        We use convention:
            - Loads -> create_load.p_mw is positive consumption (kW -> MW /1000)
            - Generators (pv/wind/battery) -> sgen.p_mw is NEGATIVE for injection into network
              (i.e., sgen.p_mw = -gen_kW/1000)
          (This sign convention is common but please ensure consistency in your analysis.)
        """
        # set loads
        from app.model.load.load import Load as OmnesLoad
        from app.model.generator.pv import PV as OmnesPV
        from app.model.generator.wind_turbine import Wind as OmnesWind
        from app.model.storage.battery import Battery as OmnesBattery

        # loads
        for ent in self.model.entities:
            if isinstance(ent, OmnesLoad):
                idx = self.load_map.get(ent.id)
                if idx is None:
                    continue
                kw = self._get_entity_profile_value(ent, timestep_idx)
                self.net.load.at[idx, "p_mw"] = float(kw) / 1000.0

        # sgens (pv, wind, battery)
        for ent in self.model.entities:
            if isinstance(ent, (OmnesPV, OmnesWind, OmnesBattery)):
                idx = self.sgen_map.get(ent.id)
                if idx is None:
                    continue
                # for PV/Wind we read their input CSV (or if not present use peak_power as simple proxy)
                kw = None
                if hasattr(ent, "input") and ent.input:
                    try:
                        kw = self._get_entity_profile_value(ent, timestep_idx)
                    except Exception:
                        kw = None
                if kw is None:
                    # fallback: use peak_power attribute if exists (assume instantaneous generation = peak)
                    kw = getattr(ent, "peak_power", 0.0)
                # convert to p_mw for pandapower and set negative for injection
                self.net.sgen.at[idx, "p_mw"] = -float(kw) / 1000.0

    def run_timestep_and_export_results(self, timestep_idx: int) -> Dict[str, Any]:
        """
        Sets timestep values, runs pandapower power flow, and returns results and writes
        basic summaries back into the Omnes model entities (adds attributes like vm_pu, loading_percent).
        """
        # set p_mw values from time series
        self.set_timestep_values(timestep_idx)

        # run power flow
        pp.runpp(self.net)

        # get bus voltages and line loadings and write back
        bus_results = {}
        for bus_id, bus_idx in self.bus_map.items():
            vm_pu = float(self.net.res_bus.at[bus_idx, "vm_pu"]) if "vm_pu" in self.net.res_bus.columns else None
            bus_results[bus_id] = {"vm_pu": vm_pu}
            # store back to Omnes bus object if you want
            # find the omnibus entity
            for ent in self.model.entities:
                if getattr(ent, "id", None) == bus_id:
                    setattr(ent, "last_vm_pu", vm_pu)

        line_results = {}
        if hasattr(self.net, "res_line"):
            for li_idx in self.net.line.index:
                name = self.net.line.at[li_idx, "name"]
                # loading_percent column is common after runpp
                loading = float(self.net.res_line.at[li_idx, "loading_percent"]) if "loading_percent" in self.net.res_line.columns else None
                line_results[name] = {"loading_percent": loading}
                # save back to model line entity
                for ent in self.model.entities:
                    if getattr(ent, "id", None) == name:
                        setattr(ent, "last_loading_percent", loading)

        results = {"buses": bus_results, "lines": line_results, "net": self.net}
        return results
