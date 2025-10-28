"""
App conversion module: pandapower_converter

Provides a converter that maps a minimal Omnes Model to a pandapower network
(pandapowerNet). The converter models a balanced (single-phase equivalent)
network and extracts time series profiles for use with pandapower simulations.

Notes
- Balanced single-phase equivalent is used. Extend for three-phase/unbalanced use.
- Batteries and flexible devices are modeled as sgen/load elements whose p_mw
  are provided externally for each timestep.
- Time series values are expected in kW and are converted to MW where needed.
"""

import logging
from typing import Optional, Union
import re

import pandapower as pp
from numpy import ndarray
from pandapower import pandapowerNet
from pandas import DataFrame

from app.conversion.converter import Converter
from app.conversion.validation_utils import (
    validate_and_normalize_time_set,
    extract_effective_time_properties,
)
from app.infra.quantity import Parameter, Quantity
from app.infra.timeseries_object import TimeseriesObject
from app.model.entity import Entity
from app.model.generator.pv import PV
from app.model.generator.wind_turbine import Wind
from app.model.grid_component.bus import Bus
from app.model.grid_component.line import Line
from app.model.load.load import Load
from app.model.model import Model
from app.model.slack import Slack
from app.model.storage.battery import Battery
from app.model.grid_component.transformer import Transformer
from utils.logging_setup import get_logger

logger = get_logger(__name__)


class PandapowerConverter(Converter):
    """
    Convert an Omnes Model into a pandapower network and associated time series.

    The converter constructs pandapower network elements (buses, lines, sgens,
    loads, ext_grids, switches) and collects time series data into the network's
    'profiles' attribute. Time series are converted to numpy arrays for use in
    simulations.

    Attributes
    ----------
    DEFAULT_TIME_SET_SIZE : int
        Default fallback size for time sets when needed.
    net : pandapowerNet
        The pandapower network being constructed.
    bus_map : dict
        Mapping from Omnes bus id to pandapower bus index.
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.net = self.create_empty_net()
        self.bus_map = {}

    def create_empty_net(self):
        net = pp.create_empty_network()
        net.profiles = {
            "load": DataFrame(),
            "renewables": DataFrame(),
            "storage": DataFrame(),
            "powerplants": DataFrame(),
        }
        return net

    def _register_converters(self):
        """
        Register entity-type -> converter method mappings.

        This initializes the internal mapping that assigns a conversion method
        to each special entity class handled by this converter.
        """
        from app.model.grid_component.bus import Bus
        from app.model.grid_component.line import Line
        from app.model.slack import Slack
        from app.model.generator.pv import PV
        from app.model.generator.wind_turbine import Wind
        from app.model.storage.battery import Battery
        from app.model.load.load import Load

        # Register entity types that need special network element creation
        self._entity_converters[Bus] = self._convert_bus_entity
        self._entity_converters[Line] = self._convert_line_entity
        self._entity_converters[Slack] = self._convert_slack_entity
        self._entity_converters[PV] = self._convert_pv_entity
        self._entity_converters[Wind] = self._convert_wind_entity
        self._entity_converters[Battery] = self._convert_battery_entity
        self._entity_converters[Load] = self._convert_load_entity
        # Transformer converter
        self._entity_converters[Transformer] = self._convert_transformer_entity

    def convert_model(
        self,
        model: Model,
        time_set: Optional[Union[int, range]] = None,
        new_freq: Optional[str] = None,
    ) -> pandapowerNet:
        """
        Convert a Model into a pandapower network and collect profiles.

        Parameters
        ----------
        model
            The Omnes Model to convert.
        time_set
            Optional time set specification (int or range). If None the model's
            defaults are used.
        new_freq
            Optional target frequency for resampling time series (e.g., '15min').

        Returns
        -------
        pandapowerNet
            The constructed pandapower network with 'profiles' and 'time_set'
            attributes populated.
        """
        # Use model defaults if not specified
        effective_freq, effective_time_set = extract_effective_time_properties(
            model, new_freq, time_set
        )

        # Reset network state for new conversion
        self.net = self.create_empty_net()
        self.bus_map = {}

        # Convert all entities to model variables
        for entity in model.entities:
            logger.info(f"Converting entity '{entity.id}'")
            entity.convert(effective_time_set, effective_freq, self)

        # Add time set information
        time_range = validate_and_normalize_time_set(
            effective_time_set, self.DEFAULT_TIME_SET_SIZE
        )
        self.net.time_set = time_range

        return self.net

    def _convert_entity_default(
        self,
        entity: Entity,
        time_set: Optional[Union[int, range]] = None,
        new_freq: Optional[str] = None,
        **kwargs,
    ):
        """
        Default conversion routine for an entity's quantities.

        Extracts parameters and time series quantities from an entity and stores
        them into the appropriate pandapower tables (e.g., net.bus, net.load)
        or into net.profiles.

        Parameters
        ----------
        entity
            The entity to process.
        time_set
            Optional time-step specification forwarded to quantity conversion.
        new_freq
            Optional frequency forwarded to quantity conversion.
        kwargs
            Internal options: 'entity_type', 'idx', and 'profile_type'.
        """
        entity_type = kwargs.pop("entity_type")
        idx = kwargs.pop("idx", 0)
        profile_type = kwargs.pop("profile_type", entity_type)
        for key, quantity in entity.quantities.items():
            converted_value = self.convert_quantity(
                quantity, name=f"{entity.id}_{key}", time_set=time_set, freq=new_freq
            )
            if converted_value is None:
                continue
            elif isinstance(quantity, Parameter):
                self.net[entity_type].loc[idx, key] = converted_value
            elif isinstance(quantity, TimeseriesObject):
                self.net.profiles[profile_type].loc[
                    :, f"{entity.id}_{key}"
                ] = converted_value

    # Wrapper methods that create network elements AND extract quantities
    def _convert_bus_entity(self, bus: Bus, time_set=None, new_freq=None):
        """Create a pandapower bus for the given Bus entity and extract quantities."""
        idx = self._convert_bus(bus)
        self._convert_entity_default(
            bus, time_set, new_freq, entity_type="bus", idx=idx
        )

    def _convert_line_entity(self, line: Line, time_set=None, new_freq=None):
        """Create a pandapower line or switch for the given Line entity and extract quantities."""
        idx, kind = self._convert_line(line)
        self._convert_entity_default(
            line, time_set, new_freq, entity_type=kind, idx=idx
        )

    def _convert_slack_entity(self, slack: Slack, time_set=None, new_freq=None):
        """Create an external grid (slack) element and extract quantities."""
        idx = self._convert_slack(slack)
        self._convert_entity_default(
            slack,
            time_set,
            new_freq,
            entity_type="ext_grid",
            idx=idx,
            profile_type="powerplants",
        )

    def _convert_pv_entity(self, pv: PV, time_set=None, new_freq=None):
        """Create a PV static generator element and extract quantities."""
        idx = self._convert_pv(pv)
        self._convert_entity_default(
            pv,
            time_set,
            new_freq,
            entity_type="sgen",
            idx=idx,
            profile_type="renewables",
        )

    def _convert_wind_entity(self, wind: Wind, time_set=None, new_freq=None):
        """Create a wind static generator element and extract quantities."""
        idx = self._convert_wind(wind)
        self._convert_entity_default(
            wind,
            time_set,
            new_freq,
            entity_type="sgen",
            idx=idx,
            profile_type="renewables",
        )

    def _convert_battery_entity(self, battery: Battery, time_set=None, new_freq=None):
        """Create a battery static generator element (modeled as sgen) and extract quantities."""
        idx = self._convert_battery(battery)
        self._convert_entity_default(
            battery,
            time_set,
            new_freq,
            entity_type="sgen",
            idx=idx,
            profile_type="storage",
        )

    def _convert_load_entity(self, load: Load, time_set=None, new_freq=None):
        """Create a load element and extract quantities."""
        idx = self._convert_load(load)
        self._convert_entity_default(
            load, time_set, new_freq, entity_type="load", idx=idx
        )

    def _convert_transformer_entity(
        self, trafo: Transformer, time_set=None, new_freq=None
    ):
        """Create a pandapower transformer (trafo) and extract quantities (if any)."""
        idx = self._convert_transformer(trafo)
        # Use 'trafo' as the pandapower table name
        self._convert_entity_default(
            trafo,
            time_set,
            new_freq,
            entity_type="trafo",
            idx=idx,
        )

    def _convert_bus(self, bus: Bus) -> int:
        """
        Create a pandapower bus from a Bus entity.

        Parameters
        ----------
        bus
            The Bus entity to convert.

        Returns
        -------
        int
            Index of the created pandapower bus.
        """
        b = pp.create_bus(
            self.net, vn_kv=bus.nominal_voltage.value / 1000.0, name=bus.id
        )
        self.bus_map[bus.id] = b
        return b

    def _convert_line(self, line: Line) -> tuple[int, str]:
        """
        Create a pandapower line or switch from a Line entity.

        A switch is created when line length, resistance or reactance are zero;
        otherwise a line with parameters is created.

        Parameters
        ----------
        line
            The Line entity to convert.

        Returns
        -------
        tuple[int, str]
            (element index, kind) where kind is 'line' or 'switch'.
        """
        if line.line_length == 0 or line.reactance == 0 or line.resistance == 0:
            idx = pp.create_switch(
                self.net,
                bus=self.bus_map[line.from_bus],
                element=self.bus_map[line.to_bus],
                et="b",
                type="CB",
            )
            kind = "switch"
        else:
            idx = pp.create_line_from_parameters(
                self.net,
                from_bus=self.bus_map[line.from_bus],
                to_bus=self.bus_map[line.to_bus],
                length_km=line.line_length.value,
                r_ohm_per_km=line.resistance.value / line.line_length.value,
                x_ohm_per_km=line.reactance.value / line.line_length.value,
                c_nf_per_km=0,
                max_i_ka=line.max_current.value / 1000.0,
                name=line.id,
            )
            kind = "line"
        return idx, kind

    def _convert_slack(self, slack: Slack) -> int:
        """
        Create an external grid (ext_grid) connected to the given bus.

        Parameters
        ----------
        slack
            The Slack entity to convert.

        Returns
        -------
        int
            Index of the created ext_grid element.
        """
        return pp.create_ext_grid(self.net, bus=self.bus_map[slack.bus], name=slack.id)

    def _convert_pv(self, pv: PV) -> int:
        """
        Create a static generator element for a PV entity.

        The static generator is initialized with p_mw equal to the PV's peak
        power (converted to MW).

        Parameters
        ----------
        pv
            The PV entity to convert.

        Returns
        -------
        int
            Index of the created static generator.
        """
        return pp.create_sgen(
            self.net,
            bus=self.bus_map[pv.bus],
            p_mw=pv.peak_power.value / 1000.0,
            q_mvar=0,
            name=pv.id,
        )

    def _convert_wind(self, wind: Wind) -> int:
        """
        Create a static generator element for a wind entity.

        Parameters
        ----------
        wind
            The Wind entity to convert.

        Returns
        -------
        int
            Index of the created static generator.
        """
        return pp.create_sgen(
            self.net,
            bus=self.bus_map[wind.bus],
            p_mw=wind.peak_power.value / 1000.0,
            q_mvar=0,
            name=wind.id,
        )

    def _convert_battery(self, battery: Battery) -> int:
        """
        Create a static generator element representing a battery.

        The battery is represented as a controllable sgen whose p_mw will be
        set externally during simulation.

        Parameters
        ----------
        battery
            The Battery entity to convert.

        Returns
        -------
        int
            Index of the created sgen element.
        """
        bus_idx = self.bus_map[battery.bus]
        return pp.create_sgen(
            self.net, bus=bus_idx, p_mw=0.0, q_mvar=0.0, name=battery.id
        )

    def _convert_load(self, load: Load) -> int:
        """
        Create a pandapower load element from a Load entity.

        Parameters
        ----------
        load
            The Load entity providing 'p_kw' and 'q_kw' quantities.

        Returns
        -------
        int
            Index of the created load element.
        """
        return pp.create_load(
            self.net,
            bus=self.bus_map[load.bus],
            p_mw=load.tags["p_kw"] / 1000.0,
            q_mvar=load.tags["q_kw"] / 1000.0,
            name=load.id,
        )

    def _convert_transformer(self, trafo: Transformer) -> int:
        """
        Create a pandapower transformer from a Transformer entity.

        Attempt to parse SN (MVA) and HV/LV kV from trafo.type string when present.
        Provide reasonable defaults if parsing fails.
        """
        # Defaults
        sn_mva = 0.16  # default from example
        vn_hv_kv = 20.0
        vn_lv_kv = 0.4
        # Try to parse e.g. "0.16 MVA 20/0.4 kV" patterns
        if isinstance(trafo.type, str):
            # parse leading SN value
            m_sn = re.search(r"(\d+(\.\d+)?)\s*MVA", trafo.type, flags=re.IGNORECASE)
            if m_sn:
                try:
                    sn_mva = float(m_sn.group(1))
                except Exception:
                    pass
            # parse hv/lv kV values like "20/0.4 kV"
            m_kv = re.search(
                r"(\d+(\.\d+)?)\s*/\s*(\d+(\.\d+)?)\s*kV",
                trafo.type,
                flags=re.IGNORECASE,
            )
            if m_kv:
                try:
                    vn_hv_kv = float(m_kv.group(1))
                    vn_lv_kv = float(m_kv.group(3))
                except Exception:
                    pass

        # fallback: if loading_max provided, consider it as kW and convert to MVA-ish guess (not strict)
        if getattr(trafo, "loading_max", None) and (not sn_mva):
            try:
                sn_mva = float(trafo.loading_max) / 1000.0
            except Exception:
                pass

        # Tap position and side
        tp_pos = int(getattr(trafo, "tappos", 0) or 0)
        tp_side = getattr(trafo, "auto_tap_side", None)
        tp_side = "hv" if str(tp_side or "").lower() == "hv" else "lv"

        # Create transformer in pandapower using parameter-based creation
        # Use moderate default short-circuit impedance values
        vsc_percent = 6.0
        vscr_percent = 0.3
        try:
            trafo_idx = pp.create_transformer_from_parameters(
                self.net,
                hv_bus=self.bus_map[trafo.hv_bus],
                lv_bus=self.bus_map[trafo.lv_bus],
                sn_mva=sn_mva,
                vn_hv_kv=vn_hv_kv,
                vn_lv_kv=vn_lv_kv,
                vsc_percent=vsc_percent,
                vscr_percent=vscr_percent,
                pfe_kw=0.0,
                i0_percent=0.0,
                tp_pos=tp_pos,
                tp_side=tp_side,
                name=trafo.id,
            )
        except Exception:
            # If parameter-based creation fails, try the simpler create_transformer (std_type may not exist)
            trafo_idx = pp.create_transformer(
                self.net,
                hv_bus=self.bus_map[trafo.hv_bus],
                lv_bus=self.bus_map[trafo.lv_bus],
                std_type=None,
                name=trafo.id,
            )
        return trafo_idx

    def convert_quantity(
        self,
        quantity: Quantity,
        name: str,
        time_set: Optional[Union[int, range]] = None,
        freq: Optional[str] = None,
    ) -> Optional[ndarray]:
        """
        Convert a Quantity to a numpy array suitable for pandapower profiles.

        - Returns None if the quantity is empty.
        - For Parameter instances, returns the parameter's scalar or array value.
        - For Timeseries-like quantities, returns values resampled to the
          requested time_set and frequency.

        Parameters
        ----------
        quantity
            The Quantity to convert.
        name
            Human-readable identifier used for logging/debugging.
        time_set
            Optional time set specification for resampling.
        freq
            Optional frequency for resampling.

        Returns
        -------
        numpy.ndarray or None
            Array of values across the time set, or None if empty.
        """
        if quantity.empty():
            return None
        if isinstance(quantity, Parameter):
            return quantity.value
        else:
            return quantity.value(time_set=time_set, freq=freq)

    def convert_back(self, model: Model) -> None:
        """
        Map simulation results from a pandapower network back onto model entities.

        Updates model entities (for example, last_vm_pu or last_loading_percent)
        with values read from the network results tables.

        Parameters
        ----------
        net
            The pandapower network containing simulation results.
        model
            The Omnes model to update.
        """
        # Example: Update bus voltages in the model
        # get bus voltages and line loadings and write back
        bus_results = {}
        for bus_id, bus_idx in self.bus_map.items():
            vm_pu = (
                float(self.net.res_bus.at[bus_idx, "vm_pu"])
                if "vm_pu" in self.net.res_bus.columns
                else None
            )
            bus_results[bus_id] = {"vm_pu": vm_pu}
            # store back to Omnes bus object if you want
            # to find the omnibus entity
            for ent in self.model.entities:
                if getattr(ent, "id", None) == bus_id:
                    setattr(ent, "last_vm_pu", vm_pu)

        line_results = {}
        if hasattr(self.net, "res_line"):
            for li_idx in self.net.line.index:
                name = self.net.line.at[li_idx, "name"]
                # loading_percent column is common after runpp
                loading = (
                    float(self.net.res_line.at[li_idx, "loading_percent"])
                    if "loading_percent" in self.net.res_line.columns
                    else None
                )
                line_results[name] = {"loading_percent": loading}
                # save back to model line entity
                for ent in self.model.entities:
                    if getattr(ent, "id", None) == name:
                        setattr(ent, "last_loading_percent", loading)

        results = {"buses": bus_results, "lines": line_results, "net": self.net}
