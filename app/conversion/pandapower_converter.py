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

from typing import Optional, Union

import pandapower as pp
from numpy import ndarray
from pandapower import pandapowerNet
from pandas import DataFrame

from app.conversion.converter import Converter
from app.conversion.validation_utils import (
    validate_and_normalize_time_range,
    extract_effective_time_properties,
)
from app.infra.quantity import Quantity
from app.infra.parameter import Parameter
from app.infra.timeseries_object import TimeseriesObject
from app.infra.util import TimeSet
from app.model.entity import Entity
from app.model.generator.pv import PV
from app.model.generator.wind_turbine import Wind
from app.model.grid_component.bus import Bus
from app.model.grid_component.line import Line
from app.model.grid_component.transformer import Transformer
from app.model.load.load import Load
from app.model.model import Model
from app.model.slack import Slack
from app.model.storage.battery import Battery
from app.infra.logging_setup import get_logger

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
    DEFAULT_TIME_RANGE_SIZE : int
        Default fallback size for time ranges when needed.
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
        from app.model.generic_entity import GenericEntity

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
        self._entity_converters[GenericEntity] = self._convert_generic_entity

    def convert_model(
        self,
        model: Model,
        time_set: Optional["TimeSet"] = None,
        **kwargs,
    ) -> pandapowerNet:
        """
        Convert a Model into a pandapower network and collect profiles.

        Parameters
        ----------
        model
            The Omnes Model to convert.
        time_set : TimeSet, optional
            TimeSet object containing time configuration. If None the model's
            defaults are used.

        Returns
        -------
        pandapowerNet
            The constructed pandapower network with 'profiles' and 'time_set'
            attributes populated.
        """
        # Use model defaults if not specified, creates effective TimeSet
        effective_time_set = extract_effective_time_properties(model, time_set)

        # Reset network state for new conversion
        self.net = self.create_empty_net()
        self.bus_map = {}

        # Convert all entities to model variables
        for _, entity in model.entities.items():
            logger.info(f"Converting entity '{entity.id}'")
            entity.convert(effective_time_set, self)

        # Add time set information
        self.net.time_set = effective_time_set

        return self.net

    def _convert_entity_default(
        self,
        entity: Entity,
        time_set: Optional["TimeSet"] = None,
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
        time_set : TimeSet, optional
            TimeSet object containing time configuration.
        kwargs
            Internal options: 'entity_type', 'idx', and 'profile_type'.
        """
        entity_type = kwargs.pop("entity_type")
        idx = kwargs.pop("idx", 0)
        profile_type = kwargs.pop("profile_type", entity_type)
        for key, quantity in entity.quantities.items():
            converted_value = self.convert_quantity(
                quantity,
                name=f"{entity.id}_{key}",
                time_set=time_set,
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
    def _convert_bus_entity(self, bus: Bus, time_set=None):
        """Create a pandapower bus for the given Bus entity and extract quantities."""
        idx = self._convert_bus(bus)
        self._convert_entity_default(bus, time_set, entity_type="bus", idx=idx)

    def _convert_line_entity(self, line: Line, time_set=None):
        """Create a pandapower line or switch for the given Line entity and extract quantities."""
        idx, kind = self._convert_line(line)
        self._convert_entity_default(line, time_set, entity_type=kind, idx=idx)

    def _convert_slack_entity(self, slack: Slack, time_set=None):
        """Create an external grid (slack) element and extract quantities."""
        idx = self._convert_slack(slack)
        self._convert_entity_default(
            slack,
            time_set,
            entity_type="ext_grid",
            idx=idx,
            profile_type="powerplants",
        )

    def _convert_pv_entity(self, pv: PV, time_set=None):
        """Create a PV static generator element and extract quantities."""
        idx = self._convert_pv(pv)
        self._convert_entity_default(
            pv,
            time_set,
            entity_type="sgen",
            idx=idx,
            profile_type="renewables",
        )

    def _convert_wind_entity(self, wind: Wind, time_set=None):
        """Create a wind static generator element and extract quantities."""
        idx = self._convert_wind(wind)
        self._convert_entity_default(
            wind,
            time_set,
            entity_type="sgen",
            idx=idx,
            profile_type="renewables",
        )

    def _convert_battery_entity(self, battery: Battery, time_set=None):
        """Create a battery static generator element (modeled as sgen) and extract quantities."""
        idx = self._convert_battery(battery)
        self._convert_entity_default(
            battery,
            time_set,
            entity_type="sgen",
            idx=idx,
            profile_type="storage",
        )

    def _convert_load_entity(self, load: Load, time_set=None):
        """Create a load element and extract quantities."""
        idx = self._convert_load(load)
        self._convert_entity_default(
            load,
            time_set,
            entity_type="load",
            idx=idx,
        )

    def _convert_transformer_entity(self, trafo: Transformer, time_set=None):
        """Create a pandapower transformer (trafo) and extract quantities (if any)."""
        idx = self._convert_transformer(trafo)
        # Use 'trafo' as the pandapower table name
        self._convert_entity_default(
            trafo,
            time_set,
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
            self.net,
            vn_kv=bus.nominal_voltage.value / 1000.0,
            name=bus.id,
            geodata=tuple(bus.coordinates.values()),
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
        # Handle both plain attributes and wrapped (with .value) attributes
        length = line.line_length.value
        resistance = line.resistance.value
        reactance = line.reactance.value
        capacitance = line.capacitance.value
        max_current = line.max_current.value

        if length == 0 or reactance == 0 or resistance == 0:
            idx = pp.create_switch(
                self.net,
                bus=self.bus_map[line.from_bus.split("_")[0]],
                element=self.bus_map[line.to_bus.split("_")[0]],
                et="b",
                type="CB",
                name=line.id,
            )
            kind = "switch"
        else:
            idx = pp.create_line_from_parameters(
                self.net,
                from_bus=self.bus_map[line.from_bus.split("_")[0]],
                to_bus=self.bus_map[line.to_bus.split("_")[0]],
                length_km=length,
                r_ohm_per_km=resistance,
                x_ohm_per_km=reactance,
                c_nf_per_km=capacitance,
                max_i_ka=max_current,
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
            self.net,
            bus=bus_idx,
            p_mw=battery.max_charge_rate.value / 1000.0,
            q_mvar=0.0,
            name=battery.id,
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
            # Default settings: constant impedance load
            const_z_p_percent=1,
            const_z_q_percent=1,
            p_mw=load.nominal_power.value / 1000.0,
            q_mvar=0.0,
            sn_mva=load.nominal_power.value / 1000.0,
            name=load.id,
        )

    def _convert_transformer(self, trafo: Transformer) -> int:
        """
        Create a pandapower transformer from a Transformer entity.

        Uses the transformer's 'type' (std_type) defined earlier via
        GenericEntity -> _convert_generic_entity when available. Falls back
        to creating a transformer from explicit numeric parameters if the
        std_type is not present.

        Returns the index of the created trafo element in the pandapower net.
        """

        # Resolve bus indices
        hv_id = trafo.from_bus
        lv_id = trafo.to_bus
        hv_idx = self.bus_map[hv_id]
        lv_idx = self.bus_map[lv_id]

        # Resolve std_type: try to extract from trafo.type.value
        std_type_name = None
        if hasattr(trafo, "type") and hasattr(trafo.type, "value"):
            std_type_name = trafo.type.value

        # Check if std_type exists in pandapower's trafo types
        if std_type_name:
            trafo_types = self.net.std_types["trafo"].keys()
            if std_type_name not in trafo_types:
                logger.warning(
                    f"Transformer type '{std_type_name}' not found in pandapower std_types."
                )
                std_type_name = None

        # Create transformer, using std_type if available
        return pp.create_transformer(
            self.net,
            hv_bus=hv_idx,
            lv_bus=lv_idx,
            sn_mva=trafo.nominal_power.value / 1000.0,
            std_type=std_type_name,
            name=trafo.id,
        )

    def _convert_generic_entity(self, entity, time_set=None, new_freq=None):
        """
        Convert a GenericEntity that represents a transformer into a pandapower
        trafo_type entry (a transformer type definition stored in the network).

        Expected parameter names on the entity (as Parameters):
        id, sR, vmHV, vmLV, va0, vmImp, pCu, pFe, iNoLoad,
        tapable, tapside, dVm, dVa, tapNeutr, tapMin, tapMax

        The method extracts these parameters from entity.quantities, applies
        reasonable unit conversions (kVA -> MVA, V -> kV, W -> kW) and
        appends a row to the network's trafo_type (or trafo_types) table. The
        function returns the std_type name created (string).
        """

        # Helper to extract parameter value from the entity.quantities mapping
        def _param(name, default=None):
            q = entity.quantities.get(name)
            if q is None:
                return default
            return getattr(q, "value", q)

        # Use the entity-provided id as the std_type name (fallback to entity.id)
        std_type_name = str(_param("id", getattr(entity, "id", "trafo_type")))

        # Build the trafo_type record. Use common pandapower field names where possible
        trafo_type_record = {
            "std_type": std_type_name,
            "sn_mva": _param("sR", None),
            "vn_hv_kv": _param("vmHV", None),
            "vn_lv_kv": _param("vmLV", None),
            # map SimBench vmImp -> vk_percent (short-circuit impedance)
            "vk_percent": _param("vmImp", None),
            # vkr_percent (resistance percent) not provided by default, set to zero
            "vkr_percent": 0.8,
            # copper and iron losses (convert names appropriately)
            "i0_percent": _param("iNoLoad", None),
            "tap_step_percent": _param("dVm", None),
            "tap_step_degree": _param("dVa", None),
            "tap_neutral": _param("tapNeutr", None),
            "tap_min": _param("tapMin", None),
            "tap_max": _param("tapMax", None),
            "pfe_kw": _param("pFe", None),
            "shift_degree": _param("va0", 150),
        }
        # write back (pandas DataFrame is mutable, but ensure attribute stays set)
        pp.create_std_type(self.net, trafo_type_record, std_type_name, element="trafo")

        logger.info(f"Created trafo type '{std_type_name}'")

        # Return the created type name so callers can reference it (if needed)
        return std_type_name

    def convert_quantity(
        self,
        quantity: Quantity,
        name: str,
        time_set: Optional["TimeSet"] = None,
    ) -> Optional[ndarray]:
        """
        Convert a Quantity to a numpy array suitable for pandapower profiles.

        - Returns None if the quantity is empty.
        - For Parameter instances, returns the parameter's scalar or array value.
        - For Timeseries-like quantities, returns values resampled to the
          requested time set configuration.

        Parameters
        ----------
        quantity
            The Quantity to convert.
        name
            Human-readable identifier used for logging/debugging.
        time_set : TimeSet, optional
            TimeSet object containing time configuration.

        Returns
        -------
        numpy.ndarray or None
            Array of values across the time range, or None if empty.
        """
        if quantity.empty():
            return None
        if isinstance(quantity, Parameter):
            return quantity.value
        else:
            # Extract time range and frequency from TimeSet
            time_range = time_set.number_of_time_steps if time_set else None
            freq = time_set.freq if time_set else None
            return quantity.value(time_set=time_range, freq=freq)

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
            for _, ent in self.model.entities.items():
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
                for _, ent in self.model.entities.items():
                    if getattr(ent, "id", None) == name:
                        setattr(ent, "last_loading_percent", loading)
