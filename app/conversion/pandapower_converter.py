"""
app/conversion/pandapower_converter.py

A converter that maps a minimal Omnes Model to a pandapowerNet.

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

from typing import Optional, Union, Dict, Any

import numpy as np
import pandapower as pp
from numpy import ndarray
from pandapower import pandapowerNet

from app.conversion.converter import Converter
from app.conversion.validation_utils import (
    validate_and_normalize_time_set,
    extract_effective_time_properties,
)
from app.infra.quantity import Parameter, Quantity
from app.model.entity import Entity
from app.model.generator.pv import PV
from app.model.generator.wind_turbine import Wind
from app.model.grid_component.bus import Bus
from app.model.grid_component.line import Line
from app.model.load.load import Load
from app.model.model import Model
from app.model.slack import Slack
from app.model.storage.battery import Battery


class PandapowerConverter(Converter):
    """
    Converts a Model class object into a pandapower network and model variables.
    """

    DEFAULT_TIME_SET_SIZE = 10

    def __init__(self):
        super().__init__()
        self.model = None
        self.net = pp.create_empty_network()
        self.bus_map = {}

    def _register_converters(self):
        """
        Register specialized converters for network elements.
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

    def convert_model(
        self,
        model: Model,
        time_set: Optional[Union[int, range]] = None,
        new_freq: Optional[str] = None,
    ) -> tuple[pandapowerNet, Dict[str, Any]]:
        """
        Convert the model to a pandapower network and extract model variables.

        Converts the model's entities into a pandapower network structure (buses, lines,
        generators, loads, etc.) and extracts their quantities as numpy arrays for time
        series simulation. Handles resampling of time series data to the specified
        frequency and time set.

        Parameters
        ----------
        model : Model
            The model to convert.
        time_set : Optional[Union[int, range]], optional
            The number of time steps to include in the output arrays.
            If None, uses model.number_of_time_steps.
        new_freq : Optional[str], optional
            The target frequency to resample time series data to (e.g., '15min', '1H').
            If None, uses model.frequency.

        Returns
        -------
        tuple[pandapower.Net, Dict[str, Any]]
            A tuple containing:
            - The pandapower network object with all elements created
            - Dictionary containing time series data as numpy arrays and time set information.
              Includes a 'time_set' key with the range of time steps.
        """
        # Use model defaults if not specified
        effective_freq, effective_time_set = extract_effective_time_properties(
            model, new_freq, time_set
        )

        # Reset network state for new conversion
        self.net = pp.create_empty_network()
        self.bus_map = {}

        # Convert all entities to model variables
        model_variables = {}
        for entity in model.entities:
            model_variables.update(
                entity.convert(effective_time_set, effective_freq, self)
            )

        # Add time set information
        time_range = validate_and_normalize_time_set(
            effective_time_set, self.DEFAULT_TIME_SET_SIZE
        )
        model_variables["time_set"] = time_range

        return self.net, model_variables

    def _convert_entity_default(
        self,
        entity: Entity,
        time_set: Optional[Union[int, range]] = None,
        new_freq: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Default entity conversion - just extract quantities without creating network elements.

        Parameters
        ----------
        entity : Entity
            The entity to convert.
        time_set : Optional[Union[int, range]], optional
            The number of time steps to represent in the resulting arrays.
        new_freq : Optional[str], optional
            The target frequency to resample time series data to.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing numpy arrays for the entity's quantities.
        """
        entity_variables = {
            f"{entity.id}.{key}": self.convert_quantity(
                quantity,
                name=f"{entity.id}.{key}",
                time_set=time_set,
                freq=new_freq,
            )
            for key, quantity in entity.quantities.items()
        }

        for sub_entity in entity.sub_entities:
            entity_variables.update(self.convert_entity(sub_entity, time_set, new_freq))

        return entity_variables

    # Wrapper methods that create network elements AND extract quantities
    def _convert_bus_entity(
        self, bus: Bus, time_set=None, new_freq=None
    ) -> Dict[str, Any]:
        """Wrapper that creates bus network element and extracts quantities."""
        self._convert_bus(bus)
        return self._convert_entity_default(bus, time_set, new_freq)

    def _convert_line_entity(
        self, line: Line, time_set=None, new_freq=None
    ) -> Dict[str, Any]:
        """Wrapper that creates line network element and extracts quantities."""
        self._convert_line(line)
        return self._convert_entity_default(line, time_set, new_freq)

    def _convert_slack_entity(
        self, slack: Slack, time_set=None, new_freq=None
    ) -> Dict[str, Any]:
        """Wrapper that creates slack network element and extracts quantities."""
        self._convert_slack(slack)
        return self._convert_entity_default(slack, time_set, new_freq)

    def _convert_pv_entity(
        self, pv: PV, time_set=None, new_freq=None
    ) -> Dict[str, Any]:
        """Wrapper that creates PV network element and extracts quantities."""
        self._convert_pv(pv)
        return self._convert_entity_default(pv, time_set, new_freq)

    def _convert_wind_entity(
        self, wind: Wind, time_set=None, new_freq=None
    ) -> Dict[str, Any]:
        """Wrapper that creates wind turbine network element and extracts quantities."""
        self._convert_wind(wind)
        return self._convert_entity_default(wind, time_set, new_freq)

    def _convert_battery_entity(
        self, battery: Battery, time_set=None, new_freq=None
    ) -> Dict[str, Any]:
        """Wrapper that creates battery network element and extracts quantities."""
        self._convert_battery(battery)
        return self._convert_entity_default(battery, time_set, new_freq)

    def _convert_load_entity(
        self, load: Load, time_set=None, new_freq=None
    ) -> Dict[str, Any]:
        """Wrapper that creates load network element and extracts quantities."""
        self._convert_load(load)
        return self._convert_entity_default(load, time_set, new_freq)

    def _convert_bus(self, bus: Bus) -> None:
        """
        Convert a Bus entity to a pandapower bus element.

        Parameters
        ----------
        bus : Bus
            The bus entity to convert
        """
        b = pp.create_bus(self.net, vn_kv=bus.nominal_voltage / 1000.0, name=bus.id)
        self.bus_map[bus.id] = b

    def _convert_line(self, line: Line) -> None:
        """
        Convert a Line entity to a pandapower line or switch element.

        Creates a switch if line length, resistance, or reactance is zero,
        otherwise creates a line from parameters.

        Parameters
        ----------
        line : Line
            The line entity to convert
        """
        if line.line_length == 0 or line.reactance == 0 or line.resistance == 0:
            pp.create_switch(
                self.net,
                bus=self.bus_map[line.from_bus],
                element=self.bus_map[line.to_bus],
                et="b",
            )
        else:
            pp.create_line_from_parameters(
                self.net,
                from_bus=self.bus_map[line.from_bus],
                to_bus=self.bus_map[line.to_bus],
                length_km=line.line_length,
                r_ohm_per_km=line.resistance / line.line_length,
                x_ohm_per_km=line.reactance / line.line_length,
                c_nf_per_km=0,
                max_i_ka=line.max_current / 1000.0,
                name=line.id,
            )

    def _convert_slack(self, slack: Slack) -> None:
        """
        Convert a Slack entity to a pandapower external grid element.

        Parameters
        ----------
        slack : Slack
            The slack bus entity to convert
        """
        pp.create_ext_grid(self.net, bus=self.bus_map[slack.bus], name=slack.id)

    def _convert_pv(self, pv: PV) -> None:
        """
        Convert a PV entity to a pandapower static generator element.

        Parameters
        ----------
        pv : PV
            The PV entity to convert
        """
        pp.create_sgen(
            self.net,
            bus=self.bus_map[pv.bus],
            p_mw=pv.peak_power / 1000.0,
            q_mvar=0,
            name=pv.id,
        )

    def _convert_wind(self, wind: Wind) -> None:
        """
        Convert a Wind entity to a pandapower static generator element.

        Parameters
        ----------
        wind : Wind
            The wind turbine entity to convert
        """
        pp.create_sgen(
            self.net,
            bus=self.bus_map[wind.bus],
            p_mw=wind.peak_power / 1000.0,
            q_mvar=0,
            name=wind.id,
        )

    def _convert_battery(self, battery: Battery) -> None:
        """
        Convert a Battery entity to a pandapower static generator element.

        The battery is modeled as a static generator with controllable sign
        (positive for discharge, negative for charge).

        Parameters
        ----------
        battery : Battery
            The battery entity to convert
        """
        bus_idx = self.bus_map[battery.bus]
        pp.create_sgen(self.net, bus=bus_idx, p_mw=0.0, q_mvar=0.0, name=battery.id)

    def _convert_load(self, load: Load) -> None:
        """
        Convert a Load entity to a pandapower load element.

        Parameters
        ----------
        load : Load
            The load entity to convert
        """
        pp.create_load(
            self.net, bus=self.bus_map[load.bus], p_mw=0.002, q_mvar=0, name=load.id
        )

    def convert_entity(
        self,
        entity: Entity,
        time_set: Optional[Union[int, range]] = None,
        new_freq: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert an Entity and its sub-entities into numpy arrays for simulation input.

        This method recursively traverses the entity hierarchy, resamples all time series
        data to the specified frequency, and converts each quantity into a numpy array.
        Network elements are created in the pandapower net as a side effect.

        Parameters
        ----------
        entity : Entity
            The root entity to convert (may have sub-entities).
        time_set : Optional[Union[int, range]], optional
            The number of time steps to represent in the resulting arrays.
            If None, uses the default time set size.
        new_freq : Optional[str], optional
            The target frequency to resample time series data to (e.g., '15min', '1H').

        Returns
        -------
        Dict[str, Any]
            A flat dictionary containing numpy arrays from the entity and its descendants.
            Keys are in the format 'entity_id.quantity_name'.
        """
        # Convert entity quantities
        entity_variables = {
            f"{entity.id}.{key}": self.convert_quantity(
                quantity,
                name=f"{entity.id}.{key}",
                time_set=time_set,
                freq=new_freq,
            )
            for key, quantity in entity.quantities.items()
        }

        # Recursively convert sub-entities
        for sub_entity in entity.sub_entities:
            entity_variables.update(self.convert_entity(sub_entity, time_set, new_freq))

        return entity_variables

    def convert_quantity(
        self,
        quantity: Quantity,
        name: str,
        time_set: Optional[Union[int, range]] = None,
        freq: Optional[str] = None,
    ) -> ndarray:
        """
        Convert a quantity to a numpy array for pandapower simulation input.

        If the quantity is empty, create a numpy array filled with NaN values.
        If the quantity is a Parameter, return its value directly.
        Otherwise, return the time series values resampled to the specified time set and frequency.

        Parameters
        ----------
        quantity : Quantity
            The quantity to convert
        name : str
            The name identifier for the quantity
        time_set : Optional[Union[int, range]], optional
            The time set specification. If None, uses default size.
        freq : Optional[str], optional
            The frequency for resampling

        Returns
        -------
        np.ndarray
            Numpy array containing the quantity values over time, or NaN array if empty
        """
        if quantity.empty():
            normalized_time_set = validate_and_normalize_time_set(
                time_set, self.DEFAULT_TIME_SET_SIZE
            )
            return np.ones(len(normalized_time_set)) * np.nan
        if isinstance(quantity, Parameter):
            return quantity.get_values()
        else:
            return quantity.get_values(time_set=time_set, freq=freq)
