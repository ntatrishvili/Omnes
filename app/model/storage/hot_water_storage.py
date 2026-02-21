from typing import Optional

from app.infra.parameter import Parameter
from app.infra.quantity_factory import (
    DefaultQuantityFactory,
    QuantityFactory,
)
from app.model.device import Vector
from app.model.storage.storage import Storage


class HotWaterStorage(Storage):
    """
    A hot water storage tank that stores thermal energy.

    The storage capacity and charge/discharge rates are calculated from the
    physical properties of water and the tank configuration.

    Attributes
    ----------
    default_vector : Vector
        The energy vector (HEAT)
    default_contributes_to : str
        The balance this device contributes to
    default_set_temperature : float
        Target water temperature in °C (default: 55°C)
    default_water_in_temperature : float
        Inlet water temperature in °C (default: 12°C)
    default_volume : float
        Tank volume in liters (default: None)
    default_time_to_charge : float
        Time to fully charge the tank in hours (default: 1.0h)
    default_time_to_discharge : float
        Time to fully discharge the tank in hours (default: 1.0h)

    Notes
    -----
    Capacity calculation:
        capacity (kWh) = (T_set - T_in) * c_water * volume

    Where:
        - c_water = 4.186 kJ/(kg·K) ≈ 0.001163 kWh/(L·K)
        - T_set = set temperature (°C)
        - T_in = water inlet temperature (°C)
        - volume = tank volume (L)

    Charge/discharge rates:
        max_charge_rate (kW) = capacity / time_to_charge
        max_discharge_rate (kW) = capacity / time_to_discharge
    """

    default_vector = Vector.HEAT
    default_contributes_to = "heat_balance"

    default_set_temperature: Optional[float] = 55.0  # °C
    default_water_in_temperature: Optional[float] = 12.0  # °C
    default_volume: Optional[float] = None  # liters
    default_time_to_charge: Optional[float] = 1.0  # hours
    default_time_to_discharge: Optional[float] = 1.0  # hours

    # Specific heat capacity of water: 4.186 kJ/(kg·K) = 4.186 kJ/(L·K)
    # Converted to kWh/(L·K): 4.186 / 3600 ≈ 0.001163
    C_WATER_KWH_PER_LITER_KELVIN = 4.186 / 3600  # kWh/(L·K)

    @staticmethod
    def _get_numeric_value(val):
        """Extract numeric value from Parameter or return value as-is."""
        if val is None:
            return None
        if isinstance(val, Parameter):
            return val.value
        if hasattr(val, "value") and not callable(val.value):
            return val.value
        return val

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        # Extract HotWaterStorage-specific parameters before calling super().__init__
        volume = kwargs.pop("volume", self.default_volume)
        set_temperature = kwargs.pop("set_temperature", self.default_set_temperature)
        water_in_temperature = kwargs.pop(
            "water_in_temperature", self.default_water_in_temperature
        )
        time_to_charge = kwargs.pop("time_to_charge", self.default_time_to_charge)
        time_to_discharge = kwargs.pop(
            "time_to_discharge", self.default_time_to_discharge
        )

        # Get numeric values for calculations
        volume_val = self._get_numeric_value(volume)
        set_temp_val = self._get_numeric_value(set_temperature)
        water_in_temp_val = self._get_numeric_value(water_in_temperature)
        time_to_charge_val = self._get_numeric_value(time_to_charge)
        time_to_discharge_val = self._get_numeric_value(time_to_discharge)

        # Calculate derived quantities if volume is provided
        capacity = kwargs.get("capacity")
        max_charge_rate = kwargs.get("max_charge_rate")
        max_discharge_rate = kwargs.get("max_discharge_rate")

        if volume_val is not None and capacity is None:
            # Calculate capacity from physical properties
            # capacity (kWh) = (T_set - T_in) * c_water * volume
            delta_t = set_temp_val - water_in_temp_val
            capacity = delta_t * self.C_WATER_KWH_PER_LITER_KELVIN * volume_val
            kwargs["capacity"] = capacity

        # Get numeric value for capacity (could have been passed as Parameter)
        capacity_val = (
            self._get_numeric_value(capacity)
            if capacity is not None
            else self._get_numeric_value(kwargs.get("capacity"))
        )

        if capacity_val is not None:
            # Calculate charge/discharge rates if not explicitly provided
            if (
                max_charge_rate is None
                and time_to_charge_val is not None
                and time_to_charge_val > 0
            ):
                kwargs["max_charge_rate"] = capacity_val / time_to_charge_val

            if (
                max_discharge_rate is None
                and time_to_discharge_val is not None
                and time_to_discharge_val > 0
            ):
                kwargs["max_discharge_rate"] = capacity_val / time_to_discharge_val

        # Call parent constructor (Storage) which creates capacity, max_charge_rate, etc.
        super().__init__(id, quantity_factory, **kwargs)

        # Create HotWaterStorage-specific quantities
        self.create_quantity("volume", input=volume)
        self.create_quantity(
            "set_temperature",
            input=set_temperature,
            default_type=Parameter,
        )
        self.create_quantity(
            "water_in_temperature",
            input=water_in_temperature,
            default_type=Parameter,
        )
        self.create_quantity(
            "time_to_charge",
            input=time_to_charge,
            default_type=Parameter,
        )
        self.create_quantity(
            "time_to_discharge",
            input=time_to_discharge,
            default_type=Parameter,
        )

    def __str__(self):
        capacity_str = ""
        if "capacity" in self.quantities and self.quantities["capacity"] is not None:
            cap_val = getattr(
                self.quantities["capacity"], "value", self.quantities["capacity"]
            )
            if cap_val is not None:
                capacity_str = f", capacity: {cap_val:.2f} kWh"
        return (
            f"Hot water storage '{self.id}' with volume: {self.quantities.get('volume')} L, "
            f"set temperature: {self.quantities.get('set_temperature')} °C"
            f"{capacity_str}"
        )
