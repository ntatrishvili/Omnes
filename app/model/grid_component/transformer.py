from typing import Optional

from app.infra.parameter import Parameter
from app.infra.quantity_factory import (
    DefaultQuantityFactory,
    QuantityFactory,
)
from app.model.grid_component.connector import Connector


class Transformer(Connector):
    """Transformer (HV/LV) model read from SimBench-like CSV.

    Parameters
    ----------
    id : str | None
        Optional identifier for the transformer.
    quantity_factory : QuantityFactory, optional
        Factory used to create time-series objects for this component. If not
        provided the DefaultQuantityFactory is used.
    **kwargs :
        Additional keyword arguments accepted (commonly present in SimBench
        CSV parsing)
        - hv_bus: identifier or name of the high-voltage bus (int or str)
        - lv_bus: identifier or name of the low-voltage bus (int or str)
        - nominal_power: rated power in MVA (float)
        - nominal_voltage_hv_side: rated HV-side voltage in kV (float)
        - nominal_voltage_lv_side: rated LV-side voltage in kV (float)
        - type: arbitrary transformer type string

        Other kwargs are forwarded to the base Connector and may be accepted
        there as well.

    Attributes
    ----------
    default_nominal_power : float
        Default nominal power in MVA.
    default_nominal_voltage_hv_side : float
        Default nominal voltage on the high-voltage side in kV.
    default_nominal_voltage_lv_side : float
        Default nominal voltage on the low-voltage side in kV.
    quantities : dict
        Mapping of quantity names to Parameter objects created from kwargs.
    """

    _quantity_excludes = ["default_type"]

    default_nominal_power: float = 0.16  # MVA
    default_nominal_voltage_hv_side: float = 20.0  # kV
    default_nominal_voltage_lv_side: float = 0.4  # kV
    default_type: str = ""

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        """Initialize a Transformer instance.

        Values provided in ``kwargs`` (for example ``nominal_power``) override
        the class defaults. Any remaining kwargs are passed to the Connector
        base class.
        """
        super().__init__(id=id, quantity_factory=quantity_factory, **kwargs)
        for quantity_name in (
            "nominal_power",
            "nominal_voltage_hv_side",
            "nominal_voltage_lv_side",
            "type",
        ):
            self.create_quantity(
                quantity_name,
                input=kwargs.pop(
                    quantity_name, getattr(self, f"default_{quantity_name}")
                ),
                default_type=Parameter,
            )

    def __str__(self):
        """String representation of the Transformer entity."""
        return (
            f"Transformer '{self.id}' {self.from_bus}--{self.to_bus} with "
            f"nominal power {self.nominal_power} MVA, "
            f"HV side nominal voltage {self.nominal_voltage_hv_side} kV, "
            f"LV side nominal voltage {self.nominal_voltage_lv_side} kV, "
            f"type '{self.type}'"
        )
