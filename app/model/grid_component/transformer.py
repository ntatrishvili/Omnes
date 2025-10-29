from typing import Optional

from app.infra.quantity import Parameter
from app.infra.timeseries_object_factory import (
    DefaultTimeseriesFactory,
    TimeseriesFactory,
)
from app.model.grid_component.connector import Connector


class Transformer(Connector):
    """Transformer (HV/LV) model read from SimBench-like CSV.

    Parameters
    ----------
    id : str | None
        Optional identifier for the transformer.
    ts_factory : TimeseriesFactory, optional
        Factory used to create time-series objects for this component. If not
        provided the DefaultTimeseriesFactory is used.
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

    default_nominal_power: float = 0.16  # MVA
    default_nominal_voltage_hv_side: float = 20.0  # kV
    default_nominal_voltage_lv_side: float = 0.4  # kV

    def __init__(
        self,
        id: Optional[str] = None,
        ts_factory: TimeseriesFactory = DefaultTimeseriesFactory(),
        **kwargs,
    ):
        """Initialize a Transformer instance.

        Values provided in ``kwargs`` (for example ``nominal_power``) override
        the class defaults. Any remaining kwargs are passed to the Connector
        base class.
        """
        super().__init__(id=id, ts_factory=ts_factory, **kwargs)
        self.quantities.update(
            {
                "nominal_power": Parameter(
                    value=kwargs.pop("nominal_power", self.default_nominal_power)
                ),
                "nominal_voltage_hv_side": Parameter(
                    value=kwargs.pop(
                        "nominal_voltage_hv_side", self.default_nominal_voltage_hv_side
                    )
                ),
                "nominal_voltage_lv_side": Parameter(
                    value=kwargs.pop(
                        "nominal_voltage_lv_side", self.default_nominal_voltage_lv_side
                    )
                ),
                "type": Parameter(value=kwargs.pop("type", "")),
            }
        )
