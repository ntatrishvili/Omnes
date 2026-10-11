"""KFKI-specific devices.

Both devices are electric loads, so they subclass ``Load``: the existing Pulp and
Pandapower converters resolve handlers through the class MRO and therefore treat
them as loads without any converter change.
"""

from typing import Optional

from app.infra.parameter import Parameter
from app.infra.quantity_factory import DefaultQuantityFactory, QuantityFactory
from app.infra.timeseries_object import TimeseriesObject
from app.model.device import Device, Vector
from app.model.load.load import Load


class HeatPump(Load):
    """Heat pump electrical demand (profile taken from the input CSV)."""

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        super().__init__(id=id, quantity_factory=quantity_factory, **kwargs)


class HeatDemand(Device):
    """Thermal demand of a building (kW thermal) on the heat balance.

    A ``Device`` of vector HEAT, deliberately not a ``Load``: the Pandapower converter
    therefore leaves it out of the electrical network, while the dispatch balances it
    against the heat pump and the hot-water storage.
    """

    default_vector = Vector.HEAT
    default_contributes_to = "heat_balance"

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        super().__init__(id=id, quantity_factory=quantity_factory, **kwargs)
        self.create_quantity(
            "p_cons", **kwargs.get("p_cons", {}), default_type=TimeseriesObject
        )


class EVCharger(Load):
    """Group of identical EV chargers at one site.

    ``n_chargers`` scales a per-charger demand profile; ``charger_kw`` is the
    rated AC power of one charger (used for the nominal power).
    """

    def __init__(
        self,
        id: Optional[str] = None,
        quantity_factory: QuantityFactory = DefaultQuantityFactory(),
        **kwargs,
    ):
        n_chargers = kwargs.pop("n_chargers", 0)
        charger_kw = kwargs.pop("charger_kw", 11.0)
        kwargs.setdefault("nominal_power", n_chargers * charger_kw)
        super().__init__(id=id, quantity_factory=quantity_factory, **kwargs)
        self.create_quantity("n_chargers", input=n_chargers, default_type=Parameter)
        self.create_quantity("charger_kw", input=charger_kw, default_type=Parameter)
