from typing import Optional

from app.infra.quantity import Parameter
from app.infra.timeseries_object_factory import TimeseriesFactory
from app.model.entity import Entity


class Storage(Entity):
    default_capacity: Optional[float] = None
    default_max_charge_rate: Optional[float] = None
    default_max_discharge_rate: Optional[float] = None
    default_charge_efficiency: Optional[float] = None
    default_discharge_efficiency: Optional[float] = None
    default_storage_efficiency: Optional[float] = None

    def __init__(self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs):
        super().__init__(id, ts_factory, **kwargs)
        self.quantities.update({
            "capacity": Parameter(value=kwargs.get("capacity", self.default_capacity)),
            "max_charge_rate": Parameter(value=kwargs.get("max_charge_rate", self.default_max_charge_rate)),
            "max_discharge_rate": Parameter(value=kwargs.get("max_discharge_rate", self.default_max_discharge_rate)),
            "charge_efficiency": Parameter(value=kwargs.get("charge_efficiency", self.default_charge_efficiency)),
            "discharge_efficiency": Parameter(value=kwargs.get("discharge_efficiency", self.default_discharge_efficiency)),
            "storage_efficiency": Parameter(value=kwargs.get("storage_efficiency", self.default_storage_efficiency)),
        })
