from typing import Optional

from app.infra.quantity import Parameter
from app.infra.timeseries_object_factory import TimeseriesFactory
from app.model.storage.storage import Storage


class HotWaterStorage(Storage):
    default_set_temperature: Optional[float] = None
    default_volume: Optional[float] = None

    def __init__(self, id: Optional[str] = None, ts_factory: TimeseriesFactory = None, **kwargs):
        super().__init__(id, ts_factory, **kwargs)
        self.quantities.update({"volume": Parameter(value=kwargs.get("volume", self.default_volume)),
            "set_temperature": Parameter(value=kwargs.get("volume", self.default_set_temperature)), })
