import numpy as np
from pandas import DataFrame, date_range

from app.model.battery import Battery
from app.model.consumer import Consumer
from app.model.pv import PV
from app.model.timeseries_object import TimeseriesObject


def test_entity_with_quantity():
    p_cons = TimeseriesObject(data=DataFrame(
        index=date_range(start="2022-07-19 00:00", end="2022-07-20 23:45", freq="15min", inclusive="both"),
        data=np.ones(192), ))
    c = Consumer(id="consumer", p_cons=p_cons)
    assert c.get_consumption().sum().sum() == 192
    print(c)


def test_two_entities():
    battery = Battery("battery", capacity=1, max_power=1.2)
    print(battery)
    assert battery.capacity == 1
    assert battery.max_power == 1.2
    p_pv = TimeseriesObject(data=DataFrame(
        index=date_range(start="2022-07-19 00:00", end="2022-07-20 23:45", freq="15min", inclusive="both"),
        data=np.ones(192), ))
    pv = PV("pv", p_pv=p_pv)
    assert pv.get_production().sum().sum() == 192
    print(pv)
