import unittest

import xarray
from pandas.core.interchange.dataframe_protocol import DataFrame

from app.infra.parameter import Parameter
from app.infra.quantity_factory import DefaultQuantityFactory


class TestDefaultQuantityFactory(unittest.TestCase):
    def test_factory_create_returns_timeseries_object(self):
        factory = DefaultQuantityFactory()
        obj = factory.create("p_test", data=xarray.DataArray([1, 2, 3, 4, 5]))
        # Should return a TimeseriesObject instance

        from app.infra.timeseries_object import TimeseriesObject

        self.assertIsInstance(obj, TimeseriesObject)

    def test_factory_create_returns_parameter(self):
        factory = DefaultQuantityFactory()
        obj = factory.create("p_test", value=42)
        # Should return a Parameter instance

        from app.infra.parameter import Parameter

        self.assertIsInstance(obj, Parameter)

    def test_factory_create_raises(self):
        factory = DefaultQuantityFactory()
        p_test = factory.create("p_test", unknown_arg=123)
        self.assertIsInstance(p_test, Parameter)


if __name__ == "__main__":
    unittest.main()
