import unittest

from app.infra.timeseries_object_factory import DefaultTimeseriesFactory


class TestDefaultTimeseriesFactory(unittest.TestCase):
    def test_factory_create_returns_timeseriesobject(self):
        factory = DefaultTimeseriesFactory()
        obj = factory.create("p_test")
        # Should return a TimeseriesObject instance

        from app.infra.timeseries_object import TimeseriesObject

        self.assertIsInstance(obj, TimeseriesObject)


if __name__ == "__main__":
    unittest.main()
