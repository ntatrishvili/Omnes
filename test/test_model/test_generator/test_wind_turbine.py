import unittest

from app.model.generator.wind_turbine import Wind


def test_wind_initialization():
    wind = Wind(id="wind1", bus="bus1")

    assert wind.id == "wind1"
    assert wind.p_out.empty() is True
    assert "p_out" in wind.quantities

    # Test string output with empty p_out
    assert "with production sum = 0" in str(wind)


if __name__ == "__main__":
    unittest.main()
