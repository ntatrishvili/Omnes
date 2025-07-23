import unittest

from app.model.generator.wind_turbine import Wind


def test_wind_initialization():
    wind = Wind(id="wind1", bus="bus1")

    assert wind.id == "wind1"
    assert wind.p_wind.empty() is True
    assert "p_wind" in wind.quantities

    # Test string output with empty p_wind
    assert "with production sum = 0" in str(wind)


if __name__ == "__main__":
    unittest.main()
