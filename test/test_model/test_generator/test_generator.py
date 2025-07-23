import unittest

from app.model.generator.generator import Generator


def test_generator_default_quantities():
    gen = Generator(id="gen1", bus="bus1")

    assert gen.id == "gen1"
    assert gen.peak_power.value == 0
    assert gen.efficiency.value == 0
    assert gen.peak_power.empty() is False
    assert gen.efficiency.empty() is False


if __name__ == "__main__":
    unittest.main()
