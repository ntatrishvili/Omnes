# Ensure project root is on sys.path so imports like `utils.*` work when running tests
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import pandas as pd

from app.infra.singleton import Singleton
from app.infra.configuration import Config


class TestConfiguration(unittest.TestCase):
    def setUp(self):
        # Ensure a fresh singleton for each test
        Singleton._instance = None

        # Create a temporary config file with only the settings used in the tests.
        # This avoids relying on the repository's config/config.ini and makes tests hermetic.
        self.temp_config = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".ini"
        )
        self.config_path = self.temp_config.name

        # Minimal config content matching what's asserted in the tests
        config_content = """
                        [time]
                        # time_set is stored as string in config file
                        time_set = 35037
                        # frequency should be parsed by Config into a pandas Timedelta
                        frequency = 15min
                        
                        [path]
                        root = /some/root
                        """
        self.temp_config.write(config_content)
        self.temp_config.flush()
        self.temp_config.close()

    def tearDown(self):
        # Clean up the temporary file
        try:
            os.remove(self.config_path)
        except OSError:
            pass

    def test_get_and_getint_and_frequency_processing(self):
        cfg = Config(config_filename=self.config_path)

        # From config: time_set = 35037 (string in file)
        self.assertEqual(cfg.get("time", "time_set"), "35037")
        # getint should parse it to an integer
        self.assertEqual(cfg.getint("time", "time_set"), 35037)

        # frequency is registered and should be converted to a pandas Timedelta
        freq = cfg.get("time", "frequency")
        self.assertIsInstance(freq, pd.Timedelta)
        self.assertEqual(freq, pd.Timedelta("15min"))

    def test_set_and_get_methods(self):
        cfg = Config(config_filename=self.config_path)

        # Use an existing section 'path' to avoid NoSection errors
        # setint / set / setarray / setboolean should update the in-memory config
        cfg.setint("path", "root", 123)
        self.assertEqual(cfg.getstr("path", "root"), "123")

        cfg.set("path", "foo", "bar")
        self.assertEqual(cfg.getstr("path", "foo"), "bar")

        cfg.setarray("path", "arr", ["1", "2", "3"])
        arr = cfg.getarray("path", "arr", dtype=int)
        self.assertEqual(arr, [1, 2, 3])

        cfg.setboolean("path", "flag", True)
        self.assertTrue(cfg.getboolean("path", "flag"))

    def test_getarray_single_value_and_has_option(self):
        cfg = Config(config_filename=self.config_path)

        # setarray with a single element should still be retrievable
        cfg.setarray("path", "single", ["only"])
        self.assertEqual(cfg.getarray("path", "single"), ["only"])

        # has_option should reflect options we've set
        self.assertTrue(cfg.has_option("path", "single"))


if __name__ == "__main__":
    unittest.main()
