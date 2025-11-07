# Ensure project root is on sys.path so imports like `utils.*` work when running tests
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import pandas as pd

from app.infra.singleton import Singleton
from utils.configuration import Config


class TestConfiguration(unittest.TestCase):
    def setUp(self):
        # Ensure a fresh singleton for each test
        Singleton._instance = None
        # Resolve the repository config file path relative to this test file
        repo_root = os.path.dirname(os.path.dirname(__file__))
        self.config_path = os.path.join(repo_root, "config", "config.ini")

    def test_get_and_getint_and_frequency_processing(self):
        cfg = Config(config_filename=self.config_path)

        # From config.ini: time_set = 35037 (string in file)
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
