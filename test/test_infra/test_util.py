import unittest

import app.conversion.pulp_converter
from app.infra import util
import pulp


class TestUtil(unittest.TestCase):
    def test_get_input_path_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            util.get_input_path("not_a_file.csv")

    def test_create_empty_pulp_var(self):
        pulp_vars = app.conversion.pulp_converter.create_empty_pulp_var("foo", 3)
        self.assertEqual(len(pulp_vars), 3)
        for i, v in enumerate(pulp_vars):
            self.assertIsInstance(v, pulp.LpVariable)
            self.assertIn(f"foo_{i}", v.name)

    def test_flatten(self):
        nested = [[1, 2], [3, [4, 5]], 6]
        flat = util.flatten(nested)
        self.assertEqual(flat, [1, 2, 3, 4, 5, 6])


if __name__ == "__main__":
    unittest.main()
