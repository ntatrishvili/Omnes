import unittest

import pulp

import app.conversion.pulp_converter
from app.infra import util


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

    def test_cast_like_none_sample_returns_value(self):
        self.assertEqual(util.cast_like('abc', None), 'abc')

    def test_cast_like_numeric_conversions(self):
        # integer from string
        self.assertEqual(util.cast_like('42', 0), 42)
        # float from string
        self.assertAlmostEqual(util.cast_like('3.14', 0.0), 3.14)
        # float from string with comma
        self.assertAlmostEqual(util.cast_like('1,234.56', 0.0), 1234.56)

    def test_cast_like_bool_and_strings(self):
        # cast_like now only converts numeric types; for non-numeric samples it returns the original value
        self.assertEqual(util.cast_like('yes', True), 'yes')
        self.assertEqual(util.cast_like('No', True), 'No')
        # non-string to bool returns the original value as well
        self.assertEqual(util.cast_like(0, True), 0)

    def test_cast_like_sequence_and_mapping(self):
        # cast_like only converts numeric target types; non-numeric samples are returned unchanged
        self.assertEqual(util.cast_like('[1,2,3]', []), '[1,2,3]')
        self.assertEqual(util.cast_like('{"a":1}', {}), '{"a":1}')
        self.assertEqual(util.cast_like([1, 2], tuple()), [1, 2])
        self.assertEqual(util.cast_like((3, 4), list()), (3, 4))
        # string sample: non-numeric sample -> return original value
        self.assertEqual(util.cast_like(123, ''), 123)

    def test_cast_like_invalid_numeric_raises(self):
        with self.assertRaises(Exception):
            util.cast_like('not a number', 0)

    def test_timesetbuilder_create_periods_and_tz(self):
        ts = util.TimesetBuilder.create(number_of_time_steps=5, time_start='2020-01-01', resolution='1D', tz='UTC')
        self.assertEqual(ts.number_of_time_steps, 5)
        self.assertEqual(ts.tz, 'UTC')
        self.assertEqual(len(ts.time_points), 5)
        # resolution should be a pandas frequency-like object
        self.assertIsNotNone(ts.resolution)


if __name__ == "__main__":
    unittest.main()
