import unittest
from unittest.mock import Mock

import numpy as np
import pulp

from app.conversion.pulp_converter import PulpConverter, create_empty_pulp_var
from app.infra.quantity import Quantity
from app.infra.parameter import Parameter
from app.infra.relation import (
    EntityReference,
    Literal,
    Operator,
    Relation,
    AssignmentExpression,
)


class TestCreateEmptyPulpVar(unittest.TestCase):
    def test_create_empty_pulp_var(self):
        """Test creation of empty PuLP variables"""
        time_set = 3
        variables = create_empty_pulp_var("test_var", time_set)

        self.assertEqual(len(variables), 3)
        for i, var in enumerate(variables):
            self.assertIsInstance(var, pulp.LpVariable)
            self.assertEqual(var.name, f"test_var_{i}")
            self.assertEqual(var.lowBound, 0)


class TestPulpConverter(unittest.TestCase):
    def setUp(self):
        self.converter = PulpConverter()

    def test_convert_quantity_empty(self):
        """Test converting empty quantity"""
        mock_quantity = Mock()
        mock_quantity.empty.return_value = True

        result = self.converter.convert_quantity(mock_quantity, "test_name", time_set=3)

        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], pulp.LpVariable)

    def test_convert_quantity_parameter(self):
        """Test converting parameter quantity"""
        mock_parameter = Mock(spec=Parameter)
        mock_parameter.empty.return_value = False
        mock_parameter.value = 42

        result = self.converter.convert_quantity(mock_parameter, "test_name")

        self.assertEqual(result, 42)

    def test_convert_quantity_regular(self):
        """Test converting regular quantity"""
        mock_quantity = Mock(spec=Quantity)
        mock_quantity.empty.return_value = False
        mock_quantity.value.return_value = [1, 2, 3, 4, 5]

        result = self.converter.convert_quantity(
            mock_quantity, "test_name", time_set=5, freq="1H"
        )

        self.assertEqual(result, [1, 2, 3, 4, 5])
        mock_quantity.value.assert_called_once_with(time_set=5, freq="1H")


class TestDynamicConstraintBuilding(unittest.TestCase):
    """Test the dynamic constraint building with dependency inversion"""

    def setUp(self):
        self.converter = PulpConverter()

    def test_simple_temporal_constraint(self):
        """Test basic temporal constraint with time offsets"""
        # battery.discharge_power(t) <= 2 * battery.discharge_power(t-1)
        relation = Relation(
            "battery.discharge_power(t) <= 2 * battery.discharge_power(t-1)",
            "discharge_ramp_constraint",
        )

        time_steps = 5
        battery_discharge = [
            pulp.LpVariable(f"battery_discharge_{t}", lowBound=0)
            for t in range(time_steps)
        ]
        objects = {"battery.discharge_power": battery_discharge}

        result = self.converter.convert_relation(relation, objects, time_steps)

        # Verify structure
        self.assertIn("discharge_ramp_constraint", result)
        constraints = result["discharge_ramp_constraint"]
        self.assertEqual(len(constraints), time_steps)

        # Verify each constraint is a PuLP constraint
        for constraint in constraints:
            self.assertIsInstance(constraint, pulp.LpConstraint)

        # Verify the constraints make sense (t=1 should reference t=0)
        # This tests that the dynamic assembly worked correctly
        constraint_str = str(constraints[1])
        self.assertIn("battery_discharge_1", constraint_str)
        self.assertIn("battery_discharge_0", constraint_str)

    def test_complex_power_balance(self):
        """Test complex multi-entity power balance constraint"""
        # pv.generation + battery.discharge + grid.import >= load.demand + battery.charge + grid.export
        relation = Relation(
            "pv.generation(t) + battery.discharge_power(t) + grid.import_power(t) >= load.demand(t) + battery.charge_power(t) + grid.export_power(t)",
            "power_balance",
        )

        time_steps = 24
        objects = {
            "pv.generation": [
                pulp.LpVariable(f"pv_gen_{t}", lowBound=0) for t in range(time_steps)
            ],
            "battery.discharge_power": [
                pulp.LpVariable(f"bat_dis_{t}", lowBound=0) for t in range(time_steps)
            ],
            "grid.import_power": [
                pulp.LpVariable(f"grid_imp_{t}", lowBound=0) for t in range(time_steps)
            ],
            "load.demand": [
                30 + 20 * np.sin(t * np.pi / 12) for t in range(time_steps)
            ],  # Load pattern
            "battery.charge_power": [
                pulp.LpVariable(f"bat_chg_{t}", lowBound=0) for t in range(time_steps)
            ],
            "grid.export_power": [
                pulp.LpVariable(f"grid_exp_{t}", lowBound=0) for t in range(time_steps)
            ],
        }

        result = self.converter.convert_relation(relation, objects, time_steps)

        # Verify all constraints generated
        constraints = result["power_balance"]
        self.assertEqual(len(constraints), time_steps)

        # Verify it correctly identifies all entities
        expected_entities = {
            "pv.generation",
            "battery.discharge_power",
            "grid.import_power",
            "load.demand",
            "battery.charge_power",
            "grid.export_power",
        }
        actual_entities = set(relation.get_ids())
        self.assertEqual(expected_entities, actual_entities)

        # Verify constraint structure by checking string representation
        constraint_str = str(constraints[12])  # Check noon constraint
        self.assertIn("pv_gen_12", constraint_str)
        self.assertIn("bat_dis_12", constraint_str)
        self.assertIn("grid_imp_12", constraint_str)

    def test_arithmetic_operations_and_precedence(self):
        """Test complex arithmetic with proper operator precedence"""
        # battery.power <= 0.8 * pv.power + 0.2 * pv.power(t-1)
        relation = Relation(
            "battery.power(t) <= 0.8 * pv.power(t) + 0.2 * pv.power(t-1)",
            "weighted_pv_constraint",
        )

        time_steps = 5
        objects = {
            "battery.power": [
                pulp.LpVariable(f"battery_{t}") for t in range(time_steps)
            ],
            "pv.power": [pulp.LpVariable(f"pv_{t}") for t in range(time_steps)],
        }

        result = self.converter.convert_relation(relation, objects, time_steps)
        constraints = result["weighted_pv_constraint"]

        # Verify correct number of constraints
        self.assertEqual(len(constraints), time_steps)

        # Check that the arithmetic is correctly structured
        # For t=2, should have: battery_2 <= 0.8*pv_2 + 0.2*pv_1
        constraint_str = str(constraints[2])
        self.assertIn("battery_2", constraint_str)
        self.assertIn("pv_2", constraint_str)
        self.assertIn("pv_1", constraint_str)  # t-1 reference

        # Test precedence: 2 + 3 * x should be 2 + (3*x), not (2+3)*x
        precedence_relation = Relation(
            "battery.power(t) >= 2 + 3 * pv.power(t)", "precedence_test"
        )
        precedence_result = self.converter.convert_relation(
            precedence_relation, objects, 3
        )

        # Should have 3 constraints
        self.assertEqual(len(precedence_result["precedence_test"]), 3)

    def test_battery_state_of_charge_dynamics(self):
        """Test complex state_of_charge dynamics with energy balance"""
        # state_of_charge(t) = state_of_charge(t-1) + efficiency * charge(t) - discharge(t) * loss_factor
        # Using multiplication instead of division for linear programming compatibility
        relation = Relation(
            "battery.state_of_charge(t) >= battery.state_of_charge(t-1) + 0.9 * battery.charge_power(t) - 1.1 * battery.discharge_power(t)",
            "soc_dynamics",
        )

        time_steps = 10
        objects = {
            "battery.state_of_charge": [
                pulp.LpVariable(f"soc_{t}", lowBound=0, upBound=100)
                for t in range(time_steps)
            ],
            "battery.charge_power": [
                pulp.LpVariable(f"charge_{t}", lowBound=0) for t in range(time_steps)
            ],
            "battery.discharge_power": [
                pulp.LpVariable(f"discharge_{t}", lowBound=0) for t in range(time_steps)
            ],
        }

        result = self.converter.convert_relation(relation, objects, time_steps)
        constraints = result["soc_dynamics"]

        self.assertEqual(len(constraints), time_steps)

        # Verify the constraint includes all terms for t=5
        constraint_str = str(constraints[5])
        self.assertIn("soc_5", constraint_str)  # Current state_of_charge
        self.assertIn("soc_4", constraint_str)  # Previous state_of_charge (t-1)
        self.assertIn("charge_5", constraint_str)  # Charge power
        self.assertIn("discharge_5", constraint_str)  # Discharge power

    def test_division_by_constant(self):
        """Test division by constants is properly handled"""
        # Test division by constant: power / 2.0
        relation = Relation("battery.power(t) / 2.0 <= 50", "half_power_limit")

        time_steps = 3
        objects = {
            "battery.power": [
                pulp.LpVariable(f"battery_{t}") for t in range(time_steps)
            ]
        }

        result = self.converter.convert_relation(relation, objects, time_steps)
        constraints = result["half_power_limit"]

        self.assertEqual(len(constraints), time_steps)
        # The division by 2.0 should be converted to multiplication by 0.5
        constraint_str = str(constraints[0])
        self.assertIn("battery_0", constraint_str)

    def test_time_bounds_handling(self):
        """Test proper handling of time bounds (t-1 at t=0, etc.)"""
        relation = Relation(
            "battery.power(t) <= battery.power(t-1) + 10", "ramp_constraint"
        )

        time_steps = 3
        battery_power = [pulp.LpVariable(f"battery_{t}") for t in range(time_steps)]
        objects = {"battery.power": battery_power}

        result = self.converter.convert_relation(relation, objects, time_steps)
        constraints = result["ramp_constraint"]

        # Should handle t=0 case gracefully (t-1 becomes t=0 due to bounds checking)
        self.assertEqual(len(constraints), time_steps)

        # At t=0, should have: battery_0 <= battery_0 + 10 (simplified to valid constraint)
        constraint_t0_str = str(constraints[0])
        self.assertIn("battery_0", constraint_t0_str)

        # At t=1, should have: battery_1 <= battery_0 + 10
        constraint_t1_str = str(constraints[1])
        self.assertIn("battery_1", constraint_t1_str)
        self.assertIn("battery_0", constraint_t1_str)

    def test_error_handling(self):
        """Test error handling for various edge cases"""
        # Test missing entity
        relation = Relation("nonexistent.power(t) <= 100", "missing_entity")
        objects = {"battery.power": [pulp.LpVariable("battery_0")]}

        with self.assertRaises(ValueError) as context:
            self.converter.convert_relation(relation, objects, 1)
        self.assertIn("nonexistent.power", str(context.exception))
        self.assertIn("not found", str(context.exception))

        # Test != operator (now supported with warning)
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            neq_relation = Relation("battery.power(t) != 50", "neq_test")
            battery_vars = [pulp.LpVariable(f"battery_{t}") for t in range(3)]
            objects = {"battery.power": battery_vars}

            # This should work now (with warning)
            result = self.converter.convert_relation(neq_relation, objects, 3)

            # Check that warning was issued
            self.assertEqual(len(w), 3)  # One warning per time step
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertIn("Not equal constraints", str(w[0].message))
            self.assertIn("Mixed Integer Programming", str(w[0].message))

            # Check that constraints were created
            constraints = result["neq_test"]
            self.assertEqual(len(constraints), 3)


class TestRealisticEnergyScenarios(unittest.TestCase):
    """Test realistic energy system scenarios with time series data"""

    def setUp(self):
        self.converter = PulpConverter()

    def create_solar_pv_pattern(self, time_steps: int):
        """Create realistic solar PV generation pattern"""
        hours = np.linspace(0, 24, time_steps)
        solar_pattern = np.maximum(0, np.sin(np.pi * (hours - 6) / 12)) ** 2
        return [
            pulp.LpVariable(f"pv_gen_{t}", lowBound=0, upBound=solar_pattern[t] * 100)
            for t in range(time_steps)
        ]

    def create_load_pattern(self, time_steps: int):
        """Create realistic load demand pattern"""
        hours = np.linspace(0, 24, time_steps)
        base_load = 30
        evening_peak = 20 * np.exp(-((hours - 19) ** 2) / 8)  # Peak at 7 PM
        return (base_load + evening_peak).tolist()

    def test_comprehensive_energy_system(self):
        """Test a comprehensive energy system with multiple constraints"""
        time_steps = 24

        # Create realistic variables
        pv_generation = self.create_solar_pv_pattern(time_steps)
        load_demand = self.create_load_pattern(time_steps)
        battery_charge = [
            pulp.LpVariable(f"bat_charge_{t}", lowBound=0, upBound=50)
            for t in range(time_steps)
        ]
        battery_discharge = [
            pulp.LpVariable(f"bat_discharge_{t}", lowBound=0, upBound=50)
            for t in range(time_steps)
        ]
        battery_soc = [
            pulp.LpVariable(f"bat_soc_{t}", lowBound=0, upBound=100)
            for t in range(time_steps)
        ]
        grid_import = [
            pulp.LpVariable(f"grid_import_{t}", lowBound=0) for t in range(time_steps)
        ]

        objects = {
            "pv.generation": pv_generation,
            "load.demand": load_demand,
            "battery.charge_power": battery_charge,
            "battery.discharge_power": battery_discharge,
            "battery.state_of_charge": battery_soc,
            "grid.import_power": grid_import,
        }

        # Test multiple realistic constraints
        constraints_to_test = [
            ("battery.charge_power(t) <= 50", "battery_charge_limit"),
            ("battery.discharge_power(t) <= 50", "battery_discharge_limit"),
            ("battery.state_of_charge(t) <= 100", "battery_soc_limit"),
            (
                "pv.generation(t) + battery.discharge_power(t) + grid.import_power(t) >= load.demand(t) + battery.charge_power(t)",
                "power_balance",
            ),
            (
                "battery.discharge_power(t) <= battery.discharge_power(t-1) + 10",
                "discharge_ramp_limit",
            ),
        ]

        for constraint_expr, constraint_name in constraints_to_test:
            with self.subTest(constraint=constraint_name):
                relation = Relation(constraint_expr, constraint_name)
                result = self.converter.convert_relation(relation, objects, time_steps)

                # Verify constraint was created
                self.assertIn(constraint_name, result)
                constraints = result[constraint_name]
                self.assertEqual(len(constraints), time_steps)

                # Verify all constraints are valid PuLP constraints
                for i, constraint in enumerate(constraints):
                    self.assertIsInstance(
                        constraint,
                        pulp.LpConstraint,
                        f"Constraint {i} in {constraint_name} is not a LpConstraint",
                    )

    def test_realistic_constraint_values(self):
        """Test that constraints produce realistic and expected values"""
        time_steps = 5

        # Create simple test case with known values
        battery_power = [
            pulp.LpVariable(f"battery_{t}", lowBound=0, upBound=100)
            for t in range(time_steps)
        ]
        objects = {"battery.power": battery_power}

        # Test simple bound constraint
        relation = Relation("battery.power(t) <= 80", "power_limit")
        result = self.converter.convert_relation(relation, objects, time_steps)

        constraints = result["power_limit"]
        self.assertEqual(len(constraints), time_steps)

        # Each constraint should be: battery_t <= 80
        for i, constraint in enumerate(constraints):
            constraint_str = str(constraint)
            self.assertIn(f"battery_{i}", constraint_str)
            self.assertIn("80", constraint_str)


class TestLegacyIntegration(unittest.TestCase):
    """Test integration with legacy code and backwards compatibility"""

    def setUp(self):
        self.converter = PulpConverter()

    def test_convert_relation_default_time_set(self):
        """Test that converter uses default time_set when none provided"""
        relation = Relation("x <= y", "simple_constraint")

        x_vars = [pulp.LpVariable(f"x_{t}") for t in range(10)]  # Default is 10
        y_vars = [pulp.LpVariable(f"y_{t}") for t in range(10)]
        objects = {"x": x_vars, "y": y_vars}

        result = self.converter.convert_relation(
            relation, objects
        )  # No time_set provided

        constraints = result["simple_constraint"]
        self.assertEqual(len(constraints), 10)  # Should use default

    def test_entity_reference_conversion_methods(self):
        """Test the converter's entity reference conversion methods directly"""
        time_steps = 5
        battery_power = [pulp.LpVariable(f"battery_{t}") for t in range(time_steps)]

        # Store objects in converter for testing
        self.converter._PulpConverter__objects = {"battery.power": battery_power}

        # Test convert_entity_reference
        entity_ref = EntityReference("battery.power", 0)  # t+0
        result = self.converter.convert_entity_reference(entity_ref, "battery.power", 2)
        self.assertEqual(result, battery_power[2])

        # Test with time offset
        entity_ref_offset = EntityReference("battery.power", -1)  # t-1
        result_offset = self.converter.convert_entity_reference(
            entity_ref_offset, "battery.power", 3
        )
        self.assertEqual(result_offset, battery_power[2])  # t=3, t-1=2

    def test_convert_literal_method(self):
        """Test the converter's literal conversion method"""
        literal = Literal(42)
        result = self.converter.convert_literal(literal, 42, 0)
        self.assertEqual(result, 42)

        # Test with float
        literal_float = Literal(3.14)
        result_float = self.converter.convert_literal(literal_float, 3.14, 0)
        self.assertEqual(result_float, 3.14)

    def test_convert_binary_expression_method(self):
        """Test the converter's binary expression conversion method"""
        # Test arithmetic operations
        result_add = self.converter.convert_binary_expression(
            None, 5, 3, Operator.ADD, 0
        )
        self.assertEqual(result_add, 8)

        result_mult = self.converter.convert_binary_expression(
            None, 4, 6, Operator.MULTIPLY, 0
        )
        self.assertEqual(result_mult, 24)

        # Test with PuLP variables
        var1 = pulp.LpVariable("var1")
        var2 = pulp.LpVariable("var2")

        result_constraint = self.converter.convert_binary_expression(
            None, var1, var2, Operator.LESS_THAN_OR_EQUAL, 0
        )
        self.assertIsInstance(result_constraint, pulp.LpConstraint)


class TestComprehensiveCoverage(unittest.TestCase):
    """Test additional functionality to improve code coverage"""

    def setUp(self):
        self.converter = PulpConverter()

    def test_convert_model_comprehensive(self):
        """Test convert_model with various scenarios to improve coverage"""
        from app.infra.quantity import Quantity
        from app.infra.util import TimesetBuilder
        from app.model.entity import Entity
        from app.model.model import Model

        # Create a dummy timeset builder for testing
        class DummyTimesetBuilder(TimesetBuilder):
            def create(self, time_kwargs=None, **kwargs):
                if time_kwargs is None:
                    time_kwargs = {}
                import pandas as pd

                from app.infra.util import TimeSet

                dates = pd.date_range(start="2023-01-01", periods=5, freq="1h")
                return TimeSet(
                    start="2023-01-01",
                    end="2023-01-01 04:00:00",
                    resolution="1H",
                    number_of_time_steps=5,
                    time_points=dates,
                    **time_kwargs,
                )

        dummy_builder = DummyTimesetBuilder()
        model = Model(id="test_model", timeset_builder=dummy_builder)

        entity = Entity(id="test_entity")
        from app.infra.timeseries_object import TimeseriesObject

        entity.quantities["power"] = TimeseriesObject(name="power", data=[])
        model.entities = {entity.id: entity}

        # Test convert_model with default parameters
        result = self.converter.convert_model(model)
        self.assertIsInstance(result, dict)
        self.assertIn("time_set", result)
        self.assertIn("test_entity.power", result)

        result_custom = self.converter.convert_model(
            model, time_set=3, new_freq="30min"
        )
        self.assertIsInstance(result_custom, dict)
        self.assertIn("time_set", result_custom)

    def test_convert_entity_reference_edge_cases(self):
        """Test edge cases in convert_entity_reference"""
        from app.infra.relation import EntityReference

        # Test with negative time offset resulting in negative actual_time
        entity_ref = EntityReference(entity_id="test", time_offset=-2)
        pulp_vars = [pulp.LpVariable(f"test_var_{i}") for i in range(3)]

        # Set up objects dictionary
        self.converter._PulpConverter__objects = {"test": pulp_vars}

        # Test negative time - should use index 0
        result = self.converter.convert_entity_reference(entity_ref, "test", 1)
        self.assertEqual(result, pulp_vars[0])

        # Test time beyond bounds - should use last index
        result_beyond = self.converter.convert_entity_reference(entity_ref, "test", 10)
        self.assertEqual(result_beyond, pulp_vars[2])  # Last element

        # Test with single variable (not list)
        single_var = pulp.LpVariable("single")
        self.converter._PulpConverter__objects = {"single_entity": single_var}
        entity_ref_single = EntityReference(entity_id="single_entity", time_offset=0)
        result_single = self.converter.convert_entity_reference(
            entity_ref_single, "single_entity", 0
        )
        self.assertEqual(result_single, single_var)

        # Test entity not found
        with self.assertRaises(ValueError) as context:
            self.converter.convert_entity_reference(entity_ref, "missing_entity", 0)
        self.assertIn(
            "Entity ID 'missing_entity' not found in provided objects",
            str(context.exception),
        )

    def test_convert_binary_expression_unsupported_operator(self):
        """Test unsupported operator in convert_binary_expression"""
        from app.infra.relation import Operator

        # Mock an unsupported operator (this would need to be a new enum value)
        # For now, we'll test the error case by patching
        with self.assertRaises(ValueError) as context:
            # Use None as an invalid operator to trigger the else clause
            self.converter.convert_binary_expression(None, 5, 3, None, 0)
        self.assertIn("Unsupported operator", str(context.exception))

    def test_multiplication_and_division_constraints(self):
        """Test multiplication and division constraints and their restrictions"""
        var1 = pulp.LpVariable("var1")
        var2 = pulp.LpVariable("var2")

        # Test valid multiplication (variable * constant)
        result_mult = self.converter.convert_binary_expression(
            None, var1, 5, Operator.MULTIPLY, 0
        )
        self.assertIsInstance(result_mult, pulp.LpAffineExpression)

        # Test valid multiplication (constant * variable)
        result_mult2 = self.converter.convert_binary_expression(
            None, 3, var2, Operator.MULTIPLY, 0
        )
        self.assertIsInstance(result_mult2, pulp.LpAffineExpression)

        # Test invalid multiplication (variable * variable)
        with self.assertRaises(ValueError) as context:
            self.converter.convert_binary_expression(
                None, var1, var2, Operator.MULTIPLY, 0
            )
        self.assertIn(
            "Multiplication of two variables is not supported", str(context.exception)
        )

        # Test valid division (variable / constant)
        result_div = self.converter.convert_binary_expression(
            None, var1, 2, Operator.DIVIDE, 0
        )
        self.assertIsInstance(result_div, pulp.LpAffineExpression)

        # Test invalid division (variable / variable)
        with self.assertRaises(ValueError) as context:
            self.converter.convert_binary_expression(
                None, var1, var2, Operator.DIVIDE, 0
            )
        self.assertIn("Division by variables is not supported", str(context.exception))

    def test_all_comparison_operators(self):
        """Test all comparison operators for complete coverage"""
        var1 = pulp.LpVariable("var1")
        var2 = pulp.LpVariable("var2")

        # Test LESS_THAN (should use <=)
        result_lt = self.converter.convert_binary_expression(
            None, var1, var2, Operator.LESS_THAN, 0
        )
        self.assertIsInstance(result_lt, pulp.LpConstraint)

        # Test GREATER_THAN (should use >=)
        result_gt = self.converter.convert_binary_expression(
            None, var1, var2, Operator.GREATER_THAN, 0
        )
        self.assertIsInstance(result_gt, pulp.LpConstraint)

        # Test EQUAL
        result_eq = self.converter.convert_binary_expression(
            None, var1, var2, Operator.EQUAL, 0
        )
        self.assertIsInstance(result_eq, pulp.LpConstraint)

    def test_create_empty_pulp_var_function(self):
        """Test the standalone create_empty_pulp_var function"""
        from app.conversion.pulp_converter import create_empty_pulp_var

        vars_list = create_empty_pulp_var("test_var", 4)
        self.assertEqual(len(vars_list), 4)
        self.assertIsInstance(vars_list[0], pulp.LpVariable)
        self.assertEqual(vars_list[0].name, "test_var_0")
        self.assertEqual(vars_list[3].name, "test_var_3")
        self.assertEqual(vars_list[0].lowBound, 0)

    def test_assignment_expression(self):
        """Test AssignmentExpression conversion to equality constraint"""
        from app.infra.relation import AssignmentExpression, EntityReference, Literal

        time_steps = 3
        battery_power = [pulp.LpVariable(f"battery_{t}") for t in range(time_steps)]
        self.converter._PulpConverter__objects = {"battery.power": battery_power}

        # Assignment: battery.power(t) = 42
        expr = AssignmentExpression(EntityReference("battery.power"), Literal(42))
        for t in range(time_steps):
            result = expr.convert(self.converter, t, time_steps)
            self.assertIsInstance(result, pulp.LpConstraint)
            constraint_str = str(result)
            self.assertIn(f"battery_{t}", constraint_str)
            self.assertIn("42", constraint_str)


if __name__ == "__main__":
    unittest.main()
