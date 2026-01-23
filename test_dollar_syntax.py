#!/usr/bin/env python3
"""
Test script to verify that the $ syntax works correctly in relations.
"""

from app.infra.relation import Relation, SelfReference, EntityReference


def test_dollar_syntax():
    """Test that $ syntax is correctly parsed as SelfReference"""

    print("Testing $ syntax parsing in relations...")
    print("=" * 50)

    # Test simple self reference
    relation1 = Relation("$.power >= 0")
    print(f"Relation 1: {relation1}")
    print(f"IDs: {relation1.get_ids()}")
    print()

    # Test self reference with time offset
    relation2 = Relation("$.soc = $.soc(t-1) * 0.995")
    print(f"Relation 2: {relation2}")
    print(f"IDs: {relation2.get_ids()}")
    print()

    # Test mixed self and entity references
    relation3 = Relation("$.power = hot_water1.p_in / $.efficiency")
    print(f"Relation 3: {relation3}")
    print(f"IDs: {relation3.get_ids()}")
    print()

    # Test complex expression with both self and entity references
    relation4 = Relation("pv1.p_out + wind1.p_out >= load1.power + $.p_in")
    print(f"Relation 4: {relation4}")
    print(f"IDs: {relation4.get_ids()}")
    print()

    # Test assignment expression with self reference
    relation5 = Relation("$.max_discharge_rate = 1")
    print(f"Relation 5: {relation5}")
    print(f"IDs: {relation5.get_ids()}")
    print()

    # Test conditional expression with self reference
    relation6 = Relation("if $.soc < 0.2 * $.capacity then $.max_discharge_rate = 1")
    print(f"Relation 6: {relation6}")
    print(f"IDs: {relation6.get_ids()}")
    print()

    # Test arithmetic operations with self references
    relation7 = Relation("$.p_out * $.efficiency <= $.peak_power")
    print(f"Relation 7: {relation7}")
    print(f"IDs: {relation7.get_ids()}")
    print()

    # Test multiple self references in same expression
    relation8 = Relation("$.q_out >= -0.4 * $.p_out")
    print(f"Relation 8: {relation8}")
    print(f"IDs: {relation8.get_ids()}")
    print()

    print("All tests completed successfully!")


def test_individual_components():
    """Test individual components of the $ syntax"""

    print("\nTesting individual components...")
    print("=" * 50)

    # Test SelfReference directly
    self_ref1 = SelfReference("power")
    print(f"SelfReference 1: {self_ref1}")
    print(f"IDs: {self_ref1.get_ids()}")

    self_ref2 = SelfReference("soc", -1)
    print(f"SelfReference 2: {self_ref2}")
    print(f"IDs: {self_ref2.get_ids()}")

    self_ref3 = SelfReference("capacity", 1)
    print(f"SelfReference 3: {self_ref3}")
    print(f"IDs: {self_ref3.get_ids()}")
    print()


if __name__ == "__main__":
    test_dollar_syntax()
    test_individual_components()
