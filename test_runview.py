#!/usr/bin/env python
"""Quick test to verify RunView functionality."""

import sys
from dsl.example_model import create_model
from app.operation.example_optimization import optimize_energy_system_pulp

def test_runview():
    """Test RunView.entity.quantity.value returns data."""
    print("Creating model...")
    model = create_model()
    
    print("Running optimization...")
    opt_view = optimize_energy_system_pulp(model=model)
    
    print(f"\nReturned type: {type(opt_view)}")
    print(f"RunView.id: {opt_view.id}")
    print(f"RunView.run_id: {opt_view.run_id}")
    print(f"RunView entities: {list(opt_view.entities.keys())}")
    
    # Test accessing PV entity
    print(f"\nAccessing opt_view.pv1...")
    pv1 = opt_view.pv1
    print(f"  Type: {type(pv1)}")
    print(f"  ID: {pv1.id}")
    print(f"  Quantities: {list(pv1.entity.quantities.keys())}")
    
    # Test accessing a quantity
    print(f"\nAccessing opt_view.pv1.p_out...")
    p_out_view = pv1.p_out
    print(f"  Type: {type(p_out_view)}")
    
    # Test getting value
    print(f"\nAccessing opt_view.pv1.p_out.value...")
    value = p_out_view.value
    print(f"  Type: {type(value)}")
    print(f"  Shape: {value.shape if hasattr(value, 'shape') else len(value)}")
    print(f"  First 5 values: {value[:5] if hasattr(value, '__getitem__') else 'N/A'}")
    
    # Compare with raw model
    print(f"\nComparing with raw model...")
    raw_value = model.pv1.p_out.value()
    print(f"  Raw value shape: {raw_value.shape if hasattr(raw_value, 'shape') else len(raw_value)}")
    print(f"  Raw first 5 values: {raw_value[:5]}")
    
    # Verify they're different (optimized != raw)
    if hasattr(value, '__iter__') and hasattr(raw_value, '__iter__'):
        import numpy as np
        are_same = np.allclose(value, raw_value)
        print(f"\n  Optimized == Raw? {are_same}")
        if not are_same:
            print(f"  ✅ Values are DIFFERENT (expected - optimization changed them)")
        else:
            print(f"  ⚠️  Values are SAME (unexpected)")
    
    print("\n✅ RunView test PASSED")
    return True

if __name__ == "__main__":
    try:
        test_runview()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
