#!/usr/bin/env python3
"""
Test server strategy selection for both standard FL and SOTA strategies.
Verifies that mclserver.py correctly selects strategies based on config.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from flwr.server.strategy import FedAvg, FedProx
import mclserver

def test_server_strategy_selection():
    """Test that server strategies are correctly selected based on config."""
    print("\n" + "="*50)
    print("Testing Server Strategy Selection")
    print("="*50)
    
    results = []
    
    # Test cases: strategy name -> expected type/behavior
    test_cases = [
        ('fedavg', FedAvg),
        ('fedprox', FedProx),
        ('fedweit', FedAvg),  # Should fallback to FedAvg for now
        ('plora', FedAvg),    # Should fallback to FedAvg for now
        ('unknown', FedAvg)   # Should fallback to FedAvg
    ]
    
    for strategy_name, expected_type in test_cases:
        try:
            print(f"\nTesting strategy: {strategy_name}")
            
            # Create mock config
            config = OmegaConf.create({
                'server': {
                    'strategy': strategy_name,
                    'fraction_fit': 1.0,
                    'fraction_eval': 1.0,
                    'min_fit': 2,
                    'min_eval': 2,
                    'num_clients': 3,
                    'num_rounds': 5,
                    'fedprox': {'mu': 0.01},  # For FedProx test
                }
            })
            
            # Mock the config loading
            original_cfg = mclserver.cfg
            mclserver.cfg = config
            
            # Test strategy creation
            strategy = mclserver.create_server_strategy()
            
            # Verify strategy type
            assert isinstance(strategy, expected_type), f"Expected {expected_type}, got {type(strategy)}"
            print(f"✓ {strategy_name} -> {type(strategy).__name__}")
            
            # Restore config
            mclserver.cfg = original_cfg
            
            results.append(True)
            
        except Exception as e:
            print(f"✗ Strategy test failed for {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n{'='*50}")
    print(f"Server Strategy Selection Test Results: {success_count}/{total_count} passed")
    print(f"{'='*50}")
    
    if success_count == total_count:
        print("🎉 All server strategy selection tests passed!")
        return True
    else:
        print("❌ Some server strategy selection tests failed!")
        return False

if __name__ == "__main__":
    test_server_strategy_selection() 