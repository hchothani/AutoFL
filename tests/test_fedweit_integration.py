#!/usr/bin/env python3
"""
Integration test for FedWeIT strategy - both client and server side.
Tests strategy instantiation, parameter handling, and basic functionality.
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.SimpleCNN import Net
from algorithms.fedweit import FedWeITStrategy, FedWeITServerStrategy
from clutils.clstrat import make_cl_strat
from config_utils import load_config
from omegaconf import OmegaConf

def test_fedweit_client_strategy():
    """Test FedWeIT client strategy instantiation and basic operations."""
    print("\n" + "="*50)
    print("Testing FedWeIT Client Strategy")
    print("="*50)
    
    try:
        # Create model
        net = Net()
        
        # Create FedWeIT strategy with config
        fedweit_config = {
            'sparsity': 0.5,
            'l1_lambda': 0.1,
            'l2_lambda': 100.0
        }
        
        strategy = FedWeITStrategy(
            model=net,
            sparsity=0.5,
            num_clients=2,
            l1_lambda=0.1,
            l2_lambda=100.0,
            device='cpu'
        )
        
        print("✓ FedWeIT strategy created successfully")
        
        # Test basic parameter operations
        base_params = strategy._get_model_params()
        print(f"✓ Base parameters extracted: {len(base_params)} layers")
        
        # Test task initialization
        strategy.train_task(
            data=[(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))) for _ in range(3)],
            task_id=0,
            client_id=0,
            epochs=1
        )
        print("✓ Task training completed")
        
        # Test sparse update generation
        sparse_update = strategy.get_sparse_update(client_id=0, task_id=0)
        assert 'masked_base' in sparse_update
        assert 'task_adaptive' in sparse_update
        print("✓ Sparse update generated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ FedWeIT client strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fedweit_server_strategy():
    """Test FedWeIT server strategy instantiation."""
    print("\n" + "="*50)
    print("Testing FedWeIT Server Strategy")
    print("="*50)
    
    try:
        # Create server strategy
        server_strategy = FedWeITServerStrategy()
        
        print("✓ FedWeIT server strategy created successfully")
        
        # Test basic aggregation logic
        client_updates = [
            {
                'masked_base': {
                    'layer1': torch.randn(10, 5),
                    'layer2': torch.randn(5, 2)
                }
            },
            {
                'masked_base': {
                    'layer1': torch.randn(10, 5),
                    'layer2': torch.randn(5, 2)
                }
            }
        ]
        
        agg_result = server_strategy.aggregate_client_updates(client_updates)
        assert 'masked_base' in agg_result
        assert 'layer1' in agg_result['masked_base']
        assert 'layer2' in agg_result['masked_base']
        print("✓ Server aggregation logic verified")
        
        return True
        
    except Exception as e:
        print(f"✗ FedWeIT server strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fedweit_config_integration():
    """Test that FedWeIT can be selected via config system."""
    print("\n" + "="*50)
    print("Testing FedWeIT Config Integration")
    print("="*50)
    
    try:
        # Create mock config for FedWeIT
        config = OmegaConf.create({
            'cl': {'strategy': 'fedweit'},
            'fedweit': {
                'sparsity': 0.5,
                'l1_lambda': 0.1,
                'l2_lambda': 100.0
            },
            'server': {'num_clients': 2},
            'training': {'learning_rate': 0.001},
            'client': {'epochs': 1, 'num_gpus': 0.0},
            'dataset': {'batch_size': 32}
        })
        
        # Temporarily save config for testing
        with open('temp_fedweit_config.yaml', 'w') as f:
            OmegaConf.save(config, f)
        
        # Mock the config loading to use our test config
        import config_utils
        original_load = config_utils.load_config
        
        def mock_load():
            return config
        
        config_utils.load_config = mock_load
        
        # Reload the clstrat module to pick up new config
        import importlib
        import clutils.clstrat
        importlib.reload(clutils.clstrat)
        
        # Test strategy creation via config
        net = Net()
        strategy, eval_plugin = clutils.clstrat.make_cl_strat(net)
        
        assert isinstance(strategy, FedWeITStrategy)
        assert eval_plugin is None  # Custom strategies don't use Avalanche eval plugin
        print("✓ FedWeIT strategy created via config system")
        
        # Restore original config loader
        config_utils.load_config = original_load
        
        # Clean up
        if os.path.exists('temp_fedweit_config.yaml'):
            os.remove('temp_fedweit_config.yaml')
        
        return True
        
    except Exception as e:
        print(f"✗ FedWeIT config integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fedweit_integration():
    """Run all FedWeIT integration tests."""
    print("Running FedWeIT Integration Tests...")
    
    results = []
    results.append(test_fedweit_client_strategy())
    results.append(test_fedweit_server_strategy())
    results.append(test_fedweit_config_integration())
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n{'='*50}")
    print(f"FedWeIT Integration Test Results: {success_count}/{total_count} passed")
    print(f"{'='*50}")
    
    if success_count == total_count:
        print("🎉 All FedWeIT integration tests passed!")
        return True
    else:
        print("❌ Some FedWeIT integration tests failed!")
        return False

if __name__ == "__main__":
    test_fedweit_integration() 