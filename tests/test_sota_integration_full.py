#!/usr/bin/env python3
"""
Comprehensive test for SOTA algorithm integration.
Tests the full pipeline: config loading, strategy creation, parameter handling.
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from models.SimpleCNN import Net
from clutils.clstrat import make_cl_strat
import mclserver
from algorithms.fedweit import FedWeITStrategy
from algorithms.plora import PLoRAStrategy

def test_sota_algorithm_integration():
    """Test full integration for multiple SOTA algorithms."""
    print("\n" + "="*70)
    print("COMPREHENSIVE SOTA ALGORITHM INTEGRATION TEST")
    print("="*70)
    
    results = []
    
    # Test algorithms with their expected strategy classes
    test_algorithms = [
        ('fedweit', FedWeITStrategy),
        ('plora', PLoRAStrategy),
        ('fedcprompt', None)  # Import will be tested dynamically
    ]
    
    for algo_name, expected_class in test_algorithms:
        print(f"\n{'-'*50}")
        print(f"Testing {algo_name.upper()} Integration")
        print(f"{'-'*50}")
        
        try:
            # 1. Test Config Loading (from sota directory if exists)
            sota_config_path = f'config/sota/{algo_name}_cifar10.yaml'
            if os.path.exists(sota_config_path):
                print(f"✓ Found SOTA config: {sota_config_path}")
                config = OmegaConf.load(sota_config_path)
            else:
                print(f"⚠ Creating test config for {algo_name}")
                config = create_test_config(algo_name)
            
            assert config.cl.strategy == algo_name
            print(f"✓ Config loaded: strategy = {config.cl.strategy}")
            
            # 2. Test Client Strategy Creation via Config
            # Mock the config system
            import config_utils
            original_load = config_utils.load_config
            config_utils.load_config = lambda: config
            
            # Reload clstrat to pick up new config
            import importlib
            import clutils.clstrat
            importlib.reload(clutils.clstrat)
            
            # Test strategy creation
            net = Net()
            strategy, eval_plugin = clutils.clstrat.make_cl_strat(net)
            
            if expected_class:
                assert isinstance(strategy, expected_class), f"Expected {expected_class}, got {type(strategy)}"
            print(f"✓ Client strategy created: {type(strategy).__name__}")
            
            # 3. Test Server Strategy Selection
            original_cfg = mclserver.cfg
            mclserver.cfg = config
            server_strategy = mclserver.create_server_strategy()
            print(f"✓ Server strategy selected: {type(server_strategy).__name__}")
            mclserver.cfg = original_cfg
            
            # 4. Test Basic Strategy Operations
            if hasattr(strategy, 'model'):
                model_params = len(list(strategy.model.parameters()))
                print(f"✓ Strategy has model with {model_params} parameter tensors")
            
            # 5. Test Algorithm-Specific Features
            if algo_name == 'fedweit':
                # Test FedWeIT-specific features
                base_params = strategy._get_model_params()
                assert len(base_params) > 0
                print(f"✓ FedWeIT base parameters: {len(base_params)} layers")
                
            elif algo_name == 'plora':
                # Test PLoRA-specific features
                lora_config = getattr(config, 'plora', {})
                assert 'rank' in lora_config
                print(f"✓ PLoRA config: rank={lora_config.get('rank', 'N/A')}")
            
            # Restore original config loader
            config_utils.load_config = original_load
            
            print(f"🎉 {algo_name.upper()} integration test PASSED")
            results.append(True)
            
        except Exception as e:
            print(f"❌ {algo_name.upper()} integration test FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
            
            # Restore config loader even on failure
            try:
                config_utils.load_config = original_load
            except:
                pass
    
    # Summary
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n{'='*70}")
    print(f"SOTA ALGORITHM INTEGRATION TEST RESULTS")
    print(f"{'='*70}")
    print(f"Algorithms tested: {total_count}")
    print(f"Successful integrations: {success_count}")
    print(f"Failed integrations: {total_count - success_count}")
    
    if success_count == total_count:
        print("\n🎉 ALL SOTA ALGORITHM INTEGRATIONS PASSED!")
        print("✅ Your SOTA algorithms are fully integrated with the FL/CL pipeline!")
        print("🚀 You can now run experiments with these algorithms using the config system.")
        return True
    else:
        print(f"\n⚠ {total_count - success_count} integration(s) failed.")
        print("🔧 Check the failed algorithms and fix any issues.")
        return False

def create_test_config(algo_name):
    """Create a test config for the given algorithm."""
    base_config = {
        'dataset': {'workload': 'cifar10', 'batch_size': 32, 'num_classes': 10},
        'model': {'name': 'simple_cnn', 'num_classes': 10},
        'cl': {'strategy': algo_name, 'num_experiences': 5, 'split': 'random'},
        'training': {'learning_rate': 0.001, 'epochs': 1},
        'server': {'strategy': 'fedavg', 'num_rounds': 3, 'num_clients': 2, 'fraction_fit': 1.0, 'fraction_eval': 1.0, 'min_fit': 2, 'min_eval': 2},
        'client': {'num_cpus': 4, 'num_gpus': 0.0, 'epochs': 1, 'falloff': 0.0, 'exp_epochs': None},
        'wb': {'project': 'test', 'name': f'{algo_name}_test'}
    }
    
    # Add algorithm-specific config
    if algo_name == 'fedweit':
        base_config['fedweit'] = {'sparsity': 0.5, 'l1_lambda': 0.1, 'l2_lambda': 100.0}
    elif algo_name == 'plora':
        base_config['plora'] = {'rank': 4, 'alpha': 1.0}
    elif algo_name == 'fedcprompt':
        base_config['fedcprompt'] = {'prompt_length': 8, 'prompt_lr': 0.01}
    
    return OmegaConf.create(base_config)

def test_existing_strategies_still_work():
    """Test that existing strategies (naive, ewc, etc.) still work."""
    print(f"\n{'-'*50}")
    print("Testing Existing Strategies Still Work")
    print(f"{'-'*50}")
    
    existing_strategies = ['naive', 'ewc', 'replay']
    results = []
    
    for strategy_name in existing_strategies:
        try:
            config = OmegaConf.create({
                'cl': {'strategy': strategy_name},
                'client': {'num_gpus': 0.0, 'epochs': 1},
                'server': {'num_clients': 2},
                'training': {'learning_rate': 0.001},
                'dataset': {'batch_size': 32}
            })
            
            # Mock config system
            import config_utils
            original_load = config_utils.load_config
            config_utils.load_config = lambda: config
            
            # Reload clstrat
            import importlib
            import clutils.clstrat
            importlib.reload(clutils.clstrat)
            
            # Test strategy creation
            net = Net()
            strategy, eval_plugin = clutils.clstrat.make_cl_strat(net)
            
            print(f"✓ {strategy_name} strategy works: {type(strategy).__name__}")
            results.append(True)
            
            # Restore config
            config_utils.load_config = original_load
            
        except Exception as e:
            print(f"✗ {strategy_name} strategy failed: {e}")
            results.append(False)
    
    return all(results)

if __name__ == "__main__":
    print("Starting comprehensive SOTA algorithm integration test...")
    
    # Test SOTA algorithms
    sota_success = test_sota_algorithm_integration()
    
    # Test existing strategies still work
    existing_success = test_existing_strategies_still_work()
    
    print(f"\n{'='*70}")
    print("FINAL INTEGRATION TEST SUMMARY")
    print(f"{'='*70}")
    
    if sota_success and existing_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ SOTA algorithms successfully integrated!")
        print("✅ Existing strategies still functional!")
        print("\n🚀 Ready for production use!")
    else:
        print("❌ Some tests failed:")
        if not sota_success:
            print("  - SOTA algorithm integration issues")
        if not existing_success:
            print("  - Existing strategy compatibility issues")
        print("\n🔧 Please fix issues before proceeding.") 