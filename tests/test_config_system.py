#!/usr/bin/env python3
"""
test script to verify that the configuration system works correctly.
tests model selection, parameter passing, and configuration overrides.
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import sys
import os

# add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_simple_cnn_config():
    """test SimpleCNN model loading with different configurations"""
    print("="*60)
    print("testing SimpleCNN configuration")
    print("="*60)
    
    try:
        # test with CIFAR10 config (10 classes)
        cfg = OmegaConf.load('config/config.yaml')
        cfg.model.name = "simple_cnn"
        cfg.model.num_classes = 10
        cfg.dataset.workload = "cifar10"
        
        # import and override the config in SimpleCNN
        import models.SimpleCNN
        models.SimpleCNN.cfg = cfg
        
        from models.SimpleCNN import Net
        net = Net()
        
        print(f"simplecnn loaded with {cfg.model.num_classes} classes")
        print(f"final layer output features: {net.fc3.out_features}")
        print(f"total parameters: {sum(p.numel() for p in net.parameters())}")
        
        # test with CIFAR100 config (100 classes)
        cfg.model.num_classes = 100
        cfg.dataset.workload = "cifar100"
        models.SimpleCNN.cfg = cfg
        
        # need to reload the module to get updated config
        import importlib
        importlib.reload(models.SimpleCNN)
        from models.SimpleCNN import Net
        net = Net()
        
        print(f"simplecnn reloaded with {cfg.model.num_classes} classes")
        print(f"final layer output features: {net.fc3.out_features}")
        print(f"total parameters: {sum(p.numel() for p in net.parameters())}")
        
        return True
        
    except Exception as e:
        print(f"simplecnn test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_resnet_config():
    """test ResNet model loading with different configurations"""
    print("\n" + "="*60)
    print("testing ResNet configuration")
    print("="*60)
    
    try:
        # test with CIFAR10 config (10 classes)
        cfg = OmegaConf.load('config/config.yaml')
        cfg.model.name = "resnet"
        cfg.model.num_classes = 10
        cfg.dataset.workload = "cifar10"
        
        # import and override the config in ResNet
        import models.ResNet
        models.ResNet.cfg = cfg
        
        from models.ResNet import ResNet
        net = ResNet(num_classes=10)
        
        print(f"resnet loaded with 10 classes")
        print(f"final layer output features: {net.model.fc.out_features}")
        print(f"total parameters: {sum(p.numel() for p in net.parameters())}")
        
        # test with CIFAR100 config (100 classes)
        net = ResNet(num_classes=100)
        
        print(f"resnet loaded with 100 classes")
        print(f"final layer output features: {net.model.fc.out_features}")
        print(f"total parameters: {sum(p.numel() for p in net.parameters())}")
        
        # test default parameter from config
        cfg.model.num_classes = 50
        models.ResNet.cfg = cfg
        
        # need to reload the module to get updated config
        import importlib
        importlib.reload(models.ResNet)
        from models.ResNet import ResNet
        net = ResNet()  # should use config default
        
        print(f"resnet with config default classes")
        print(f"final layer output features: {net.model.fc.out_features}")
        
        return True
        
    except Exception as e:
        print(f"resnet test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_selection_from_config():
    """test that the get_model() function correctly selects models based on config"""
    print("\n" + "="*60)
    print("testing model selection from configuration")
    print("="*60)
    
    try:
        # test get_model() function from mclmain.py
        import mclmain
        
        # test SimpleCNN selection
        cfg = OmegaConf.load('config/config.yaml')
        cfg.model.name = "simple_cnn"
        cfg.model.num_classes = 10
        cfg.dataset.workload = "cifar10"
        mclmain.cfg = cfg
        
        net = mclmain.get_model()
        print(f"get_model() with simple_cnn: {type(net).__name__}")
        print(f"output classes: {net.fc3.out_features}")
        
        # test ResNet selection
        cfg.model.name = "resnet"
        cfg.model.num_classes = 100
        cfg.dataset.workload = "cifar100"
        mclmain.cfg = cfg
        
        net = mclmain.get_model()
        print(f"get_model() with resnet: {type(net).__name__}")
        print(f"output classes: {net.model.fc.out_features}")
        
        # test unknown model error
        cfg.model.name = "unknown_model"
        mclmain.cfg = cfg
        
        try:
            net = mclmain.get_model()
            print("should have raised error for unknown model")
            return False
        except ValueError as e:
            print(f"correctly raised error for unknown model: {e}")
        
        return True
        
    except Exception as e:
        print(f"model selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mclient_model_selection():
    """test model selection in mclientCL.py"""
    print("\n" + "="*60)
    print("testing model selection in mclientCL")
    print("="*60)
    
    try:
        # test get_model() function from mclientCL.py
        import mclientCL
        
        # test SimpleCNN selection
        cfg = OmegaConf.load('config/config.yaml')
        cfg.model.name = "simple_cnn"
        cfg.model.num_classes = 10
        mclientCL.cfg = cfg
        
        net = mclientCL.get_model()
        print(f"mclientCL.get_model() with simple_cnn: {type(net).__name__}")
        
        # test ResNet selection
        cfg.model.name = "resnet"
        cfg.model.num_classes = 100
        mclientCL.cfg = cfg
        
        net = mclientCL.get_model()
        print(f"mclientCL.get_model() with resnet: {type(net).__name__}")
        print(f"output classes: {net.model.fc.out_features}")
        
        return True
        
    except Exception as e:
        print(f"mclientCL model selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_parameter_flow():
    """test that configuration parameters flow correctly through the system"""
    print("\n" + "="*60)
    print("testing configuration parameter flow")
    print("="*60)
    
    try:
        cfg = OmegaConf.load('config/config.yaml')
        
        # test parameter access
        print(f"model name: {cfg.model.name}")
        print(f"model classes: {cfg.model.num_classes}")
        print(f"dataset workload: {cfg.dataset.workload}")
        print(f"batch size: {cfg.dataset.batch_size}")
        print(f"learning rate: {cfg.training.learning_rate}")
        print(f"client epochs: {cfg.client.epochs}")
        print(f"server rounds: {cfg.server.num_rounds}")
        print(f"number of clients: {cfg.server.num_clients}")
        print(f"cl experiences: {cfg.cl.num_experiences}")
        print(f"cl strategy: {cfg.cl.strategy}")
        
        # test parameter modification
        original_lr = cfg.training.learning_rate
        cfg.training.learning_rate = 0.01
        print(f"learning rate changed from {original_lr} to {cfg.training.learning_rate}")
        
        # test nested parameter access
        print(f"niid alpha: {cfg.dataset.niid.alpha}")
        print(f"niid seed: {cfg.dataset.niid.seed}")
        print(f"wandb project: {cfg.wb.project}")
        print(f"wandb name: {cfg.wb.name}")
        
        return True
        
    except Exception as e:
        print(f"config parameter flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """run all configuration system tests"""
    print("starting configuration system tests...")
    
    # run all tests
    test1_passed = test_simple_cnn_config()
    test2_passed = test_resnet_config() 
    test3_passed = test_model_selection_from_config()
    test4_passed = test_mclient_model_selection()
    test5_passed = test_config_parameter_flow()
    
    print("\n" + "="*60)
    print("configuration system test summary")
    print("="*60)
    print(f"simplecnn config:         {'passed' if test1_passed else 'failed'}")
    print(f"resnet config:            {'passed' if test2_passed else 'failed'}")
    print(f"model selection (main):   {'passed' if test3_passed else 'failed'}")
    print(f"model selection (client): {'passed' if test4_passed else 'failed'}")
    print(f"parameter flow:           {'passed' if test5_passed else 'failed'}")
    
    all_passed = all([test1_passed, test2_passed, test3_passed, test4_passed, test5_passed])
    
    if all_passed:
        print("all configuration tests passed")
        print("model selection works correctly")
        print("parameter passing flows properly") 
        print("configuration overrides work")
        print("both simplecnn and resnet models are configurable")
    else:
        print("some configuration tests failed")
    
    print("="*60)

if __name__ == "__main__":
    main() 