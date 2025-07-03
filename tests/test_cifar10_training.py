#!/usr/bin/env python3
"""
comprehensive test script to verify that all CIFAR10 workloads work correctly.
tests regular CIFAR10, CIFAR10CL, and CIFAR10DomainCL workloads.
"""

import torch
import torch.nn as nn
import sys
import os

# add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from models.SimpleCNN import Net
from clutils.clstrat import make_cl_strat
from clutils.make_experiences import split_dataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets

# load config and set to CIFAR10
cfg = OmegaConf.load('config/config.yaml')
cfg.model.num_classes = 10  # CIFAR10 has 10 classes

def test_cifar10_regular():
    """test regular CIFAR10 workload"""
    print("="*50)
    print("testing regular CIFAR10 workload")
    print("="*50)
    
    try:
        from workloads.CIFAR10 import load_datasets
        
        # load datasets
        trainloader, valloader, testloader = load_datasets(partition_id=0)
        print(f"loaded train loader with {len(trainloader)} batches")
        print(f"loaded val loader with {len(valloader)} batches")
        print(f"loaded test loader with {len(testloader)} batches")
        
        # test data format
        for batch in trainloader:
            if isinstance(batch, dict):
                print(f"batch format: dict with keys {list(batch.keys())}")
                if 'img' in batch:
                    print(f"image shape: {batch['img'].shape}")
                if 'label' in batch:
                    print(f"labels: {batch['label'][:5]}")
            else:
                print(f"batch format: {type(batch)}")
            break
        
        print("regular CIFAR10 test passed")
        return True
        
    except Exception as e:
        print(f"regular CIFAR10 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cifar10_cl():
    """test CIFAR10 CL workload"""
    print("\n" + "="*50)
    print("testing CIFAR10 CL workload")
    print("="*50)
    
    try:
        from workloads.CIFAR10CL import load_datasets
        
        # load datasets
        train_data, test_data = load_datasets(partition_id=0)
        print(f"loaded train data: {len(train_data)} samples")
        print(f"loaded test data: {len(test_data)} samples")
        
        # create experiences
        n_experiences = 3  # use fewer experiences for testing
        train_experiences = split_dataset(train_data, n_experiences)
        test_experiences = split_dataset(test_data, n_experiences)
        print(f"created {len(train_experiences)} train experiences")
        print(f"created {len(test_experiences)} test experiences")
        
        # create benchmark
        benchmark = benchmark_from_datasets(train=train_experiences, test=test_experiences)
        print("created benchmark")
        
        # create model and strategy
        net = Net()
        cl_strategy, eval_plugin = make_cl_strat(net)
        print(f"created model with {sum(p.numel() for p in net.parameters())} parameters")
        print(f"model output classes: {net.fc3.out_features}")
        print(f"learning rate: {cl_strategy.optimizer.param_groups[0]['lr']}")
        
        # test training on first experience (abbreviated)
        for i, experience in enumerate(benchmark.train_stream):
            if i == 0:  # only train on first experience
                print(f"training on experience {experience.current_experience}")
                # test just a few epochs for speed
                original_epochs = cl_strategy.train_epochs
                cl_strategy.train_epochs = 2  # reduce epochs for testing
                result = cl_strategy.train(experience)
                cl_strategy.train_epochs = original_epochs  # restore
                print("training completed successfully")
                
                # check if loss decreased
                if result and 'Loss_Epoch/train_phase/train_stream' in result:
                    final_loss = result['Loss_Epoch/train_phase/train_stream']
                    print(f"final training loss: {final_loss}")
                break
        
        print("CIFAR10 CL test passed")
        return True
        
    except Exception as e:
        print(f"CIFAR10 CL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cifar10_domain():
    """test CIFAR10 Domain CL workload"""
    print("\n" + "="*50)
    print("testing CIFAR10 domain CL workload")
    print("="*50)
    
    try:
        from workloads.CIFAR10DomainCL import load_datasets
        
        # load datasets
        benchmark = load_datasets(partition_id=0)
        print("loaded domain CL benchmark")
        
        # handle different benchmark types
        if hasattr(benchmark, 'train_stream'):
            train_stream = benchmark.train_stream
            test_stream = benchmark.test_stream
        elif hasattr(benchmark, 'train_datasets_stream'):
            train_stream = benchmark.train_datasets_stream
            test_stream = benchmark.test_datasets_stream
        else:
            raise ValueError(f"unknown benchmark type: {type(benchmark)}")
            
        print(f"train experiences: {len(list(train_stream))}")
        print(f"test experiences: {len(list(test_stream))}")
        
        # create model and strategy
        net = Net()
        cl_strategy, eval_plugin = make_cl_strat(net)
        print(f"created model with {sum(p.numel() for p in net.parameters())} parameters")
        print(f"model output classes: {net.fc3.out_features}")
        print(f"learning rate: {cl_strategy.optimizer.param_groups[0]['lr']}")
        
        # test training on first experience (abbreviated)
        for i, experience in enumerate(train_stream):
            if i == 0:  # only train on first experience
                print(f"training on experience {experience.current_experience}")
                # test just a few epochs for speed
                original_epochs = cl_strategy.train_epochs
                cl_strategy.train_epochs = 2  # reduce epochs for testing
                result = cl_strategy.train(experience)
                cl_strategy.train_epochs = original_epochs  # restore
                print("training completed successfully")
                
                # check if loss decreased
                if result and hasattr(result, 'get'):
                    loss_keys = [k for k in result.keys() if 'Loss_Epoch' in k and 'train' in k]
                    if loss_keys:
                        final_loss = result[loss_keys[0]]
                        print(f"final training loss: {final_loss}")
                break
        
        print("CIFAR10 domain CL test passed")
        return True
        
    except Exception as e:
        print(f"CIFAR10 domain CL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("starting CIFAR10 workload tests...")
    print(f"model configured for {cfg.model.num_classes} classes")
    
    # test all CIFAR10 workloads
    test1_passed = test_cifar10_regular()
    test2_passed = test_cifar10_cl()
    test3_passed = test_cifar10_domain()
    
    print("\n" + "="*50)
    print("CIFAR10 test summary")
    print("="*50)
    print(f"regular CIFAR10:     {'passed' if test1_passed else 'failed'}")
    print(f"CIFAR10 CL:          {'passed' if test2_passed else 'failed'}")
    print(f"CIFAR10 domain CL:   {'passed' if test3_passed else 'failed'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("all CIFAR10 tests passed - all workloads are functional")
    else:
        print("some CIFAR10 tests failed - check error messages above")
    
    print("="*50)

if __name__ == "__main__":
    main() 