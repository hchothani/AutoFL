#!/usr/bin/env python3
"""
Detailed test for PLoRA strategy on real CIFAR10 data.
Verifies accuracy improvement after training.
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.SimpleCNN import Net
from algorithms.plora import PLoRAStrategy
from workloads.CIFAR10 import load_datasets


def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch['img']
                y = batch['label']
            else:
                x, y = batch
            if isinstance(x, list):
                x = torch.stack(x)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0

def test_plora_cifar10():
    print("\n" + "="*50)
    print("testing PLoRAStrategy on real CIFAR10 data")
    print("="*50)
    try:
        # Load CIFAR10 data (single client/partition)
        trainloader, valloader, testloader = load_datasets(partition_id=0)
        print(f"Loaded train: {len(trainloader)} batches, val: {len(valloader)}, test: {len(testloader)}")
        # Create model and strategy
        net = Net()
        plora = PLoRAStrategy(net, plora_config={'rank': 4, 'alpha': 1.0}, num_clients=1)
        # Evaluate before training
        acc_before = evaluate_accuracy(plora.model, testloader)
        print(f"Test accuracy before training: {acc_before:.4f}")
        # Train for a few epochs
        epochs = 3
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for batch in trainloader:
                if isinstance(batch, dict):
                    x = batch['img']
                    y = batch['label']
                else:
                    x, y = batch
                if isinstance(x, list):
                    x = torch.stack(x)
                plora.model.train()
                lora_params = [p for n, m in plora.model.named_modules() if hasattr(m, 'lora_a') for p in [m.lora_a, m.lora_b]]
                optimizer = torch.optim.Adam(lora_params, lr=0.001)
                optimizer.zero_grad()
                out = plora.model(x)
                loss = torch.nn.CrossEntropyLoss()(out, y)
                loss.backward()
                optimizer.step()
            acc = evaluate_accuracy(plora.model, valloader)
            print(f"  Validation accuracy: {acc:.4f}")
        # Evaluate after training
        acc_after = evaluate_accuracy(plora.model, testloader)
        print(f"Test accuracy after training: {acc_after:.4f}")
        print(f"Accuracy improvement: {acc_after - acc_before:.4f}")
        assert acc_after > acc_before, "Accuracy did not improve after training!"
        print("PloRA CIFAR10 test passed.")
        return True
    except Exception as e:
        print(f"PloRA CIFAR10 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_plora_cifar10() 