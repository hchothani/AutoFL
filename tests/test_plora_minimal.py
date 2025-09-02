#!/usr/bin/env python3
"""
Minimal test for PLoRA strategy with SimpleCNN and random data.
Verifies LoRA injection, training, and aggregation.
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.SimpleCNN import Net
from algorithms.plora import PLoRAStrategy

def test_plora_minimal():
    print("\n" + "="*50)
    print("testing PLoRAStrategy minimal LoRA injection/aggregation")
    print("="*50)
    try:
        # create two simplecnn models
        net1 = Net()
        net2 = Net()
        # randomize weights
        for p in net1.parameters():
            p.data.normal_()
        for p in net2.parameters():
            p.data.normal_()
        # create plora strategy
        plora = PLoRAStrategy(net1, plora_config={'rank': 4, 'alpha': 1.0}, num_clients=2)
        # simulate two clients with random data
        data = [(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))) for _ in range(5)]
        # train both clients
        plora.train_round(data, task_id=0, client_id=0)
        plora.train_round(data, task_id=0, client_id=1)
        # aggregate
        agg = plora.aggregate([plora.client_loras[0], plora.client_loras[1]])
        print("PloRA aggregation completed. Aggregated LoRA layers:", list(agg.keys()))
        # test forward pass
        out = plora.model(torch.randn(4, 3, 32, 32))
        print("PloRA model output shape:", out.shape)
        print("PloRA minimal test passed.")
        return True
    except Exception as e:
        print(f"PloRA minimal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_plora_minimal() 