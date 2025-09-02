import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from algorithms.sacfl import SacFLEncoderDecoder, SacFLStrategy

def make_mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 16),
        nn.ReLU(),
        nn.Linear(16, out_dim)
    )

def test_sacfl_strategy():
    # Simple encoder/decoder
    encoder = make_mlp(8, 4)
    decoder = make_mlp(4, 2)
    model = SacFLEncoderDecoder(encoder, decoder)
    config = {
        'shift_threshold': 0.01,  # low to trigger shift
        'memory_size': 10,
        'contrastive_temperature': 0.5,
        'contrastive_weight': 1.0
    }
    strategy = SacFLStrategy(model, config, num_clients=2)

    # Fake data: 2 batches per client
    torch.manual_seed(42)
    data_client0 = [(torch.randn(4, 8), torch.randint(0, 2, (4,))) for _ in range(2)]
    data_client1 = [(torch.randn(4, 8), torch.randint(0, 2, (4,))) for _ in range(2)]

    # Simulate a round for each client
    enc0, dec0 = strategy.train_round(data_client0, task_id=0, client_id=0)
    enc1, dec1 = strategy.train_round(data_client1, task_id=0, client_id=1)

    # Aggregate encoder
    agg_enc = strategy.aggregate_encoder([enc0, enc1])
    print("Aggregated encoder weights (first layer):", list(agg_enc.parameters())[0].flatten()[:5])

    # Check decoder locality
    print("Client 0 decoder weights (first layer):", list(strategy.client_decoders[0].parameters())[0].flatten()[:5])
    print("Client 1 decoder weights (first layer):", list(strategy.client_decoders[1].parameters())[0].flatten()[:5])

    # Check shift detection and reset
    triggered0 = strategy.detect_shift(0)
    triggered1 = strategy.detect_shift(1)
    print(f"Shift detected for client 0: {triggered0}")
    print(f"Shift detected for client 1: {triggered1}")
    assert triggered0 or triggered1, "At least one client should trigger shift detection in this test."

    # Evaluate
    acc0 = strategy.evaluate(data_client0, client_id=0)
    acc1 = strategy.evaluate(data_client1, client_id=1)
    print(f"Client 0 accuracy: {acc0:.3f}")
    print(f"Client 1 accuracy: {acc1:.3f}")

if __name__ == "__main__":
    test_sacfl_strategy() 