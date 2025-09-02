import torch
import torch.nn as nn
from algorithms.stamp import STAMPStrategy

# Simple CNN for MNIST-like data
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(26*26*16, num_classes)
    def forward(self, x):
        print(f"[DEBUG] Input shape: {x.shape}")
        x = self.conv[0](x)
        print(f"[DEBUG] After conv: {x.shape}")
        x = self.conv[1](x)
        x = self.conv[2](x)
        print(f"[DEBUG] After flatten: {x.shape}")
        return self.fc(x)


def test_stamp_minimal():
    print("\n" + "="*50)
    print("testing STAMPStrategy minimal workflow")
    print("="*50)
    try:
        num_clients = 2
        num_classes = 3
        batch_size = 4
        input_shape = (1, 28, 28)
        # create model
        net = SimpleCNN(num_classes=num_classes)
        # create random data for two clients
        data_clients = []
        for c in range(num_clients):
            data = []
            for _ in range(batch_size):
                x = torch.randn(input_shape)
                y = torch.randint(0, num_classes, (1,)).item()
                data.append((x, torch.tensor(y)))
            data_clients.append(data)
        # create STAMP strategy
        stamp = STAMPStrategy(net, stamp_config={'coreset_size': 8, 'feature_layer': 'conv', 'device': 'cpu'}, num_clients=num_clients)
        # run local training for each client
        client_protos = []
        for cid in range(num_clients):
            xs = torch.stack([x if x.dim() == 3 else x.unsqueeze(0) for x, y in data_clients[cid]])
            ys = torch.stack([y for x, y in data_clients[cid]])
            coreset, protos = stamp.train_round((xs, ys), task_id=0, client_id=cid, lambda_proto=1.0)
            print(f"client {cid} coreset size: {len(coreset)}; prototypes: {list(protos.keys())}")
            client_protos.append(protos)
        # aggregate
        agg_coreset, global_protos = stamp.aggregate([stamp.get_coreset_params(0), stamp.get_coreset_params(1)], client_protos)
        print(f"aggregated coreset size: {len(agg_coreset)}; global prototypes: {list(global_protos.keys())}")
        print("STAMP minimal test passed.")
        return True
    except Exception as e:
        print(f"STAMP minimal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_stamp_minimal() 