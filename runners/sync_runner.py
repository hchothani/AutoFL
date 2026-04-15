import time
from typing import Dict, Optional, Tuple, List
import numpy as np

import flwr as fl
import torch
import wandb

from clients.sync_client import SyncSimulatedClient

def run_sync_simulation(cfg, model_fn, train_loaders, test_loaders, global_test_loader, device, wandb_enabled):
    """Pure synchronous FL execution loop using standard Flower logic."""
    num_rounds = cfg.server.num_rounds
    num_clients = cfg.server.num_clients
    
    print(f"\n[Sync Runner] Initializing synchronous simulation for {num_rounds} rounds...")

    start_time = time.time();

    # 1. Centralized Evaluator (Runs on the Server)
    def evaluate_fn(server_round: int, parameters: fl.common.NDArrays, config: Dict) -> Optional[Tuple[float, Dict]]:
        model = model_fn().to(device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for batch in global_test_loader:
                if isinstance(batch, dict):
                    images, labels = batch.get("img", batch.get("x")).to(device), batch.get("label", batch.get("y")).to(device)
                else:
                    images, labels = batch[0].to(device), batch[1].to(device)
                    
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                correct += outputs.max(1)[1].eq(labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        elapsed_time = time.time() - start_time
        
        print(f"[Round {server_round} | {elapsed_time:.1f}] Global Eval - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        if wandb_enabled:
            wandb.log({
                "sync/loss": avg_loss, 
                "sync/accuracy": accuracy,
                "sync/round": server_round,
                "sync/elapsed_time": elapsed_time
            }, step=server_round)
            
        return avg_loss, {"accuracy": accuracy, "elapsed_time": elapsed_time}

    # 2. Strategy Initialization
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=cfg.server.fraction_fit,
        min_fit_clients=cfg.server.min_fit,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn
    )

    # 3. Client Factory (Spins up clients on demand)
    def client_fn(cid: str) -> fl.client.Client:
        client_idx = int(cid)
        return SyncSimulatedClient(
            cid=cid,
            model_fn=model_fn,
            train_loader=train_loaders[client_idx],
            test_loader=test_loaders[client_idx],
            device=device,
            cfg=cfg
        ).to_client()

    # 4. Execute Simulation
#    start_time = time.time()
    
    # Run the Flower simulation engine
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": cfg.client.num_cpus, "num_gpus": cfg.client.num_gpus},
        ray_init_args={
            "num_cpus":  cfg.get("async", {}).get("max_workers", 4),
            "num_gpus": 1
        },
    )
    
    total_elapsed = time.time() - start_time
    print(f"\n[Sync Runner] Simulation Complete in {total_elapsed:.1f}s")
    
    # 5. Extract Final Metrics
    final_loss = history.losses_centralized[-1][1] if history.losses_centralized else 0.0
    final_acc = history.metrics_centralized.get("accuracy", [(0, 0.0)])[-1][1] if history.metrics_centralized else 0.0

    return {
        "final_loss": final_loss,
        "final_accuracy": final_acc,
        "total_updates": num_rounds * int(num_clients * cfg.server.fraction_fit),
        "elapsed_time": total_elapsed,
    }
