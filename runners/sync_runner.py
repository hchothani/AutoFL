import collections
import collections.abc
collections.Sequence = collections.abc.Sequence

import time
from typing import Dict, Optional, Tuple, List
import numpy as np

import flwr as fl
import torch
import wandb

from clients.sync_client import SyncSimulatedClient

def run_sync_simulation(cfg, model_fn, train_loaders, test_loaders, global_test_loaders, device, wandb_enabled):
    """Pure synchronous FL execution loop using standard Flower logic."""
    num_rounds = cfg.server.num_rounds
    num_clients = cfg.server.num_clients

    cl_enabled = cfg.get("cl", {}).get("enabled", False)
    num_phases = cfg.get("cl", {}).get("num_experiences", 1) if cl_enabled else 1
    rounds_per_phase = max(1, num_rounds // num_phases)

    phase_max_accs = [0.0] * num_phases
    initial_phase_acc = [None] * num_phases
    seen_phases = set()

    print(f"\n[Sync Runner] Initializing synchronous simulation for {num_rounds} rounds...")

    start_time = time.time()

    # 1. Centralized Evaluator
    def evaluate_fn(server_round: int, parameters: fl.common.NDArrays, config: Dict) -> Optional[Tuple[float, Dict]]:
        model = model_fn().to(device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        current_phase = 0 if server_round == 0 else min((server_round - 1) // rounds_per_phase, num_phases - 1)
        
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_phases_loss = 0.0
        total_correct = 0
        total_total = 0
        metrics_dict = {}
        phases_accuracies = []
        
        for phase_idx, phase_loader in enumerate(global_test_loaders):
            with torch.no_grad():
                for batch in phase_loader:
                    if isinstance(batch, dict):
                        images, labels = batch.get("img", batch.get("x")).to(device), batch.get("label", batch.get("y")).to(device)
                    else:
                        images, labels = batch[0].to(device), batch[1].to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    phase_loss += loss.item() * labels.size(0)
                    correct += outputs.max(1)[1].eq(labels).sum().item()
                    total += labels.size(0)
            total_total += total
            total_correct += correct
            phase_accuracy = correct / max(total, 1)
            phase_accuracies.append(phase_accuracy)
            total_phases_loss += phase_loss / max(total, 1)
            metrics[f"phase_{phase_idx}_accuracy"] = phase_accuracy

        avg_loss = total_phases_loss / max(total, 1)
        avg_accuracy = total_correct / max(total, 1)

        if server_round == 0 or initial_phase_acc[0] is None:
            for p in range(num_phases):
                initial_phase_acc[p] = phase_accuracies[p]
        seen_phases.add(current_phase) 
        phase_max_accs[current_phase] = max()

        bwt, fwt = 0.0, 0.0
        
        # Calculate BWT (Average performance drop on past tasks)
        if current_phase > 0:
            bwt = sum(phase_accuracies[p] - phase_max_accs[p] for p in range(current_phase)) / current_phase
            
        # Calculate FWT (Zero-shot improvement on future tasks vs initial init)
        if current_phase < num_phases - 1:
            fwt = sum(phase_accuracies[p] - initial_phase_acc[p] for p in range(current_phase + 1, num_phases)) / (num_phases - current_phase - 1)

        # Calculate Average Seen Accuracy
        avg_seen_acc = sum(phase_accuracies[p] for p in seen_phases) / len(seen_phases)

        metrics_dict["bwt"] = bwt
        metrics_dict["fwt"] = fwt
        metrics_dict["avg_seen_acc"] = avg_seen_acc
        
        elapsed_time = time.time() - start_time
        
        print(f"[Phase: {current_phase}][Round {server_round} | {elapsed_time:.1f}] Global Eval - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        if wandb_enabled:
            log_dict = {
                "sync/loss": avg_loss, 
                "sync/accuracy": avg_accuracy,
                "sync/round": server_round,
                "sync/elapsed_time": elapsed_time
            }
            if cl_enabled:
                for k, v in metrics_dict.items():
                    log_dict[f"sync/{k}"] = v
            wandb.log(log_dict, step=server_round)

            
        return avg_loss, log_dict

    # Setup Contextual Configuration
    def on_fit_config_fn(server_round: int) -> Dict[str, fl.common.Scalar]:        
        current_phase = min((server_round - 1) // rounds_per_phase, num_phases - 1)
        print(f"[Server] Round {server_round} starting. Instructing all vehicles to use Phase {current_phase}")
        return {"current_phase": current_phase}

    # 2. Strategy Initialization
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=cfg.server.fraction_fit,
        min_fit_clients=cfg.server.min_fit,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_fit_config_fn
    )

    # 3. Client Factory
    def client_fn(cid: str) -> fl.client.Client:
        client_idx = int(cid)
        return SyncSimulatedClient(
            cid=cid,
            model_fn=model_fn,
            train_loaders=train_loaders[client_idx], # Passed as list
            test_loaders=test_loaders[client_idx],   # Passed as list
            device=device,
            cfg=cfg
        ).to_client()

    # 4. Execute Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": cfg.client.num_cpus, "num_gpus": cfg.client.num_gpus},
        ray_init_args={
            "ignore_reinit_error": True,
        },
    )
    
    total_elapsed = time.time() - start_time
    print(f"\n[Sync Runner] Simulation Complete in {total_elapsed:.1f}s")
    
    # 5. Extract Final Metrics
    final_loss = history.losses_centralized[-1][1] if history.losses_centralized else 0.0
    final_acc = history.metrics_centralized.get("sync/accuracy", [(0, 0.0)])[-1][1] if history.metrics_centralized else 0.0

    return {
        "final_loss": final_loss,
        "final_accuracy": final_acc,
        "total_updates": num_rounds * int(num_clients * cfg.server.fraction_fit),
        "elapsed_time": total_elapsed,
    }
