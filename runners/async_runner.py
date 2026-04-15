import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import wandb

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from algorithms.async_fl import AsynchronousStrategy, AsyncHistory
from clients.async_client import create_simulated_clients

def get_async_config(cfg: DictConfig) -> Dict[str, Any]:
    """Extract async configuration from the config."""
    async_cfg = cfg.get("async", {})
    if isinstance(async_cfg, DictConfig):
        async_cfg = OmegaConf.to_container(async_cfg, resolve=True)
    return {
        "total_train_time": async_cfg.get("total_train_time", 300),
        "waiting_interval": async_cfg.get("waiting_interval", 10),
        "max_workers": async_cfg.get("max_workers", 4),
        "aggregation_strategy": async_cfg.get("aggregation_strategy", "fedasync"),
        "staleness_alpha": async_cfg.get("staleness_alpha", 0.5),
        "fedasync_mixing_alpha": async_cfg.get("fedasync_mixing_alpha", 0.9),
        "fedasync_a": async_cfg.get("fedasync_a", 0.5),
        "use_staleness": async_cfg.get("use_staleness", True),
        "use_sample_weighing": async_cfg.get("use_sample_weighing", True),
        "send_gradients": async_cfg.get("send_gradients", False),
        "server_artificial_delay": async_cfg.get("server_artificial_delay", False),
        "is_streaming": async_cfg.get("is_streaming", False),
        "client_local_delay": async_cfg.get("client_local_delay", False),
        "simulate_delay": async_cfg.get("simulate_delay", True),
        "min_delay": async_cfg.get("min_delay", 0.5),
        "max_delay": async_cfg.get("max_delay", 3.0),
    }

def evaluate_global_model(model: torch.nn.Module, params: List[np.ndarray], test_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Evaluate model with given parameters."""
    state_dict = model.state_dict()
    for key, param in zip(state_dict.keys(), params):
        state_dict[key] = torch.tensor(param).to(device)
    model.load_state_dict(state_dict)

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                images, labels = batch.get("img", batch.get("x")).to(device), batch.get("label", batch.get("y")).to(device)
            elif isinstance(batch, (tuple, list)):
                images, labels = batch[0].to(device), batch[1].to(device)
            else:
                continue

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / max(total, 1), correct / max(total, 1)

def run_async_simulation(cfg, async_cfg, model_fn, train_loaders, test_loaders, global_test_loader, device, wandb_enabled):
    """Run async FL simulation exactly as originally written."""
    num_clients = len(train_loaders)

    print(f"\nCreating {num_clients} simulated clients...")
    clients = create_simulated_clients(
        num_clients=num_clients,
        model_fn=model_fn,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        device=device,
        local_epochs=cfg.client.local_epochs,
        learning_rate=cfg.client.learning_rate,
        simulate_delay=async_cfg["simulate_delay"],
        min_delay=async_cfg["min_delay"],
        max_delay=async_cfg["max_delay"],
    )

    global_model = model_fn().to(device)
    global_params = [val.cpu().numpy() for _, val in global_model.state_dict().items()]

    total_samples = sum(len(loader.dataset) for loader in train_loaders)
    async_strategy = AsynchronousStrategy(
        total_samples=total_samples,
        staleness_alpha=async_cfg["staleness_alpha"],
        fedasync_mixing_alpha=async_cfg["fedasync_mixing_alpha"],
        fedasync_a=async_cfg["fedasync_a"],
        num_clients=num_clients,
        async_aggregation_strategy=async_cfg["aggregation_strategy"],
        use_staleness=async_cfg["use_staleness"],
        use_sample_weighing=async_cfg["use_sample_weighing"],
    )

    history = AsyncHistory()
    param_lock = Lock()
    current_params = ndarrays_to_parameters(global_params)

    total_train_time = async_cfg["total_train_time"]
    waiting_interval = async_cfg["waiting_interval"]
    max_workers = async_cfg["max_workers"]

    initial_loss, initial_acc = evaluate_global_model(global_model, global_params, global_test_loader, device)
    
    if wandb_enabled:
        wandb.log({"async/loss": initial_loss, "async/accuracy": initial_acc, "async/updates": 0, "async/elapsed_time": 0.0}, step=0)

    start_time = time.time()
    end_time = start_time + total_train_time
    update_count = 0

    def train_client(client_idx: int) -> tuple:
        from flwr.common import FitIns
        client = clients[client_idx]
        with param_lock:
            params = current_params
        config = {"start_timestamp": time.time()}
        return client_idx, client.fit(FitIns(parameters=params, config=config))

    def aggregate_result(client_idx: int, fit_res):
        nonlocal current_params, update_count
        t_diff = time.time() - fit_res.metrics.get("start_timestamp", time.time())
        with param_lock:
            current_params = async_strategy.average(current_params, fit_res.parameters, t_diff, fit_res.num_examples)
            update_count += 1
        return t_diff

    executor = ThreadPoolExecutor(max_workers=max_workers)
    active_futures, client_queue = set(), list(range(num_clients))
    eval_counter, last_eval_time = 0, start_time

    for _ in range(min(max_workers, num_clients)):
        if client_queue:
            client_idx = client_queue.pop(0)
            active_futures.add((executor.submit(train_client, client_idx), client_idx))

    while time.time() < end_time and (active_futures or client_queue):
        completed = [(f, c) for f, c in list(active_futures) if f.done()]
        for future, client_idx in completed:
            active_futures.discard((future, client_idx))
            try:
                _, fit_res = future.result()
                t_diff = aggregate_result(client_idx, fit_res)
                print(f"[t={time.time() - start_time:.1f}s] Client {client_idx} completed (loss: {fit_res.metrics.get('loss', 0):.4f})")
            except Exception as e:
                pass
            
            if time.time() < end_time:
                active_futures.add((executor.submit(train_client, client_idx), client_idx))

        if time.time() - last_eval_time >= waiting_interval:
            eval_counter += 1
            with param_lock:
                eval_params = parameters_to_ndarrays(current_params)
            loss, acc = evaluate_global_model(global_model, eval_params, global_test_loader, device)
            print(f"\\n[t={time.time() - start_time:.1f}s] Evaluation {eval_counter}: Loss: {loss:.4f}, Accuracy: {acc:.4f}\\n")
            
            if wandb_enabled:
                wandb.log({"async/loss": loss, "async/accuracy": acc, "async/updates": update_count, "async/elapsed_time": time.time() - start_time}, step=eval_counter)
            last_eval_time = time.time()
        time.sleep(0.1)

    executor.shutdown(wait=True)
    
    with param_lock:
        final_params = parameters_to_ndarrays(current_params)
    final_loss, final_acc = evaluate_global_model(global_model, final_params, global_test_loader, device)

    return {
        "final_loss": final_loss,
        "final_accuracy": final_acc,
        "total_updates": update_count,
        "elapsed_time": time.time() - start_time,
    }
