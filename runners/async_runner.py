# Patch for old torch versions to be able to work with modern data transforms
import collections
import collections.abc

collections.Sequence = collections.abc.Sequence

import time
from concurrent.futures import ThreadPoolExecutor
import ray
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

# @ray.remote
# def execute_ray_client(client_idx: int, client, params, start_timestamp) -> tuple:
#    from flwr.common import FitIns
#    config = {"start_timestamp": start_timestamp}
#    return client_idx, client.fit(FitIns(parameters=params, config=config))


@ray.remote
class AsyncRayClientActor:
    def __init__(self, client_idx, client_obj):
        self.client_idx = client_idx
        self.client = client_obj

    def fit(self, params, start_timestamp):
        from flwr.common import FitIns

        config = {"start_timestamp": start_timestamp}
        # The heavy PyTorch DataLoader stays in memory; only weights are processed
        fit_res = self.client.fit(FitIns(parameters=params, config=config))
        return self.client_idx, fit_res


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


def evaluate_global_model(
    model: torch.nn.Module,
    params: List[np.ndarray],
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
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
                images, labels = batch.get("img", batch.get("x")).to(device), batch.get(
                    "label", batch.get("y")
                ).to(device)
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


def run_async_simulation(
    cfg,
    async_cfg,
    model_fn,
    train_loaders,
    test_loaders,
    global_test_loader,
    device,
    wandb_enabled,
):
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

    initial_loss, initial_acc = evaluate_global_model(
        global_model, global_params, global_test_loader, device
    )

    if wandb_enabled:
        wandb.log(
            {
                "async/loss": initial_loss,
                "async/accuracy": initial_acc,
                "async/updates": 0,
                "async/elapsed_time": 0.0,
            },
            step=0,
        )

    start_time = time.time()
    end_time = start_time + total_train_time
    update_count = 0

    #    def train_client(client_idx: int) -> tuple:
    #        from flwr.common import FitIns
    #        client = clients[client_idx]
    #        with param_lock:
    #            params = current_params
    #        config = {"start_timestamp": time.time()}
    #        return client_idx, client.fit(FitIns(parameters=params, config=config))

    def aggregate_result(client_idx: int, fit_res):
        nonlocal current_params, update_count
        t_diff = time.time() - fit_res.metrics.get("start_timestamp", time.time())
        with param_lock:
            current_params = async_strategy.average(
                current_params, fit_res.parameters, t_diff, fit_res.num_examples
            )
            update_count += 1
        return t_diff

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    active_tasks = {}
    client_queue = list(range(num_clients))
    eval_counter, last_eval_time = 0, start_time

    # Spawn Initial Batch of Clients
    print(f"\nDeploying {num_clients} stateful vehicle actors to the Ray cluster...")
    ray_actors = {}

    # 1. Boot up the permanent Actors applying your strict Blueprint
    for i in range(num_clients):
        actor = AsyncRayClientActor.options(
            num_cpus=cfg.client.num_cpus, num_gpus=cfg.client.num_gpus
        ).remote(client_idx=i, client_obj=clients[i])
        ray_actors[i] = actor

    active_tasks = {}
    eval_counter, last_eval_time = 0, start_time

    # 2. Command ALL actors to begin training immediately
    # Ray's internal scheduler naturally bottlenecks them based on main.py
    for client_idx, actor in ray_actors.items():
        with param_lock:
            params = current_params
        task = actor.fit.remote(params, time.time())
        active_tasks[task] = client_idx

    # 3. Continuous Execution Loop
    while time.time() < end_time and active_tasks:
        ready_tasks, _ = ray.wait(list(active_tasks.keys()), num_returns=1, timeout=0.1)

        for task in ready_tasks:
            client_idx = active_tasks.pop(task)
            try:
                returned_client_idx, fit_res = ray.get(task)
                t_diff = aggregate_result(returned_client_idx, fit_res)
                print(
                    f"[t={time.time() - start_time:.1f}s] Vehicle {client_idx} completed (loss: {fit_res.metrics.get('loss', 0):.4f})"
                )
            except Exception as e:
                print(f"[Error] Vehicle {client_idx} failed: {e}")

            # RE-SUBMIT IMMEDIATELY (passing only lightweight params)
            if time.time() < end_time:
                with param_lock:
                    params = current_params
                new_task = ray_actors[client_idx].fit.remote(params, time.time())
                active_tasks[new_task] = client_idx

        # Evaluation Block
        if time.time() - last_eval_time >= waiting_interval:
            eval_counter += 1
            with param_lock:
                eval_params = parameters_to_ndarrays(current_params)
            loss, acc = evaluate_global_model(
                global_model, eval_params, global_test_loader, device
            )
            print(
                f"\n[t={time.time() - start_time:.1f}s] Evaluation {eval_counter}: Loss: {loss:.4f}, Accuracy: {acc:.4f}\n"
            )

            if wandb_enabled:
                wandb.log(
                    {
                        "async/loss": loss,
                        "async/accuracy": acc,
                        "async/updates": update_count,
                        "async/elapsed_time": time.time() - start_time,
                    },
                    step=eval_counter,
                )
            last_eval_time = time.time()

    # Clean up actors
    for actor in ray_actors.values():
        ray.kill(actor)
    ray.shutdown()

    with param_lock:
        final_params = parameters_to_ndarrays(current_params)
    final_loss, final_acc = evaluate_global_model(
        global_model, final_params, global_test_loader, device
    )

    return {
        "final_loss": final_loss,
        "final_accuracy": final_acc,
        "total_updates": update_count,
        "elapsed_time": time.time() - start_time,
    }
