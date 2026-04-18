import collections
import collections.abc
collections.Sequence = collections.abc.Sequence

import time
import json
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

# Used for Context -> Move to Utils when cleaning
def calculate_cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0: return 1.0 
    sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    return 1.0 - sim

# Used for LoRA -> Move to Utils when cleaning
def split_arrays(flat_arrays: List[np.ndarray], base_idx: List[int], lora_idx: List[int]):
    """Slices a flat list of network weights into Base weights and LoRA weights."""
    return [flat_arrays[i] for i in base_idx], [flat_arrays[i] for i in lora_idx]

def combine_arrays(base_arrays: List[np.ndarray], lora_arrays: List[np.ndarray], base_idx: List[int], lora_idx: List[int], total_len: int):
    """Zips Base and LoRA weights back into a single flat list for the client."""
    combined = [None] * total_len
    for i, val in zip(base_idx, base_arrays): combined[i] = val
    for i, val in zip(lora_idx, lora_arrays): combined[i] = val
    return combined

@ray.remote
class AsyncRayClientActor:
    def __init__(self, client_idx, client_obj):
        self.client_idx = client_idx
        self.client = client_obj
        
    def fit(self, params, start_timestamp, current_phase):
        from flwr.common import FitIns
        config = {
            "start_timestamp": start_timestamp,
            "current_phase": current_phase
        }
        fit_res = self.client.fit(FitIns(parameters=params, config=config))
        return self.client_idx, fit_res, current_phase

def get_async_config(cfg: DictConfig) -> Dict[str, Any]:
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

def evaluate_global_model(model: torch.nn.Module, params: List[np.ndarray], test_loaders: List[DataLoader], device: torch.device) -> tuple[float, float]:
    state_dict = model.state_dict()
    for key, param in zip(state_dict.keys(), params):
        state_dict[key] = torch.tensor(param).to(device)
    model.load_state_dict(state_dict)

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_phases_loss = 0.0
    total_correct = 0
    total_total = 0
    metrics_dict = {}

    for phase_idx, phase_loader in enumerate(test_loaders):
        phase_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in phase_loader:
                if isinstance(batch, dict):
                    images, labels = batch.get("img", batch.get("x")).to(device), batch.get("label", batch.get("y")).to(device)
                elif isinstance(batch, (tuple, list)):
                    images, labels = batch[0].to(device), batch[1].to(device)
                else:
                    continue

                outputs = model(images)
                loss = criterion(outputs, labels)
                phase_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        total_total += total
        total_correct += correct
        phase_accuracy = correct / max(total, 1)
        total_phases_loss += phase_loss / max(total, 1)
        metrics_dict[f"phase_{phase_idx}_accuracy"] = phase_accuracy

    total_loss = total_phases_loss / len(test_loaders)
    total_accuracy = total_correct / max(total_total, 1)
    metrics_dict[f"accuracy"] = total_accuracy
    return total_loss, metrics_dict

def run_async_simulation(cfg, async_cfg, model_fn, train_loaders, test_loaders, global_test_loaders, device, wandb_enabled):
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
    global_keys = list(global_model.state_dict().keys())
    global_params = [val.cpu().numpy() for _, val in global_model.state_dict().items()]

    # Identify which tensors are Base vs LoRA
    base_indices = [i for i, k in enumerate(global_keys) if "lora" not in k]
    lora_indices = [i for i, k in enumerate(global_keys) if "lora" in k]
    use_lora = len(lora_indices) > 0  # Automatically detects if factory used PEFT
    total_param_len = len(global_keys)

    # --- 2. INITIALIZE STATE TRACKERS ---
    global_arrays = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    init_base, init_lora = split_arrays(global_arrays, base_indices, lora_indices)

    global_base_params = ndarrays_to_parameters(init_base)
    context_adapters = {0: ndarrays_to_parameters(init_lora)} if use_lora else {}
    
    # Helper to spawn independent strategy aggregators for Base and each Context
    def create_strategy(samples, custom_alpha = None):
        alpha = custom_alpha if custom_alpha is not None else async_cfg["fedasync_mixing_alpha"]
        return AsynchronousStrategy(
            total_samples=samples,
            staleness_alpha=async_cfg["staleness_alpha"],
            fedasync_mixing_alpha=alpha,
            fedasync_a=async_cfg["fedasync_a"],
            num_clients=num_clients,
            async_aggregation_strategy=async_cfg["aggregation_strategy"],
            use_staleness=async_cfg["use_staleness"],
            use_sample_weighing=async_cfg["use_sample_weighing"]
        )

    param_lock = Lock()

    # Calculate Phase Timings
    total_train_time = async_cfg["total_train_time"]
    cl_enabled = cfg.get("cl", {}).get("enabled", False)
    num_phases = cfg.get("cl", {}).get("num_experiences", 1) if cl_enabled else 1
    phase_duration = total_train_time / num_phases

    phase_total_samples = [
        sum(len(loaders[p].dataset) for loaders in train_loaders)
        for p in range(num_phases)
    ]

    base_strategy = create_strategy(phase_total_samples[0])
    context_strategies = {0: create_strategy(phase_total_samples[0])} if use_lora else {}
    total_samples = sum(len(loaders[0].dataset) for loaders in train_loaders)

#    async_strategy = AsynchronousStrategy(
#        total_samples=phase_total_samples[0],
#        staleness_alpha=async_cfg["staleness_alpha"],
#        fedasync_mixing_alpha=async_cfg["fedasync_mixing_alpha"],
#        fedasync_a=async_cfg["fedasync_a"],
#        num_clients=num_clients,
#        async_aggregation_strategy=async_cfg["aggregation_strategy"],
#        use_staleness=async_cfg["use_staleness"],
#        use_sample_weighing=async_cfg["use_sample_weighing"],
#    )

    # Variables for Prototypes
    server_context_prototypes = []
    context_distance_threshold = cfg.get("context", {}).get("threshold", 0.15)
    context_assignments = {}

    history = AsyncHistory()
    param_lock = Lock()

    total_train_time = async_cfg["total_train_time"]
    waiting_interval = async_cfg["waiting_interval"]

    # Calculate Initial Metrics
    phase_max_accs = [0.0] * num_phases
    seen_phases = set()
    current_global_phase = 0
    initial_loss, initial_metrics = evaluate_global_model(global_model, global_params, global_test_loaders, device)
    initial_phase_acc = [initial_metrics[f"phase_{i}_accuracy"] for i in range(num_phases)] 
    
    if wandb_enabled:
        log_dict = {
            "async/loss": initial_loss,
            "async/updates": 0,
            "async/elapsed_time": 0.0
        }
        for k, v in initial_metrics.items():
            log_dict[f"async/{k}"] = v
        wandb.log(log_dict, step=0)

    start_time = time.time()
    end_time = start_time + total_train_time
    update_count = 0

    def aggregate_result(client_idx: int, fit_res, phase_idx: int):
        nonlocal current_params, update_count, server_context_prototypes
        t_diff = time.time() - fit_res.metrics.get("start_timestamp", time.time())
        proto_str = fit_res.metrics.get("prototype", None)

        # Context Switching
        if proto_str is not None:
            incoming_proto = np.array(json.loads(proto_str))
            with param_lock:
                if len(server_context_prototypes) == 0:
                    server_context_prototypes.append(incoming_proto)
                    assigned_context = 0
                    print(f"  [Context Bank] Vehicle {client_idx} established Initial Context 0.")
                else:
                    distances = [calculate_cosine_distance(incoming_proto, p) for p in server_context_prototypes]
                    min_dist = min(distances)
                    closest_idx = distances.index(min_dist)
                    if min_dist < context_distance_threshold:
                        server_context_prototypes[closest_idx] = (0.9 * server_context_prototypes[closest_idx]) + (0.1 * incoming_proto)
                        assigned_context = closest_idx
                        print(f"  [Context Bank] Vehicle {client_idx} assigned Context {assigned_context}")
                    else:
                        server_context_prototypes.append(incoming_proto)
                        assigned_context = len(server_context_prototypes) - 1
                        print(f"  [Context Bank] Vehicle {client_idx} generated NEW Context {assigned_context} (Cos Dist: {min_dist:.3f})")
                context_assignments[client_idx] = assigned_context

        with param_lock:
            # 1. Unpack and slice incoming weights
            incoming_arrays = parameters_to_ndarrays(fit_res.parameters)
            inc_base, inc_lora = split_arrays(incoming_arrays, base_indices, lora_indices)

            # 2. Update the Global Base (Universal Knowledge)
            old_base = parameters_to_ndarrays(global_base_params)
            base_strategy.total_samples = phase_total_samples[phase_idx]
            global_base_params = base_strategy.average(
                global_base_params,
                ndarrays_to_parameters(inc_base),
                t_diff, fit_res.num_examples
            )
            new_base = parameters_to_ndarrays(global_base_params)
            base_shift = calculate_weight_shift(old_base, new_base)

            # 3. Update the Context Adapter (Domain Specialization)
            lora_shift = 0.0
            if use_lora:
                if assigned_context not in context_adapters:
                    # Initialize brand new adapter and strategy for a new context
                    context_adapters[assigned_context] = ndarrays_to_parameters(inc_lora)
                    context_strategies[assigned_context] = create_strategy(phase_total_samples[phase_idx])
                    lora_shift = "INITIALIZED"
                else:
                    # --- TRACK LORA SHIFT ---
                    old_lora = parameters_to_ndarrays(context_adapters[assigned_context])
                    context_strategies[assigned_context].total_samples = phase_total_samples[phase_idx]
                    context_adapters[assigned_context] = context_strategies[assigned_context].average(
                        context_adapters[assigned_context], ndarrays_to_parameters(inc_lora), t_diff, fit_res.num_examples
                    )
                    new_lora = parameters_to_ndarrays(context_adapters[assigned_context])
                    lora_shift = f"{calculate_weight_shift(old_lora, new_lora):.4f}"
                

                print(f"  [Weight Shift] Vehicle {client_idx} | Base: {base_shift:.4f} | LoRA {assigned_context}: {lora_shift}")

            update_count += 1
            
        return t_diff

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    print(f"\nDeploying {num_clients} stateful vehicle actors to the Ray cluster...")
    ray_actors = {}
    
    for i in range(num_clients):
        actor = AsyncRayClientActor.options(
            num_cpus=cfg.client.num_cpus,
            num_gpus=cfg.client.num_gpus
        ).remote(client_idx=i, client_obj=clients[i])
        ray_actors[i] = actor

    active_tasks = {}
    eval_counter, last_eval_time = 0, start_time

    # Command ALL actors to begin training
    for client_idx, actor in ray_actors.items():
        with param_lock:
            # Safely zip the initial weights for Context 0
            if use_lora:
                combined = combine_arrays(
                    parameters_to_ndarrays(global_base_params),
                    parameters_to_ndarrays(context_adapters[0]),
                    base_indices, lora_indices, total_param_len
                )
                params = ndarrays_to_parameters(combined)
            else:
                params = global_base_params
        task = actor.fit.remote(params, time.time(), current_phase=0)
        active_tasks[task] = client_idx

    while time.time() < end_time and active_tasks:
        ready_tasks, _ = ray.wait(list(active_tasks.keys()), num_returns=1, timeout=0.1) 
        
        for task in ready_tasks:
            client_idx = active_tasks.pop(task)
            try:
                returned_client_idx, fit_res, returned_phase = ray.get(task)
                t_diff = aggregate_result(returned_client_idx, fit_res, returned_phase)
                print(f"[Phase: {returned_phase}] [t={time.time() - start_time:.1f}s] Vehicle {client_idx} completed (loss: {fit_res.metrics.get('loss', 0):.4f})")
            except Exception as e:
                print(f"[Error] Vehicle {client_idx} failed: {e}")
            
            if time.time() < end_time:
                with param_lock:
                    assigned_context = context_assignments.get(client_idx, 0)
                    
                    if use_lora:
                        assigned_adapter = context_adapters.get(assigned_context, context_adapters[0])
                        combined = combine_arrays(
                            parameters_to_ndarrays(global_base_params),
                            parameters_to_ndarrays(assigned_adapter),
                            base_indices, lora_indices, total_param_len
                        )
                        params = ndarrays_to_parameters(combined)
                    else:
                        params = global_base_params
                
                # Determine current phase based on wall-clock time
                elapsed = time.time() - start_time
                current_phase = min(int(elapsed / phase_duration), num_phases - 1)
                if current_phase > current_global_phase:
                    print(f"\n{'='*50}")
                    print(f"[Server] SHIFTING PHASE: Transitioning to Phase {current_phase} at t={elapsed:.1f}s")
                    print(f"\n{'='*50}")                
                    current_global_phase = current_phase
                new_task = ray_actors[client_idx].fit.remote(params, time.time(), current_phase)
                active_tasks[new_task] = client_idx
        # Evaluation Block
        if time.time() - last_eval_time >= waiting_interval:
            eval_counter += 1
            with param_lock:
                if use_lora:
                    latest_ctx = max(context_adapters.keys())
                    combined_eval = combine_arrays(
                        parameters_to_ndarrays(global_base_params),
                        parameters_to_ndarrays(context_adapters[latest_ctx]),
                        base_indices, lora_indices, total_param_len
                    )
                    eval_params = combined_eval
                else:
                    eval_params = parameters_to_ndarrays(global_base_params)
            loss, metrics_dict = evaluate_global_model(global_model, eval_params, global_test_loaders, device)

            # CL Math for Metrics
            phase_accuracies = [metrics_dict[f"phase_{i}_accuracy"] for i in range(num_phases)]
            seen_phases.add(current_global_phase)
            phase_max_accs[current_global_phase] = max(phase_max_accs[current_global_phase], phase_accuracies[current_global_phase])
            bwt, fwt = 0.0, 0.0
            if current_global_phase > 0:
                bwt = sum(phase_accuracies[p] - phase_max_accs[p] for p in range(current_global_phase)) / current_global_phase
            if current_global_phase < num_phases - 1:
                fwt = sum(phase_accuracies[p] - initial_phase_acc[p] for p in range(current_global_phase + 1, num_phases)) / (num_phases - current_global_phase -1)
            avg_seen_acc = sum(phase_accuracies[p] for p in seen_phases) / len(seen_phases)

            # Creating Log Dict
            metrics_dict["bwt"] = bwt
            metrics_dict["fwt"] = fwt
            metrics_dict["avg_seen_acc"] = avg_seen_acc

            acc = metrics_dict.pop("accuracy")
            print(f"\n[t={time.time() - start_time:.1f}s] Evaluation {eval_counter}: Loss: {loss:.4f}, Accuracy: {acc:.4f}, BWT: {bwt:.4f}, FWT: {fwt:.4f}\n")
            
            if wandb_enabled:
                log_dict = {
                    "async/loss": loss,
                    "async/accuracy": acc,
                    "async/updates": update_count,
                    "async/elapsed_time": time.time() - start_time
                }
                if cl_enabled:
                    for k, v in metrics_dict.items():
                        if k != "accuracy": log_dict[f"async/{k}"] = v
                wandb.log(log_dict, step=eval_counter)
            last_eval_time = time.time()

    for actor in ray_actors.values():
        ray.kill(actor)
    ray.shutdown()    

    with param_lock:
        if use_lora:
            latest_ctx = max(context_adapters.keys())
            final_params = combine_arrays(
                parameters_to_ndarrays(global_base_params),
                parameters_to_ndarrays(context_adapters[latest_ctx]),
                base_indices, lora_indices, total_param_len
            )
        else:
            final_params = parameters_to_ndarrays(global_base_params)
    final_loss, final_metrics = evaluate_global_model(global_model, final_params, global_test_loaders, device)

    return {
        "final_loss": final_loss,
        "final_accuracy": final_metrics.get("accuracy"),
        "total_updates": update_count,
        "elapsed_time": time.time() - start_time,
    }
