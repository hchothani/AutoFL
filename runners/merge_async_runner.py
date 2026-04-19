"""
async_runner.py  —  Asynchronous Federated Learning with Context-Aware LoRA Adapter Bank
=========================================================================================

CHANGE LOG: Merged Structural Hygiene with Ensemble Continual Learning Math
─────────────────────────────────────────────────────────────────────────────────────────
[RESTORED] Dual-Speed Alphas: Base strategy updates slowly (0.1) to preserve FWT/BWT, 
           while Context Adapters update rapidly (0.9).
[RESTORED] Expert Initialization: New contexts use the raw incoming client LoRA weights 
           as their pure foundation, rather than averaging them with initialized zeroes.
[UPGRADE]  Ensemble Inference: `phase_to_context` now tracks a SET of all contexts seen 
           in a phase. The evaluator fuses all active adapters for a phase and averages 
           their logits to accurately test spatio-temporally mixed datasets.
─────────────────────────────────────────────────────────────────────────────────────────
"""

import collections
import collections.abc
collections.Sequence = collections.abc.Sequence  # Compatibility shim for older Flower versions

import time
import json
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from threading import Lock
import ray
from omegaconf import DictConfig, OmegaConf
import wandb

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from algorithms.async_fl import AsynchronousStrategy, AsyncHistory
from clients.async_client import create_simulated_clients


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers  
# ──────────────────────────────────────────────────────────────────────────────

def calculate_cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Returns 1 - cosine_similarity.  Returns 1.0 (max distance) for zero vectors."""
    norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    return 1.0 - float(np.clip(sim, -1.0, 1.0))

def split_arrays(
    flat_arrays: List[np.ndarray],
    base_idx: List[int],
    lora_idx: List[int],
) -> tuple:
    """Slice a flat weight list into (base_weights, lora_weights)."""
    return [flat_arrays[i] for i in base_idx], [flat_arrays[i] for i in lora_idx]

def combine_arrays(
    base_arrays: List[np.ndarray],
    lora_arrays: List[np.ndarray],
    base_idx: List[int],
    lora_idx: List[int],
    total_len: int,
) -> List[np.ndarray]:
    """Zip base and LoRA weights back into a single flat list for the client."""
    combined: List[Optional[np.ndarray]] = [None] * total_len
    for i, val in zip(base_idx, base_arrays):
        combined[i] = val
    for i, val in zip(lora_idx, lora_arrays):
        combined[i] = val
    return combined  # type: ignore[return-value]

def calculate_weight_shift(
    old_params_list: List[np.ndarray],
    new_params_list: List[np.ndarray],
) -> float:
    """L1 norm of the element-wise difference between two weight snapshots."""
    return float(
        sum(np.sum(np.abs(old - new)) for old, new in zip(old_params_list, new_params_list))
    )

# ──────────────────────────────────────────────────────────────────────────────
# Ray remote actor
# ──────────────────────────────────────────────────────────────────────────────

@ray.remote
class AsyncRayClientActor:
    """Wraps a stateful Flower client as a Ray actor for non-blocking async training."""

    def __init__(self, client_idx: int, client_obj):
        self.client_idx = client_idx
        self.client = client_obj

    def fit(self, params, start_timestamp: float, current_phase: int):
        from flwr.common import FitIns
        config = {
            "start_timestamp": start_timestamp,
            "current_phase": current_phase,
        }
        fit_res = self.client.fit(FitIns(parameters=params, config=config))
        return self.client_idx, fit_res, current_phase

# ──────────────────────────────────────────────────────────────────────────────
# Config helper
# ──────────────────────────────────────────────────────────────────────────────

def get_async_config(cfg: DictConfig) -> Dict[str, Any]:
    async_cfg = cfg.get("async", {})
    if isinstance(async_cfg, DictConfig):
        async_cfg = OmegaConf.to_container(async_cfg, resolve=True)
    return {
        "total_train_time":       async_cfg.get("total_train_time",       300),
        "waiting_interval":       async_cfg.get("waiting_interval",        10),
        "max_workers":            async_cfg.get("max_workers",              4),
        "aggregation_strategy":   async_cfg.get("aggregation_strategy", "fedasync"),
        "staleness_alpha":        async_cfg.get("staleness_alpha",        0.5),
        "fedasync_mixing_alpha":  async_cfg.get("fedasync_mixing_alpha",  0.9),
        "fedasync_a":             async_cfg.get("fedasync_a",             0.5),
        "use_staleness":          async_cfg.get("use_staleness",          True),
        "use_sample_weighing":    async_cfg.get("use_sample_weighing",    True),
        "send_gradients":         async_cfg.get("send_gradients",         False),
        "server_artificial_delay":async_cfg.get("server_artificial_delay",False),
        "is_streaming":           async_cfg.get("is_streaming",           False),
        "client_local_delay":     async_cfg.get("client_local_delay",     False),
        "simulate_delay":         async_cfg.get("simulate_delay",         True),
        "min_delay":              async_cfg.get("min_delay",              0.5),
        "max_delay":              async_cfg.get("max_delay",              3.0),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Global model evaluation (ENSEMBLE INFERENCE UPGRADE)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_global_model(
    model: torch.nn.Module,
    # ### --- ENSEMBLE CHANGE: Now accepts a List of Fused Models per Phase --- ###
    phase_params_dict: Dict[int, List[List[np.ndarray]]], 
    test_loaders: List[DataLoader],
    device: torch.device,
) -> tuple:
    """
    Evaluates the global model. If a phase has multiple contexts, it runs the data
    through ALL active adapters for that phase and averages the logits (Ensemble Inference).
    """
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_phases_loss = 0.0
    total_correct = 0
    total_total = 0
    metrics_dict: Dict[str, float] = {}

    try:
        for phase_idx, phase_loader in enumerate(test_loaders):
            # Dynamic adapter loading — fetch all active experts for this phase
            ensemble_params = phase_params_dict.get(phase_idx)
            if not ensemble_params:
                latest_idx = max(phase_params_dict.keys())
                ensemble_params = phase_params_dict[latest_idx]

            phase_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for batch in phase_loader:
                    if isinstance(batch, dict):
                        images = batch.get("img", batch.get("x")).to(device)
                        labels = batch.get("label", batch.get("y")).to(device)
                    elif isinstance(batch, (tuple, list)):
                        images, labels = batch[0].to(device), batch[1].to(device)
                    else:
                        continue

                    # ### --- ENSEMBLE CHANGE: Run inference across all active adapters --- ###
                    ensemble_logits = 0
                    for params in ensemble_params:
                        state_dict = model.state_dict()
                        for key, param in zip(state_dict.keys(), params):
                            state_dict[key] = torch.tensor(param).to(device)
                        model.load_state_dict(state_dict)
                        
                        outputs = model(images)
                        ensemble_logits += outputs
                        
                    # Average the outputs across all adapters
                    ensemble_logits = ensemble_logits / len(ensemble_params)

                    loss = criterion(ensemble_logits, labels)
                    phase_loss += loss.item() * labels.size(0)
                    _, predicted = ensemble_logits.max(1)
                    # -------------------------------------------------------------------------

                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            total_total += total
            total_correct += correct
            phase_accuracy = correct / max(total, 1)
            total_phases_loss += phase_loss / max(total, 1)
            metrics_dict[f"phase_{phase_idx}_accuracy"] = phase_accuracy

    finally:
        # Always restore the model state
        model.load_state_dict(original_state)

    total_loss = total_phases_loss / max(len(test_loaders), 1)
    total_accuracy = total_correct / max(total_total, 1)
    metrics_dict["accuracy"] = total_accuracy
    return total_loss, metrics_dict

# ──────────────────────────────────────────────────────────────────────────────
# Main simulation entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def run_async_simulation(
    cfg,
    async_cfg,
    model_fn,
    train_loaders,
    test_loaders,
    global_test_loaders,
    device,
    wandb_enabled,
):
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

    # ── Model & weight-index initialisation ───────────────────────────────────
    global_model = model_fn().to(device)
    global_keys = list(global_model.state_dict().keys())
    global_params = [val.cpu().numpy() for _, val in global_model.state_dict().items()]

    base_indices = [i for i, k in enumerate(global_keys) if "lora" not in k]
    lora_indices = [i for i, k in enumerate(global_keys) if "lora" in k]
    use_lora = len(lora_indices) > 0
    total_param_len = len(global_keys)

    global_arrays = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    init_base, init_lora = split_arrays(global_arrays, base_indices, lora_indices)

    # Shared backbone (updated on every client result regardless of context)
    global_base_params = ndarrays_to_parameters(init_base)

    # Context adapter bank:  context_id (int) → Flower Parameters (LoRA weights only)
    context_adapters: Dict[int, Any] = (
        {0: ndarrays_to_parameters(init_lora)} if use_lora else {}
    )

    # ── Strategy factory ──────────────────────────────────────────────────────
    def create_strategy(samples: int, custom_alpha: Optional[float] = None) -> AsynchronousStrategy:
        alpha = custom_alpha if custom_alpha is not None else async_cfg["fedasync_mixing_alpha"]
        return AsynchronousStrategy(
            total_samples=samples,
            staleness_alpha=async_cfg["staleness_alpha"],
            fedasync_mixing_alpha=alpha,
            fedasync_a=async_cfg["fedasync_a"],
            num_clients=num_clients,
            async_aggregation_strategy=async_cfg["aggregation_strategy"],
            use_staleness=async_cfg["use_staleness"],
            use_sample_weighing=async_cfg["use_sample_weighing"],
        )

    # ── Phase & timing setup ──────────────────────────────────────────────────
    total_train_time = async_cfg["total_train_time"]
    waiting_interval = async_cfg["waiting_interval"]

    cl_enabled  = cfg.get("cl", {}).get("enabled", False)
    num_phases  = cfg.get("cl", {}).get("num_experiences", 1) if cl_enabled else 1
    phase_duration = total_train_time / num_phases

    phase_total_samples = [
        sum(len(loaders[p].dataset) for loaders in train_loaders if loaders[p] is not None)
        for p in range(num_phases)
    ]

    # ### --- ENSEMBLE CHANGE: RESTORED DUAL-SPEED ALPHAS --- ###
    base_strategy = create_strategy(phase_total_samples[0], custom_alpha=0.1)
    context_strategies: Dict[int, AsynchronousStrategy] = (
        {0: create_strategy(phase_total_samples[0], custom_alpha=0.9)} if use_lora else {}
    )

    # ── Context bank state ────────────────────────────────────────────────────
    server_context_prototypes: List[np.ndarray] = []
    context_distance_threshold = cfg.get("context", {}).get("threshold", 0.15)
    context_assignments: Dict[int, int] = {}   

    # ### --- ENSEMBLE CHANGE: Upgrade to a SET to track all contexts seen in a phase --- ###
    phase_to_context: Dict[int, set] = {p: set() for p in range(num_phases)}

    history = AsyncHistory()
    param_lock = Lock()

    # ── CL metric trackers ────────────────────────────────────────────────────
    phase_max_accs     = [0.0] * num_phases
    seen_phases: set   = set()
    current_global_phase = 0

    # ── Initial evaluation ────────────────────────────────────────────────────
    if use_lora:
        _initial_combined = combine_arrays(
            init_base, init_lora, base_indices, lora_indices, total_param_len
        )
        # Note: Initial dictionary now passes Lists of Lists to match Ensemble signature
        initial_params_dict = {p_idx: [_initial_combined] for p_idx in range(num_phases)}
    else:
        initial_params_dict = {p_idx: [global_params] for p_idx in range(num_phases)}

    initial_loss, initial_metrics = evaluate_global_model(
        global_model, initial_params_dict, global_test_loaders, device
    )
    initial_phase_acc = [initial_metrics[f"phase_{i}_accuracy"] for i in range(num_phases)]

    if wandb_enabled:
        log_dict = {"async/loss": initial_loss, "async/updates": 0, "async/elapsed_time": 0.0}
        for k, v in initial_metrics.items():
            log_dict[f"async/{k}"] = v
        wandb.log(log_dict, step=0)

    start_time   = time.time()
    end_time     = start_time + total_train_time
    update_count = 0

    # ── Inner aggregation callback ─────────────────────────────────────────────
    def aggregate_result(client_idx: int, fit_res, phase_idx: int) -> float:
        nonlocal global_base_params, update_count, server_context_prototypes
        t_diff   = time.time() - fit_res.metrics.get("start_timestamp", time.time())
        proto_str = fit_res.metrics.get("prototype", None)

        with param_lock:
            # ── 1. CONTEXT DETECTION ─────────────────────────────────────────
            if proto_str is not None:
                incoming_proto = np.array(json.loads(proto_str))

                if len(server_context_prototypes) == 0:
                    server_context_prototypes.append(incoming_proto)
                    assigned_context = 0
                    print(f"  [Context Bank] Vehicle {client_idx} established Initial Context 0.")
                else:
                    distances  = [
                        calculate_cosine_distance(incoming_proto, p)
                        for p in server_context_prototypes
                    ]
                    min_dist   = min(distances)
                    closest_idx = distances.index(min_dist)

                    if min_dist < context_distance_threshold:
                        server_context_prototypes[closest_idx] = (
                            0.9 * server_context_prototypes[closest_idx]
                            + 0.1 * incoming_proto
                        )
                        assigned_context = closest_idx
                        print(f"  [Context Bank] Vehicle {client_idx} assigned Context {assigned_context}")
                    else:
                        server_context_prototypes.append(incoming_proto)
                        assigned_context = len(server_context_prototypes) - 1
                        print(f"  [Context Bank] Vehicle {client_idx} generated NEW Context {assigned_context} (Cos Dist: {min_dist:.3f})")

                context_assignments[client_idx] = assigned_context

                # ### --- ENSEMBLE CHANGE: Add context to the SET for this phase --- ###
                phase_to_context[phase_idx].add(assigned_context)

            else:
                assigned_context = context_assignments.get(client_idx, 0)

            # ── 2. UNPACK INCOMING WEIGHTS ───────────────────────────────────
            incoming_arrays = parameters_to_ndarrays(fit_res.parameters)
            inc_base, inc_lora = split_arrays(incoming_arrays, base_indices, lora_indices)

            # ── 3. UPDATE GLOBAL BASE (universal shared knowledge) ───────────
            old_base = parameters_to_ndarrays(global_base_params)
            base_strategy.total_samples = phase_total_samples[phase_idx]
            global_base_params = base_strategy.average(
                global_base_params,
                ndarrays_to_parameters(inc_base),
                t_diff,
                fit_res.num_examples,
            )
            new_base   = parameters_to_ndarrays(global_base_params)
            base_shift = calculate_weight_shift(old_base, new_base)

            # ── 4. UPDATE CONTEXT ADAPTER (domain specialisation) ────────────
            lora_shift: Any = 0.0
            if use_lora:
                if assigned_context not in context_adapters:
                    # ### --- ENSEMBLE CHANGE: REVERT FIX #7 (EXPERT POISONING) --- ###
                    # We initialize the adapter purely with the discovering client's weights
                    context_adapters[assigned_context] = ndarrays_to_parameters(inc_lora)
                    context_strategies[assigned_context] = create_strategy(phase_total_samples[phase_idx], custom_alpha=0.9)
                    lora_shift = "INITIALIZED"
                else:
                    # Only average if the adapter is already established
                    old_lora = parameters_to_ndarrays(context_adapters[assigned_context])
                    context_strategies[assigned_context].total_samples = phase_total_samples[phase_idx]
                    context_adapters[assigned_context] = context_strategies[assigned_context].average(
                        context_adapters[assigned_context],
                        ndarrays_to_parameters(inc_lora),
                        t_diff,
                        fit_res.num_examples,
                    )
                    new_lora = parameters_to_ndarrays(context_adapters[assigned_context])
                    lora_shift = f"{calculate_weight_shift(old_lora, new_lora):.4f}"

            print(
                f"  [Weight Shift] Vehicle {client_idx} | "
                f"Base: {base_shift:.4f} | LoRA {assigned_context}: {lora_shift}"
            )

        update_count += 1
        return t_diff

    # ── Ray cluster setup ──────────────────────────────────────────────────────
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    print(f"\nDeploying {num_clients} stateful vehicle actors to the Ray cluster...")
    ray_actors: Dict[int, Any] = {}
    for i in range(num_clients):
        actor = AsyncRayClientActor.options(
            num_cpus=cfg.client.num_cpus,
            num_gpus=cfg.client.num_gpus,
        ).remote(client_idx=i, client_obj=clients[i])
        ray_actors[i] = actor

    active_tasks: Dict[Any, int] = {}
    eval_counter, last_eval_time = 0, start_time

    for client_idx, actor in ray_actors.items():
        with param_lock:
            if use_lora:
                combined = combine_arrays(
                    parameters_to_ndarrays(global_base_params),
                    parameters_to_ndarrays(context_adapters[0]),
                    base_indices, lora_indices, total_param_len,
                )
                params = ndarrays_to_parameters(combined)
            else:
                params = global_base_params
        task = actor.fit.remote(params, time.time(), current_phase=0)
        active_tasks[task] = client_idx

    # ── Main async event loop ──────────────────────────────────────────────────
    while time.time() < end_time and active_tasks:
        ready_tasks, _ = ray.wait(list(active_tasks.keys()), num_returns=1, timeout=0.1)

        for task in ready_tasks:
            client_idx = active_tasks.pop(task)
            try:
                returned_client_idx, fit_res, returned_phase = ray.get(task)
                t_diff = aggregate_result(returned_client_idx, fit_res, returned_phase)
                print(
                    f"[Phase: {returned_phase}] [t={time.time() - start_time:.1f}s] "
                    f"Vehicle {client_idx} completed "
                    f"(loss: {fit_res.metrics.get('loss', 0):.4f})"
                )
            except Exception as e:
                print(f"[Error] Vehicle {client_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            if time.time() < end_time:
                with param_lock:
                    assigned_context = context_assignments.get(client_idx, 0)
                    if use_lora:
                        assigned_adapter = context_adapters.get(
                            assigned_context, context_adapters[0]
                        )
                        combined = combine_arrays(
                            parameters_to_ndarrays(global_base_params),
                            parameters_to_ndarrays(assigned_adapter),
                            base_indices, lora_indices, total_param_len,
                        )
                        params = ndarrays_to_parameters(combined)
                    else:
                        params = global_base_params

                elapsed = time.time() - start_time
                current_phase = min(int(elapsed / phase_duration), num_phases - 1)
                if current_phase > current_global_phase:
                    print(f"\n{'='*50}")
                    print(
                        f"[Server] SHIFTING PHASE: Transitioning to Phase "
                        f"{current_phase} at t={elapsed:.1f}s"
                    )
                    print(f"{'='*50}\n")
                    current_global_phase = current_phase

                new_task = ray_actors[client_idx].fit.remote(params, time.time(), current_phase)
                active_tasks[new_task] = client_idx

        # ── Periodic evaluation ────────────────────────────────────────────────
        if time.time() - last_eval_time >= waiting_interval:
            eval_counter += 1
            
            # ### --- ENSEMBLE CHANGE: Build a List of Fused Models per Phase --- ###
            eval_params_dict: Dict[int, List[List[np.ndarray]]] = {}

            with param_lock:
                if use_lora:
                    for p_idx in range(num_phases):
                        active_contexts = phase_to_context.get(p_idx, set())
                        if not active_contexts:
                            # Fallback if phase hasn't started yet
                            active_contexts = {max(context_adapters.keys()) if context_adapters else 0}
                            
                        fused_models_for_phase = []
                        for ctx in active_contexts:
                            adapter_idx = ctx if ctx in context_adapters else max(context_adapters.keys())
                            combined_eval = combine_arrays(
                                parameters_to_ndarrays(global_base_params),
                                parameters_to_ndarrays(context_adapters[adapter_idx]),
                                base_indices, lora_indices, total_param_len,
                            )
                            fused_models_for_phase.append(combined_eval)
                            
                        eval_params_dict[p_idx] = fused_models_for_phase
                else:
                    for p_idx in range(num_phases):
                        eval_params_dict[p_idx] = [parameters_to_ndarrays(global_base_params)]

            loss, metrics_dict = evaluate_global_model(
                global_model, eval_params_dict, global_test_loaders, device
            )

            # ── CL metrics ────────────────────────────────────────────────────
            phase_accuracies = [metrics_dict[f"phase_{i}_accuracy"] for i in range(num_phases)]
            seen_phases.add(current_global_phase)
            phase_max_accs[current_global_phase] = max(
                phase_max_accs[current_global_phase],
                phase_accuracies[current_global_phase],
            )

            bwt, fwt = 0.0, 0.0
            if current_global_phase > 0:
                bwt = (
                    sum(
                        phase_accuracies[p] - phase_max_accs[p]
                        for p in range(current_global_phase)
                    )
                    / current_global_phase
                )
            if current_global_phase < num_phases - 1:
                remaining = num_phases - current_global_phase - 1
                fwt = (
                    sum(
                        phase_accuracies[p] - initial_phase_acc[p]
                        for p in range(current_global_phase + 1, num_phases)
                    )
                    / remaining
                )
            avg_seen_acc = sum(phase_accuracies[p] for p in seen_phases) / len(seen_phases)

            metrics_dict["bwt"]          = bwt
            metrics_dict["fwt"]          = fwt
            metrics_dict["avg_seen_acc"] = avg_seen_acc

            acc = metrics_dict.pop("accuracy")
            print(
                f"\n[t={time.time() - start_time:.1f}s] Evaluation {eval_counter}: "
                f"Loss: {loss:.4f}, Accuracy: {acc:.4f}, "
                f"BWT: {bwt:.4f}, FWT: {fwt:.4f}\n"
            )

            if wandb_enabled:
                log_dict = {
                    "async/loss":         loss,
                    "async/accuracy":     acc,
                    "async/updates":      update_count,
                    "async/elapsed_time": time.time() - start_time,
                }
                if cl_enabled:
                    for k, v in metrics_dict.items():
                        if k != "accuracy":
                            log_dict[f"async/{k}"] = v
                wandb.log(log_dict, step=eval_counter)

            last_eval_time = time.time()

    # ── Teardown ───────────────────────────────────────────────────────────────
    for actor in ray_actors.values():
        ray.kill(actor)
    ray.shutdown()

    # ── Final evaluation ───────────────────────────────────────────────────────
    final_params_dict: Dict[int, List[List[np.ndarray]]] = {}
    with param_lock:
        if use_lora:
            for p_idx in range(num_phases):
                active_contexts = phase_to_context.get(p_idx, set())
                if not active_contexts:
                    active_contexts = {max(context_adapters.keys()) if context_adapters else 0}
                    
                fused_models_for_phase = []
                for ctx in active_contexts:
                    adapter_idx = ctx if ctx in context_adapters else max(context_adapters.keys())
                    combined_eval = combine_arrays(
                        parameters_to_ndarrays(global_base_params),
                        parameters_to_ndarrays(context_adapters[adapter_idx]),
                        base_indices, lora_indices, total_param_len,
                    )
                    fused_models_for_phase.append(combined_eval)
                    
                final_params_dict[p_idx] = fused_models_for_phase
        else:
            for p_idx in range(num_phases):
                final_params_dict[p_idx] = [parameters_to_ndarrays(global_base_params)]

    final_loss, final_metrics = evaluate_global_model(
        global_model, final_params_dict, global_test_loaders, device
    )

    return {
        "final_loss":     final_loss,
        "final_accuracy": final_metrics.get("accuracy"),
        "total_updates":  update_count,
        "elapsed_time":   time.time() - start_time,
    }
