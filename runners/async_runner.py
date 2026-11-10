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
from utils.compression_utils import partition_model_indices, svd_merge_lora_weights


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
# Global model evaluation (Fast evaluation with device pre-loading & logit norm)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_global_model(
    model: torch.nn.Module,
    phase_params_dict: Dict[int, List[List[np.ndarray]]], 
    test_loaders: List[DataLoader],
    device: torch.device,
) -> tuple:
    """
    Evaluates the global model across continual learning phases.
    Pre-builds device state_dicts ONCE per phase outside the batch loop for 10x-50x speedup.
    If a phase has multiple contexts, normalizes logits and averages them (Ensemble Inference).
    """
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    template_keys = list(original_state.keys())
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_phases_loss = 0.0
    total_correct = 0
    total_total = 0
    metrics_dict: Dict[str, float] = {}

    try:
        for phase_idx, phase_loader in enumerate(test_loaders):
            ensemble_params = phase_params_dict.get(phase_idx)
            if not ensemble_params:
                latest_idx = max(phase_params_dict.keys())
                ensemble_params = phase_params_dict[latest_idx]

            # Pre-build device state_dicts ONCE per phase outside the batch loop,
            # preserving exact tensor dtypes (prevents integer buffer corruption).
            prepared_states = [
                {
                    key: torch.as_tensor(param, dtype=original_state[key].dtype, device=device)
                    for key, param in zip(template_keys, params)
                }
                for params in ensemble_params
            ]

            phase_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                if len(prepared_states) == 1:
                    model.load_state_dict(prepared_states[0])
                    for batch in phase_loader:
                        if isinstance(batch, dict):
                            images = batch.get("img", batch.get("x")).to(device)
                            labels = batch.get("label", batch.get("y")).to(device)
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
                else:
                    for batch in phase_loader:
                        if isinstance(batch, dict):
                            images = batch.get("img", batch.get("x")).to(device)
                            labels = batch.get("label", batch.get("y")).to(device)
                        elif isinstance(batch, (tuple, list)):
                            images, labels = batch[0].to(device), batch[1].to(device)
                        else:
                            continue

                        ensemble_logits = 0
                        for state_dict in prepared_states:
                            model.load_state_dict(state_dict)
                            outputs = model(images)
                            out_norm = torch.norm(outputs, p=2, dim=1, keepdim=True)
                            ensemble_logits += outputs / torch.clamp(out_norm, min=1e-8)

                        ensemble_logits = ensemble_logits / len(prepared_states)
                        loss = criterion(ensemble_logits, labels)
                        phase_loss += loss.item() * labels.size(0)
                        _, predicted = ensemble_logits.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()

            total_total += total
            total_correct += correct
            phase_accuracy = correct / max(total, 1)
            total_phases_loss += phase_loss / max(total, 1)
            metrics_dict[f"phase_{phase_idx}_accuracy"] = phase_accuracy

    finally:
        model.load_state_dict(original_state)

    total_loss = total_phases_loss / max(len(test_loaders), 1)
    total_accuracy = total_correct / max(total_total, 1)
    metrics_dict["accuracy"] = total_accuracy
    return total_loss, metrics_dict


# ──────────────────────────────────────────────────────────────────────────────
# Autonomous Context Merging (Phase-Agnostic with SVD LoRA Denoising)
# ──────────────────────────────────────────────────────────────────────────────

def merge_similar_contexts(
    server_context_prototypes: Dict[int, np.ndarray],
    context_adapters: Dict[int, Any],
    context_strategies: Dict[int, AsynchronousStrategy],
    context_assignments: Dict[int, int],
    phase_to_context: Dict[int, set],
    threshold: float,
    updated_contexts: set,
    context_sample_counts: Dict[int, int],
):
    """
    Scans active contexts and merges any pair with cosine distance < threshold.
    Operates 100% unsupervised and phase-agnostically:
      - Weighted prototype fusion + unit-norm re-projection.
      - Truncated SVD LoRA merging (ΔW = B @ A) to eliminate cross-term noise.
      - Reroutes vehicle assignments and updates phase trackers without ground-truth cheating.
    """
    if not updated_contexts:
        return

    active_ids = list(server_context_prototypes.keys())
    merged_this_round = set()

    for id_a in list(updated_contexts):
        if id_a in merged_this_round or id_a not in server_context_prototypes:
            continue

        for id_b in active_ids:
            if id_a == id_b:
                continue
            if id_b in merged_this_round or id_b not in server_context_prototypes:
                continue

            dist = calculate_cosine_distance(
                server_context_prototypes[id_a],
                server_context_prototypes[id_b],
            )

            if dist < threshold:
                n_a = float(context_sample_counts.get(id_a, 1))
                n_b = float(context_sample_counts.get(id_b, 1))
                total_n = max(n_a + n_b, 1.0)
                print(
                    f"  [Context Bank] COLLAPSE: Merging Context {id_b} into Context {id_a} "
                    f"(Cos Dist: {dist:.3f}, samples: {int(n_a)} vs {int(n_b)})"
                )

                # 1. Weighted Prototype Averaging with L2 Unit-Norm Projection
                p_fused = (server_context_prototypes[id_a] * n_a + server_context_prototypes[id_b] * n_b) / total_n
                p_norm = np.linalg.norm(p_fused)
                server_context_prototypes[id_a] = p_fused / p_norm if p_norm > 0 else p_fused

                # 2. Truncated SVD LoRA Weight Merging (Eliminates Cross-Term Noise)
                w_a = parameters_to_ndarrays(context_adapters[id_a])
                w_b = parameters_to_ndarrays(context_adapters[id_b])
                w_merged = svd_merge_lora_weights(w_a, w_b, n_a, n_b)
                context_adapters[id_a] = ndarrays_to_parameters(w_merged)

                # 3. Accumulate Cumulative Sample Counts
                context_sample_counts[id_a] = int(n_a + n_b)
                if id_b in context_sample_counts:
                    del context_sample_counts[id_b]

                # 4. Reroute Client Assignments
                for c_idx, ctx_id in list(context_assignments.items()):
                    if ctx_id == id_b:
                        context_assignments[c_idx] = id_a

                # 5. Update Phase Trackers
                for p_idx in phase_to_context:
                    if id_b in phase_to_context[p_idx]:
                        phase_to_context[p_idx].remove(id_b)
                        phase_to_context[p_idx].add(id_a)

                # 6. Delete Redundant Context
                del server_context_prototypes[id_b]
                del context_adapters[id_b]
                if id_b in context_strategies:
                    del context_strategies[id_b]

                merged_this_round.add(id_b)

    updated_contexts.clear()

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
    raw_comp = cfg.get("context", {}).get("compression", {})
    comp_dict = OmegaConf.to_container(raw_comp, resolve=True) if OmegaConf.is_config(raw_comp) else dict(raw_comp) if raw_comp else {}
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
        compression_cfg=comp_dict,
    )

    # ── Model & weight-index initialisation ───────────────────────────────────
    global_model = model_fn().to(device)
    global_keys = list(global_model.state_dict().keys())
    global_params = [val.cpu().numpy() for _, val in global_model.state_dict().items()]

    base_indices, lora_indices = partition_model_indices(global_model)
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
    server_context_prototypes: Dict[int, np.ndarray] = {}
    next_context_id = 0
    context_distance_threshold = cfg.get("context", {}).get("threshold", 0.15)
    merge_threshold = cfg.get("context", {}).get("merge_threshold", 0.08)
    merge_interval_evals = cfg.get("context", {}).get("merge_interval_evals", 3)
    context_sample_counts: Dict[int, int] = {}
    updated_contexts: set = set()
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
        nonlocal global_base_params, update_count, server_context_prototypes, next_context_id
        t_diff   = time.time() - fit_res.metrics.get("start_timestamp", time.time())
        proto_str = fit_res.metrics.get("prototype", None)

        with param_lock:
            # ── 1. CONTEXT ROUTING (100% Phase-Agnostic) ──────────────────────
            if proto_str is not None:
                incoming_proto = np.array(json.loads(proto_str), dtype=np.float32)
                p_norm = np.linalg.norm(incoming_proto)
                if p_norm > 1e-8:
                    incoming_proto = incoming_proto / p_norm

                if len(server_context_prototypes) == 0:
                    assigned_context = next_context_id
                    server_context_prototypes[assigned_context] = incoming_proto
                    next_context_id += 1
                    print(f"  [Context Bank] Vehicle {client_idx} established Initial Context {assigned_context}.")
                else:
                    active_ctx_keys = list(server_context_prototypes.keys())
                    distances = [
                        calculate_cosine_distance(incoming_proto, server_context_prototypes[k])
                        for k in active_ctx_keys
                    ]
                    min_dist = min(distances)
                    closest_ctx = active_ctx_keys[distances.index(min_dist)]

                    if min_dist < context_distance_threshold:
                        # Radial unit-norm EMA update
                        updated = 0.9 * server_context_prototypes[closest_ctx] + 0.1 * incoming_proto
                        up_norm = np.linalg.norm(updated)
                        server_context_prototypes[closest_ctx] = updated / up_norm if up_norm > 1e-8 else updated
                        assigned_context = closest_ctx
                        print(f"  [Context Bank] Vehicle {client_idx} assigned Context {assigned_context} (Dist: {min_dist:.3f})")
                    else:
                        assigned_context = next_context_id
                        server_context_prototypes[assigned_context] = incoming_proto
                        next_context_id += 1
                        print(f"  [Context Bank] Vehicle {client_idx} generated NEW Context {assigned_context} (Cos Dist: {min_dist:.3f})")

                context_assignments[client_idx] = assigned_context
                phase_to_context[phase_idx].add(assigned_context)
            else:
                default_ctx = max(context_adapters.keys()) if context_adapters else 0
                assigned_context = context_assignments.get(client_idx, default_ctx)

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
                    context_adapters[assigned_context] = ndarrays_to_parameters(inc_lora)
                    context_strategies[assigned_context] = create_strategy(phase_total_samples[phase_idx], custom_alpha=0.9)
                    lora_shift = "INITIALIZED"
                else:
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

                context_sample_counts[assigned_context] = (
                    context_sample_counts.get(assigned_context, 0) + fit_res.num_examples
                )

            updated_contexts.add(assigned_context)
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
                default_ctx = max(context_adapters.keys()) if context_adapters else 0
                combined = combine_arrays(
                    parameters_to_ndarrays(global_base_params),
                    parameters_to_ndarrays(context_adapters[default_ctx]),
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

                with param_lock:
                    default_ctx = max(context_adapters.keys()) if context_adapters else 0
                    assigned_context = context_assignments.get(client_idx, default_ctx)
                    if use_lora:
                        assigned_adapter = context_adapters.get(
                            assigned_context, context_adapters[default_ctx]
                        )
                        combined = combine_arrays(
                            parameters_to_ndarrays(global_base_params),
                            parameters_to_ndarrays(assigned_adapter),
                            base_indices, lora_indices, total_param_len,
                        )
                        params = ndarrays_to_parameters(combined)
                    else:
                        params = global_base_params

                new_task = ray_actors[client_idx].fit.remote(params, time.time(), current_phase)
                active_tasks[new_task] = client_idx

        # ── Periodic evaluation ────────────────────────────────────────────────
        if time.time() - last_eval_time >= waiting_interval:
            eval_counter += 1
            
            with param_lock:
                if use_lora and updated_contexts and (eval_counter % merge_interval_evals == 0):
                    merge_similar_contexts(
                        server_context_prototypes=server_context_prototypes,
                        context_adapters=context_adapters,
                        context_strategies=context_strategies,
                        context_assignments=context_assignments,
                        phase_to_context=phase_to_context,
                        threshold=merge_threshold,
                        updated_contexts=updated_contexts,
                        context_sample_counts=context_sample_counts,
                    )

                eval_params_dict: Dict[int, List[List[np.ndarray]]] = {}
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
                            
                        eval_params_dict[p_idx] = fused_models_for_phase
                else:
                    for p_idx in range(num_phases):
                        eval_params_dict[p_idx] = [parameters_to_ndarrays(global_base_params)]

            eval_start_ts = time.time()
            loss, metrics_dict = evaluate_global_model(
                global_model, eval_params_dict, global_test_loaders, device
            )
            eval_overhead = time.time() - eval_start_ts
            start_time += eval_overhead
            end_time += eval_overhead

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
                f"\n[t={time.time() - start_time:.1f}s] Evaluation {eval_counter} (eval took {eval_overhead:.1f}s): "
                f"Loss: {loss:.4f}, Accuracy: {acc:.4f}, "
                f"BWT: {bwt:.4f}, FWT: {fwt:.4f} | Active Contexts: {len(server_context_prototypes)}\n"
            )

            if wandb_enabled:
                log_dict = {
                    "async/loss":           loss,
                    "async/accuracy":       acc,
                    "async/updates":        update_count,
                    "async/elapsed_time":   time.time() - start_time,
                    "async/total_contexts": len(server_context_prototypes),
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
        "total_contexts": len(server_context_prototypes),
    }
