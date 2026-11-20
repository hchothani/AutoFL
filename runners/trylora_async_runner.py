"""
async_runner.py  —  Asynchronous Federated Learning with Context-Aware LoRA Adapter Bank
=========================================================================================

CHANGE LOG (all fixes relative to original):
─────────────────────────────────────────────────────────────────────────────────────────
FIX #1  [CRITICAL]  aggregate_result — NameError when client sends no prototype.
            `assigned_context` was only set inside `if proto_str is not None`, but was
            read unconditionally afterwards.  Added an else-branch that falls back to
            context_assignments.get(client_idx, 0).

FIX #2  [CRITICAL]  aggregate_result — Race condition between context assignment and
            LoRA update.  Two separate `with param_lock:` blocks with a gap between them
            meant another thread could overwrite context_assignments[client_idx] between
            the write (block 1) and the read (block 2).  Merged into ONE atomic lock
            acquisition covering the full detection→assignment→update pipeline.

FIX #3  [CRITICAL]  evaluate_global_model — Model weights permanently mutated by
            evaluation.  Each phase-loop iteration called model.load_state_dict() and
            left the model holding the last phase's weights.  Snapshot the state dict
            before the loop and restore it unconditionally after (even on exception).

FIX #4  [DESIGN]   Phase index ≠ Context index conflation.  Evaluation and the final
            params dict both assumed context_id == phase_idx, which is never guaranteed
            when context detection is prototype-driven.  Introduced `phase_to_context`
            (Dict[int, int]) that records which context was most recently dominant for
            each phase, and all evaluation lookups now route through it.

FIX #5  [DESIGN]   Cold-start adapter mismatch.  All clients were dispatched with
            context 0's adapter before any context assignments existed.  This is
            unavoidable at t=0 but is now explicitly documented; clients receive their
            correct adapter on every re-dispatch after their first result is aggregated.

FIX #6  [BUG]      Duplicate declarations of `param_lock` and `total_train_time`.
            The second `param_lock = Lock()` silently replaced the first object, risking
            orphaned locks if anything had acquired the original.  Both duplicates removed;
            each variable is now declared exactly once.

FIX #7  [MINOR]    New context adapters initialized from raw, unweighted client LoRA
            weights.  A freshly discovered context now starts from `init_lora` (the
            model's own LoRA initialization, typically zeros) and the arriving client's
            update is immediately blended in through the strategy's average() call —
            applying staleness and sample weighting correctly from the very first update.
            The if/else was restructured so aggregation always runs (new OR existing).

FIX #8  [MINOR]    Initial evaluation used `global_params` (raw state-dict list) for
            all phases, bypassing the adapter bank composition used everywhere else.
            Now builds `initial_params_dict` via combine_arrays(init_base, init_lora)
            for LoRA models, keeping the parameter composition fully consistent.

FIX #9  [MINOR]    Wrong type annotation on evaluate_global_model.
            `phase_params_dict` was annotated `List[np.ndarray]` but is a
            `Dict[int, List[np.ndarray]]`.  Corrected.

FIX #10 [NOTE]     Lock usage in aggregate_result is architecturally correct (one merged
            block per FIX #2) but is currently redundant because aggregate_result is
            always called serially from the main event loop after ray.get().  The lock
            is retained for forward-safety if the call pattern is ever parallelised.
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
# Utility helpers  (move to utils.py when cleaning)
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
# Global model evaluation
# ──────────────────────────────────────────────────────────────────────────────

# FIX #9: Corrected type annotation — phase_params_dict is Dict[int, List[np.ndarray]]
#         (was incorrectly annotated as List[np.ndarray]).
def evaluate_global_model(
    model: torch.nn.Module,
    phase_params_dict: Dict[int, List[np.ndarray]],
    test_loaders: List[DataLoader],
    device: torch.device,
) -> tuple:
    """
    Evaluate the global model on each phase's test loader using the adapter that
    was assigned to that phase.  Falls back to the most recently trained adapter
    for phases whose adapter has not been initialised yet.

    FIX #3: Saves the model's state dict before evaluation and restores it
    afterwards (including on exception), so evaluation never leaves the model
    in a dirty state.
    """
    # FIX #3 — snapshot current weights so evaluation is side-effect-free
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_phases_loss = 0.0
    total_correct = 0
    total_total = 0
    metrics_dict: Dict[str, float] = {}

    try:
        for phase_idx, phase_loader in enumerate(test_loaders):
            # Dynamic adapter loading — fall back to latest available if phase not yet seen
            params = phase_params_dict.get(phase_idx)
            if params is None:
                latest_idx = max(phase_params_dict.keys())
                params = phase_params_dict[latest_idx]

            # Load the (base + adapter) weights for this phase
            state_dict = model.state_dict()
            for key, param in zip(state_dict.keys(), params):
                state_dict[key] = torch.tensor(param).to(device)
            model.load_state_dict(state_dict)

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

    finally:
        # FIX #3 — always restore the model, even if an exception was raised mid-loop
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

    # Auto-detect LoRA by inspecting parameter key names
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
    # FIX #6: total_train_time declared ONCE here (original had a second identical
    #         assignment at line 222 that silently shadowed the first).
    total_train_time = async_cfg["total_train_time"]
    waiting_interval = async_cfg["waiting_interval"]

    cl_enabled  = cfg.get("cl", {}).get("enabled", False)
    num_phases  = cfg.get("cl", {}).get("num_experiences", 1) if cl_enabled else 1
    phase_duration = total_train_time / num_phases

    phase_total_samples = [
        sum(len(loaders[p].dataset) for loaders in train_loaders)
        for p in range(num_phases)
    ]

    base_strategy    = create_strategy(phase_total_samples[0])
    context_strategies: Dict[int, AsynchronousStrategy] = (
        {0: create_strategy(phase_total_samples[0])} if use_lora else {}
    )

    # ── Context bank state ────────────────────────────────────────────────────
    server_context_prototypes: List[np.ndarray] = []
    context_distance_threshold = cfg.get("context", {}).get("threshold", 0.15)
    context_assignments: Dict[int, int] = {}   # client_idx → context_id

    # FIX #4: phase_to_context records which context was last dominant for each
    #         phase so that evaluation uses the semantically correct adapter
    #         instead of naively assuming phase_idx == context_idx.
    phase_to_context: Dict[int, int] = {p: 0 for p in range(num_phases)}

    history = AsyncHistory()

    # FIX #6: param_lock declared ONCE here (original had a second `Lock()` at
    #         line 220 that silently discarded the first object).
    param_lock = Lock()

    # ── CL metric trackers ────────────────────────────────────────────────────
    phase_max_accs     = [0.0] * num_phases
    seen_phases: set   = set()
    current_global_phase = 0

    # ── Initial evaluation ────────────────────────────────────────────────────
    # FIX #8: Build initial_params_dict via combine_arrays so the parameter
    #         composition (base ∥ LoRA) is identical to every subsequent eval.
    #         Previously `global_params` (the raw full state-dict list) was used
    #         directly, bypassing the adapter bank composition.
    if use_lora:
        _initial_combined = combine_arrays(
            init_base, init_lora, base_indices, lora_indices, total_param_len
        )
        initial_params_dict = {p_idx: _initial_combined for p_idx in range(num_phases)}
    else:
        initial_params_dict = {p_idx: global_params for p_idx in range(num_phases)}

    # FIX #3: evaluate_global_model now restores model state internally — safe to call.
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
        """
        Process one completed client round:
          1. Detect / update the client's context from its prototype vector.
          2. Aggregate the base weights (universal knowledge, all clients).
          3. Aggregate the LoRA adapter for the assigned context only.

        FIX #2: The original had two separate `with param_lock:` blocks with a gap
        between them.  This created a window where another thread could overwrite
        context_assignments[client_idx] between the write (block 1) and the read
        (block 2).  Everything is now one atomic critical section.
        """
        nonlocal global_base_params, update_count, server_context_prototypes

        t_diff   = time.time() - fit_res.metrics.get("start_timestamp", time.time())
        proto_str = fit_res.metrics.get("prototype", None)

        # FIX #2: Single, merged lock acquisition — context detection, base update,
        #         and adapter update are all atomic with respect to param_lock.
        # FIX #10: Lock is currently redundant (aggregate_result is called serially
        #          from the main event loop) but retained for forward-safety.
        with param_lock:

            # ── 1. CONTEXT DETECTION ─────────────────────────────────────────
            if proto_str is not None:
                incoming_proto = np.array(json.loads(proto_str))

                if len(server_context_prototypes) == 0:
                    # First-ever prototype — bootstrap context 0
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
                        # Close enough — EMA-update the matched prototype centroid
                        server_context_prototypes[closest_idx] = (
                            0.9 * server_context_prototypes[closest_idx]
                            + 0.1 * incoming_proto
                        )
                        assigned_context = closest_idx
                        print(f"  [Context Bank] Vehicle {client_idx} assigned Context {assigned_context}")
                    else:
                        # Novel distribution — spawn a new context
                        server_context_prototypes.append(incoming_proto)
                        assigned_context = len(server_context_prototypes) - 1
                        print(
                            f"  [Context Bank] Vehicle {client_idx} generated NEW Context "
                            f"{assigned_context} (Cos Dist: {min_dist:.3f})"
                        )

                context_assignments[client_idx] = assigned_context

                # FIX #4: Record which context was dominant for this phase so
                #         evaluation can route to the right adapter later.
                phase_to_context[phase_idx] = assigned_context

            else:
                # FIX #1: assigned_context was UNBOUND here in the original when
                #         proto_str is None (client sent no prototype).  Fall back
                #         to the most recently known assignment for this client.
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
                    # FIX #7: New context — initialise from init_lora (the model's
                    #         clean LoRA init, typically zeros) rather than using the
                    #         raw client weights directly.  The client's update is
                    #         blended in immediately below via strategy.average(),
                    #         applying staleness & sample weighting from round one.
                    context_adapters[assigned_context]   = ndarrays_to_parameters(init_lora)
                    context_strategies[assigned_context] = create_strategy(phase_total_samples[phase_idx])
                    lora_shift = "INITIALIZED"

                # Aggregate the incoming LoRA into the context adapter (new OR existing)
                old_lora = parameters_to_ndarrays(context_adapters[assigned_context])
                context_strategies[assigned_context].total_samples = phase_total_samples[phase_idx]
                context_adapters[assigned_context] = context_strategies[assigned_context].average(
                    context_adapters[assigned_context],
                    ndarrays_to_parameters(inc_lora),
                    t_diff,
                    fit_res.num_examples,
                )
                new_lora = parameters_to_ndarrays(context_adapters[assigned_context])
                # Don't overwrite the "INITIALIZED" label on the very first update
                if lora_shift != "INITIALIZED":
                    lora_shift = f"{calculate_weight_shift(old_lora, new_lora):.4f}"

            print(
                f"  [Weight Shift] Vehicle {client_idx} | "
                f"Base: {base_shift:.4f} | LoRA {assigned_context}: {lora_shift}"
            )

        # update_count lives outside the lock — aggregate_result is always called
        # serially from the main thread, so no synchronisation is required.
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

    # FIX #5: All clients cold-start with context 0's adapter because context
    #         assignments cannot be known before the first training round.  This
    #         is unavoidable at t=0.  Every re-dispatch after the first round
    #         uses the client's true assigned context adapter (see loop below).
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
                # Do not re-dispatch a failed actor — skip to next ready task
                continue

            if time.time() < end_time:
                # Re-dispatch the client with its context-specific adapter
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

                # Determine current phase from elapsed wall-clock time
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
            eval_params_dict: Dict[int, List[np.ndarray]] = {}

            with param_lock:
                if use_lora:
                    for p_idx in range(num_phases):
                        # FIX #4: Route through phase_to_context so we load the
                        #         adapter that was actually dominant for this phase,
                        #         not just context_adapters[p_idx] (which assumed
                        #         phase_idx == context_idx — never guaranteed).
                        context_for_phase = phase_to_context.get(p_idx, 0)
                        adapter_idx = (
                            context_for_phase
                            if context_for_phase in context_adapters
                            else max(context_adapters.keys())
                        )
                        combined_eval = combine_arrays(
                            parameters_to_ndarrays(global_base_params),
                            parameters_to_ndarrays(context_adapters[adapter_idx]),
                            base_indices, lora_indices, total_param_len,
                        )
                        eval_params_dict[p_idx] = combined_eval
                else:
                    for p_idx in range(num_phases):
                        eval_params_dict[p_idx] = parameters_to_ndarrays(global_base_params)

            # FIX #3: Model state is restored inside evaluate_global_model — safe.
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
    final_params_dict: Dict[int, List[np.ndarray]] = {}
    with param_lock:
        if use_lora:
            for p_idx in range(num_phases):
                # FIX #4: Use phase_to_context here too, consistent with mid-run evals.
                context_for_phase = phase_to_context.get(p_idx, 0)
                adapter_idx = (
                    context_for_phase
                    if context_for_phase in context_adapters
                    else max(context_adapters.keys())
                )
                final_params_dict[p_idx] = combine_arrays(
                    parameters_to_ndarrays(global_base_params),
                    parameters_to_ndarrays(context_adapters[adapter_idx]),
                    base_indices, lora_indices, total_param_len,
                )
        else:
            for p_idx in range(num_phases):
                final_params_dict[p_idx] = parameters_to_ndarrays(global_base_params)

    # FIX #3: Model state restored internally — no extra cleanup needed.
    final_loss, final_metrics = evaluate_global_model(
        global_model, final_params_dict, global_test_loaders, device
    )

    return {
        "final_loss":     final_loss,
        "final_accuracy": final_metrics.get("accuracy"),
        "total_updates":  update_count,
        "elapsed_time":   time.time() - start_time,
    }
