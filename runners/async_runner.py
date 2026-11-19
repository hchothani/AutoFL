"""
async_runner.py  —  Asynchronous Federated Learning with Context-Aware LoRA Adapter Bank
=========================================================================================

CHANGE LOG: 100% Phase-Blind Routing & Aggregation + Post-Training Prototypes + 5-Worker Pool
─────────────────────────────────────────────────────────────────────────────────────────
[UPGRADE]  100% Phase-Blind Routing & Aggregation: Removed `context_to_phase` and per-phase
           `total_samples` overrides from `aggregate_result`. Incoming prototypes are
           matched globally across all stored contexts, and `eval_phase_idx` is used
           strictly as a passive observer for BWT/FWT benchmark evaluation.
[RESTORED] Real-World Post-Training Prototype Extraction: Vehicles adapt locally via SGD
           before extracting penultimate features, ensuring distinct domains separate
           cleanly in cosine space without needing a phase oracle.
[UPGRADE]  5-Worker Memory Hygiene: Keeps `max_workers=5` while capping Ray Plasma at
           1 GiB, bounding per-worker CUDA memory fraction to 0.16 (~3.76 GiB/worker),
           dropping duplicate driver train_loaders, and running periodic `gc.collect()`.
[UPGRADE]  Streaming Single-Phase Evaluation: Decodes `global_base_params` once per
           evaluation and assembles/frees per-phase ensemble states one phase at a time,
           eliminating the t=900s final-evaluation RAM spike.
[UPGRADE]  Non-Intrusive SimpleCNN Path + High-Dim QR Projection: Leaves 84-D SimpleCNN
           completely untouched (no QR decomposition). For high-dim backbones (D > 84),
           applies stateless intra-vector channel standardization + orthonormal QR.
[UPGRADE]  Real-World Context-Aware BatchNorm & Head Isolation: Stores BN running stats
           & classifier head inside each Context Adapter so pure `model.eval()` never
           suffers cross-context running_mean drift.
─────────────────────────────────────────────────────────────────────────────────────────
"""

import collections
import collections.abc
collections.Sequence = collections.abc.Sequence  # Compatibility shim for older Flower versions

import gc
import time
import json
import random
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from threading import Lock
import ray
from omegaconf import DictConfig, OmegaConf
import wandb

from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from algorithms.async_fl import AsynchronousStrategy, AsyncHistory


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers  
# ──────────────────────────────────────────────────────────────────────────────

def get_projection_matrix(d_in: int, d_out: int = 84, seed: int = 42) -> np.ndarray:
    """
    Generates a deterministic orthonormal projection matrix W_proj in R^(d_out x d_in)
    using thin QR decomposition (Johnson-Lindenstrauss lemma).
    Guarantees W_proj @ W_proj.T = I_(d_out) for bounded metric distortion.
    """
    if d_in == d_out:
        return np.eye(d_out, dtype=np.float32)
    if d_in < d_out:
        return np.eye(d_out, d_in, dtype=np.float32)

    rng = np.random.RandomState(seed)
    G = rng.randn(d_in, d_out)
    Q, _ = np.linalg.qr(G)
    return Q.T.astype(np.float32)


def calculate_cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Returns 1 - cosine_similarity. Returns 1.0 (max distance) for zero vectors."""
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


def _find_classifier_module(model: torch.nn.Module) -> torch.nn.Module:
    """
    Locates the final non-LoRA nn.Linear classifier layer (e.g., fc3.base_layer,
    fc.base_layer, or classifier.1.base_layer) so forward hooks capture the full
    penultimate representation rather than a rank-r LoRA sub-layer.
    """
    last_linear = None
    for name, module in model.named_modules():
        if "lora_" not in name and isinstance(module, torch.nn.Linear):
            last_linear = module
    if last_linear is None:
        raise RuntimeError("Could not locate final nn.Linear classifier module for prototype extraction.")
    return last_linear


# ──────────────────────────────────────────────────────────────────────────────
# Persistent Ray Worker Actor (Zero-Deserialization-Leak Execution Engine)
# ──────────────────────────────────────────────────────────────────────────────

@ray.remote
class AsyncWorkerActor:
    """
    A persistent Ray worker actor that maintains a single resident GPU model instance.
    Trains assigned clients on demand using CPU parameter payloads without deserializing
    CUDA tensors or client objects across Ray IPC.
    """

    def __init__(
        self,
        model_fn: Any,
        train_loaders_ref: Any,
        local_epochs: int = 3,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        simulate_delay: bool = False,
        min_delay: float = 0.5,
        max_delay: float = 3.0,
        device_str: str = "cuda:0",
        gpu_mem_fraction: float = 0.16,
    ):
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, device=self.device)
        self.model = model_fn().to(self.device)
        self.train_loaders = (
            ray.get(train_loaders_ref)
            if isinstance(train_loaders_ref, ray.ObjectRef)
            else train_loaders_ref
        )
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.simulate_delay = simulate_delay
        self.min_delay = min_delay
        self.max_delay = max_delay

    def fit(
        self,
        client_idx: int,
        params: Any,
        start_timestamp: float,
        current_phase: int,
    ) -> Tuple[int, FitRes, int]:
        start_time = time.time()
        # `current_phase` only selects the simulated time-window data stream on the vehicle
        active_train_loader = self.train_loaders[client_idx][current_phase]
        num_examples = len(active_train_loader.dataset)

        if self.simulate_delay:
            download_delay = random.uniform(self.min_delay / 2, self.max_delay / 2)
            time.sleep(download_delay)

        # Load incoming CPU weights into the resident GPU model in-place
        params_nd = parameters_to_ndarrays(params)
        state_dict = self.model.state_dict()
        for key, p in zip(state_dict.keys(), params_nd):
            state_dict[key] = torch.as_tensor(p, dtype=state_dict[key].dtype, device=self.device)
        self.model.load_state_dict(state_dict)
        del params_nd

        # ── Local Training ─────────────────────────────────────────────────────
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0
        for _ in range(self.local_epochs):
            for batch in active_train_loader:
                if isinstance(batch, dict):
                    images = batch.get("img", batch.get("x")).to(self.device)
                    labels = batch.get("label", batch.get("y")).to(self.device)
                elif isinstance(batch, (tuple, list)):
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue

                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Free optimizer and gradient buffers before prototype forward pass
        self.model.zero_grad(set_to_none=True)
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Real-World Post-Training Prototype Extraction ──────────────────────
        self.model.eval()
        feature_sum = None
        feature_count = 0

        def feature_hook(module, inp, out):
            nonlocal feature_sum, feature_count
            batch_feats = inp[0].detach().cpu().numpy()
            if batch_feats.ndim > 2:
                batch_feats = batch_feats.reshape(batch_feats.shape[0], -1)

            if feature_sum is None:
                feature_sum = np.sum(batch_feats, axis=0)
            else:
                feature_sum += np.sum(batch_feats, axis=0)
            feature_count += batch_feats.shape[0]

        classifier_layer = _find_classifier_module(self.model)
        hook_handle = classifier_layer.register_forward_hook(feature_hook)

        try:
            with torch.no_grad():
                for batch in active_train_loader:
                    if isinstance(batch, dict):
                        images = batch.get("img", batch.get("x")).to(self.device)
                    elif isinstance(batch, (tuple, list)):
                        images = batch[0].to(self.device)
                    else:
                        continue
                    _ = self.model(images)
        finally:
            hook_handle.remove()

        if feature_count > 0:
            raw_proto = (feature_sum / feature_count).astype(np.float32)
            client_prototype = raw_proto.tolist()
        else:
            client_prototype = None

        if self.simulate_delay:
            upload_delay = random.uniform(self.min_delay / 2, self.max_delay / 2)
            time.sleep(upload_delay)

        new_params = [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]
        self.model.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        # No phase metadata is sent to the server in FitRes.metrics
        metrics = {
            "loss": avg_loss,
            "training_time": elapsed,
            "start_timestamp": start_timestamp,
            "client_id": str(client_idx),
        }
        if client_prototype is not None:
            metrics["prototype"] = json.dumps(client_prototype)

        fit_res = FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(new_params),
            num_examples=num_examples,
            metrics=metrics,
        )
        del new_params
        return client_idx, fit_res, current_phase


# ──────────────────────────────────────────────────────────────────────────────
# Config helper
# ──────────────────────────────────────────────────────────────────────────────

def get_async_config(cfg: DictConfig) -> Dict[str, Any]:
    async_cfg = cfg.get("async", {})
    if isinstance(async_cfg, DictConfig):
        async_cfg = OmegaConf.to_container(async_cfg, resolve=True)
    return {
        "total_train_time":           async_cfg.get("total_train_time",       900),
        "waiting_interval":           async_cfg.get("waiting_interval",        10),
        "max_workers":                async_cfg.get("max_workers",              5),
        "aggregation_strategy":       async_cfg.get("aggregation_strategy", "fedasync"),
        "staleness_alpha":            async_cfg.get("staleness_alpha",        0.5),
        "fedasync_mixing_alpha":      async_cfg.get("fedasync_mixing_alpha",  0.9),
        "fedasync_a":                 async_cfg.get("fedasync_a",             0.5),
        "use_staleness":              async_cfg.get("use_staleness",          True),
        "use_sample_weighing":        async_cfg.get("use_sample_weighing",    True),
        "send_gradients":             async_cfg.get("send_gradients",         False),
        "server_artificial_delay":    async_cfg.get("server_artificial_delay",False),
        "is_streaming":               async_cfg.get("is_streaming",           False),
        "client_local_delay":         async_cfg.get("client_local_delay",     False),
        "simulate_delay":             async_cfg.get("simulate_delay",         True),
        "min_delay":                  async_cfg.get("min_delay",              0.5),
        "max_delay":                  async_cfg.get("max_delay",              3.0),
        "client_selection_strategy":  async_cfg.get("client_selection_strategy", "round_robin"),
        "loss_temperature":           async_cfg.get("loss_temperature",       1.0),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Client Scheduler
# ──────────────────────────────────────────────────────────────────────────────

def get_next_client(strategy: str, registry: dict, available_clients: list, temperature: float = 1.0) -> int:
    """Selects the next client from the available pool based on the defined strategy."""
    if not available_clients:
        raise ValueError("No clients available to schedule.")

    if strategy == "round_robin":
        return available_clients.pop(0)
    
    elif strategy == "data_weighted":
        weights = [registry[c]["data"] for c in available_clients]
        selected = random.choices(available_clients, weights=weights, k=1)[0]
        available_clients.remove(selected)
        return selected
        
    elif strategy == "loss_weighted":
        losses = [registry[c]["loss"] for c in available_clients]
        max_loss = max(losses) if losses else 0
        exp_weights = [math.exp((l - max_loss) / temperature) for l in losses]
        selected = random.choices(available_clients, weights=exp_weights, k=1)[0]
        available_clients.remove(selected)
        return selected
        
    return available_clients.pop(0)


# ──────────────────────────────────────────────────────────────────────────────
# Streaming Global Model Evaluation (1 phase in memory at a time)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_global_model_streaming(
    model: torch.nn.Module,
    base_nd: List[np.ndarray],
    lora_nd_map: Dict[int, List[np.ndarray]],
    phase_contexts_map: Dict[int, set],
    use_lora: bool,
    base_indices: List[int],
    lora_indices: List[int],
    total_param_len: int,
    test_loaders: List[DataLoader],
    device: torch.device,
) -> tuple:
    """
    Evaluates one phase at a time and frees each phase's assembled state_dicts
    immediately after scoring so peak host RAM never scales with num_phases * num_contexts.
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
            if use_lora:
                active_contexts = phase_contexts_map.get(phase_idx, set())
                if not active_contexts:
                    active_contexts = {max(lora_nd_map.keys()) if lora_nd_map else 0}

                fallback_idx = max(lora_nd_map.keys())
                ensemble_params = [
                    combine_arrays(
                        base_nd,
                        lora_nd_map[ctx if ctx in lora_nd_map else fallback_idx],
                        base_indices,
                        lora_indices,
                        total_param_len,
                    )
                    for ctx in active_contexts
                ]
            else:
                ensemble_params = [base_nd]

            prepared_states = [
                {
                    key: torch.as_tensor(param, dtype=original_state[key].dtype, device=device)
                    for key, param in zip(template_keys, params)
                }
                for params in ensemble_params
            ]
            del ensemble_params

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

            del prepared_states
            total_total += total
            total_correct += correct
            phase_accuracy = correct / max(total, 1)
            total_phases_loss += phase_loss / max(total, 1)
            metrics_dict[f"phase_{phase_idx}_accuracy"] = phase_accuracy
    finally:
        model.load_state_dict(original_state)
        del original_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    # ── Model & weight-index initialisation ───────────────────────────────────
    global_model = model_fn().to(device)
    global_keys = list(global_model.state_dict().keys())

    # Context-Aware BatchNorm & Head Isolation:
    # - For SimpleCNN (has_bn == False): Strictly uses "lora" in k (100% untouched).
    # - For ResNet-18 / MobileNetV2 (has_bn == True): Keeps BatchNorm running stats & final
    #   classifier head inside each Context Adapter so `model.eval()` never suffers BN drift.
    has_bn = any("running_mean" in k for k in global_keys)
    if has_bn:
        context_key_tokens = (
            "lora", "bn", "running_", "num_batches_tracked",
            "downsample.1", "fc", "linear", "classifier",
        )
        lora_indices = [
            i for i, k in enumerate(global_keys)
            if any(tok in k for tok in context_key_tokens)
        ]
        lora_set = set(lora_indices)
        base_indices = [i for i in range(len(global_keys)) if i not in lora_set]
    else:
        base_indices = [i for i, k in enumerate(global_keys) if "lora" not in k]
        lora_indices = [i for i, k in enumerate(global_keys) if "lora" in k]

    use_lora = len(lora_indices) > 0
    total_param_len = len(global_keys)

    global_arrays = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    init_base, init_lora = split_arrays(global_arrays, base_indices, lora_indices)
    del global_arrays

    global_base_params = ndarrays_to_parameters(init_base)
    context_adapters: Dict[int, Any] = {0: ndarrays_to_parameters(init_lora)} if use_lora else {}

    # ── Phase-blind strategy factory ──────────────────────────────────────────
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
    client_strategy  = async_cfg["client_selection_strategy"]
    temperature      = async_cfg["loss_temperature"]
    print(f"[Scheduler] Client selection strategy: {client_strategy}")

    cl_enabled  = cfg.get("cl", {}).get("enabled", False)
    num_phases  = cfg.get("cl", {}).get("num_experiences", 1) if cl_enabled else 1
    phase_duration = total_train_time / num_phases

    # Fixed phase-agnostic sample scale (never indexed by phase_idx during aggregation)
    nominal_total_samples = max(
        sum(len(loaders[0].dataset) for loaders in train_loaders if loaders[0] is not None),
        1,
    )

    base_strategy = create_strategy(nominal_total_samples, custom_alpha=0.1)
    context_strategies: Dict[int, AsynchronousStrategy] = (
        {0: create_strategy(nominal_total_samples, custom_alpha=0.9)} if use_lora else {}
    )

    # ── Context bank state (100% phase-blind) ─────────────────────────────────
    server_context_prototypes: Dict[int, np.ndarray] = {}
    next_context_id = 0
    context_distance_threshold = float(cfg.get("context", {}).get("threshold", 0.15))
    context_assignments: Dict[int, int] = {}

    # Passive write-only observer used solely for benchmark evaluation (BWT / FWT)
    phase_to_context: Dict[int, set] = {p: set() for p in range(num_phases)}

    # Prototype Compression State
    target_proto_dim: int = int(cfg.get("context", {}).get("target_proto_dim", 84))
    w_proj: Optional[np.ndarray] = None

    param_lock = Lock()
    phase_max_accs = [0.0] * num_phases
    seen_phases: set = set()
    current_global_phase = 0

    # ── Initial evaluation ────────────────────────────────────────────────────
    initial_loss, initial_metrics = evaluate_global_model_streaming(
        model=global_model,
        base_nd=init_base,
        lora_nd_map={0: init_lora} if use_lora else {},
        phase_contexts_map=phase_to_context,
        use_lora=use_lora,
        base_indices=base_indices,
        lora_indices=lora_indices,
        total_param_len=total_param_len,
        test_loaders=global_test_loaders,
        device=device,
    )
    del init_base, init_lora
    gc.collect()
    initial_phase_acc = [initial_metrics[f"phase_{i}_accuracy"] for i in range(num_phases)]

    if wandb_enabled:
        log_dict = {"async/loss": initial_loss, "async/updates": 0, "async/elapsed_time": 0.0}
        for k, v in initial_metrics.items():
            log_dict[f"async/{k}"] = v
        wandb.log(log_dict, step=0)

    start_time   = time.time()
    end_time     = start_time + total_train_time
    update_count = 0

    # ── Inner aggregation callback (100% phase-blind decision logic) ──────────
    def aggregate_result(client_idx: int, fit_res, eval_phase_idx: int) -> float:
        nonlocal global_base_params, update_count, server_context_prototypes, next_context_id
        nonlocal w_proj

        t_diff    = time.time() - fit_res.metrics.get("start_timestamp", time.time())
        proto_str = fit_res.metrics.get("prototype", None)

        with param_lock:
            # ── 1. PHASE-BLIND CONTEXT ROUTING ───────────────────────────────
            if proto_str is not None:
                raw_proto = np.array(json.loads(proto_str), dtype=np.float32)

                if raw_proto.shape[0] <= target_proto_dim:
                    raw_norm = np.linalg.norm(raw_proto)
                    incoming_proto = (raw_proto / raw_norm) if raw_norm > 1e-8 else raw_proto
                else:
                    if w_proj is None:
                        w_proj = get_projection_matrix(
                            d_in=raw_proto.shape[0],
                            d_out=target_proto_dim,
                            seed=42,
                        )
                        print(f"  [Context Bank] Initialized QR W_proj: {raw_proto.shape[0]}-D -> {target_proto_dim}-D")

                    proto_std = np.std(raw_proto)
                    p_centered = (raw_proto - np.mean(raw_proto)) / (proto_std if proto_std > 1e-8 else 1.0)
                    raw_norm = np.linalg.norm(p_centered)
                    p_unit = (p_centered / raw_norm) if raw_norm > 1e-8 else p_centered
                    p_projected = np.dot(w_proj, p_unit)
                    p_norm = np.linalg.norm(p_projected)
                    incoming_proto = (p_projected / p_norm) if p_norm > 1e-8 else p_projected

                # Compare against ALL stored context prototypes globally (no phase oracle)
                all_ctx_keys = list(server_context_prototypes.keys())

                if len(all_ctx_keys) == 0:
                    assigned_context = next_context_id
                    server_context_prototypes[assigned_context] = incoming_proto
                    next_context_id += 1
                    print(f"  [Context Bank] Vehicle {client_idx} established Initial Context {assigned_context}.")
                else:
                    distances = [
                        calculate_cosine_distance(incoming_proto, server_context_prototypes[k])
                        for k in all_ctx_keys
                    ]
                    min_dist = min(distances)
                    closest_ctx = all_ctx_keys[distances.index(min_dist)]

                    if min_dist < context_distance_threshold:
                        updated = 0.9 * server_context_prototypes[closest_ctx] + 0.1 * incoming_proto
                        up_norm = np.linalg.norm(updated)
                        server_context_prototypes[closest_ctx] = updated / up_norm if up_norm > 1e-8 else updated
                        assigned_context = closest_ctx
                        print(f"  [Context Bank] Vehicle {client_idx} assigned Context {assigned_context} (Cos Dist: {min_dist:.3f})")
                    else:
                        assigned_context = next_context_id
                        server_context_prototypes[assigned_context] = incoming_proto
                        next_context_id += 1
                        print(f"  [Context Bank] Vehicle {client_idx} generated NEW Context {assigned_context} (Cos Dist: {min_dist:.3f})")

                context_assignments[client_idx] = assigned_context
            else:
                default_ctx = max(context_adapters.keys()) if context_adapters else 0
                assigned_context = context_assignments.get(client_idx, default_ctx)

            # Passive evaluation-only bookkeeping (never read by routing or aggregation)
            phase_to_context[eval_phase_idx].add(assigned_context)

            # ── 2. UNPACK INCOMING WEIGHTS ───────────────────────────────────
            incoming_arrays = parameters_to_ndarrays(fit_res.parameters)
            inc_base, inc_lora = split_arrays(incoming_arrays, base_indices, lora_indices)
            del incoming_arrays

            # ── 3. UPDATE GLOBAL BASE (phase-blind) ──────────────────────────
            old_base = parameters_to_ndarrays(global_base_params)
            global_base_params = base_strategy.average(
                global_base_params, ndarrays_to_parameters(inc_base), t_diff, fit_res.num_examples,
            )
            new_base = parameters_to_ndarrays(global_base_params)
            base_shift = calculate_weight_shift(old_base, new_base)
            del inc_base, old_base, new_base

            # ── 4. UPDATE SPECIALIZED LORA ADAPTER (phase-blind) ─────────────
            lora_shift: Any = 0.0
            if use_lora:
                if assigned_context not in context_adapters:
                    context_adapters[assigned_context] = ndarrays_to_parameters(inc_lora)
                    context_strategies[assigned_context] = create_strategy(nominal_total_samples, custom_alpha=0.9)
                    lora_shift = "INITIALIZED"
                else:
                    old_lora = parameters_to_ndarrays(context_adapters[assigned_context])
                    context_adapters[assigned_context] = context_strategies[assigned_context].average(
                        context_adapters[assigned_context], ndarrays_to_parameters(inc_lora), t_diff, fit_res.num_examples,
                    )
                    new_lora = parameters_to_ndarrays(context_adapters[assigned_context])
                    lora_shift = f"{calculate_weight_shift(old_lora, new_lora):.4f}"
                    del old_lora, new_lora
            del inc_lora

            print(f"  [Weight Shift] Vehicle {client_idx} | Base: {base_shift:.4f} | LoRA {assigned_context}: {lora_shift}")

        update_count += 1
        if update_count % 10 == 0:
            gc.collect()
        return t_diff

    # ── Ray Cluster & Worker Pool Setup ─────────────────────────────────────────
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            object_store_memory=1 * 1024**3,  # Cap Plasma shared memory at 1 GiB
        )

    max_workers = async_cfg.get("max_workers")
    if not max_workers:
        total_cpus = ray.cluster_resources().get("CPU", 1.0)
        cpus_per_client = cfg.client.num_cpus or 1.0
        max_workers = max(1, int(total_cpus // cpus_per_client))
        print(f"[Ray] Auto-detected hardware concurrency limit: {max_workers} active tasks")

    # Bound per-worker CUDA memory so `max_workers` actors fit comfortably inside 23.52 GiB
    worker_gpu_fraction = min(0.25, 0.80 / max(max_workers, 1))

    print(f"\nLoading {num_clients} client train loaders into Ray Shared Memory (Plasma)...")
    train_loaders_ref = ray.put(train_loaders)

    print(f"\nDeploying pool of {max_workers} persistent Worker Actors (gpu_mem_fraction={worker_gpu_fraction:.2f})...")
    workers = [
        AsyncWorkerActor.options(
            num_cpus=cfg.client.num_cpus,
            num_gpus=cfg.client.num_gpus,
        ).remote(
            model_fn=model_fn,
            train_loaders_ref=train_loaders_ref,
            local_epochs=cfg.client.local_epochs,
            learning_rate=cfg.client.learning_rate,
            momentum=cfg.client.get("momentum", 0.0),
            weight_decay=cfg.client.get("weight_decay", 0.0),
            simulate_delay=async_cfg["simulate_delay"],
            min_delay=async_cfg["min_delay"],
            max_delay=async_cfg["max_delay"],
            device_str=str(device),
            gpu_mem_fraction=worker_gpu_fraction,
        )
        for _ in range(max_workers)
    ]

    # ── Initial Dispatch ────────────────────────────────────────────────────────
    client_registry = {i: {"data": 1.0, "loss": 1.0} for i in range(num_clients)}
    available_clients = list(range(num_clients))
    unique_vehicles_trained = set()
    
    active_tasks: Dict[Any, Tuple[Any, int]] = {}
    eval_counter, last_eval_time = 0, start_time
    
    for worker in workers:
        if not available_clients:
            break
        client_idx = get_next_client(client_strategy, client_registry, available_clients, temperature)
        unique_vehicles_trained.add(client_idx)
        print(f"  [Scheduler] Init Dispatch -> Assigned Vehicle {client_idx} to a persistent worker.")
        
        with param_lock:
            if use_lora:
                combined = combine_arrays(
                    parameters_to_ndarrays(global_base_params),
                    parameters_to_ndarrays(context_adapters[0]),
                    base_indices, lora_indices, total_param_len,
                )
                params = ndarrays_to_parameters(combined)
                del combined
            else:
                params = global_base_params
                
        task = worker.fit.remote(client_idx, params, time.time(), current_phase=0)
        del params
        active_tasks[task] = (worker, client_idx)

    # Free driver copies of train_loaders once all workers have initialized
    del train_loaders, train_loaders_ref
    gc.collect()

    # ── Main async event loop ──────────────────────────────────────────────────
    while time.time() < end_time and active_tasks:
        ready_tasks, _ = ray.wait(list(active_tasks.keys()), num_returns=1, timeout=0.1)

        for task in ready_tasks:
            worker, returned_client_idx = active_tasks.pop(task)
            
            try:
                client_idx_from_tuple, fit_res, returned_phase = ray.get(task)
                t_diff = aggregate_result(client_idx_from_tuple, fit_res, returned_phase)
                
                client_registry[client_idx_from_tuple]["data"] = fit_res.num_examples
                client_registry[client_idx_from_tuple]["loss"] = fit_res.metrics.get("loss", 1.0)
                available_clients.append(client_idx_from_tuple)
                
                print(
                    f"[Phase: {returned_phase}] [t={time.time() - start_time:.1f}s] "
                    f"Vehicle {client_idx_from_tuple} completed (loss: {fit_res.metrics.get('loss', 0):.4f})"
                )
                del fit_res
            except Exception as e:
                print(f"[Error] Vehicle {returned_client_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                available_clients.append(returned_client_idx)
                
            # Dispatch a new task to the worker that just completed
            if time.time() < end_time:
                elapsed = time.time() - start_time
                current_phase = min(int(elapsed / phase_duration), num_phases - 1)
                
                if current_phase > current_global_phase:
                    print(f"\n{'='*50}")
                    print(f"[Server] SHIFTING PHASE: Transitioning to Phase {current_phase} at t={elapsed:.1f}s")
                    print(f"{'='*50}\n")
                    current_global_phase = current_phase

                next_client = get_next_client(client_strategy, client_registry, available_clients, temperature)
                unique_vehicles_trained.add(next_client)
                
                print(
                    f"  [Scheduler] Handoff: Vehicle {returned_client_idx} finished -> "
                    f"Dispatched Vehicle {next_client} to the freed worker. "
                    f"(Unique vehicles seen: {len(unique_vehicles_trained)}/{num_clients})"
                )
                
                with param_lock:
                    default_ctx = max(context_adapters.keys()) if context_adapters else 0
                    assigned_context = context_assignments.get(next_client, default_ctx)
                    if use_lora:
                        assigned_adapter = context_adapters.get(assigned_context, context_adapters[default_ctx])
                        combined = combine_arrays(
                            parameters_to_ndarrays(global_base_params),
                            parameters_to_ndarrays(assigned_adapter),
                            base_indices, lora_indices, total_param_len,
                        )
                        params = ndarrays_to_parameters(combined)
                        del combined
                    else:
                        params = global_base_params

                new_task = worker.fit.remote(next_client, params, time.time(), current_phase)
                del params
                active_tasks[new_task] = (worker, next_client)

        # ── Periodic evaluation (Streaming 1 phase at a time) ──────────────────
        if time.time() - last_eval_time >= waiting_interval:
            eval_start_ts = time.time()
            eval_counter += 1
            with param_lock:
                base_nd = parameters_to_ndarrays(global_base_params)
                lora_nd_map = (
                    {ctx: parameters_to_ndarrays(adapter) for ctx, adapter in context_adapters.items()}
                    if use_lora else {}
                )
                phase_contexts_snap = {p: set(ctxs) for p, ctxs in phase_to_context.items()}

            loss, metrics_dict = evaluate_global_model_streaming(
                model=global_model,
                base_nd=base_nd,
                lora_nd_map=lora_nd_map,
                phase_contexts_map=phase_contexts_snap,
                use_lora=use_lora,
                base_indices=base_indices,
                lora_indices=lora_indices,
                total_param_len=total_param_len,
                test_loaders=global_test_loaders,
                device=device,
            )
            del base_nd, lora_nd_map, phase_contexts_snap
            gc.collect()

            eval_overhead = time.time() - eval_start_ts
            start_time += eval_overhead
            end_time   += eval_overhead

            phase_accuracies = [metrics_dict[f"phase_{i}_accuracy"] for i in range(num_phases)]
            seen_phases.add(current_global_phase)
            phase_max_accs[current_global_phase] = max(phase_max_accs[current_global_phase], phase_accuracies[current_global_phase])

            bwt, fwt = 0.0, 0.0
            if current_global_phase > 0:
                bwt = sum(phase_accuracies[p] - phase_max_accs[p] for p in range(current_global_phase)) / current_global_phase
            if current_global_phase < num_phases - 1:
                remaining = num_phases - current_global_phase - 1
                fwt = sum(phase_accuracies[p] - initial_phase_acc[p] for p in range(current_global_phase + 1, num_phases)) / remaining
            avg_seen_acc = sum(phase_accuracies[p] for p in seen_phases) / len(seen_phases)

            metrics_dict["bwt"] = bwt
            metrics_dict["fwt"] = fwt
            metrics_dict["avg_seen_acc"] = avg_seen_acc

            acc = metrics_dict.pop("accuracy")
            print(
                f"\n[t={time.time() - start_time:.1f}s] Evaluation {eval_counter} (eval took {eval_overhead:.1f}s): "
                f"Loss: {loss:.4f}, Accuracy: {acc:.4f}, BWT: {bwt:.4f}, FWT: {fwt:.4f} | "
                f"Active Contexts: {len(server_context_prototypes)}"
            )
            print(f"  [Scheduler Health] {len(unique_vehicles_trained)} out of {num_clients} total vehicles have participated so far.\n")

            if wandb_enabled:
                log_dict = {
                    "async/loss":         loss,
                    "async/accuracy":     acc,
                    "async/updates":      update_count,
                    "async/elapsed_time": time.time() - start_time,
                    "async/total_contexts": len(server_context_prototypes),
                }
                if cl_enabled:
                    for k, v in metrics_dict.items():
                        if k != "accuracy":
                            log_dict[f"async/{k}"] = v
                wandb.log(log_dict, step=eval_counter)

            last_eval_time = time.time()

    # ── Teardown (Reclaim all worker & Ray memory BEFORE final evaluation) ─────
    for worker in workers:
        ray.kill(worker)
    del workers, active_tasks
    gc.collect()
    ray.shutdown()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Final evaluation (Streaming 1 phase at a time) ─────────────────────────
    with param_lock:
        base_nd = parameters_to_ndarrays(global_base_params)
        lora_nd_map = (
            {ctx: parameters_to_ndarrays(adapter) for ctx, adapter in context_adapters.items()}
            if use_lora else {}
        )
        phase_contexts_snap = {p: set(ctxs) for p, ctxs in phase_to_context.items()}

    final_loss, final_metrics = evaluate_global_model_streaming(
        model=global_model,
        base_nd=base_nd,
        lora_nd_map=lora_nd_map,
        phase_contexts_map=phase_contexts_snap,
        use_lora=use_lora,
        base_indices=base_indices,
        lora_indices=lora_indices,
        total_param_len=total_param_len,
        test_loaders=global_test_loaders,
        device=device,
    )
    del base_nd, lora_nd_map, phase_contexts_snap
    gc.collect()

    return {
        "final_loss":     final_loss,
        "final_accuracy": final_metrics.get("accuracy"),
        "total_updates":  update_count,
        "elapsed_time":   time.time() - start_time,
        "total_contexts": len(server_context_prototypes),
    }
