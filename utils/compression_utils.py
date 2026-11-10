"""
utils/compression_utils.py
========================================================================================
Modular Prototype Compression & Dimensionality Reduction for AutoFL Context Detection
----------------------------------------------------------------------------------------
Compresses high-dimensional feature representations (e.g., 512-D for ResNet-18,
1280-D for MobileNetV2) down to compact dimensions (e.g., 84-D for SimpleCNN scale).

Features:
  1. Anisotropic Cone Decoupling: Channel standardization ((p - mean) / std) to remove
     the shared positive DC offset caused by ReLU activations across classes.
  2. Johnson-Lindenstrauss Orthogonal Random Projection: Deterministic QR-based
     orthogonal projection (W @ W^T = I) to preserve pairwise geometry with bounded distortion.
  3. L2 Normalization: Produces unit vectors on the hypersphere for stable cosine clustering.
  4. Truncated SVD LoRA Merging: Eliminates cross-term noise when consolidating redundant contexts.
  5. Universal Model Partitioning: Categorizes model parameters into shared backbone vs.
     context-specific adapters (LoRA + classifier head + normalization stats).
========================================================================================
"""

from typing import List, Union, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

# Cache for projection matrices: (d_in, d_out, seed) -> np.ndarray
_PROJECTION_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def get_orthogonal_projection_matrix(
    d_in: int,
    d_out: int = 84,
    seed: int = 42
) -> np.ndarray:
    """
    Generates or retrieves a deterministic orthogonal projection matrix W in R^{d_out x d_in}
    using thin QR decomposition of a standard normal random matrix.
    
    Guarantees: W @ W.T = I_{d_out}
    """
    if d_in <= d_out:
        return np.eye(d_in, dtype=np.float32)

    cache_key = (d_in, d_out, seed)
    if cache_key in _PROJECTION_CACHE:
        return _PROJECTION_CACHE[cache_key]

    rng = np.random.RandomState(seed)
    # Gaussian random matrix in R^{d_in x d_out}
    G = rng.randn(d_in, d_out).astype(np.float32)
    # QR decomposition yields orthogonal Q in R^{d_in x d_out} such that Q.T @ Q = I_{d_out}
    Q, _ = np.linalg.qr(G)
    # Projection matrix W in R^{d_out x d_in}
    W = Q.T.astype(np.float32)

    _PROJECTION_CACHE[cache_key] = W
    return W


def compress_prototype(
    prototype: Union[np.ndarray, List[float]],
    target_dim: int = 84,
    centering: bool = True,
    seed: int = 42
) -> np.ndarray:
    """
    Compresses an incoming prototype vector to target_dim using intra-vector standardization,
    unit L2 normalization, and deterministic orthogonal random projection.

    Args:
        prototype: 1D array or list of features (e.g. 512-D for ResNet, 1280-D for MobileNet).
        target_dim: Desired target dimensionality (default: 84 to match SimpleCNN scale).
        centering: If True, applies channel standardization ((p - mean) / std) to eliminate
                   the anisotropic positive ReLU cone.
        seed: Random seed for deterministic projection matrix generation.

    Returns:
        1D unit-normalized numpy array in R^{target_dim} (or R^{d_in} if d_in <= target_dim).
    """
    p = np.asarray(prototype, dtype=np.float32).ravel()
    d_in = p.shape[0]

    # Non-QR path for models whose native dimension is already <= target_dim (e.g. SimpleCNN 84-D)
    if d_in <= target_dim:
        norm = np.linalg.norm(p)
        return (p / norm) if norm > 1e-12 else p

    # 1. Intra-vector channel standardization (zero mean, unit variance)
    if centering:
        std = np.std(p)
        p = (p - np.mean(p)) / (std if std > 1e-8 else 1.0)

    # 2. Native space L2 unit normalization
    norm_native = np.linalg.norm(p)
    if norm_native > 1e-12:
        p = p / norm_native

    # 3. Deterministic Orthonormal QR Random Projection (d_in -> target_dim)
    W = get_orthogonal_projection_matrix(d_in=d_in, d_out=target_dim, seed=seed)
    p_proj = W @ p

    # 4. Target space L2 unit normalization
    norm_target = np.linalg.norm(p_proj)
    if norm_target > 1e-12:
        p_proj = p_proj / norm_target

    return p_proj


def find_classifier_module(model: nn.Module) -> nn.Module:
    """
    Finds the outermost classification layer of a neural network (or PEFT wrapped model)
    whose input[0] receives the raw feature vector from the convolutional/feature backbone.

    Guarantees:
      - ResNet-18: Hooks self.fc (input shape: [batch, 512])
      - SimpleCNN: Hooks self.fc3 (input shape: [batch, 84])
      - MobileNet: Hooks self.backbone.classifier (input shape: [batch, 1280])
      - Prevents internal PEFT LoRA submodules (such as lora_B) from hijacking the hook.
    """
    underlying = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        underlying = model.base_model.model
    elif hasattr(model, "base_model"):
        underlying = model.base_model

    # Check common classifier attribute names on underlying model or wrapped backbone
    targets = [underlying]
    if hasattr(underlying, "backbone"):
        targets.append(getattr(underlying, "backbone"))

    for target in targets:
        for attr in ["fc", "fc3", "classifier", "heads", "head", "linear", "out_proj"]:
            if hasattr(target, attr):
                return getattr(target, attr)

    # Fallback: scan top-level children of underlying in reverse order
    children = list(underlying.children())
    if children:
        return children[-1]

    return underlying


def partition_model_indices(model: nn.Module) -> Tuple[List[int], List[int]]:
    """
    Universally partitions a model's state_dict keys into (base_indices, lora_indices).

    Context Adapter indices (lora_indices) include:
      1. Any parameter containing 'lora' (PEFT low-rank adapter weights).
      2. Any parameter/buffer belonging to the classifier head (found via find_classifier_module).
      3. Any parameter/buffer belonging to normalization layers (isinstance _BatchNorm or _InstanceNorm).
         This ensures running statistics and normalization affine weights are tracked per context
         rather than corrupting the shared base model across non-IID client updates.

    Base indices (base_indices) include:
      The shared universal feature extraction backbone (e.g. convolution filters).

    Works generically across ResNet, MobileNet, SimpleCNN, Transformers, etc. with zero
    hardcoded architecture string assumptions.
    """
    global_keys = list(model.state_dict().keys())

    # 1. Identify normalization layer keys dynamically via PyTorch reflection
    norm_keys = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, (nn.modules.batchnorm._BatchNorm, nn.modules.instancenorm._InstanceNorm)):
            for param_name, _ in mod.named_parameters(recurse=False):
                norm_keys.add(f"{mod_name}.{param_name}" if mod_name else param_name)
            for buf_name, _ in mod.named_buffers(recurse=False):
                norm_keys.add(f"{mod_name}.{buf_name}" if mod_name else buf_name)

    # 2. Identify classifier head keys dynamically
    head_module = find_classifier_module(model)
    head_keys = set()
    for mod_name, mod in model.named_modules():
        if mod is head_module:
            for param_name, _ in mod.named_parameters():
                head_keys.add(f"{mod_name}.{param_name}" if mod_name else param_name)
            for buf_name, _ in mod.named_buffers():
                head_keys.add(f"{mod_name}.{buf_name}" if mod_name else buf_name)

    # 3. Partition keys into (base_indices, lora_indices)
    lora_indices = []
    base_indices = []
    for idx, key in enumerate(global_keys):
        if "lora" in key or key in head_keys or key in norm_keys:
            lora_indices.append(idx)
        else:
            base_indices.append(idx)

    return base_indices, lora_indices


def svd_merge_lora_weights(
    w_a: List[np.ndarray],
    w_b: List[np.ndarray],
    n_a: float,
    n_b: float,
) -> List[np.ndarray]:
    """
    Fuses two LoRA weight parameter lists using Truncated SVD on the true 
    low-rank delta matrices (B @ A) to eliminate cross-term noise.
    Falls back to sample-weighted averaging for 1D or non-paired tensors.
    """
    total_n = max(n_a + n_b, 1e-8)
    w_merged: List[np.ndarray] = []
    i = 0

    while i < len(w_a):
        # Check if index i and i+1 form a 2D LoRA weight pair (A, B) or (B, A)
        if i + 1 < len(w_a) and w_a[i].ndim == 2 and w_a[i + 1].ndim == 2:
            m1_a, m2_a = w_a[i], w_a[i + 1]
            m1_b, m2_b = w_b[i], w_b[i + 1]

            # ── Case 1: m1 is A (r, in), m2 is B (out, r) ────────────────────
            if m1_a.shape[0] == m2_a.shape[1] and m1_a.shape == m1_b.shape and m2_a.shape == m2_b.shape:
                rank = m1_a.shape[0]
                delta_a = np.dot(m2_a, m1_a)  # (out, in)
                delta_b = np.dot(m2_b, m1_b)
                delta_merged = (n_a * delta_a + n_b * delta_b) / total_n

                U, S, Vt = np.linalg.svd(delta_merged, full_matrices=False)
                sqrt_s = np.sqrt(np.maximum(S[:rank], 0.0))

                B_new = U[:, :rank] * sqrt_s
                A_new = sqrt_s[:, None] * Vt[:rank, :]

                w_merged.append(A_new)
                w_merged.append(B_new)
                i += 2
                continue

            # ── Case 2: m1 is B (out, r), m2 is A (r, in) ────────────────────
            elif m1_a.shape[1] == m2_a.shape[0] and m1_a.shape == m1_b.shape and m2_a.shape == m2_b.shape:
                rank = m1_a.shape[1]
                delta_a = np.dot(m1_a, m2_a)  # (out, in)
                delta_b = np.dot(m1_b, m2_b)
                delta_merged = (n_a * delta_a + n_b * delta_b) / total_n

                U, S, Vt = np.linalg.svd(delta_merged, full_matrices=False)
                sqrt_s = np.sqrt(np.maximum(S[:rank], 0.0))

                B_new = U[:, :rank] * sqrt_s
                A_new = sqrt_s[:, None] * Vt[:rank, :]

                w_merged.append(B_new)
                w_merged.append(A_new)
                i += 2
                continue

        # Fallback: simple sample-weighted averaging for biases, BN stats, or un-paired tensors
        w_merged.append((w_a[i] * n_a + w_b[i] * n_b) / total_n)
        i += 1

    return w_merged
