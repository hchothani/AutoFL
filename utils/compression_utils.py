"""
utils/compression_utils.py
========================================================================================
Modular Prototype Compression & Dimensionality Reduction for AutoFL Context Detection
----------------------------------------------------------------------------------------
Compresses high-dimensional feature representations (e.g., 512-D for ResNet-18,
1280-D for MobileNetV2) down to compact dimensions (e.g., 84-D for SimpleCNN scale).

Features:
  1. Anisotropic Cone Decoupling: Coordinate centering (p - mean(p)) to remove the
     shared positive DC offset caused by ReLU activations across classes.
  2. Johnson-Lindenstrauss Orthogonal Random Projection: Deterministic QR-based
     orthogonal projection (W @ W^T = I) to preserve pairwise geometry with bounded distortion.
  3. L2 Normalization: Produces unit vectors on the hypersphere for stable cosine clustering.
  4. Zero Model Intrusion: Applied purely at prototype extraction and context clustering,
     leaving neural network weights, backpropagation, and loss functions standard.
========================================================================================
"""

from typing import List, Union, Dict, Tuple
import numpy as np

# Cache for projection matrices: (d_in, d_out, seed) -> np.ndarray
_PROJECTION_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def get_orthogonal_projection_matrix(
    d_in: int,
    d_out: int = 84,
    seed: int = 42
) -> np.ndarray:
    """
    Generates or retrieves a deterministic orthogonal projection matrix W in R^{d_out x d_in}
    using QR decomposition of a standard normal random matrix.
    
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
    Compresses an incoming prototype vector to target_dim using centering and
    deterministic orthogonal random projection.

    Args:
        prototype: 1D array or list of features (e.g. 512-D for ResNet, 1280-D for MobileNet).
        target_dim: Desired target dimensionality (default: 84 to match SimpleCNN scale).
        centering: If True, subtracts the coordinate mean to eliminate the anisotropic ReLU cone.
        seed: Random seed for deterministic projection matrix generation.

    Returns:
        1D unit-normalized numpy array in R^{target_dim} (or R^{d_in} if d_in <= target_dim).
    """
    p = np.asarray(prototype, dtype=np.float32).ravel()
    d_in = p.shape[0]

    # 1. Feature Centering (removes the common DC bias from ReLU activations)
    if centering:
        p = p - np.mean(p)

    # 2. Orthogonal Random Projection (if input dimension exceeds target_dim)
    if d_in > target_dim:
        W = get_orthogonal_projection_matrix(d_in=d_in, d_out=target_dim, seed=seed)
        p = W @ p

    # 3. L2 Normalization (ensures unit norm for clean cosine distance metrics)
    norm = np.linalg.norm(p)
    if norm > 1e-12:
        p = p / norm

    return p
