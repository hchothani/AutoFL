"""
tests/test_prototype_compression.py
========================================================================================
Unit and Dispersion Tests for External Prototype-Level Compression
========================================================================================
Verifies:
  1. Orthogonality of projection matrix (W @ W.T = I).
  2. Prototype compression output dimensions and unit L2 normalization.
  3. De-biasing of the anisotropic ReLU cone (cosine separation restoration).
  4. Standard ResNet-18 and MobileNetV2 architecture preservation (no in-model bottlenecks).
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from utils.compression_utils import (
    get_orthogonal_projection_matrix,
    compress_prototype,
    find_classifier_module,
)
from models.resnet import resnet18
from models.simple_cnn import SimpleCNN
from models.mobilenet import MobileNet
from utils.peft_utils import wrap_with_lora
from omegaconf import OmegaConf


def test_projection_matrix_orthogonality():
    print("\n--- Test 1: Projection Matrix Orthogonality ---")
    d_in, d_out, seed = 512, 84, 42
    W = get_orthogonal_projection_matrix(d_in, d_out, seed)

    assert W.shape == (84, 512), f"Expected shape (84, 512), got {W.shape}"
    # Verify W @ W.T = I_84
    WWt = W @ W.T
    I = np.eye(84, dtype=np.float32)
    diff = np.max(np.abs(WWt - I))
    print(f"Max deviation from identity |W @ W.T - I|: {diff:.2e}")
    assert diff < 1e-5, f"W is not orthogonal! Max deviation: {diff}"

    # Verify cache idempotence
    W2 = get_orthogonal_projection_matrix(d_in, d_out, seed)
    assert W is W2, "Matrix should be retrieved from cache"

    # Verify identity for d_in <= d_out
    W_identity = get_orthogonal_projection_matrix(84, 84, seed)
    assert np.allclose(W_identity, np.eye(84)), "d_in <= d_out must return identity"
    print("✓ Projection matrix orthogonality passed.")


def test_compress_prototype_shape_and_norm():
    print("\n--- Test 2: Output Shape & Normalization ---")
    rng = np.random.RandomState(123)

    # 512-D prototype (ResNet-18)
    proto_512 = rng.randn(512)
    z_84 = compress_prototype(proto_512, target_dim=84, centering=True, seed=42)
    assert z_84.shape == (84,), f"Expected shape (84,), got {z_84.shape}"
    norm = np.linalg.norm(z_84)
    assert np.isclose(norm, 1.0, atol=1e-5), f"Expected unit norm, got {norm}"

    # 1280-D prototype (MobileNetV2)
    proto_1280 = rng.randn(1280)
    z_1280 = compress_prototype(proto_1280, target_dim=84, centering=True, seed=42)
    assert z_1280.shape == (84,), f"Expected shape (84,), got {z_1280.shape}"
    assert np.isclose(np.linalg.norm(z_1280), 1.0, atol=1e-5)

    # 84-D prototype (SimpleCNN)
    proto_84 = rng.randn(84)
    z_same = compress_prototype(proto_84, target_dim=84, centering=True, seed=42)
    assert z_same.shape == (84,), f"Expected shape (84,), got {z_same.shape}"
    assert np.isclose(np.linalg.norm(z_same), 1.0, atol=1e-5)
    print("✓ Shape and normalization test passed.")


def test_anisotropic_cone_debiasing():
    print("\n--- Test 3: Anisotropic ReLU Cone De-biasing ---")
    rng = np.random.RandomState(999)

    # Simulate deep ReLU feature vectors: positive coordinates with large common DC offset (+3.0)
    dc_offset = 3.0
    task1_feature = np.maximum(0, rng.randn(512) + dc_offset)
    task2_feature = np.maximum(0, rng.randn(512) + dc_offset)

    # Raw Cosine Distance before compression
    raw_sim = np.dot(task1_feature, task2_feature) / (
        np.linalg.norm(task1_feature) * np.linalg.norm(task2_feature)
    )
    raw_cos_dist = 1.0 - raw_sim
    print(f"Raw 512-D Cosine Distance (with DC cone bias):     {raw_cos_dist:.4f}")

    # Compressed Cosine Distance with centering + orthogonal projection
    z1 = compress_prototype(task1_feature, target_dim=84, centering=True, seed=42)
    z2 = compress_prototype(task2_feature, target_dim=84, centering=True, seed=42)
    compressed_cos_dist = 1.0 - np.dot(z1, z2)
    print(f"Compressed 84-D Cosine Distance (de-biased + projected): {compressed_cos_dist:.4f}")

    # Raw cosine distance is heavily suppressed by the common DC offset (< 0.15)
    # Centering + projection restores semantic separation (> 0.40)
    assert raw_cos_dist < 0.20, f"Expected small raw distance, got {raw_cos_dist}"
    assert compressed_cos_dist > 0.40, (
        f"Expected de-biased distance to expand, got {compressed_cos_dist}"
    )
    print(
        f"✓ Dynamic range expansion confirmed: {raw_cos_dist:.4f} -> {compressed_cos_dist:.4f}"
    )


def test_standard_model_architectures():
    print("\n--- Test 4: Standard Model Architecture Verification ---")
    # ResNet-18
    model_res = resnet18(num_classes=10, in_channels=3)
    assert not hasattr(model_res, "bottleneck"), "ResNet must not have a bottleneck attribute!"
    assert not hasattr(model_res, "ln_bottleneck"), "ResNet must not have ln_bottleneck attribute!"
    assert isinstance(model_res.fc, nn.Linear)
    assert model_res.fc.in_features == 512, f"Expected in_features 512, got {model_res.fc.in_features}"
    assert model_res.fc.out_features == 10

    dummy_x = torch.zeros(2, 3, 32, 32)
    out_res = model_res(dummy_x)
    assert out_res.shape == (2, 10), f"Expected output (2, 10), got {out_res.shape}"
    print("  ResNet-18 forward pass output shape: (2, 10) [Standard]")

    # MobileNetV2
    model_mob = MobileNet(num_classes=10, in_channels=3, version="v2")
    assert not hasattr(model_mob, "bottleneck"), "MobileNet must not have a bottleneck attribute!"
    out_mob = model_mob(dummy_x)
    assert out_mob.shape == (2, 10), f"Expected output (2, 10), got {out_mob.shape}"
    print("  MobileNetV2 forward pass output shape: (2, 10) [Standard]")
    print("✓ Model architectures successfully verified.")


def test_peft_hook_dimension_integrity():
    print("\n--- Test 5: PEFT LoRA Hook Dimension Integrity ---")
    cfg = OmegaConf.create({
        "lora": {"enabled": True, "r": 8, "alpha": 16, "dropout": 0.05}
    })

    test_models = [
        ("ResNet-18", resnet18(num_classes=10), 512),
        ("SimpleCNN", SimpleCNN(num_classes=10), 84),
        ("MobileNetV2", MobileNet(num_classes=10), 1280),
    ]

    for model_name, raw_model, expected_dim in test_models:
        peft_model = wrap_with_lora(raw_model, cfg)
        target_layer = find_classifier_module(peft_model)

        hooked_dim = []
        def make_hook(acc):
            def hook(mod, inp, out):
                acc.append(inp[0].shape[-1])
            return hook

        handle = target_layer.register_forward_hook(make_hook(hooked_dim))
        dummy = torch.randn(2, 3, 32, 32)
        _ = peft_model(dummy)
        handle.remove()

        actual_dim = hooked_dim[0]
        print(f"  {model_name}: Hooked {type(target_layer).__name__} -> Feature Dim: {actual_dim} (Expected: {expected_dim})")
        assert actual_dim == expected_dim, (
            f"PEFT hook hijacked! Expected {expected_dim}-D backbone feature, got {actual_dim}-D!"
        )
        assert actual_dim != 8, f"Critical Bug: {model_name} hooked LoRA internal 8-D bottleneck!"

    print("✓ PEFT hook dimension integrity verified: zero hijacking detected.")


if __name__ == "__main__":
    print("===================================================================")
    print(" Running External Prototype-Level Compression Verification Suite  ")
    print("===================================================================")
    test_projection_matrix_orthogonality()
    test_compress_prototype_shape_and_norm()
    test_anisotropic_cone_debiasing()
    test_standard_model_architectures()
    test_peft_hook_dimension_integrity()
    print("\n===================================================================")
    print(" ALL TESTS PASSED SUCCESSFULLY! ")
    print("===================================================================")
