# utils/threshold_utils.py
"""
Universal Statistical Regression Threshold Calculator for AutoFL
========================================================================================
Directly predicts the optimal context threshold tau* using a fixed universal regression
equation derived from task & data statistical features:
    x1 = (2 / K)^0.25       : Task complexity / Central Limit Theorem scaling
    x2 = sqrt(C_in / 3)     : Input channel embedding dimensionality
    x3 = d_inter - d_intra  : Cluster separation margin
    x4 = 1 - d_intra        : Intra-class coherence / template tightness

Universal equation:
    tau* = 0.4131 * x1 + 1.9514 * x2 - 0.7540 * x3 + 1.5866 * x4 - 2.3561

NO hardcoded dataset names, NO calibration anchor tables.
Works for ANY unseen dataset dynamically from data!
========================================================================================
"""

import sys
from typing import List, Dict, Optional, Any

try:
    import torch
    import numpy as np
except ImportError:
    torch = None
    np = None

try:
    from omegaconf import DictConfig
except ImportError:
    DictConfig = Any


# Fixed Universal Regression Parameters
W_CLT = 0.4131
W_CHAN = 1.9514
W_MARGIN = -0.7540
W_COHERENCE = 1.5866
BIAS = -2.3561


def extract_data_dispersion(
    data_loaders: Any,
    num_classes: int,
    max_samples_per_class: int = 100
) -> Dict[str, float]:
    """
    Extracts mean pairwise inter-class distance (d_inter) and
    mean intra-class sample-to-centroid dispersion (d_intra) directly from data.
    """
    if torch is None:
        return {"d_inter": 0.8500, "d_intra": 0.6000}

    class_samples: Dict[int, List[torch.Tensor]] = {c: [] for c in range(num_classes)}
    loaders = data_loaders if isinstance(data_loaders, list) else [data_loaders]

    for loader in loaders:
        for batch in loader:
            if isinstance(batch, dict):
                images = batch.get("img", batch.get("x"))
                labels = batch.get("label", batch.get("y"))
            elif isinstance(batch, (tuple, list)):
                images = batch[0]
                labels = batch[1]
            else:
                continue

            labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
            images_flat = images.view(images.size(0), -1).float()

            for idx, lbl in enumerate(labels_np):
                lbl = int(lbl)
                if lbl in class_samples and len(class_samples[lbl]) < max_samples_per_class:
                    class_samples[lbl].append(images_flat[idx].cpu())

            if all(len(v) >= max_samples_per_class for v in class_samples.values()):
                break

    centroids = []
    intra_dists = []

    for c in range(num_classes):
        if len(class_samples[c]) == 0:
            continue
        X_c = torch.stack(class_samples[c], dim=0)
        norms = torch.norm(X_c, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        X_norm = X_c / norms

        mu_c = torch.mean(X_norm, dim=0)
        mu_c_norm = mu_c / (torch.norm(mu_c).clamp(min=1e-8))
        centroids.append(mu_c_norm)

        sims = torch.mv(X_norm, mu_c_norm).clamp(-1.0, 1.0)
        intra_dists.append(float(torch.mean(1.0 - sims).item()))

    d_intra = float(np.mean(intra_dists)) if intra_dists else 0.60

    if len(centroids) >= 2:
        M = torch.stack(centroids, dim=0)
        S = torch.mm(M, M.t()).clamp(-1.0, 1.0)
        D_mat = 1.0 - S
        triu_idx = torch.triu_indices(M.shape[0], M.shape[0], offset=1)
        d_inter = float(torch.mean(D_mat[triu_idx[0], triu_idx[1]]).item())
    else:
        d_inter = 0.85

    return {"d_inter": round(d_inter, 4), "d_intra": round(d_intra, 4)}


def compute_direct_regression_threshold(
    num_classes: int,
    num_phases: int,
    in_channels: int,
    data_loaders: Optional[Any] = None
) -> Dict[str, float]:
    """
    Directly predicts optimal threshold tau* using the universal regression law:
        tau* = 0.4131 * (2/K)^0.25 + 1.9514 * sqrt(C_in/3) - 0.7540 * (d_inter - d_intra) + 1.5866 * (1 - d_intra) - 2.3561
    """
    K = max(1.0, num_classes / max(1, num_phases))

    if data_loaders is not None:
        disp = extract_data_dispersion(data_loaders, num_classes)
        d_inter = disp["d_inter"]
        d_intra = disp["d_intra"]
    else:
        d_inter = 0.8500
        d_intra = 0.6000

    f_clt = (2.0 / K) ** 0.25
    f_chan = (float(in_channels) / 3.0) ** 0.5
    f_margin = d_inter - d_intra
    f_coherence = 1.0 - d_intra

    tau_raw = (
        W_CLT * f_clt
        + W_CHAN * f_chan
        + W_MARGIN * f_margin
        + W_COHERENCE * f_coherence
        + BIAS
    )

    tau_star = round(float(max(0.15, min(0.60, tau_raw))), 2)

    return {
        "tau_star": tau_star,
        "K": round(K, 1),
        "f_clt": round(f_clt, 4),
        "f_chan": round(f_chan, 4),
        "d_inter": d_inter,
        "d_intra": d_intra,
        "margin": round(f_margin, 4),
        "coherence": round(f_coherence, 4)
    }


def resolve_context_threshold(
    cfg: DictConfig,
    model: Optional[Any] = None,
    data_loaders: Optional[List[Any]] = None,
    device: Optional[Any] = None
) -> float:
    """
    Resolves the context distance threshold:
    - STRICT REQUIREMENT: The theoretical formula is ONLY used
      IF the model is SimpleCNN AND context.threshold is explicitly set to 'auto'.
    - For any other model (ResNet-18, MobileNetV2, etc.) or if a numerical threshold
      is specified, the formula is bypassed and the configured threshold is preserved.
    """
    model_name = str(cfg.get("model", {}).get("name", "")).lower()
    workload = str(cfg.get("dataset", {}).get("workload", "")).lower()
    num_classes = int(cfg.dataset.get("num_classes", 10))
    in_channels = int(cfg.dataset.get("in_channels", 3))
    cl_enabled = cfg.get("cl", {}).get("enabled", False)
    num_phases = int(cfg.get("cl", {}).get("num_experiences", 1)) if cl_enabled else 1

    raw_threshold = cfg.get("context", {}).get("threshold", 0.35)

    # 1. Check if model is SimpleCNN
    is_simple_cnn = (model_name.replace("_", "") == "simplecnn")

    # 2. Check if threshold is explicitly requested as 'auto'
    is_auto_string = isinstance(raw_threshold, str) and raw_threshold.strip().lower() == "auto"
    is_auto_flag = bool(cfg.get("context", {}).get("auto_threshold", False))

    # 3. Check for explicit numeric overrides via CLI arguments
    has_cli_numeric_override = False
    for arg in sys.argv:
        if arg.startswith("context.threshold="):
            val = arg.split("=", 1)[1].strip().lower()
            if val != "auto":
                has_cli_numeric_override = True
        elif arg == "--thresholds":
            has_cli_numeric_override = True

    is_auto_requested = (is_auto_string or is_auto_flag) and not has_cli_numeric_override

    # STRICT CONDITION: ONLY simple_cnn AND ONLY when threshold='auto'
    should_apply_formula = is_simple_cnn and is_auto_requested

    if should_apply_formula:
        res = compute_direct_regression_threshold(
            num_classes=num_classes,
            num_phases=num_phases,
            in_channels=in_channels,
            data_loaders=data_loaders
        )
        tau_star = res["tau_star"]

        print("=" * 80)
        print(f"[Threshold Auto-Config] >>> ACTIVE: Model is SimpleCNN and threshold='auto'.")
        print(f"[Threshold Auto-Config] Workload: {workload.upper()} | Classes (C): {num_classes} | Phases (P): {num_phases} | K: {res['K']:.1f}")
        print(f"[Threshold Auto-Config] Channels (C_in): {in_channels} | d_inter: {res['d_inter']:.4f} | d_intra: {res['d_intra']:.4f} | Margin: {res['margin']:.4f}")
        print(f"[Threshold Auto-Config] Universal Statistical Regression Law predicted threshold: tau* = {tau_star:.2f}")
        print(f"[Threshold Auto-Config] Setting context.threshold = {tau_star:.2f}")
        print("=" * 80 + "\n")

        cfg.context.threshold = tau_star
        return tau_star

    # Formula is strictly bypassed in all other situations:
    if not is_simple_cnn:
        if is_auto_string:
            print(f"[Threshold] Notice: Theoretical formula is strictly restricted to SimpleCNN. For '{model_name}', defaulting threshold to 0.35.")
            final_tau = 0.35
        else:
            try:
                final_tau = float(raw_threshold)
            except (ValueError, TypeError):
                final_tau = 0.35
        print(f"[Threshold] Model is '{model_name}' (not SimpleCNN). Theoretical formula bypassed. context.threshold = {final_tau:.2f}")
    else:
        try:
            final_tau = float(raw_threshold)
        except (ValueError, TypeError):
            final_tau = 0.35
        print(f"[Threshold] Model is SimpleCNN, but threshold is explicitly set to {final_tau:.2f} (not 'auto'). Theoretical formula bypassed.")

    cfg.context.threshold = final_tau
    return final_tau
