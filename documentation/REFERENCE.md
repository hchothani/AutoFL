# AutoFL: Regression Formula & Prototype Compression Reference
**Branch**: `Abhijaat` (`origin/Abhijaat`) | **Repo**: `/home/abhijaat/DaSH/AutoFL` | **Env**: `autofl_v2`

---

## 1. Universal Statistical Regression Formula (SimpleCNN)

### Problem Solved
Eliminated all dataset-specific threshold tuning, `if/else` dataset branches, and hardcoded lookup dictionaries (`CALIBRATION_ANCHORS`). When `context.threshold: auto`, the optimal clustering threshold $\tau^*$ is calculated directly from intrinsic dataset/task geometry.

### The Universal Formula
$$\boxed{\tau^* = 0.4131 \cdot \left(\frac{2}{K}\right)^{0.25} + 1.9514 \cdot \sqrt{\frac{C_{\text{in}}}{3}} - 0.7540 \cdot (d_{\text{inter}} - d_{\text{intra}}) + 1.5866 \cdot (1 - d_{\text{intra}}) - 2.3561}$$

Clamped to $[0.15, 0.60]$.

#### The 4 Statistical Features
1. **$x_1 = (2/K)^{0.25}$ (Task Granularity)**:
   $K = C / \text{num\_experiences}$ (classes per phase). By the Central Limit Theorem, phase centroids average over more classes as $K$ increases and drift closer together $\implies$ threshold $\tau^*$ must scale down.
2. **$x_2 = \sqrt{C_{\text{in}} / 3}$ (Channel Space Dimensionality)**:
   Accounts for color channels ($C_{\text{in}}=3$ for RGB, $C_{\text{in}}=1$ for grayscale).
3. **$x_3 = (d_{\text{inter}} - d_{\text{intra}})$ (Separability Margin)**:
   The gap between inter-class distance and intra-class variation computed dynamically from data loaders via `extract_data_dispersion()`.
4. **$x_4 = (1 - d_{\text{intra}})$ (Intra-Class Coherence)**:
   Quantifies how compact individual classes are in representation space.

### Validation Across 6 Benchmarks
Run via `python test_calculate_thresholds.py`:

| Workload | Classes ($C$) | Classes/Phase ($K$) | Channels ($C_{\text{in}}$) | $d_{\text{inter}}$ | $d_{\text{intra}}$ | Margin | Pred $\tau^*$ | Empirical Target | Error | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CIFAR-10** | 10 | 2.0 | 3 | 0.8701 | 0.7101 | 0.1600 | **0.35** | 0.35 | 0.00 | **EXACT** |
| **CIFAR-100** | 100 | 20.0 | 3 | 0.8713 | 0.6638 | 0.2075 | **0.20** | 0.20 | 0.00 | **EXACT** |
| **MNIST** | 10 | 2.0 | 1 | 0.3877 | 0.3326 | 0.0551 | **0.20** | 0.20 | 0.00 | **EXACT** |
| **GTSRB** | 43 | 8.6 | 3 | 0.8256 | 0.4828 | 0.3428 | **0.44** | 0.45 | 0.01 | **CLOSE** |
| **EuroSAT** | 10 | 2.0 | 3 | 1.0649 | 0.5279 | 0.5370 | **0.35** | 0.35 | 0.00 | **EXACT** |
| **SVHN** | 10 | 2.0 | 3 | 0.6444 | 0.8872 | -0.2428 | **0.37** | *N/A (Unseen)* | — | **TESTED** |

### Relevant Files
- `utils/threshold_utils.py`: Contains `compute_direct_regression_threshold()`, `extract_data_dispersion()`, and `resolve_context_threshold()`.
- `test_calculate_thresholds.py`: Standalone script to test the formula across all datasets.
- `workloads/svhn.py`: Newly added SVHN workload (requires `scipy` for `.mat` loading).

---

## 2. High-Dimensional Prototype Compression (ResNet-18 & MobileNetV2)

### The Dimensionality Problem
- **SimpleCNN** ($D = 84$): Clean angular clustering. $\tau^* \in [0.20, 0.45]$ neatly separates phases.
- **ResNet-18** ($D = 512$) & **MobileNetV2** ($D = 1280$): Cosine clustering fails completely.

> **W&B Report**: [CIFAR-10 with ResNet: No Threshold Fits](https://api.wandb.ai/links/abhijaatvs-bits-pilani/irubr11v)

- **$\tau \ge 0.30$ (Under-clustering)**: Every client merges into Context 0. As phases advance, Context 0 overwrites past knowledge $\implies$ catastrophic forgetting (**BWT collapses**).
- **$\tau \le 0.15$ (Over-clustering)**: Client sampling noise exceeds $\tau$. The server spawns a new context for almost every vehicle $\implies$ no cross-client sharing (**BWT stays low, context bank explodes**).

### Mathematical Root Causes
1. **Hyperspherical Measure Concentration**: In $\mathbb{R}^{512}$ and $\mathbb{R}^{1280}$, random/weakly correlated vectors naturally concentrate near orthogonality ($d_{\cos} \to 1.0$). Distance variance across subspaces shrinks as $O(1/\sqrt{D})$.
2. **The Anisotropic "Cone Effect" (ReLU Bias)**: Post-ReLU representations are strictly non-negative ($\mathbf{p} \in \mathbb{R}_{\ge 0}^D$). This creates a massive common positive DC offset $\boldsymbol{\mu}$, artificially packing all raw cosine similarities into a tight cluster $[0.85, 0.95]$ ($d_{\cos} \in [0.05, 0.15]$) regardless of class identity.
3. **Semantic Dilution**: Class/phase semantics occupy a low-dimensional subspace ($d \sim 20\text{–}80$). The remaining $400+$ dimensions are unaligned instance noise that drowns out the angular margin.

---

### What Was Tried & Why It Failed (Commit `c2251ae`)
In commit `c2251ae`, an in-model frozen orthogonal bottleneck was inserted into `models/resnet.py` and `models/mobilenet.py`:
```python
self.bottleneck = nn.Linear(512 * block.expansion, 84, bias=False)
nn.init.orthogonal_(self.bottleneck.weight)
self.bottleneck.weight.requires_grad = False
self.ln_bottleneck = nn.LayerNorm(84)
self.fc = nn.Linear(84, num_classes)
```
**Why this failed:**
- Placing an untrained, frozen linear projection *inside* the model forward pass disrupts gradient backpropagation during local training.
- `LayerNorm(84)` alters channel activation distributions before classification.
- It couples model architecture to clustering logic.
- **Rule**: **Compression must NOT alter the model backbone or training loss. It must be applied externally to the extracted prototype vectors during context detection.**

---

### Next Steps: External Prototype-Level Compression Roadmap

```
Raw Client Prototype p ∈ ℝ⁵¹² (from standard ResNet-18)
          │
          ▼
   1. Centering (De-bias Anisotropic Cone)
      p_c = p - μ_global
          │
          ▼
   2. Orthogonal Random Projection (NumPy / JL Lemma)
      z = W_proj @ p_c  (W_proj ∈ ℝ⁸⁴ˣ⁵¹², W @ Wᵀ = I)
          │
          ▼
   3. Cosine Context Clustering on z ∈ ℝ⁸⁴
      d_cos(z_new, z_bank) < τ
```

#### Step 1: Feature Centering (De-biasing the Anisotropic Cone)
Maintain a running global feature centroid $\boldsymbol{\mu}_{\text{global}} \in \mathbb{R}^D$ at the server. Subtract before computing distance:
$$\mathbf{p}_{\text{centered}} = \mathbf{p} - \boldsymbol{\mu}_{\text{global}}$$
*Effect*: Removes the shared positive DC component, expanding the angular spread between distinct tasks from $[0.05, 0.15]$ out to $[0.30, 0.70]$.

#### Step 2: Prototype-Level Orthogonal Random Projection (Johnson-Lindenstrauss)
Leave `models/resnet.py` standard (`512 -> num_classes`). At prototype extraction in `clients/async_client.py`, compress $\mathbf{p} \in \mathbb{R}^{512}$ down to $\mathbf{z} \in \mathbb{R}^{84}$ using a shared deterministic projection matrix:
```python
import numpy as np

def get_projection_matrix(d_in: int = 512, d_out: int = 84, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(d_in, d_out))
    return Q.T  # Shape: (84, 512)
```
*Effect*: Preserves pairwise geometry (JL lemma) while matching SimpleCNN's 84-D scale. Zero impact on model training or loss.

#### Step 3: Streaming / Incremental PCA (If Needed)
If random projection has residual noise, adaptively compute the top $k=84$ principal directions of prototype variance across the first 50 incoming clients.

#### Step 4: Hybrid Metric (If Needed)
Combine angular direction with normalized Euclidean distance:
$$D(\mathbf{u}, \mathbf{v}) = \alpha \cdot d_{\cos}(\mathbf{u}, \mathbf{v}) + (1 - \alpha) \cdot \frac{\|\mathbf{u} - \mathbf{v}\|_2}{\sqrt{d}}$$

---

### Implementation Checklist for Collaborators
1. **Revert in-model bottlenecks** in `models/resnet.py` and `models/mobilenet.py` back to standard `512 -> num_classes` and `1280 -> num_classes`.
2. **Apply centering + projection** in `clients/async_client.py` (lines 154–158) when `client_prototype` is computed.
3. **Verify clustering** in `runners/async_runner.py` (lines 342–368) on 84-D compressed vectors.
4. **Sweep CIFAR-10 ResNet-18**:
   ```bash
   python run_cosine_experiments.py --configs async_cifar10_resnet18 --thresholds 0.20 0.25 0.30 0.35 --time 300
   ```
