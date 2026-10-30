# High-Dimensional Prototype Compression & Context Clustering in AutoFL
**Status & Investigation Report for Collaborators**  
**Git Branch**: `Abhijaat` (`origin/Abhijaat`)  
**Repository**: `/home/abhijaat/DaSH/AutoFL`  
**Latest Reference Commit**: `f4a5cbe` (*Added regression for SimpleCNN formula, and a new dataset SVHN*)  

---

## 1. Executive Summary & Problem Context

AutoFL implements **Asynchronous Federated Continual Learning (FCL)** for vehicular edge environments (RSUs interacting with transient vehicles). 

### How Context Detection Works
1. **Local Training**: Clients train locally on their current phase/experience using LoRA adapters on top of a shared base model.
2. **Prototype Extraction**: At the end of local training, each client extracts a **Data Signature / Prototype** $\mathbf{p} \in \mathbb{R}^D$ (the average feature representation across local data batches immediately before the classification layer).
3. **Server Context Clustering**: When a vehicle uploads its update to the RSU, the server compares the incoming prototype $\mathbf{p}$ against all existing cluster prototypes in the `Context Bank` using **cosine distance**:
   $$d_{\cos}(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
   - **If $\min(d_{\cos}) < \tau$**: Assign update to the existing context cluster (updating that context's LoRA adapter).
   - **If $\min(d_{\cos}) \ge \tau$**: Spawn a **NEW context cluster** (initializing a new LoRA adapter for this discovered task/phase).
4. **Target Metric**: **Backward Transfer (BWT)** and Final Average Accuracy across continual learning phases.

---

## 2. The Core Issue: Why SimpleCNN Works but ResNet-18 & MobileNetV2 Fail

### A. SimpleCNN (Feature Dimension $D = 84$)
- Feature representations are compact and constrained.
- Different class mixtures / phases map to distinctly separated angular sectors.
- Intra-class variance remains tightly bounded ($d_{\text{intra}} \approx 0.15$), while inter-phase separation is wide ($d_{\text{inter}} \approx 0.50$).
- **Result**: Context clustering functions reliably. Optimal thresholds $\tau^* \in [0.20, 0.45]$ cleanly separate tasks without human tuning.

### B. ResNet-18 ($D = 512$) and MobileNetV2 ($D = 1280$)
When moving to larger architectures, context clustering completely breaks down:

> **W&B Empirical Benchmark Report**:  
> [CIFAR-10 with ResNet: No Threshold Fits (Weights & Biases)](https://api.wandb.ai/links/abhijaatvs-bits-pilani/irubr11v)

#### The Observed Dilemma
* **$\tau$ too high ($\ge 0.30$)**: Under-clustering / context collapse. Every client update merges into Context 0. As clients cycle through new continual learning phases, Context 0 overwrites past knowledge, causing catastrophic forgetting (**BWT collapses**).
* **$\tau$ too low ($\le 0.15$)**: Over-clustering / context explosion. Client-to-client sampling noise exceeds $\tau$, causing the server to generate a new context for almost every single vehicle transaction. Clients never collaborate on shared contexts (**BWT remains low, compute/memory blows up**).
* **Conclusion**: There is **no viable threshold $\tau$** in the original feature space for CIFAR-10 on ResNet-18.

---

## 3. Mathematical & Geometric Root Causes

1. **Curse of Dimensionality (Measure Concentration in Hyperspherical Space)**:
   In high dimensions ($\mathbb{R}^{512}$ or $\mathbb{R}^{1280}$), random or weakly correlated vectors naturally concentrate around orthogonality ($d_{\cos} \to 1.0$). The variance of cosine distances across random subspaces shrinks proportionally to $1/\sqrt{D}$.
2. **The Anisotropic "Cone Effect" (ReLU Feature Bias)**:
   In deep networks, post-ReLU feature vectors are strictly non-negative ($\mathbf{p} \in \mathbb{R}_{\ge 0}^D$). This introduces a large shared common-mode DC component across all tasks, squeezing raw cosine angles into a very narrow cone where cosine similarities hover unnaturally around $0.85\text{–}0.95$ ($d_{\cos} \in [0.05, 0.15]$), regardless of true semantic differences.
3. **Semantic Dilution**:
   Only a small subset of the 512 or 1280 feature dimensions encode task-discriminative information (class boundaries). The remaining hundreds of dimensions capture sample-specific details, lighting, or background noise. When computing a single cosine distance over all 512 dimensions, the semantic margin $(d_{\text{inter}} - d_{\text{intra}})$ is drowned out by noise.

---

## 4. What Was Previously Attempted & Why It Did Not Work

### Previous Attempt: In-Model Frozen Orthogonal Bottleneck (Commit `c2251ae`)
In commit `c2251ae`, an architectural change was added directly to `models/resnet.py` and `models/mobilenet.py`:
```python
# models/resnet.py lines 43-47
self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
self.bottleneck = nn.Linear(512 * block.expansion, 84, bias=False)
nn.init.orthogonal_(self.bottleneck.weight)
self.bottleneck.weight.requires_grad = False
self.ln_bottleneck = nn.LayerNorm(84)
self.fc = nn.Linear(84, num_classes)
```

### Why This Architectural Approach Failed
1. **Disrupted Optimization**: Placing an untrained, frozen linear projection directly inside the forward pass of the model severely restricts gradient backpropagation during local client training.
2. **LayerNorm Artifacts**: LayerNorm normalizes across the feature dimension before classification, distorting the relative magnitudes of class activations.
3. **Architectural Coupling**: Modifying the neural network backbone couples the model architecture to the context clustering logic, preventing clean reuse of standard pretrained models.
4. **Core Lesson**: **Compression must NOT alter the model backbone or training loss. It must be applied externally to the extracted prototype vectors during context detection.**

---

## 5. What We Are Doing Next: Prototype Feature-Space Compression Roadmap

We are implementing prototype compression entirely in NumPy/SciPy at the extraction and server clustering boundary, leaving `ResNet-18` and `MobileNetV2` model architectures in their standard form (`512 -> num_classes` and `1280 -> num_classes`).

```
                    ┌───────────────────────────────────────────────┐
                    │ Client: Standard Forward Pass on ResNet-18    │
                    │ Feature Layer: 512-D (untouched architecture) │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                       Raw Client Prototype p ∈ ℝ⁵¹²
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Step 1: Centering (De-bias Anisotropic Cone)  │
                    │         p_c = p - μ_running                   │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Step 2: Orthogonal Random Projection (JL)     │
                    │         z = W_proj @ p_c  (W_proj ∈ ℝ⁸⁴ˣ⁵¹²)  │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                       Compressed Prototype z ∈ ℝ⁸⁴
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Step 3: Cosine Context Clustering             │
                    │         d_cos(z_new, z_bank) < τ              │
                    └───────────────────────────────────────────────┘
```

### Method 1: Feature Centering (Cone Decoupling)
- **Mechanism**: Track a global running centroid $\boldsymbol{\mu}_{\text{global}} \in \mathbb{R}^D$ on the server. Center incoming prototypes:
  $$\mathbf{p}_{\text{centered}} = \mathbf{p} - \boldsymbol{\mu}_{\text{global}}$$
- **Why It Works**: Eliminates the dominant shared DC component created by ReLU activations, expanding the angular divergence between distinct tasks from $[0.05, 0.15]$ out to $[0.30, 0.70]$.

### Method 2: Pure Prototype-Level Orthogonal Random Projection (Johnson-Lindenstrauss)
- **Mechanism**: Generate a deterministic, fixed projection matrix $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{d \times D}$ ($d=84, D=512$) using a fixed seed:
  ```python
  rng = np.random.RandomState(42)
  Q, _ = np.linalg.qr(rng.randn(D, d))
  W_proj = Q.T  # Shape: (84, 512)
  ```
- **Why It Works**: According to the Johnson-Lindenstrauss lemma, orthogonal projection preserves pairwise geometry with bounded distortion while reducing the dimensionality to match SimpleCNN ($84$-D).
- **Zero Impact on Model Training**: Model weights, LoRA adapters, and backpropagation remain 100% standard.

### Method 3: Streaming Principal Component Analysis (Online PCA / SVD)
- If static random projection has residual noise, adaptively compute the top $k=84$ principal directions that capture maximum inter-task variance across received prototypes.

### Method 4: Direction + Normalized Magnitude Hybrid Metric
- If angular distance alone remains noisy, measure a joint metric:
  $$D(\mathbf{u}, \mathbf{v}) = \alpha \cdot d_{\cos}(\mathbf{u}, \mathbf{v}) + (1 - \alpha) \cdot \frac{\|\mathbf{u} - \mathbf{v}\|_2}{\sqrt{d}}$$

---

## 6. Project Files & Directory Guide

| Component | File Path | Role in Investigation |
| :--- | :--- | :--- |
| **Active Git Branch** | `Abhijaat` | Branch containing all current progress and configs. |
| **Model Definitions** | `models/resnet.py`<br>`models/mobilenet.py`<br>`models/simple_cnn.py` | Model architectures. The in-model bottleneck in `resnet.py` and `mobilenet.py` should be reverted to standard. |
| **Client Prototype Extraction** | `clients/async_client.py`<br>`clients/sync_client.py` | Hooks the penultimate linear layer (`lines 116-158`) and computes `feature_sum / feature_count`. This is where centering / projection can be applied. |
| **Server Context Clustering** | `runners/async_runner.py` | Lines `335-375`: Computes cosine distance against `server_context_prototypes` and decides whether to assign or create a context. |
| **Threshold Calculation** | `utils/threshold_utils.py` | Implements the universal statistical regression law predicting $\tau^*$ for `SimpleCNN`. |
| **Experiment Configurations** | `config/experiments/cosine_exp/` | YAML configs for all models and datasets (`async_cifar10_resnet18.yaml`, `async_cifar10_simple_cnn.yaml`, etc.). |
| **Batch Experiment Runner** | `run_cosine_experiments.py` | Automates threshold sweeps, multi-run executions, and CSV logging into `outputs/cosine_sweep_summary.csv`. |
| **Verification & Inspection** | `test_calculate_thresholds.py` | Lightweight script to verify data dispersion and threshold predictions across datasets. |
