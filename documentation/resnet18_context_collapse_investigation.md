# Incident Report: ResNet-18 Context Collapse & PEFT Hook Hijacking
**Date**: November 10, 2026  
**Component**: Client Prototype Extraction & Context Clustering  
**Affected Experiment**: `async_cifar10_resnet18` (`--thresholds 0.25 0.30 0.35 --time 180`)  
**Status**: Root Cause Identified & Verified  

---

## 1. Summary of Observed Symptoms

During evaluation of external prototype compression on `async_cifar10_resnet18`, all threshold sweeps produced failing results:

```
=========================================================================================================
Config                         | Dataset    | Model          | tau    | Accuracy   | Contexts  | BWT      | Status
---------------------------------------------------------------------------------------------------------
async_cifar10_resnet18         | cifar10    | resnet18       | 0.25   | 9.55%      | 1         | 0.1372   | SUCCESS
async_cifar10_resnet18         | cifar10    | resnet18       | 0.30   | 10.90%     | 2         | 0.1713   | SUCCESS
async_cifar10_resnet18         | cifar10    | resnet18       | 0.35   | 10.05%     | 1         | 0.0000   | SUCCESS
=========================================================================================================
```

### Key Anomalies from W&B Telemetry
1. **Accuracy Collapsed to Random Guess**: CIFAR-10 has 10 balanced classes. Overall test accuracy of **$9.55\%\text{--}10.90\%$** matches uniform random guessing.
2. **Context Collapse**: Only **1 context** was formed across all 5 continual learning phases (2 contexts for $\tau=0.30$). AutoFL failed to separate tasks.
3. **Severe Catastrophic Forgetting**:
   - `phase_0_accuracy`: **$0.0\%$** (completely forgotten)
   - `phase_3_accuracy`: **$0.0\%$** (completely forgotten)
   - `phase_2_accuracy`: **$3.0\%$**
   - `phase_1_accuracy`: **$28.0\%$**
   - `phase_4_accuracy`: **$16.0\%$**
   - Average across 5 phases: $(0 + 0.28 + 0.03 + 0 + 0.16) / 5 \approx \mathbf{9.4\%}$.
4. **Under-training**: Across the entire 180-second simulation, only **22 updates** completed across all 5 clients combined ($\approx 4$ updates total per phase).

---

## 2. Root Cause Analysis

### Root Cause 1: PEFT LoRA Hook Hijacking (Extracted 8-D Instead of 512-D)
In `clients/async_client.py` (and `clients/sync_client.py`), feature extraction was attached via:

```python
last_linear_layer = None
for module in self.model.modules():
    if isinstance(module, torch.nn.Linear):
        last_linear_layer = module
```

When ResNet-18 is wrapped with LoRA via `get_peft_model()`, the classification layer `self.fc` is replaced with a `peft.tuners.lora.layer.Linear` container holding three submodules:
1. `fc.base_layer` (`torch.nn.Linear(512, 10)`)
2. `fc.lora_A.default` (`torch.nn.Linear(512, 8)`)
3. `fc.lora_B.default` (`torch.nn.Linear(8, 10)`)

Because `self.model.modules()` traverses submodules in depth-first order, `isinstance(module, torch.nn.Linear)` evaluated `True` on `fc.lora_B.default` last.

#### The Consequence
* The forward hook was attached to **`lora_B.default`**, whose input tensor shape is **`torch.Size([batch, 8])`** (the 8-dimensional low-rank bottleneck), **not** the 512-dimensional output of ResNet-18's pooling layer.
* When `compress_prototype(raw_proto, target_dim=84)` received this 8-D vector, since $d_{\text{in}} = 8 \le 84$, it bypassed the 512 $\to$ 84 orthogonal projection matrix entirely.
* In LoRA, matrix $B$ is initialized to **all zeros** at startup.
* The extracted 8-D vectors across clients in all phases were near-zero/uninformative noise, keeping pairwise cosine distances artificially tiny ($d_{\cos} < \tau$).
* **The server merged all 5 phases into Context 0.**

### Root Cause 2: Context Collapse Caused Catastrophic Forgetting
* Because all clients were merged into Context 0, every phase repeatedly updated and overwrote **Adapter 0**.
* By Phase 4, Adapter 0 only retained knowledge of Phase 4 classes (classes 8 and 9).
* When evaluated on earlier phase test sets (e.g., Phase 0: classes 0 and 1; Phase 3: classes 6 and 7), accuracy was **$0.0\%$**.

### Root Cause 3: Under-training with `--time 180`
* In `config/experiments/cosine_exp/async_cifar10_resnet18.yaml`, `total_train_time` is configured to **900s** (180s per phase).
* Passing `--time 180` compressed the 5 continual learning phases into **36 seconds per phase**.
* With OMNeT++ latency simulation and Ray actor dispatch, only 22 client updates occurred in total. ResNet-18 (11 million parameters) cannot converge in 4 gradient steps per client.

---

## 3. Verified Diagnostic Proof

We executed an isolated diagnostic on the exact PEFT-wrapped ResNet-18 model:

```python
# Diagnostics: Inspecting hooked modules
Last linear layer hooked by naive loop:
  Name: base_model.model.fc.lora_B.default
  Module: Linear(in_features=8, out_features=10, bias=False)
  Hook input[0] shape: torch.Size([4, 8])   <-- BUG: 8-D bottleneck, not 512-D!

Directly targeting the outer classifier:
  fc_module type: <class 'peft.tuners.lora.layer.Linear'>
  fc hook input[0] shape: torch.Size([4, 512])  <-- CORRECT: True 512-D feature vector!
```

---

## 4. Planned Resolution

1. **Fix Classifier Hook Detection**:
   In `clients/async_client.py` and `clients/sync_client.py`, identify the outermost classifier head rather than the innermost submodule:
   ```python
   # Detect outermost classifier layer (works with vanilla PyTorch and PEFT LoRA)
   classifier_layer = None
   if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model"):
       # PEFT wrapped model
       inner = self.model.base_model.model
       classifier_layer = getattr(inner, "fc", None) or getattr(inner, "classifier", None)
   if classifier_layer is None:
       classifier_layer = getattr(self.model, "fc", None) or getattr(self.model, "classifier", None)
   if classifier_layer is None:
       # Fallback for SimpleCNN or custom architectures
       for module in self.model.children():
           if isinstance(module, torch.nn.Linear):
               classifier_layer = module
   ```
   This guarantees that `input[0]` receives the true **512-D** feature representation from the backbone.

2. **Restore Adequate Training Time**:
   Run with standard `total_train_time: 900` (or at least 600s) to allow ResNet-18 sufficient gradient updates across phases.

