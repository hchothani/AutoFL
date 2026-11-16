# AutoFL Benchmark Suite: Memory Optimization & Configuration Sync

**Target:** 24-Run Continual Federated Learning (CFL) Benchmark Suite  
**Date:** November 2026  

---

## 1. Executive Summary & Root Cause Analysis

During multi-phase asynchronous federated learning runs with **ResNet-18**, training runs were crashing with a `torch.OutOfMemoryError: CUDA out of memory` after extended runtimes (~400s with 5 workers, ~600s with 3 workers at batch size 32, and ~750s at batch size 16).

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.
GPU 0 total capacity: 11.63 GiB, free: ~4 - 14 MiB.
```

### The Three Root Causes:
1. **Conv2d LoRA Architecture Overhead:** In ResNet-18, every `Conv2d` module is wrapped in a LoRA adapter, and the base model parameters are explicitly unfrozen for dual-training (`param.requires_grad = True`). During backpropagation, PyTorch must store forward-pass activation graphs for both the base convolution tensors and the LoRA tensors simultaneously.
2. **Cumulative Continual Learning Context Ensembles:** Every 10 seconds, `evaluate_global_model()` evaluates all seen experiences using an ensemble of all discovered contexts in the Adapter Bank. As the simulation crosses continual learning phases (Phase 0 -> Phase 1 -> Phase 2 -> Phase 3), the number of discovered context adapters grows, and the evaluation ensemble expands on the GPU.
3. **No Garbage Collection Between Worker Tasks:** In `SimulatedAsyncClient.fit()`, local `optimizer` instances and `.grad` tensor buffers were never explicitly deleted or freed (`set_to_none=True`). PyTorch's caching allocator held onto these unreleased memory blocks across hundreds of asynchronous vehicle dispatches, leading to gradual heap fragmentation.

---

## 2. Code Changes Applied

### A. Automatic Virtual Segment Allocation (`main.py`)
To prevent CUDA memory fragmentation across long 1200-second simulations, PyTorch is configured before CUDA initialization to allocate dynamic virtual memory segments:

```python
# main.py
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```

### B. Worker-Level Gradient & Optimizer Deallocation (`clients/async_client.py`)
In `SimulatedAsyncClient.fit()`, after local training completes:
- Gradients are explicitly reset to `None` (`self.model.zero_grad(set_to_none=True)`), which immediately frees the backing GPU memory blocks.
- The local `optimizer` state is deleted (`del optimizer`).
- `torch.cuda.empty_cache()` releases inactive cached blocks back to the CUDA memory pool.

```python
# clients/async_client.py (inside fit())
avg_loss = total_loss / max(num_batches, 1)

# Free local optimizer and gradients to prevent VRAM accumulation across async dispatches
self.model.zero_grad(set_to_none=True)
del optimizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

The same cleanup is executed after prototype extraction:
```python
new_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
self.model.zero_grad(set_to_none=True)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### C. Evaluation State Cleanup (`runners/async_runner.py`)
In `evaluate_global_model()`, the temporary ensemble model states are freed, and `torch.cuda.empty_cache()` is called in the `finally` block:

```python
# runners/async_runner.py (evaluate_global_model)
finally:
    model.load_state_dict(original_state)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

And in the main event loop after periodic evaluations:
```python
# runners/async_runner.py (run_async_simulation)
loss, metrics_dict = evaluate_global_model(
    global_model, eval_params_dict, global_test_loaders, device
)
del eval_params_dict
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

---

## 3. Configuration Standards for the 24 Benchmark Runs

All 24 benchmark runs across both `simple_cnn` and `resnet18` are now standardized to identical high-throughput settings:

| Parameter | Value (All 24 Runs) | Rationale |
| :--- | :---: | :--- |
| **`client.batch_size`** | `32` | Original baseline batch optimization dynamics |
| **`async.max_workers`** | `5` | High concurrency pool across Ray workers |
| **`client.num_gpus`** | `0.20` | 5 concurrent workers = 1.0 GPU |
| **`async.total_train_time`** | `900` s | Standard 15-minute budget (180s per phase) |
| **`context.threshold`** | `auto` | Universal regression law ($x_1, x_2, x_3, x_4 \to \tau^*$) |
| **`context.compression.enabled`** | `true` | 84-D orthogonal prototype projection |
| **`cl.num_experiences`** | `5` | 5-phase Continual Learning |

---

## 4. How to Sync & Run

### For Romir:
1. **Pull the latest code from `origin/Abhijaat`:**
   ```bash
   git pull origin Abhijaat
   ```
2. **Clear any lingering Ray daemons:**
   ```bash
   ray stop --force
   pkill -9 -f "ray::"
   ```
3. **Execute runs:**
   - Run all SimpleCNN benchmarks:
     ```bash
     python run_benchmark_24.py --filter simple_cnn
     ```
   - Run all ResNet-18 benchmarks:
     ```bash
     python run_benchmark_24.py --filter resnet18
     ```
   - Run any specific experiment:
     ```bash
     python main.py --config-path config/experiments/benchmark_24 --config-name resnet18_cifar100_10c_iid
     ```

