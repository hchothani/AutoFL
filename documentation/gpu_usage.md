## Overview
can automatically detect and use available GPUs for training. The system uses Ray for distributed computing, which handles GPU allocation across federated clients.

## GPU Detection
- Models and data are automatically moved to GPU when available
- Device selection: `torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`
- Each client can use a fractional GPU allocation

## Configuration

### GPU Allocation per Client
In your experiment config, set the GPU fraction:

```yaml
client:
  num_gpus: 0.2  # 20% of GPU per client (5 clients can share 1 GPU)
  # or
  num_gpus: 1.0  # Full GPU per client (Ray will manage sharing)
```


## Example Configurations

### Fast GPU Training (CIFAR10)
```bash
python mclmain.py --config-path config/experiments --config-name cifar10_gpu_fast
```