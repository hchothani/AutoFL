# AutoFL: A Modular Federated Continual Learning Framework for Autonomous Vehicles

Built on top of PyTorch, Ray, and Flower, AutoFL is heavily optimized to eliminate GPU VRAM bottlenecks, simulate realistic data heterogeneity, and completely decouple neural network architectures from dataset constraints.

---

## ✨ Core Architecture & Features

AutoFL has recently undergone a massive architectural overhaul to support infinite scaling of models and datasets without codebase bloat.

* ** True Asynchronous Execution (Ray Actors):** Legacy Python thread pools have been deprecated. AutoFL uses a dedicated Ray Actor pool for asynchronous FL. Each simulated client is an isolated OS process with dedicated CPU/GPU fractions, enabling true parallel training without VRAM clashes or Global Interpreter Lock (GIL) bottlenecks.
* ** The Plugin Registry:** Neural networks (`models/`) and datasets (`workloads/`) are strictly decoupled. They operate as lazy-loaded plugins, meaning the framework only imports the exact math and data required for the specific run, drastically reducing memory overhead.
* ** The "Metadata Handshake":** You never have to manually hardcode `in_channels` or `num_classes` again. When a dataset plugin loads, it returns a metadata dictionary detailing its physical reality (e.g., MNIST: 1 channel, 10 classes). The framework dynamically reads this and chemically alters the neural network's architecture to match the data on the fly using Python signature inspection.
* ** Statistical Heterogeneity:** Native support for both standard **IID** and **Dirichlet Non-IID** data partitioning. Easily simulate pathological data skew across clients by tweaking a single `alpha` parameter.
* ** Algorithm Library:** Native support for FedAvg, AsyncFedAvg, WeightedAsyncFedAvg, AsyncFedED, and FedProto. Async strategies support routing either model parameters or gradients natively.

---

## 📁 Repository Structure

The codebase is organized by strict separation of concerns.

```text
AutoFL/
├── main.py                 # The singular entry point for all simulations.
├── algorithms/             # Server-side aggregation math (FedAvg, FedProto, etc.).
├── clients/                # Isolated client state and training loops (Sync & Async).
├── config/                 # OmegaConf YAML configurations & experiment blueprints.
├── models/                 # Pure PyTorch neural network blueprints (Plugins).
├── runners/                # Orchestration engines (Ray Async Pool & Flower Sync Engine).
├── utils/                  # Core framework engines (Model Factory, Data Partitioner, Logger).
├── workloads/              # Dataset fetchers, augmentations, and standardizers (Plugins).
└── tests/                  # Lightning-fast architectural unit tests.

## The Execution Workflow

1. **Configuration Load** : ```main.py``` reads ```config.yaml```
2. __Data Fetch__ : the ```data_loader.py``` asks the ```workloads``` router for the requested dataset. The dataset returns the raw data alongside physical metadata(channels, classes).

## Running With Hydra

Use Hydra group overrides to pick experiments and runtime modes:

```bash
# Default composed run (sync + simple_cnn + cifar10)
python main.py

# Select a predefined experiment
python main.py experiments=sync_cifar10
python main.py experiments=async_cifar10_gpu

# Ad-hoc overrides
python main.py runtime=async model=resnet18 dataset.workload=cifar100
```
 
