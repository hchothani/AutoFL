# SOTA Federated Continual Learning Algorithms

This directory contains configuration files for state-of-the-art (SOTA) federated continual learning algorithms integrated into the AutoFL framework.

## Available Algorithms

### 1. **FedWeIT** - Federated Weighted Inter-client Transfer
- **Paper**: [Federated Continual Learning with Weighted Inter-client Transfer](https://proceedings.mlr.press/v139/yoon21b.html) (ICML 2021)
- **Config**: `fedweit_cifar10.yaml`
- **Key Features**: 
  - Task-adaptive parameters with learnable masks
  - Inter-client knowledge transfer via attention weights
  - Custom server aggregation for masked parameters
- **Parameters**:
  - `sparsity`: Target sparsity for task-adaptive parameters (0-1)
  - `l1_lambda`: L1 regularization for sparsity
  - `l2_lambda`: Retroactive update regularization

### 2. **PLoRA** - Parameter-Efficient Low-Rank Adaptation for FL
- **Paper**: Based on LoRA adaptation for federated learning
- **Config**: `plora_cifar10.yaml`
- **Key Features**:
  - Parameter-efficient fine-tuning using low-rank adaptation
  - Reduced communication overhead
  - Standard FedAvg aggregation
- **Parameters**:
  - `rank`: LoRA rank (dimensionality of adaptation)
  - `alpha`: LoRA alpha scaling factor

### 3. **FedCPrompt** - Federated Class-Incremental Learning via Prompting
- **Paper**: Federated Class-Incremental Learning via Prompting
- **Config**: `fedcprompt_cifar10.yaml`
- **Key Features**:
  - Prompt-based continual learning
  - Class-incremental learning support
  - Standard FedAvg aggregation
- **Parameters**:
  - `prompt_length`: Length of prompt tokens
  - `prompt_lr`: Learning rate for prompt optimization

### 4. **Other Available Algorithms**
All algorithms in `/algorithms/` are integrated:
- **FedET**: Federated Learning with Elastic Transformers
- **FedGEM**: Federated Generative Episodic Memory
- **FedMA**: Federated Model Aggregation
- **FedProto**: Federated Prototype Learning
- **FedRCIL**: Federated Rehearsal-based Class-Incremental Learning
- **FedRep**: Federated Representation Learning
- **SacFL**: Scalable Federated Continual Learning
- **STAMP**: Sparse Training and Model Pruning

## Configuration Structure

Each SOTA algorithm config follows this structure:

```yaml
# Dataset configuration
dataset:
  workload: cifar10
  batch_size: 32
  num_classes: 10

# Model configuration  
model:
  name: simple_cnn
  num_classes: 10

# Continual learning configuration
cl:
  strategy: <algorithm_name>  # e.g., fedweit, plora, fedcprompt
  num_experiences: 5
  split: random

# Algorithm-specific parameters
<algorithm_name>:
  param1: value1
  param2: value2

# Training configuration
training:
  learning_rate: 0.001
  epochs: 3
  optimizer: adam

# Server configuration
server:
  strategy: <server_strategy>  # fedavg or custom (e.g., fedweit)
  num_rounds: 10
  num_clients: 3
  fraction_fit: 1.0
  fraction_eval: 1.0
  min_fit: 3
  min_eval: 3

# Client configuration
client:
  num_cpus: 4
  num_gpus: 0.0
  epochs: 3
  falloff: 0.0
  exp_epochs: null

# Weights & Biases logging
wb:
  project: autofl-sota-testing
  name: <experiment_name>
```

## How to Run Experiments

### Method 1: Using Predefined SOTA Configs

```bash
# Run FedWeIT on CIFAR10
python mclmain.py --config-path config/sota --config-name fedweit_cifar10

# Run PLoRA on CIFAR10
python mclmain.py --config-path config/sota --config-name plora_cifar10

# Run FedCPrompt on CIFAR10
python mclmain.py --config-path config/sota --config-name fedcprompt_cifar10
```

### Method 2: Using Hydra Overrides

```bash
# Run any algorithm with custom parameters
python mclmain.py cl.strategy=fedweit fedweit.sparsity=0.7

# Run PLoRA with different rank
python mclmain.py cl.strategy=plora plora.rank=8 plora.alpha=2.0

# Run with GPU acceleration
python mclmain.py cl.strategy=fedweit client.num_gpus=1.0
```

### Method 3: Create Custom Configs

1. Copy an existing config from `config/sota/`
2. Modify the algorithm and parameters
3. Save with a new name
4. Run with `--config-name your_config`

## Server Strategy Selection

The framework automatically selects the appropriate server strategy:

- **Standard FL Strategies**: `fedavg`, `fedprox`, `fedopt`
- **Custom SOTA Strategies**: 
  - `fedweit` → Custom FedWeIT aggregation (currently uses FedAvg as fallback)
  - Most others → Standard FedAvg aggregation

## Example: Custom FedWeIT Experiment

```yaml
# config/sota/my_fedweit_experiment.yaml
dataset:
  workload: cifar100
  batch_size: 64

model:
  name: resnet
  num_classes: 100

cl:
  strategy: fedweit
  num_experiences: 10

fedweit:
  sparsity: 0.3
  l1_lambda: 0.05
  l2_lambda: 50.0

server:
  strategy: fedweit
  num_rounds: 20
  num_clients: 5

client:
  num_gpus: 1.0
  epochs: 5

wb:
  project: my-sota-experiments
  name: fedweit_cifar100_resnet_custom
```

Run with:
```bash
python mclmain.py --config-path config/sota --config-name my_fedweit_experiment
```

## Testing Your Setup

Verify the integration works:

```bash
# Test all SOTA algorithms
python tests/test_sota_integration_full.py

# Test specific algorithm
python tests/test_fedweit_integration.py
python tests/test_plora_minimal.py

# Test server strategy selection
python tests/test_server_strategy_selection.py
```

## Adding New Algorithms

1. **Implement the algorithm** in `/algorithms/your_algorithm.py`
2. **Add config support** in `config/cl/default.yaml`
3. **Update strategy factory** in `clutils/clstrat.py`
4. **Add server logic** (if needed) in `mclserver.py`
5. **Create experiment configs** in `config/sota/`
6. **Add tests** in `/tests/`

## Notes

- **Memory Management**: Large models may require GPU acceleration (`client.num_gpus > 0`)
- **Communication**: SOTA algorithms may have different communication patterns
- **Convergence**: Different algorithms may require different hyperparameters for optimal performance
- **Custom Server Logic**: Some algorithms (like FedWeIT) benefit from custom server aggregation

## Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Errors**: Set `client.num_gpus=0.0` for CPU-only execution
3. **Memory Issues**: Reduce batch size or model complexity
4. **Config Errors**: Check that all required fields are present

### Getting Help:

- Check existing test files for examples
- Refer to algorithm papers for hyperparameter guidance
- Use `python mclmain.py --help` for Hydra options 