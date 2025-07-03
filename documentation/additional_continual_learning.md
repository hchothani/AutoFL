
## continual learning strategies

### 1. elastic weight consolidation (ewc)
protects important weights from being overwritten when learning new tasks.

```yaml
cl:
  strategy: ewc
  ewc_lambda: 0.4  # importance weight (0.1-10.0)
  ewc_decay_factor: null  # optional decay over time
  ewc_keep_importance_data: false
```

### 2. experience replay
stores samples from previous tasks to prevent forgetting.

```yaml
cl:
  strategy: replay
  replay_mem_size: 200  # buffer size
  replay_selection: random  # random, herding, closest_to_mean
```

### 3. hybrid (ewc + replay)
combines both ewc and experience replay for better performance.

```yaml
cl:
  strategy: hybrid
  ewc_lambda: 0.4
  replay_mem_size: 300
```

## benchmark datasets

### permuted mnist
classic cl benchmark where pixel permutations create different tasks.

```yaml
dataset:
  workload: permuted_mnist
  batch_size: 64
  num_classes: 10

cl:
  num_experiences: 5  # number of permutations
```

**characteristics:**
- task 0: identity (no permutation)
- tasks 1-n: random pixel permutations
- same classes (0-9) across all tasks
- good for testing catastrophic forgetting

### split cifar10
class-incremental learning where classes are split across tasks.

```yaml
dataset:
  workload: split_cifar10
  batch_size: 64
  num_classes: 10  # total classes

cl:
  num_experiences: 5  # splits 10 classes into 5 tasks (2 classes each)
```

**characteristics:**
- classes 0-1 in task 0, classes 2-3 in task 1, etc.
- different classes per task (true class-incremental)
- tests ability to learn new categories

## model architectures

### mobilenet
efficient mobile-optimized cnn architectures.

```yaml
model:
  name: mobilenet
  version: v2  # v2, v3_small, v3_large
  pretrained: false  # use imagenet weights
  num_classes: 10
```

## non-iid data distribution

enhanced support for heterogeneous federated learning scenarios.

```yaml
dataset:
  split: niid  # iid, niid, niid_label, niid_quantity
  niid:
    alpha: 0.5  # dirichlet concentration (lower = more heterogeneous)
    min_samples: 10  # minimum samples per client
    
    # label heterogeneity
    classes_per_client: 2  # for niid_label
    
    # quantity heterogeneity  
    quantity_skew: 0.5  # sample count variation
```

**distribution types:**
- **iid**: uniform random distribution
- **niid**: dirichlet distribution (configurable heterogeneity)
- **niid_label**: different classes per client
- **niid_quantity**: different sample counts per client

## federated learning strategies

beyond fedavg, support for advanced fl algorithms.

```yaml
server:
  strategy: fedprox  # fedavg, fedprox, scaffold, fednova, fedopt
  
  # strategy-specific parameters
  fedprox:
    mu: 0.01  # proximal term weight
    
  scaffold:
    eta_l: 1.0  # local lr multiplier
    eta_g: 1.0  # global lr multiplier
```

## example configurations

### 1. permuted mnist with ewc
```bash
python mclmain.py --config-path config/experiments --config-name permuted_mnist_ewc
```

### 2. split cifar10 with replay
```bash
python mclmain.py --config-path config/experiments --config-name split_cifar10_replay
```

### 3. hybrid strategy with mobilenet
```bash
python mclmain.py --config-path config/experiments --config-name cifar10_hybrid_gpu
```

## creating custom experiments

### minimal ewc experiment
```yaml
# my_ewc_experiment.yaml
dataset:
  workload: cifar10
  batch_size: 64

model:
  name: mobilenet
  version: v2

cl:
  strategy: ewc
  num_experiences: 4
  ewc_lambda: 0.4

server:
  num_clients: 3
  num_rounds: 5
```

### advanced non-iid setup
```yaml
dataset:
  workload: split_cifar10
  split: niid
  niid:
    alpha: 0.1  # high heterogeneity
    classes_per_client: 1  # one class per client initially

cl:
  strategy: hybrid
  ewc_lambda: 0.4
  replay_mem_size: 500

server:
  strategy: fedprox
  fedprox:
    mu: 0.01
```