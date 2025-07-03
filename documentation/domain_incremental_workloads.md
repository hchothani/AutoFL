# autofl domain incremental workloads

guide for running domain incremental continual federated learning experiments.

## quick start

run experiments using `mclmain.py`:

```bash
# default: cifar10 + simple_cnn + naive strategy
python mclmain.py

# use pre-made experiment configs
python mclmain.py --config-path config/experiments --config-name cifar10_domain
python mclmain.py --config-path config/experiments --config-name cifar100_resnet
python mclmain.py --config-path config/experiments --config-name quick_test

# command line overrides
python mclmain.py dataset.workload=cifar100 model.name=resnet cl.strategy=domain
```

## available workloads

### classic datasets
- **cifar10**: 10-class image classification
- **cifar100**: 100-class image classification

### continual learning benchmarks
- **permuted_mnist**: pixel permutation tasks (standard cl benchmark)
- **split_cifar10**: class-incremental cifar10 (2 classes per task)

### other driving datasets   (requires way more work still for these in ML architecture side - need YOLO or some implementation for segmentation)
- **bdd100k_10k**: 10k driving images subset (domain split by weather + time-of-day) - WORKING with real data (small subset)
- **bdd100k**/**bdd100k_v2**: 100k driving images (domain split by weather + time-of-day) - just extrapolated but issue being dataset too big
- **kitti**/**kitti_v2**: autonomous-driving street scenes (domain split by road-type + weather) - very large dataset, testing with subset

> **_v2** workloads use improved dataset handling pipeline in `workloads/BDD100KDomainCLV2.py` and `workloads/KITTIDomainCLV2.py`.
> 
> **bdd100k_10k** uses BDD100K images and labels for domain classification with weather/time-of-day splits.

## dataset preparation

For BDD100K 10k (needs to be already extracted in data/ folder):
- Images: `data/bdd100k_images_10k/10k/{train,val,test}/`
- Labels: `data/bdd100k_labels/100k/{train,val,test}/`

The workload automatically loads and pairs images with their JSON annotations.

For other datasets, run the helper script:

```bash
# BDD100K full dataset (≈ 70 GB extracted)
python datasets/prepare_datasets.py bdd100k --target ./data

# KITTI detection images (≈ 12 GB extracted)
python datasets/prepare_datasets.py kitti --target ./data
```

## models

- **simple_cnn**: lightweight (~62k params), good for cifar
- **resnet**: powerful (~11m params), better for complex datasets  
- **mobilenet**: efficient mobile architecture
  - **v2**: ~3.5m params, balanced performance/efficiency
  - **v3_small**: ~2.5m params, most efficient
  - **v3_large**: ~5.4m params, best performance

## continual learning strategies

- **naive**: standard incremental learning
- **domain**: domain-shift incremental learning with transformations
- **ewc**: elastic weight consolidation (prevents forgetting important weights)
- **replay**: experience replay (stores old samples in buffer)  
- **hybrid**: combines ewc + replay for best performance

### domain definitions (cifar10/100)
1. **original**: standard normalization
2. **brightness**: brightness variation
3. **contrast**: contrast variation  
4. **blur**: gaussian blur

each domain split into 2 experiences = 8 total experiences per client.

## example experiment configs

see `config/experiments/` for ready-to-use configurations:

- `cifar10_domain.yaml`: cifar10 + simple_cnn + domain cl
- `cifar100_resnet.yaml`: cifar100 + resnet + naive cl  
- `cifar100_domain.yaml`: cifar100 + resnet + domain cl
- `quick_test.yaml`: fast test with minimal resources
- `bdd100k_10k_domain.yaml`: BDD100K 10k + SimpleCNN + domain CL (recommended start)
- `bdd100k_domain.yaml`: BDD100K + ResNet + domain CL
- `kitti_domain.yaml`: KITTI + ResNet + domain CL
- `bdd100k_quick_test.yaml`: fast CPU-only sanity check

## key configuration options

```yaml
dataset:
  workload: cifar10  # or cifar100, bdd100k, kitti
  batch_size: 32

model:
  name: simple_cnn   # or resnet
  
cl:
  strategy: naive    # or domain
  num_experiences: 5

training:
  learning_rate: 0.001  # 0.0001 for resnet
  epochs: 5

server:
  num_rounds: 5
  num_clients: 5

wb:
  project: autofl-testing
  name: auto_generated  # format: {dataset}_{strategy}_{model}
```

## wandb integration

experiments automatically tracked with names like:
- `cifar10_naive_simple_cnn`
- `cifar100_domain_resnet`

view at: https://wandb.ai/username/autofl-testing

## domain definitions (bdd100k_10k)

1. **clear_day**: clear weather, daytime
2. **clear_night**: clear weather, nighttime 
3. **rainy_day**: rainy/partly cloudy weather, daytime
4. **cloudy**: cloudy/overcast weather, any time

Each domain split into 2 experiences = 8 total experiences per client. 