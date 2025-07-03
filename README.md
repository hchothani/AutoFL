## Federated Continual Learning library

the goal is to combine and make a standardized framework for federated continual learning.

Run experiments using `mclmain.py`:

```
python mclmain.py --config-path config/experiments --config-name cifar10_naive
```

additional information can be found [here](documentation/)


## TODOs

#### Phase 1 -> Simple Federated Continual Learning 

- [x]  Implement FL with Flower
- [x]  Understand how CL can work on each client
- [x]  Setup Individual client CL with Avalanche
- [x]  Integrate Avalanche with Flower
- [ ]  Basic Tests to ensure successful integration
- [ ]  Extend Base for Benchmarks
    - [x]  Add Workloads → Look at [CIFAR10CL.py] on dev 
        - [x]  **BDD100K domain incremental workload** → Had a smaller subset of this dataset
        - [ ]  **KITTI domain incremental workload** → Autonomous driving scenarios - added code similar to bdd but dataset too large to verify
        - [x] **CORe50 benchmark** added dataset and different stratergies
    - [x]  Add Models → MobileNet, ResNet etc
    - [x]  Add FL Strats
    - [x]  Add CL Strats → Buffer Replay, EWC, Hybrid --> **implemented with this branch** (added these three for now)
    - [x]  Add provisions for NIID
- [ ]  Make robust evaluation metrics and logging
    - [x]  Currently we are just printing and logging cl metrics of each experience and stream and then the final aggregate metrics for FL
    - [x]  Understanding which FL and CL metrics are required
    - [x]  Integrating and Logging using WandB
    - [ ]  Provision for local backup → not really  necessary immediately
- [x]  Centralize Configuration
    - [x]  Temp through yaml file
    - [ ]  Long term → Integrate Hydra
- [ ]  Adding documentation
    - [x] Doccumentation for DL and workloads 
    - [ ] Working FCL vs CL etc
