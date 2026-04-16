## Federated Continual Learning library

the goal is to combine and make a standardized framework for federated continual learning.

Run experiments using `mclmain.py`:

```
python mclmain.py --config-path config/experiments --config-name cifar10_naive
```

additional information can be found [here](documentation/)

## Latency-aware simulation and analysis

- Enable OMNeT++-derived latency traces via the `latency` section in your config. See [`documentation/latency_simulation.md`](documentation/latency_simulation.md) for the full walkthrough.
- Each run emits CSV logs with per-client timing, aggregated metrics, and server aggregation latency under `outputs/<timestamp>_*`.
- Compare baseline vs. latency-injected runs with `python omnet-data/plot_latency_comparison.py --baseline <run_dir> --latency <run_dir>` to generate accuracy and runtime plots alongside a summary table.


## TODOs

### For Cleaning
[ ] NIID Integration
[ ] Documentation
[ ] Tests

### Phase Wise Implementation -> Romir
[ ] Modify to accomodate multiple client loaders
[ ] Forgetting Metrics
...

### Distillation Loss
[ ] Modify train loop to have distillation loss 
[ ] Mention Hyperparams

### Adapter Bank
[ ] Calculate Adapters
[ ] Maintain an Adapter Bank on the Server for Async first
