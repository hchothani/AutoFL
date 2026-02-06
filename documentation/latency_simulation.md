# Latency-Aware AutoFL Workflows

This guide explains how to inject OMNeT++-derived latency traces into AutoFL simulations, collect timing metrics, and compare baseline versus latency-aware runs.

## 1. Configure latency simulation

Update your experiment configuration (for example `config/config.yaml` or a Hydra override) with the new `latency` block:

```yaml
latency:
  enabled: true                       # toggle latency injection
  csv_path: omnet-data/latency_with_10cars2RSU_30.09.2025.csv
  sampling_mode: mean                 # mean | trace | random | chunk
  scaling_factor: 1.0                 # global multiplier applied to delays
  threshold_multiplier: 1.0           # std-dev multiplier for drop threshold
  drop_behavior: skip                 # skip | remove
  sleep: false                        # legacy flag (physical sleeping is disabled)
  sleep_log: true                     # add sampled latency to metrics without sleeping
  upload_multiplier: 1.0              # scale generated throughput (upload)
  download_multiplier: 1.0            # scale received throughput (download)
  max_clients: 10                     # number of clients embedded in the trace
  skip_if_slow_aggregation: false     # drop round when aggregation exceeds OMNeT expectation (mean mode)
  skip_if_slow_margin: 1.0            # multiplier applied to expected max delay before skipping
  log_round_time_variance: false      # emit with/without latency round timings for comparison
```

All parameters can be overridden per experiment run.

### Choose a latency-aware server strategy

Select the Flower strategy via the `server.strategy` field. Set it to
`latency_aware_fedavg` (default) to enable dropout-aware aggregation and
aggregation-time logging. If you need the unmodified Flower strategy for
comparison, use `server.strategy: vanilla_fedavg` or provide the fully qualified
class path such as `flwr.server.strategy.FedAvg`. Any built-in Flower strategy
name (e.g. `fedprox`, `scaffold`) continues to work, and custom classes can be
registered under `algorithms/`.

### Chunk sampling mode

Set `latency.sampling_mode: chunk` to slice each client's delay trace into
`server.num_rounds` contiguous windows (e.g., samples `0–199`, `200–399`, …).
Each round replays the mean delay of its window, preserving the sequential
behaviour captured by OMNeT++ while remaining deterministic.

### Skip slow aggregations

When the mean sampler is active you can protect FedAvg by enabling
`latency.skip_if_slow_aggregation: true`. The server compares the measured
aggregation time with the maximum OMNeT-derived network delay (scaled via
`skip_if_slow_margin`) and discards the update when the server is slower than
expected.

### Compare round timings in one run

Set `latency.log_round_time_variance: true`.
Inspect `client_round_metrics.csv` for `timing/round_total_s` (simulated),
`timing/round_wall_clock_s` (measured) and `timing/round_latency_component_s` (injected).
Reference `aggregate_metrics.csv` (`local/average/round_time_variance_s` and
`local/average/simulated_latency_s`) to quantify the injected component.

## 2. Run baseline and latency experiments

1. **Baseline**: keep `latency.enabled = false`, run `python mclmain.py`.
2. **Latency-aware**: enable the block above and rerun `python mclmain.py`.

Each execution creates a timestamped directory under `outputs/` containing:

- `client_round_metrics.csv`: per-client timing, latency sample, and participation data.
- `server_round_metrics.csv`: aggregation time, accepted results, and drop list per round.
- `aggregate_metrics.csv`: round-level aggregates (accuracy, forgetting, timing) for both fit and eval phases.

The exact location is stored in `cfg.logging.run_output_dir` and logged to Weights & Biases as part of the run configuration.

## 3. Compare runs visually

Use the helper script to generate side-by-side plots:

```bash
python omnet-data/plot_latency_comparison.py \
  --baseline outputs/20250607_231648-75smut70_latency_off \
  --latency outputs/20250608_101530_latency_on \
  --output outputs/comparisons/run1
```

Artifacts produced:

- `accuracy_vs_rounds.png`: global accuracy (eval phase) versus round for each run.
- `runtime_comparison.png`: server aggregation time and average client round time versus round.
- `comparison_summary.csv`: summary table with mean accuracy, fit round time, aggregation time, and drop rate per run.

## 4. Recommended metrics for vehicular FL

Beyond accuracy and timing, consider tracking:

- **Client availability ratio**: fraction of scheduled rounds where each vehicle participates (captured via `latency/dropped`).
- **Latency threshold violations**: count of rounds exceeding the configurable `mean + k·σ` envelope.
- **Effective throughput**: time-weighted mean of generated/received throughput (logged per client in `client_round_metrics.csv`).
- **Handover volatility**: serving-cell transitions per round (available from OMNeT++ traces; integrate when modelling RSU changes).
- **Energy or compute budget**: approximate per-round energy/compute if available (placeholder for future telemetry integration).
- **Forgetting score drift**: monitor `cumalative_forgetting_measure`/`stepwise_forgetting_measure` alongside latency-induced dropouts to flag knowledge erosion.

These metrics help benchmark different radio stacks or scheduling policies (e.g., DSRC vs. C-V2X, single RSU vs. multi-RSU handover).

## 5. Tips for scaling latency profiles

- Use `latency.scaling_factor` to compress or dilate OMNeT++ delays so that at least one round completes in the target wall-clock budget.
- The upload/download multipliers allow modelling asymmetric links (e.g., congested uplink).
- Switch `sampling_mode` to `trace` for replaying the exact time series, or `random` for stochastic sampling.
- Configure `drop_behavior: remove` to permanently exclude chronically slow vehicles during the simulation.

## 6. Data destinations

Metrics are emitted to both Weights & Biases (if enabled) and CSV files. Downstream notebooks or dashboards can load these CSVs directly for bespoke analyses.
