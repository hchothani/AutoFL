from typing import List, Tuple
from flwr.common import Metrics
from logging import INFO, WARNING
from flwr.common.logger import log

import wandb
import os
from omegaconf import OmegaConf
from pathlib import Path

import numpy as np
import json

# Setup Config
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config_utils import load_config

cfg = load_config()
from utils.latency_simulator import get_runtime_recorder

# Only initialize wandb if mode is not disabled
if cfg.wb.get("mode", "online") != "disabled":
    wandb.init(
        project=cfg.wb.project,
        name=cfg.wb.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wb.get("mode", "online"),
    )
else:
    # Create a dummy run for disabled mode
    class DummyWandB:
        def log(self, *args, **kwargs):
            pass

        class plot:
            @staticmethod
            def confusion_matrix(*args, **kwargs):
                return None

    wandb = DummyWandB()

NUM_ROUNDS = cfg.server.num_rounds
LOCAL_EPOCHS = cfg.client.epochs
NUM_CLIENTS = cfg.server.num_clients


def _collect_metric_list(metrics: List[Tuple[int, Metrics]], key: str):
    values = []
    for _, metric in metrics:
        if key not in metric:
            return None
        values.append(metric[key])
    return values


def _load_accuracy_vectors(raw_values):
    vectors = []
    for value in raw_values:
        if isinstance(value, str):
            try:
                vectors.append(json.loads(value))
            except json.JSONDecodeError:
                return None
        else:
            vectors.append(value)
    return vectors


# State of all rounds metrics


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,
        "local_epochs": cfg.client.epochs,
        "num_rounds": cfg.server.num_rounds,
    }
    return config


def eval_config(server_round: int):
    config = {
        "server_round": server_round,
        "local_epochs": cfg.client.epochs,
        "num_rounds": cfg.server.num_rounds,
    }
    return config


def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    client_accuracies = [m["stream_accuracy"] for _, m in metrics]
    client_losses = [m["stream_loss"] for _, m in metrics]
    w_accuracies = [num_examples * m["stream_accuracy"] for num_examples, m in metrics]
    w_losses = [num_examples * m["stream_loss"] for num_examples, m in metrics]
    pid = [m["pid"] for _, m in metrics]
    rnd = metrics[0][1]["server_round"]

    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)

    cum_forgetting = _collect_metric_list(metrics, "cumalative_forgetting_measure")
    step_forgetting = _collect_metric_list(metrics, "stepwise_forgetting_measure")
    accuracy_per_exp_raw = _collect_metric_list(metrics, "accuracy_per_experience")
    accuracy_per_exp_vectors = (
        _load_accuracy_vectors(accuracy_per_exp_raw)
        if accuracy_per_exp_raw is not None
        else None
    )

    eval_metrics = {
        "global/average/accuracy": sum(w_accuracies) / total_examples,
        "global/client/accuracy": {id: acc for id, acc in zip(pid, client_accuracies)},
        "global/average/loss": sum(w_losses) / total_examples,
        "global/client/loss": {id: loss for id, loss in zip(pid, client_losses)},
    }

    if cum_forgetting is not None:
        avg_cum = sum(cum_forgetting) / len(cum_forgetting)
        eval_metrics.update(
            {
                "global/average/cumalative_forgetting": avg_cum,
                "global/client/cumalative_forgetting": {
                    id: cmfm for id, cmfm in zip(pid, cum_forgetting)
                },
            }
        )
    else:
        log(
            WARNING,
            "Missing 'cumalative_forgetting_measure' in evaluation metrics; skipping aggregate logging for this field.",
        )

    if step_forgetting is not None:
        avg_step = sum(step_forgetting) / len(step_forgetting)
        eval_metrics.update(
            {
                "global/average/stepwise_forgetting": avg_step,
                "global/client/stepwise_forgetting": {
                    id: swfm for id, swfm in zip(pid, step_forgetting)
                },
            }
        )
    else:
        log(
            WARNING,
            "Missing 'stepwise_forgetting_measure' in evaluation metrics; skipping aggregate logging for this field.",
        )

    if accuracy_per_exp_vectors is not None:
        weighted_accuracy_pexp = [
            sum(w * val for w, val in zip(examples, values)) / total_examples
            for values in zip(*accuracy_per_exp_vectors)
        ]
        eval_metrics["global/experience/accuracy"] = {
            id: acc for id, acc in zip(pid, weighted_accuracy_pexp)
        }
    else:
        log(
            WARNING,
            "Missing 'accuracy_per_experience' in evaluation metrics; skipping per-experience aggregates.",
        )

    wandb.log(eval_metrics, step=rnd)

    recorder = get_runtime_recorder()
    if recorder is not None:
        entries = [(num_examples, dict(m)) for num_examples, m in metrics]
        recorder.log_client_round(rnd, entries)
        aggregate_row = {
            "global/average/accuracy": eval_metrics["global/average/accuracy"],
            "global/average/loss": eval_metrics["global/average/loss"],
        }
        if "global/average/cumalative_forgetting" in eval_metrics:
            aggregate_row["global/average_cumalative_forgetting"] = eval_metrics[
                "global/average/cumalative_forgetting"
            ]
        if "global/average/stepwise_forgetting" in eval_metrics:
            aggregate_row["global/average_stepwise_forgetting"] = eval_metrics[
                "global/average/stepwise_forgetting"
            ]
        recorder.log_aggregate_metrics(rnd, "eval", aggregate_row)

    result_metrics = {
        "global/average_accuracy": eval_metrics["global/average/accuracy"],
        "global/average_loss": eval_metrics["global/average/loss"],
    }
    if cum_forgetting is not None:
        result_metrics["global/average_cumalative_forgetting"] = sum(
            cum_forgetting
        ) / len(cum_forgetting)
    if step_forgetting is not None:
        result_metrics["global/average_stepwise_forgetting"] = sum(
            step_forgetting
        ) / len(step_forgetting)

    return result_metrics


def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Calculate Metrics After Fit of Clients"""

    # Per Client Acc and Loss
    client_acc = [m["stream_acc"] for _, m in metrics]
    client_loss = [m["stream_loss"] for _, m in metrics]
    # Weighted Acc and Loss
    w_accuracies = [num_examples * m["stream_acc"] for num_examples, m in metrics]
    w_losses = [num_examples * m["stream_loss"] for num_examples, m in metrics]
    # Forgetting Measures
    cumalative_forgetting_measures = [
        m["cumalative_forgetting_measure"] for _, m in metrics
    ]
    stepwise_forgetting_measures = [
        m["stepwise_forgetting_measure"] for _, m in metrics
    ]
    network_times = [m.get("latency/expected_network_time_s", 0.0) for _, m in metrics]
    training_times = [m.get("timing/training_s", 0.0) for _, m in metrics]
    round_times = [m.get("timing/round_total_s", 0.0) for _, m in metrics]
    round_wall_clock_times = [
        m.get("timing/round_wall_clock_s", m.get("timing/round_total_s", 0.0))
        for _, m in metrics
    ]
    simulated_latency_components = [
        m.get("timing/simulated_latency_s", 0.0) for _, m in metrics
    ]
    download_times = [m.get("latency/download_time_s", 0.0) for _, m in metrics]
    upload_times = [m.get("latency/upload_time_s", 0.0) for _, m in metrics]
    has_round_baseline = any("timing/round_without_latency_s" in m for _, m in metrics)
    round_times_without_latency = (
        [m.get("timing/round_without_latency_s", 0.0) for _, m in metrics]
        if has_round_baseline
        else []
    )
    latency_components = (
        [m.get("timing/round_latency_component_s", 0.0) for _, m in metrics]
        if has_round_baseline
        else []
    )

    # Round and Partition Id's
    rnd = metrics[0][1]["round"]
    pid = [m["pid"] for _, m in metrics]

    accuracy_per_exp_pc = [json.loads(m["accuracy_per_experience"]) for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    weighted_accuracy_per_exp = [
        sum(w * val for w, val in zip(examples, values)) / sum(examples)
        for values in zip(*accuracy_per_exp_pc)
    ]

    fit_metrics = {
        "local/average/accuracy": sum(w_accuracies) / sum(examples),
        "local/client/accuracy": {id: acc for id, acc in zip(pid, client_acc)},
        "local/average/loss": sum(w_losses) / sum(examples),
        "local/client/loss": {id: loss for id, loss in zip(pid, client_loss)},
        "local/average/cumalative_forgetting": sum(cumalative_forgetting_measures)
        / len(cumalative_forgetting_measures),
        "local/client/cumalative_forgetting": {
            id: cmfm for id, cmfm in zip(pid, cumalative_forgetting_measures)
        },
        "local/average/stepwise_forgetting": sum(stepwise_forgetting_measures)
        / len(stepwise_forgetting_measures),
        "local/client/stepwise_forgetting": {
            id: swfm for id, swfm in zip(pid, stepwise_forgetting_measures)
        },
        "local/experience/accuracy": {
            id: acc for id, acc in zip(pid, weighted_accuracy_per_exp)
        },
        "local/average/network_time_s": (
            float(sum(network_times) / len(network_times)) if network_times else 0.0
        ),
        "local/average/training_time_s": (
            float(sum(training_times) / len(training_times)) if training_times else 0.0
        ),
        "local/average/round_time_s": (
            float(sum(round_times) / len(round_times)) if round_times else 0.0
        ),
        "local/average/round_wall_clock_s": (
            float(sum(round_wall_clock_times) / len(round_wall_clock_times))
            if round_wall_clock_times
            else 0.0
        ),
        "local/average/simulated_latency_s": (
            float(sum(simulated_latency_components) / len(simulated_latency_components))
            if simulated_latency_components
            else 0.0
        ),
        "local/average/download_time_s": (
            float(sum(download_times) / len(download_times)) if download_times else 0.0
        ),
        "local/average/upload_time_s": (
            float(sum(upload_times) / len(upload_times)) if upload_times else 0.0
        ),
    }

    if round_times_without_latency:
        avg_round_without_latency = float(
            sum(round_times_without_latency) / len(round_times_without_latency)
        )
        avg_latency_component = (
            float(sum(latency_components) / len(latency_components))
            if latency_components
            else 0.0
        )
        fit_metrics["local/average/round_time_without_latency_s"] = (
            avg_round_without_latency
        )
        fit_metrics["local/average/latency_component_s"] = avg_latency_component
        fit_metrics["local/average/round_time_variance_s"] = (
            fit_metrics["local/average/round_time_s"] - avg_round_without_latency
        )

    # Logging to Wandb
    wandb.log(fit_metrics, step=rnd)

    recorder = get_runtime_recorder()
    if recorder is not None:
        entries = [(num_examples, dict(m)) for num_examples, m in metrics]
        recorder.log_client_round(rnd, entries)
        aggregate_row = {
            "local/average/accuracy": fit_metrics["local/average/accuracy"],
            "local/average/loss": fit_metrics["local/average/loss"],
            "local/average/cumalative_forgetting": fit_metrics[
                "local/average/cumalative_forgetting"
            ],
            "local/average/stepwise_forgetting": fit_metrics[
                "local/average/stepwise_forgetting"
            ],
            "local/average/network_time_s": fit_metrics.get(
                "local/average/network_time_s", 0.0
            ),
            "local/average/training_time_s": fit_metrics.get(
                "local/average/training_time_s", 0.0
            ),
            "local/average/round_time_s": fit_metrics.get(
                "local/average/round_time_s", 0.0
            ),
            "local/average/round_wall_clock_s": fit_metrics.get(
                "local/average/round_wall_clock_s", 0.0
            ),
            "local/average/simulated_latency_s": fit_metrics.get(
                "local/average/simulated_latency_s", 0.0
            ),
            "local/average/download_time_s": fit_metrics.get(
                "local/average/download_time_s", 0.0
            ),
            "local/average/upload_time_s": fit_metrics.get(
                "local/average/upload_time_s", 0.0
            ),
        }
        if round_times_without_latency:
            aggregate_row["local/average/round_time_without_latency_s"] = (
                fit_metrics.get("local/average/round_time_without_latency_s", 0.0)
            )
            aggregate_row["local/average/round_time_variance_s"] = fit_metrics.get(
                "local/average/round_time_variance_s", 0.0
            )
        recorder.log_aggregate_metrics(rnd, "fit", aggregate_row)

    return {
        "local/average_accuracy": sum(w_accuracies) / sum(examples),
        "local/average_loss": sum(w_losses) / sum(examples),
        "local/average_cumalative_forgetting": sum(cumalative_forgetting_measures)
        / len(cumalative_forgetting_measures),
        "local/average_stepwise_forgetting": sum(stepwise_forgetting_measures)
        / len(stepwise_forgetting_measures),
        "local/average_network_time": (
            float(sum(network_times) / len(network_times)) if network_times else 0.0
        ),
        "local/average_round_time": (
            float(sum(round_times) / len(round_times)) if round_times else 0.0
        ),
        "local/average_round_time_without_latency": (
            float(sum(round_times_without_latency) / len(round_times_without_latency))
            if round_times_without_latency
            else 0.0
        ),
        "local/average_round_time_variance": fit_metrics.get(
            "local/average/round_time_variance_s", 0.0
        ),
    }
