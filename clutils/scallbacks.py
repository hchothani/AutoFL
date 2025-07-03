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

# Only initialize wandb if mode is not disabled
if cfg.wb.get('mode', 'online') != 'disabled':
    wandb.init(
        project=cfg.wb.project,
        name=cfg.wb.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wb.get('mode', 'online')
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

# State of all rounds metrics
wexpacc_byround = []
gcf_per_exp_running =[0 for _ in range (NUM_ROUNDS)]

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,  
        "local_epochs": cfg.client.epochs, 
        "num_rounds": cfg.server.num_rounds
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

    exp_accuracy_ds = [json.loads(m["accuracy_per_experience"]) for _, m in metrics]
    client_accuracy = [m["stream_accuracy"] for _, m in metrics]  # use stream_accuracy
    client_loss = [m["stream_loss"] for _, m in metrics]  # use stream_loss
    examples = [num_examples for num_examples, _ in metrics]
    
    try:
        wexpacc = [sum(w * val for w, val in zip(examples, values))/sum(examples) for values in zip(*exp_accuracy_ds)]
        wexpacc_byround.append(wexpacc)
        latest_wexpacc = wexpacc_byround[-1]

        # Safe array indexing to prevent index errors
        for i, w in enumerate(wexpacc_byround):
            if i < len(wexpacc_byround[i]) and i < len(latest_wexpacc):
                gcf_per_exp_running[i] = wexpacc_byround[i][i] - latest_wexpacc[i]

        gcf = sum(gcf_per_exp_running)/NUM_ROUNDS if NUM_ROUNDS > 0 else 0
    except Exception as e:
        print(f"Error in forgetting calculation: {e}")
        gcf = 0
        wexpacc = [0] * len(exp_accuracy_ds[0]) if exp_accuracy_ds else [0]

    # Log all metrics in a single wandb.log call for better organization
    try:
        wandb_metrics = {
            # Global metrics
            "global/accuracy": sum(w_accuracies) / sum(examples) if sum(examples) > 0 else 0,
            "global/loss": sum(w_losses) / sum(examples) if sum(examples) > 0 else 0,
            "global/forgetting_measure": gcf,
            "round": rnd,
        }

        # Per-experience metrics
        for i, expacc in enumerate(wexpacc, start=1):
            if isinstance(expacc, (int, float)) and not np.isnan(expacc):
                wandb_metrics[f"global/experience_{i}/accuracy"] = float(expacc)

        # Per-client metrics
        for i, (acc, cid) in enumerate(zip(client_accuracy, pid), start=1):
            if isinstance(acc, (int, float)) and not np.isnan(acc):
                wandb_metrics[f"client/{cid}/accuracy"] = float(acc)
            if i-1 < len(client_loss) and isinstance(client_loss[i-1], (int, float)) and not np.isnan(client_loss[i-1]):
                wandb_metrics[f"client/{cid}/loss"] = float(client_loss[i-1])

        # Per-experience forgetting metrics
        for i, g in enumerate(gcf_per_exp_running):
            if isinstance(g, (int, float)) and not np.isnan(g):
                wandb_metrics[f"global/experience_{i}/forgetting"] = float(g)

        # Log all metrics at once
        wandb.log(wandb_metrics, step=rnd)
        print(f"Evaluation metrics logged successfully for round {rnd}")
    except Exception as e:
        print(f"Error logging evaluation metrics: {e}")
        # Continue without logging rather than crashing

    return {
        "global_accuracy": sum(w_accuracies) / sum(examples),
        "global_loss": sum(w_losses) / sum(examples),
        "global_wexp_accuracy": ",".join(map(str, wexpacc)),
        "global_client_accuracy": ",".join(map(str, client_accuracy)),
        "global_client_loss": ",".join(map(str, client_loss)),
        "global_exp_accuracy": json.dumps(exp_accuracy_ds),
        "gcf": float(gcf),
        "gcf_per_exp": json.dumps(gcf_per_exp_running),
    }

def fit_metrics_aggregation_fn(metrics: list) -> dict:
    # metrics: List[Tuple[num_examples, client_metrics_dict]]
    accuracies = [num_examples * m["stream_acc"] for num_examples, m in metrics]
    losses = [num_examples * m["stream_loss"] for num_examples, m in metrics]
    # forgetting = [m["cumalative_forgetting_measure"] for _, m in metrics]
    forgetting = [m.get("cumalative_forgetting_measure", 0) for _, m in metrics]
    stepwise_forgetting = [m["stepwise_forgetting_measure"] for _, m in metrics]
    exp_accs = [json.loads(m["accuracy_per_experience"]) for _, m in metrics]
    
    # Extract client IDs and round info for individual logging
    client_ids = [m["pid"] for _, m in metrics]
    rnd = metrics[0][1]["round"] if metrics else 1
    # Optionally, confusion matrices
    confusion_matrices = [json.loads(m["confusion_matrix"]) for _, m in metrics if "confusion_matrix" in m]

    avg_acc = sum(accuracies) / sum(num_examples for num_examples, _ in metrics)
    avg_loss = sum(losses) / sum(num_examples for num_examples, _ in metrics)
    avg_forgetting = np.mean(forgetting)
    avg_stepwise_forgetting = np.mean(stepwise_forgetting)
    avg_exp_acc = np.mean(exp_accs, axis=0).tolist()  # Per-experience average

    # BWT and FWT (example, you may need to adjust based on your history storage)
    bwt = avg_exp_acc[-1] - np.mean(avg_exp_acc[:-1]) if len(avg_exp_acc) > 1 else 0
    fwt = avg_exp_acc[0] - np.mean(avg_exp_acc[1:]) if len(avg_exp_acc) > 1 else 0

    # Aggregate confusion matrix if available
    if confusion_matrices:
        agg_conf_matrix = np.sum(confusion_matrices, axis=0)
        wandb_conf_matrix = wandb.plot.confusion_matrix(
            probs=None,
            y_true=None,
            preds=None,
            confusion_matrix=agg_conf_matrix.tolist(),
            class_names=[str(i) for i in range(len(agg_conf_matrix))]
        )
    else:
        wandb_conf_matrix = None

    # Log to wandb - fix the list logging issue
    log_dict = {
        "global/accuracy": avg_acc,
        "global/loss": avg_loss,
        "global/forgetting": avg_forgetting,
        "global/stepwise_forgetting": avg_stepwise_forgetting,
        "global/BWT": bwt,
        "global/FWT": fwt,
        "round": rnd,
    }
    
    # Log per-experience accuracy as individual metrics instead of a list
    for i, exp_acc in enumerate(avg_exp_acc):
        log_dict[f"global/experience_{i+1}_accuracy"] = exp_acc
    
    # Add individual client metrics
    for i, (client_id, _, m) in enumerate(zip(client_ids, range(len(metrics)), [m for _, m in metrics])):
        log_dict[f"client/{client_id}/accuracy"] = m["stream_acc"]
        log_dict[f"client/{client_id}/loss"] = m["stream_loss"]
        log_dict[f"client/{client_id}/stepwise_forgetting"] = m["stepwise_forgetting_measure"]
    
    try:
        print("Logging to wandb:", log_dict)  # Debug print
        wandb.log(log_dict, step=rnd)
        print(f"Fit metrics logged successfully for round {rnd}")
    except Exception as e:
        print(f"Error logging fit metrics: {e}")
        # Continue without logging rather than crashing

    
    return {
        "global_accuracy": avg_acc,
        "global_loss": avg_loss,
        "global_forgetting": avg_forgetting,
        "global_stepwise_forgetting": avg_stepwise_forgetting,
        "global_BWT": bwt,
        "global_FWT": fwt,
        "global_experience_accuracy": json.dumps(avg_exp_acc),
    }



    
