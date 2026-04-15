import flwr
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

import wandb
import os

from clutils.clmetrics import (
    evaluate_metrics_aggregation_fn,
    fit_metrics_aggregation_fn,
)

NUM_ROUNDS = 5
NUM_CLIENTS = 5


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 3,
        "num_rounds": NUM_ROUNDS,
    }
    return config


def eval_config(server_round: int):
    config = {"server_round": server_round, "local_epochs": 3, "num_rounds": NUM_ROUNDS}
    return config


# Create FedAvg strategy
strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=NUM_CLIENTS,  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=eval_config,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
)


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training
    config = ServerConfig(NUM_ROUNDS)

    return ServerAppComponents(strategy=strategy, config=config)
