import flwr
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

import wandb
import os
from omegaconf import OmegaConf

from clutils.scallbacks import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn, fit_config, eval_config

#Setting up Configuration
from config_utils import load_config
cfg = load_config()

NUM_ROUNDS = cfg.server.num_rounds
NUM_CLIENTS = cfg.server.num_clients

# Create FedAvg strategy
strategy = FedAvg(
    fraction_fit=cfg.server.fraction_fit,  
    fraction_evaluate=cfg.server.fraction_eval, 
    min_fit_clients=cfg.server.min_fit,  
    min_evaluate_clients=cfg.server.min_eval, 
    min_available_clients=cfg.server.num_clients,  # Wait until all 10 clients are available
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=eval_config,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn
)

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training
    config = ServerConfig(cfg.server.num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


