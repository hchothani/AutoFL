import flwr
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedProx, FedOpt

import wandb
import os
from omegaconf import OmegaConf

from clutils.scallbacks import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn, fit_config, eval_config

# Import custom SOTA server strategies
from algorithms.fedweit import FedWeITServerStrategy

#Setting up Configuration
from config_utils import load_config
cfg = load_config()

NUM_ROUNDS = cfg.server.num_rounds
NUM_CLIENTS = cfg.server.num_clients

def create_server_strategy():
    """Create and return the appropriate server strategy based on configuration."""
    
    # Get strategy name from config
    strategy_name = getattr(cfg.server, 'strategy', 'fedavg').lower()
    print(f"[Server] Using server strategy: {strategy_name}")
    
    # Common parameters for standard Flower strategies
    common_params = {
        'fraction_fit': cfg.server.fraction_fit,
        'fraction_evaluate': cfg.server.fraction_eval,
        'min_fit_clients': cfg.server.min_fit,
        'min_evaluate_clients': cfg.server.min_eval,
        'min_available_clients': cfg.server.num_clients,
        'on_fit_config_fn': fit_config,
        'on_evaluate_config_fn': eval_config,
        'evaluate_metrics_aggregation_fn': evaluate_metrics_aggregation_fn,
        'fit_metrics_aggregation_fn': fit_metrics_aggregation_fn
    }
    
    # Standard Flower strategies
    if strategy_name == 'fedavg':
        return FedAvg(**common_params)
    
    elif strategy_name == 'fedprox':
        # FedProx-specific parameters
        proximal_mu = getattr(cfg.server, 'fedprox', {}).get('mu', 0.01)
        return FedProx(proximal_mu=proximal_mu, **common_params)
    
    # Note: Scaffold and FedNova not available in current Flower version
    elif strategy_name in ['scaffold', 'fednova']:
        print(f"[Server] {strategy_name} not available in current Flower version, using FedAvg")
        return FedAvg(**common_params)
    
    elif strategy_name == 'fedopt':
        # FedOpt-specific parameters
        server_optimizer = getattr(cfg.server, 'fedopt', {}).get('server_optimizer', 'adam')
        server_lr = getattr(cfg.server, 'fedopt', {}).get('server_lr', 1.0)
        beta1 = getattr(cfg.server, 'fedopt', {}).get('beta1', 0.9)
        beta2 = getattr(cfg.server, 'fedopt', {}).get('beta2', 0.999)
        return FedOpt(
            server_optimizer=server_optimizer,
            server_lr=server_lr,
            beta1=beta1,
            beta2=beta2,
            **common_params
        )
    
    # Custom SOTA strategies
    elif strategy_name == 'fedweit':
        print("[Server] FedWeIT strategy selected")
        print("[Server] Note: Using FedAvg for now. Full FedWeIT server aggregation requires custom Flower strategy implementation.")
        print("[Server] Custom FedWeIT aggregation logic is available in algorithms/fedweit.py")
        # TODO: Implement full FedWeIT server strategy integration with Flower
        return FedAvg(**common_params)
    
    # For other SOTA strategies that don't need custom server aggregation
    elif strategy_name in ['plora', 'fedcprompt', 'fedet', 'fedgem', 'fedma', 'fedproto', 'fedrcil', 'fedrep', 'sacfl', 'stamp']:
        print(f"[Server] Using FedAvg for SOTA strategy: {strategy_name}")
        print(f"[Server] Note: {strategy_name} uses custom client logic but standard server aggregation")
        return FedAvg(**common_params)
    
    else:
        print(f"[Server] Warning: Unknown strategy '{strategy_name}', falling back to FedAvg")
        return FedAvg(**common_params)

# Create the appropriate strategy
strategy = create_server_strategy()

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for specified number of rounds
    config = ServerConfig(cfg.server.num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


