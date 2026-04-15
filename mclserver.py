import importlib
import inspect

from flwr.common import Context
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from clutils.scallbacks import (
    evaluate_metrics_aggregation_fn,
    fit_metrics_aggregation_fn,
    fit_config,
    eval_config,
)
from config_utils import load_config
from algorithms import STRATEGY_REGISTRY, LatencyAwareFedAvg
from omegaconf import DictConfig, OmegaConf

cfg = load_config()

NUM_ROUNDS = cfg.server.num_rounds
NUM_CLIENTS = cfg.server.num_clients


def _normalize_strategy_name(name: str) -> str:
    return name.replace("-", "_").lower()


def _resolve_strategy_class(strategy_name: str):
    if not strategy_name:
        raise ValueError("Strategy name must be provided in configuration.")

    if "." in strategy_name:
        module_name, _, class_name = strategy_name.rpartition(".")
        if not module_name or not class_name:
            raise ValueError(
                f"Invalid fully qualified strategy path '{strategy_name}'."
            )
        module = importlib.import_module(module_name)
        try:
            return getattr(module, class_name)
        except AttributeError as exc:
            raise ValueError(
                f"Class '{class_name}' not found in module '{module_name}'."
            ) from exc

    normalized = _normalize_strategy_name(strategy_name)
    if normalized in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[normalized]

    builtin_map = {
        "vanilla_fedavg": FedAvg,
        "flwr_fedavg": FedAvg,
    }
    if normalized in builtin_map:
        return builtin_map[normalized]

    module = importlib.import_module("flwr.server.strategy")
    for attr_name in dir(module):
        if attr_name.lower() == normalized:
            attr = getattr(module, attr_name)
            if inspect.isclass(attr):
                return attr

    candidate = "".join(part.capitalize() for part in normalized.split("_"))
    if hasattr(module, candidate):
        attr = getattr(module, candidate)
        if inspect.isclass(attr):
            return attr

    raise ValueError(
        f"Unknown server strategy '{strategy_name}'. "
        "Provide a fully qualified path or register it in algorithms."
    )


StrategyCls = _resolve_strategy_class(
    cfg.server.get("strategy", "latency_aware_fedavg")
)

strategy_kwargs = {
    "fraction_fit": cfg.server.fraction_fit,
    "fraction_evaluate": cfg.server.fraction_eval,
    "min_fit_clients": cfg.server.min_fit,
    "min_evaluate_clients": cfg.server.min_eval,
    "min_available_clients": cfg.server.num_clients,
    "on_fit_config_fn": fit_config,
    "on_evaluate_config_fn": eval_config,
    "evaluate_metrics_aggregation_fn": evaluate_metrics_aggregation_fn,
    "fit_metrics_aggregation_fn": fit_metrics_aggregation_fn,
}

raw_latency_cfg = cfg.get("latency", {})
if isinstance(raw_latency_cfg, DictConfig):
    latency_cfg = OmegaConf.to_container(raw_latency_cfg, resolve=True)
elif isinstance(raw_latency_cfg, dict):
    latency_cfg = raw_latency_cfg
else:
    latency_cfg = {}

if issubclass(StrategyCls, LatencyAwareFedAvg):
    strategy_kwargs["latency_cfg"] = latency_cfg

strategy = StrategyCls(**strategy_kwargs)


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for specified number of rounds
    config = ServerConfig(cfg.server.num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
