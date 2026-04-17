"""Factory for building Flower server strategies from config."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict

import flwr as fl


def _filter_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(callable_obj)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _build_common_kwargs(
    cfg, num_clients: int, evaluate_fn: Callable[..., Any]
) -> Dict[str, Any]:
    return {
        "fraction_fit": cfg.server.fraction_fit,
        "fraction_evaluate": cfg.server.fraction_eval,
        "min_fit_clients": cfg.server.min_fit,
        "min_evaluate_clients": cfg.server.min_eval,
        "min_available_clients": num_clients,
        "evaluate_fn": evaluate_fn,
    }


def build_server_strategy(cfg, num_clients: int, evaluate_fn: Callable[..., Any]):
    """Create a Flower strategy object based on cfg.server.strategy."""
    strategy_name = str(cfg.server.strategy).strip().lower()
    common_kwargs = _build_common_kwargs(cfg, num_clients, evaluate_fn)

    if strategy_name == "fedavg":
        return fl.server.strategy.FedAvg(**common_kwargs)

    if strategy_name == "fedprox":
        kwargs = dict(common_kwargs)
        kwargs["proximal_mu"] = cfg.server.fedprox.mu
        return fl.server.strategy.FedProx(
            **_filter_kwargs(fl.server.strategy.FedProx, kwargs)
        )

    if strategy_name == "fedopt":
        optimizer_name = str(cfg.server.fedopt.server_optimizer).strip().lower()
        strategy_map = {
            "adam": "FedAdam",
            "adagrad": "FedAdagrad",
            "yogi": "FedYogi",
            "sgd": "FedAvg",
        }
        strategy_class_name = strategy_map.get(optimizer_name)
        if strategy_class_name is None:
            supported = sorted(strategy_map.keys())
            raise ValueError(
                f"Unsupported server.fedopt.server_optimizer '{optimizer_name}'. Supported: {supported}"
            )

        strategy_class = getattr(fl.server.strategy, strategy_class_name, None)
        if strategy_class is None:
            raise RuntimeError(
                f"Flower strategy '{strategy_class_name}' is not available in the installed Flower version."
            )

        kwargs = dict(common_kwargs)
        kwargs.update(
            {
                "eta": cfg.server.fedopt.server_lr,
                "beta_1": cfg.server.fedopt.beta1,
                "beta_2": cfg.server.fedopt.beta2,
            }
        )
        return strategy_class(**_filter_kwargs(strategy_class, kwargs))

    if strategy_name in {"scaffold", "fednova"}:
        raise NotImplementedError(
            f"Strategy '{strategy_name}' is configured but not implemented yet in this runtime."
        )

    raise ValueError(f"Unsupported server.strategy '{strategy_name}'.")
