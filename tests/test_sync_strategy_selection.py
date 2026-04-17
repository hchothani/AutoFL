from omegaconf import OmegaConf

from runners.strategy_factory import build_server_strategy


def test_strategy_factory_rejects_unknown_strategy():
    cfg = OmegaConf.create(
        {
            "server": {
                "strategy": "unknown",
                "fraction_fit": 1.0,
                "fraction_eval": 1.0,
                "min_fit": 2,
                "min_eval": 2,
                "fedprox": {"mu": 0.01},
                "fedopt": {
                    "server_optimizer": "adam",
                    "server_lr": 1.0,
                    "beta1": 0.9,
                    "beta2": 0.999,
                },
            }
        }
    )

    try:
        build_server_strategy(cfg, num_clients=4, evaluate_fn=lambda *_: None)
        assert False, "Expected ValueError for unsupported strategy"
    except ValueError as exc:
        assert "Unsupported server.strategy" in str(exc)
