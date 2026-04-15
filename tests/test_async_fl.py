#!/usr/bin/env python3
"""
Unit tests for async FL components.
Tests the AsynchronousStrategy, AsyncClientManager, and AsyncHistory classes.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_async_strategy_imports():
    """Test that async strategy components can be imported."""
    print("=" * 50)
    print("Testing async FL imports")
    print("=" * 50)

    try:
        from algorithms.async_fl import (
            AsynchronousStrategy,
            AsyncServer,
            AsyncClientManager,
            AsyncHistory,
        )

        print("✓ All async FL components imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_async_strategy_fedasync():
    """Test FedAsync aggregation strategy."""
    print("\n" + "=" * 50)
    print("Testing FedAsync aggregation")
    print("=" * 50)

    try:
        from algorithms.async_fl import AsynchronousStrategy
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

        # Create strategy
        strategy = AsynchronousStrategy(
            total_samples=1000,
            staleness_alpha=0.5,
            fedasync_mixing_alpha=0.9,
            fedasync_a=0.5,
            num_clients=10,
            async_aggregation_strategy="fedasync",
            use_staleness=True,
            use_sample_weighing=True,
            send_gradients=False,
            server_artificial_delay=False,
        )
        print("✓ AsynchronousStrategy created")

        # Create test parameters
        global_weights = [np.ones((10, 10)), np.ones((10,))]
        client_weights = [np.ones((10, 10)) * 2, np.ones((10,)) * 2]

        global_params = ndarrays_to_parameters(global_weights)
        client_params = ndarrays_to_parameters(client_weights)

        # Test aggregation
        result = strategy.average(
            global_params, client_params, t_diff=1.0, num_samples=100
        )
        result_arrays = parameters_to_ndarrays(result)

        # Result should be between global and client values
        assert result_arrays[0].shape == (10, 10)
        assert 1.0 < result_arrays[0][0, 0] < 2.0
        print(
            f"✓ FedAsync aggregation works (result value: {result_arrays[0][0, 0]:.4f})"
        )

        return True
    except Exception as e:
        print(f"✗ FedAsync test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_async_strategy_asyncfeded():
    """Test AsyncFedED aggregation strategy."""
    print("\n" + "=" * 50)
    print("Testing AsyncFedED aggregation")
    print("=" * 50)

    try:
        from algorithms.async_fl import AsynchronousStrategy
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

        # Create strategy
        strategy = AsynchronousStrategy(
            total_samples=1000,
            staleness_alpha=0.5,
            fedasync_mixing_alpha=0.9,
            fedasync_a=0.5,
            num_clients=10,
            async_aggregation_strategy="asyncfeded",
            use_staleness=True,
            use_sample_weighing=True,
            send_gradients=False,
            server_artificial_delay=False,
        )
        print("✓ AsynchronousStrategy (AsyncFedED) created")

        # Create test parameters
        global_weights = [np.zeros((5, 5))]
        client_weights = [np.ones((5, 5))]

        global_params = ndarrays_to_parameters(global_weights)
        client_params = ndarrays_to_parameters(client_weights)

        # Test aggregation
        result = strategy.average(
            global_params, client_params, t_diff=0.5, num_samples=100
        )
        result_arrays = parameters_to_ndarrays(result)

        # Result should move toward client values
        assert result_arrays[0].shape == (5, 5)
        assert result_arrays[0][0, 0] > 0
        print(
            f"✓ AsyncFedED aggregation works (result value: {result_arrays[0][0, 0]:.4f})"
        )

        return True
    except Exception as e:
        print(f"✗ AsyncFedED test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_staleness_weights():
    """Test staleness weighting functions."""
    print("\n" + "=" * 50)
    print("Testing staleness weighting functions")
    print("=" * 50)

    try:
        from algorithms.async_fl import AsynchronousStrategy

        strategy = AsynchronousStrategy(
            total_samples=1000,
            staleness_alpha=0.5,
            fedasync_mixing_alpha=0.9,
            fedasync_a=0.5,
            num_clients=10,
            async_aggregation_strategy="fedasync",
            use_staleness=True,
            use_sample_weighing=False,
        )

        # Test polynomial decay
        weight_0 = strategy.get_staleness_weight_coeff_fedasync_poly(t_diff=0)
        weight_1 = strategy.get_staleness_weight_coeff_fedasync_poly(t_diff=1)
        weight_5 = strategy.get_staleness_weight_coeff_fedasync_poly(t_diff=5)

        assert weight_0 == 1.0, "t_diff=0 should give weight 1.0"
        assert weight_1 < weight_0, "Higher staleness should give lower weight"
        assert weight_5 < weight_1, "Higher staleness should give lower weight"
        print(
            f"✓ Polynomial decay: t=0 -> {weight_0:.4f}, t=1 -> {weight_1:.4f}, t=5 -> {weight_5:.4f}"
        )

        # Test hinge function
        weight_b3 = strategy.get_staleness_weight_coeff_fedasync_hinge(
            t_diff=3, a=10, b=4
        )
        weight_b5 = strategy.get_staleness_weight_coeff_fedasync_hinge(
            t_diff=5, a=10, b=4
        )

        assert weight_b3 == 1.0, "t_diff < b should give weight 1.0"
        assert weight_b5 < 1.0, "t_diff > b should give weight < 1.0"
        print(
            f"✓ Hinge function: t=3 (b=4) -> {weight_b3:.4f}, t=5 (b=4) -> {weight_b5:.4f}"
        )

        # Test PAFLM
        weight_paflm = strategy.get_staleness_weight_coeff_paflm(t_diff=2)
        assert weight_paflm > 0, "PAFLM weight should be positive"
        print(f"✓ PAFLM staleness: t=2 -> {weight_paflm:.4f}")

        return True
    except Exception as e:
        print(f"✗ Staleness weighting test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_async_history():
    """Test AsyncHistory for timestamp-based metrics."""
    print("\n" + "=" * 50)
    print("Testing AsyncHistory")
    print("=" * 50)

    try:
        from algorithms.async_fl import AsyncHistory

        history = AsyncHistory()
        print("✓ AsyncHistory created")

        # Add some async metrics
        history.add_loss_centralized_async(timestamp=1.0, loss=1.5)
        history.add_loss_centralized_async(timestamp=2.0, loss=1.2)
        history.add_loss_centralized_async(timestamp=3.0, loss=0.9)

        history.add_metrics_centralized_async(metrics={"accuracy": 0.7}, timestamp=1.0)
        history.add_metrics_centralized_async(metrics={"accuracy": 0.8}, timestamp=2.0)

        history.add_metrics_distributed_fit_async(
            client_id="0",
            metrics={"loss": 0.5, "samples": 100},
            timestamp=1.5,
        )

        # Check metrics were recorded
        assert len(history.losses_centralized_async) == 3
        assert len(history.metrics_centralized_async["accuracy"]) == 2
        assert "0" in history.metrics_distributed_fit_async["loss"]
        print("✓ Metrics recorded correctly")

        # Check summary
        summary = history.get_async_metrics_summary()
        assert summary["num_centralized_evaluations"] == 3
        assert summary["final_centralized_loss"] == 0.9
        print(f"✓ Summary: {summary}")

        return True
    except Exception as e:
        print(f"✗ AsyncHistory test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_async_client_manager():
    """Test AsyncClientManager for client state tracking."""
    print("\n" + "=" * 50)
    print("Testing AsyncClientManager")
    print("=" * 50)

    try:
        from algorithms.async_fl import AsyncClientManager

        manager = AsyncClientManager()
        print("✓ AsyncClientManager created")

        # Check initial state
        assert manager.num_free() == 0
        assert manager.num_available() == 0
        print("✓ Initial state correct (no clients)")

        # Note: Full testing would require mock ClientProxy objects
        # For now, just verify the manager can be instantiated

        return True
    except Exception as e:
        print(f"✗ AsyncClientManager test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_async_server_init():
    """Test AsyncServer initialization."""
    print("\n" + "=" * 50)
    print("Testing AsyncServer initialization")
    print("=" * 50)

    try:
        from algorithms.async_fl import (
            AsyncServer,
            AsyncClientManager,
            AsynchronousStrategy,
        )
        from flwr.server.strategy import FedAvg

        # Create components
        strategy = FedAvg()
        client_manager = AsyncClientManager()
        async_strategy = AsynchronousStrategy(
            total_samples=50000,
            num_clients=10,
        )

        # Create server
        server = AsyncServer(
            strategy=strategy,
            client_manager=client_manager,
            async_strategy=async_strategy,
            total_train_time=60,
            waiting_interval=5,
            max_workers=2,
        )
        print("✓ AsyncServer created")

        # Check attributes
        assert server.total_train_time == 60
        assert server.waiting_interval == 5
        assert server.max_workers == 2
        print("✓ Server attributes correct")

        return True
    except Exception as e:
        print(f"✗ AsyncServer test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all async FL tests."""
    print("Starting Async FL component tests...")

    results = {
        "imports": test_async_strategy_imports(),
        "fedasync": test_async_strategy_fedasync(),
        "asyncfeded": test_async_strategy_asyncfeded(),
        "staleness": test_staleness_weights(),
        "history": test_async_history(),
        "client_manager": test_async_client_manager(),
        "server": test_async_server_init(),
    }

    print("\n" + "=" * 50)
    print("Async FL Test Summary")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ passed" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("All async FL tests passed!")
        return 0
    else:
        print("Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    exit(main())
