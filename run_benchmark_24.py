#!/usr/bin/env python3
"""
run_benchmark_24.py
========================================================================================
Automated 24-Run Benchmark Suite Runner for AutoFL
----------------------------------------------------------------------------------------
Matrix:
  Models (2):    simple_cnn, resnet18
  Datasets (3):  cifar10, cifar100, gtsrb
  Clients (2):   5, 10
  Splits (2):    iid, niid
  Total:         24 runs

Usage Examples:
  # Run all 24 benchmark experiments sequentially:
  python run_benchmark_24.py --all

  # Run only CIFAR-10 experiments:
  python run_benchmark_24.py --filter cifar10

  # Run only ResNet18 experiments:
  python run_benchmark_24.py --filter resnet18

  # Run a specific single experiment:
  python run_benchmark_24.py --config resnet18_cifar10_5c_niid

  # Run with custom train time (e.g., 300s quick test instead of default 900s):
  python run_benchmark_24.py --all --time 300
========================================================================================
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "config" / "experiments" / "benchmark_24"
OUTPUT_DIR = REPO_ROOT / "outputs"


def cleanup_ray():
    """Aggressively terminates lingering Ray workers to prevent VRAM memory leaks."""
    try:
        subprocess.run(["ray", "stop", "--force"], capture_output=True, check=False)
    except Exception:
        pass
    try:
        subprocess.run(["pkill", "-9", "-f", "ray::"], capture_output=True, check=False)
    except Exception:
        pass
    time.sleep(1.5)


def get_all_benchmark_configs():
    files = sorted(CONFIG_DIR.glob("*.yaml"))
    return [f.stem for f in files if f.stem != "top_config"]


def run_experiment(config_name: str, train_time: int = None):
    print("\n" + "=" * 80)
    print(f"[BENCHMARK-24] Launching Experiment: {config_name}")
    if train_time:
        print(f"               Overriding Train Time: {train_time}s")
    print("=" * 80)

    cleanup_ray()

    cmd = [
        sys.executable, "-u", "main.py",
        "--config-path", str(CONFIG_DIR),
        "--config-name", config_name,
    ]
    if train_time:
        cmd.append(f"async.total_train_time={train_time}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    start_ts = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
            cwd=str(REPO_ROOT),
        )
        ret_code = process.wait()
        elapsed = time.time() - start_ts
        if ret_code == 0:
            print(f"[BENCHMARK-24] Completed {config_name} in {elapsed:.1f}s.")
        else:
            print(f"[BENCHMARK-24] FAILED {config_name} with exit code {ret_code} in {elapsed:.1f}s.")
        return ret_code == 0
    except KeyboardInterrupt:
        print("\n[BENCHMARK-24] Interrupted by user.")
        cleanup_ray()
        sys.exit(1)
    finally:
        cleanup_ray()


def main():
    parser = argparse.ArgumentParser(description="AutoFL 24-Run Benchmark Suite Runner")
    parser.add_argument("--all", action="store_true", help="Run all 24 benchmark experiments")
    parser.add_argument("--config", type=str, default=None, help="Run a specific config name")
    parser.add_argument("--filter", type=str, default=None, help="Substring filter (e.g. 'cifar10', 'resnet18', '5c')")
    parser.add_argument("--time", type=int, default=None, help="Override async.total_train_time in seconds")
    parser.add_argument("--list", action="store_true", help="List all 24 configs and exit")

    args = parser.parse_args()

    all_configs = get_all_benchmark_configs()

    if args.list:
        print(f"\nDiscovered {len(all_configs)} Benchmark Configs:")
        for i, c in enumerate(all_configs, 1):
            print(f"  {i:2d}. {c}")
        return

    configs_to_run = []
    if args.config:
        clean_name = args.config.replace(".yaml", "")
        if clean_name in all_configs:
            configs_to_run = [clean_name]
        else:
            print(f"Config '{clean_name}' not found in {CONFIG_DIR}.")
            return
    elif args.filter:
        configs_to_run = [c for c in all_configs if args.filter.lower() in c.lower()]
        print(f"Filtered {len(configs_to_run)} configs matching '{args.filter}'.")
    elif args.all:
        configs_to_run = all_configs
    else:
        parser.print_help()
        print("\nTip: Run with --all to execute all 24 runs, or --list to see the experiment matrix.")
        return

    print(f"\nStarting benchmark queue of {len(configs_to_run)} experiments...")
    for idx, cfg_name in enumerate(configs_to_run, 1):
        print(f"\n>>> Queue Progress: [{idx}/{len(configs_to_run)}] <<<")
        success = run_experiment(cfg_name, train_time=args.time)
        if not success:
            print(f"Warning: {cfg_name} failed. Continuing to next experiment...")


if __name__ == "__main__":
    main()

