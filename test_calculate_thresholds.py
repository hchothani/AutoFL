#!/usr/bin/env python3
"""
test_calculate_thresholds.py
========================================================================================
Test Runner for Universal Regression Threshold Calculator in utils/threshold_utils.py
========================================================================================
Directly calls compute_direct_regression_threshold() from utils.threshold_utils.py.
Predicts optimal tau* directly from:
    - K = C / P (task complexity)
    - C_in (input channels)
    - d_inter (inter-class centroid distance)
    - d_intra (intra-class dispersion)
Formula:
    tau* = 0.4131 * (2/K)^0.25 + 1.9514 * sqrt(C_in/3) - 0.7540 * (d_inter - d_intra) + 1.5866 * (1 - d_intra) - 2.3561

NO CALIBRATION_ANCHORS dictionary! NO hardcoded dataset names!
========================================================================================
Usage:
    python test_calculate_thresholds.py
========================================================================================
"""

import sys
import argparse
from torch.utils.data import DataLoader

from workloads import load_workload
from utils.threshold_utils import (
    compute_direct_regression_threshold,
    W_CLT, W_CHAN, W_MARGIN, W_COHERENCE, BIAS
)


ALL_DATASETS = ["cifar10", "cifar100", "mnist", "gtsrb", "eurosat", "svhn"]

TARGET_THRESHOLDS = {
    "cifar10": 0.35,
    "cifar100": 0.20,
    "mnist": 0.20,
    "gtsrb": 0.45,
    "eurosat": 0.35,
    "svhn": None  # Unseen out-of-sample dataset!
}


def main():
    parser = argparse.ArgumentParser(description="Test Universal Regression Threshold in utils.threshold_utils")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory where datasets are stored")
    parser.add_argument("--phases", type=int, default=5, help="Number of continual learning phases (P)")
    args = parser.parse_args()

    print("=" * 115)
    print(f"{'Universal Statistical Regression Threshold Evaluation':^115}")
    print(f"{'(Calling utils.threshold_utils.compute_direct_regression_threshold directly)':^115}")
    print("=" * 115)

    print(f"[Universal Equation Parameters]")
    print(f"  tau* = {W_CLT:.4f} * (2/K)^0.25 + {W_CHAN:.4f} * sqrt(C_in/3) + ({W_MARGIN:.4f}) * Margin + {W_COHERENCE:.4f} * Coherence + ({BIAS:.4f})\n")

    results = []

    for name in ALL_DATASETS:
        print(f"[Loading] Processing workload: {name.upper()}...")
        try:
            _, test_dataset, metadata = load_workload(name, args.data_dir)
        except Exception as e:
            print(f"[Error] Failed to load {name}: {e}")
            continue

        num_classes = metadata["num_classes"]
        in_channels = metadata["in_channels"]
        target = TARGET_THRESHOLDS[name]

        loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

        # Directly call utils.threshold_utils
        res = compute_direct_regression_threshold(
            num_classes=num_classes,
            num_phases=args.phases,
            in_channels=in_channels,
            data_loaders=[loader]
        )

        tau_star = res["tau_star"]
        if target is not None:
            diff = abs(tau_star - target)
            status = "EXACT" if diff <= 0.01 else ("CLOSE" if diff <= 0.03 else f"DIFF ({diff:+.2f})")
            target_str = f"{target:.2f}"
            diff_str = f"{diff:.2f}"
        else:
            diff = 0.0
            status = "UNSEEN"
            target_str = "N/A"
            diff_str = "N/A"

        print(f"  -> {name.upper():<10} | K={res['K']:<5.1f} | d_inter={res['d_inter']:<6.4f} | d_intra={res['d_intra']:<6.4f} | "
              f"Margin={res['margin']:<6.4f} | Pred tau*={tau_star:.2f} (Target: {target_str}) -> {status}")

        results.append({
            "name": name,
            "classes": num_classes,
            "K": res["K"],
            "in_channels": in_channels,
            "d_inter": res["d_inter"],
            "d_intra": res["d_intra"],
            "margin": res["margin"],
            "tau_star": tau_star,
            "target": target_str,
            "diff": diff_str,
            "status": status
        })

    print("\n" + "=" * 120)
    print(f"{'UNIVERSAL REGRESSION SUMMARY TABLE':^120}")
    print("=" * 120)
    hdr = (
        f"| {'Workload':<10} | {'Classes':<7} | {'K':<5} | {'C_in':<4} | "
        f"{'d_inter':<9} | {'d_intra':<9} | {'Margin':<9} | {'Pred tau*':<10} | {'Target':<7} | {'Diff':<6} | {'Status':<10} |"
    )
    print(hdr)
    print("|" + "-" * 12 + "|" + "-" * 9 + "|" + "-" * 7 + "|" + "-" * 6 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 12 + "|" + "-" * 9 + "|" + "-" * 8 + "|" + "-" * 12 + "|")

    for r in results:
        row = (
            f"| {r['name']:<10} | {r['classes']:<7} | {r['K']:<5.1f} | {r['in_channels']:<4} | "
            f"{r['d_inter']:<9.4f} | {r['d_intra']:<9.4f} | {r['margin']:<9.4f} | {r['tau_star']:<10.2f} | "
            f"{r['target']:<7} | {r['diff']:<6} | {r['status']:<10} |"
        )
        print(row)
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
