#!/usr/bin/env python3
"""
Kaggle-specific batch testing script.
Optimized for Kaggle notebook environment.
"""

import os
import sys
import re
from pathlib import Path


# Configuration for Kaggle
KAGGLE_CONFIG = {
    "dataset_root": "/kaggle/input/casiab",
    "dataset_partition": "./misc/partitions/CASIA-B_include_005.json",
    "output_dir": "/kaggle/working/batch_test_results",
    # Models to test
    "models": {
        'HSTL': {
            'save_name': 'HSTL',
            'checkpoint_base': '/kaggle/input/project-hstl/output/CASIA-B/HSTL/HSTL/checkpoints',
            'iterations': [80000],
            'backbone_type': 'conv3d',
            'loss_type': 'triplet',
        },
        'kaggle': {
            'save_name': 'kaggle',
            'checkpoint_base': '/kaggle/input/project-hstl/output/CASIA-B/HSTL/kaggle/checkpoints',
            'iterations': [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000],
            'backbone_type': 'conv3d',
            'loss_type': 'triplet',},
        "p3d": {
            "save_name": "p3d",
            "checkpoint_base": "/kaggle/input/project-hstl/output/CASIA-B/HSTL/p3d/checkpoints",
            "iterations": [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000,
            ],
            "backbone_type": "p3d",
            "loss_type": "triplet",
        },
        'med-circleloss': {
            'save_name': 'med-circleloss',
            'checkpoint_base': '/kaggle/input/project-hstl/output/CASIA-B/HSTL/med-circleloss/checkpoints',
            'iterations': [5000, 10000, 15000, 18000, 21000, 24000, 27000, 30000, 33000, 36000, 39000, 42000, 45000, 48000, 51000, 54000, 57000, 60000, 63000, 66000, 69000, 72000, 75000, 78000, 81000, 84000, 87000, 90000, 92500, 95000, 97500, 100000],
            'backbone_type': 'conv3d',
            'loss_type': 'circle',
        },
        'p3d-circleloss': {
            'save_name': 'p3d-circleloss',
            'checkpoint_base': '/kaggle/input/project-hstl/output/CASIA-B/HSTL/p3d-circleloss/checkpoints',
            'iterations': [2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000, 32500, 35000, 37500, 40000, 42500, 45000, 47500, 50000, 52500, 55000, 57500, 60000, 62500, 65000, 67500, 70000, 72500, 75000, 77500, 80000, 82500, 85000, 87500, 90000, 92500, 95000, 97500, 100000],
            'backbone_type': 'p3d',
            'loss_type': 'circle',
        }
    },
}


def parse_log_results(log_content):
    """Parse test results from log output."""
    results = {}

    lines = log_content.split("\n")

    for i, line in enumerate(lines):
        # Include identical-view cases
        if "Rank-1 (Include identical-view cases)" in line:
            if i + 1 < len(lines):
                match = re.search(
                    r"NM:\s+([\d.]+),\s+BG:\s+([\d.]+),\s+CL:\s+([\d.]+)", lines[i + 1]
                )
                if match:
                    results["include_NM"] = float(match.group(1))
                    results["include_BG"] = float(match.group(2))
                    results["include_CL"] = float(match.group(3))

        # Exclude identical-view cases
        if "Rank-1 (Exclude identical-view cases)" in line:
            if i + 1 < len(lines):
                match = re.search(
                    r"NM:\s+([\d.]+),\s+BG:\s+([\d.]+),\s+CL:\s+([\d.]+)", lines[i + 1]
                )
                if match:
                    results["exclude_NM"] = float(match.group(1))
                    results["exclude_BG"] = float(match.group(2))
                    results["exclude_CL"] = float(match.group(3))

        # Angle-wise results
        if "Rank-1 of each angle" in line:
            angles = {"NM": [], "BG": [], "CL": []}
            for j in range(i + 1, min(i + 10, len(lines))):
                for condition in ["NM", "BG", "CL"]:
                    if f"{condition}:" in lines[j] and "[" in lines[j]:
                        angles[condition] = lines[j].split(f"{condition}:")[1].strip()
            results["angles"] = angles

    return results


def test_single_checkpoint(
    model_name,
    iteration,
    checkpoint_path,
    dataset_root,
    partition,
    backbone_type,
    loss_type,
):
    """Test a single checkpoint."""

    print(f"\nTesting {model_name} - Iteration {iteration}")
    print("-" * 80)

    # Import here to avoid issues
    # sys.path.insert(0, '/kaggle/working')
    # from lib.main import run_model, initialization
    # from lib.utils.common import config_loader

    # Create temp config
    temp_config_path = f"/kaggle/working/temp_test_{model_name}_{iteration}.yaml"

    config_yaml = f"""data_cfg:
  dataset_name: CASIA-B
  dataset_root: {dataset_root}
  dataset_partition: {partition}
  num_workers: 4
  remove_no_gallery: false
  test_dataset_name: CASIA-B

evaluator_cfg:
  enable_distributed: false
  enable_float16: false
  restore_ckpt_strict: true
  restore_hint: {iteration}
  save_name: {model_name}
  sampler:
    batch_size: 1
    sample_type: all_ordered
    type: InferenceSampler

model_cfg:
  model: HSTL
  channels: [32, 64, 128]
  class_num: 74
  backbone_type: {backbone_type}
  loss_type: {loss_type}
"""

    with open(temp_config_path, "w") as f:
        f.write(config_yaml)

    # Run test
    try:
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "lib/main.py",
                "--cfgs",
                temp_config_path,
                "--phase",
                "test",
                "--iter",
                str(iteration),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        output = result.stdout + result.stderr
        results = parse_log_results(output)

        # Clean up
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

        return results, output

    except Exception as e:
        print(f"ERROR: {e}")
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        return None, str(e)


def run_batch_test():
    """Run batch testing for all models."""

    config = KAGGLE_CONFIG
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # Test each model
    for model_key, model_info in config["models"].items():
        print(f"\n\n{'=' * 80}")
        print(f"Testing Model: {model_key}")
        print(f"{'=' * 80}")

        all_results[model_key] = {}

        for iteration in model_info["iterations"]:
            checkpoint_file = f"{model_info['save_name']}-{iteration:05d}.pt"
            checkpoint_path = os.path.join(
                model_info["checkpoint_base"], checkpoint_file
            )

            if not os.path.exists(checkpoint_path):
                print(f"SKIP: Checkpoint not found - {checkpoint_file}")
                continue

            results, log_output = test_single_checkpoint(
                model_info["save_name"],
                iteration,
                checkpoint_path,
                config["dataset_root"],
                config["dataset_partition"],
                model_info["backbone_type"],
                model_info["loss_type"],
            )

            all_results[model_key][iteration] = results
            print(results)

            # Save individual log
            log_dir = os.path.join(output_dir, "logs", model_key)
            os.makedirs(log_dir, exist_ok=True)

            log_file = os.path.join(log_dir, f"{model_key}_{iteration}.log")
            with open(log_file, "w") as f:
                f.write(log_output)

    # Generate summary
    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append("BATCH TEST RESULTS SUMMARY")
    summary_lines.append("=" * 100)

    for model_key, iterations in all_results.items():
        summary_lines.append(f"\n{model_key}:")
        summary_lines.append("-" * 100)
        summary_lines.append(
            f"{'Iter':<10} {'NM_Incl':<10} {'BG_Incl':<10} {'CL_Incl':<10} {'NM_Excl':<10} {'BG_Excl':<10} {'CL_Excl':<10}"
        )
        summary_lines.append("-" * 100)

        for iteration, result in sorted(iterations.items()):
            if result:
                summary_lines.append(
                    f"{iteration:<10} "
                    f"{result.get('include_NM', 0):<10.1f} "
                    f"{result.get('include_BG', 0):<10.1f} "
                    f"{result.get('include_CL', 0):<10.1f} "
                    f"{result.get('exclude_NM', 0):<10.1f} "
                    f"{result.get('exclude_BG', 0):<10.1f} "
                    f"{result.get('exclude_CL', 0):<10.1f}"
                )

    summary_lines.append("\n" + "=" * 100)
    summary_text = "\n".join(summary_lines)

    print("\n\n" + summary_text)

    # Save summary
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    run_batch_test()
