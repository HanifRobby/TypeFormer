"""
08_run_main_matrix.py — Orchestrate all 9 experiment configurations.

Runs B0 (separate script logic), B1, B2, N1-N4 for all E values.
F1 and FN require pre-trained FiLM checkpoints (run 06_train_film.py first).

Usage:
    conda run -n Typeformer python scripts/08_run_main_matrix.py
    conda run -n Typeformer python scripts/08_run_main_matrix.py --configs b2 n4
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

CONFIGS = {
    "b0": "config/experiments/b0_baseline.yaml",
    "b1": "config/experiments/b1_centroid_eucl.yaml",
    "b2": "config/experiments/b2_centroid_cosine.yaml",
    "n1": "config/experiments/n1_znorm.yaml",
    "n2": "config/experiments/n2_tnorm.yaml",
    "n3": "config/experiments/n3_snorm.yaml",
    "n4": "config/experiments/n4_asnorm.yaml",
    "f1": "config/experiments/f1_film_only.yaml",
    "fn": "config/experiments/fn_full_system.yaml",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all experiment configurations.")
    p.add_argument("--configs", nargs="+",
                   choices=list(CONFIGS.keys()) + ["all"],
                   default=["all"],
                   help="Which configs to run (default: all)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def run_config(config_key: str, batch_size: int, no_cache: bool) -> bool:
    """Run one config via subprocess. Returns True on success."""
    config_path = CONFIGS[config_key]
    print(f"\n{'='*60}")
    print(f"Running: {config_key}  ({config_path})")
    print(f"{'='*60}")

    if config_key == "b0":
        script = str(ROOT / "scripts" / "02_reproduce_baseline.py")
        cmd = [sys.executable, script, "--all-E", f"--batch-size={batch_size}"]
    elif config_key in ("f1", "fn"):
        print(f"  Skipping {config_key}: requires pre-trained FiLM (run 06_train_film.py)")
        return True
    else:
        script = str(ROOT / "scripts" / "07_evaluate_single.py")
        cmd = [sys.executable, script, f"--config={config_path}",
               f"--batch-size={batch_size}"]

    if no_cache:
        cmd.append("--no-cache")

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  ERROR: {config_key} failed with return code {result.returncode}")
        return False
    return True


def main() -> None:
    args = parse_args()

    configs_to_run = list(CONFIGS.keys()) if "all" in args.configs else args.configs

    print(f"Running configurations: {configs_to_run}")

    failed = []
    for key in configs_to_run:
        ok = run_config(key, args.batch_size, args.no_cache)
        if not ok:
            failed.append(key)

    print(f"\n{'='*60}")
    print(f"Matrix run complete.")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"All {len(configs_to_run)} configurations completed successfully.")


if __name__ == "__main__":
    main()
