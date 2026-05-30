"""
09_statistical_analysis.py — Wilcoxon signed-rank tests + bootstrap CI summary.

Loads per-user EER CSVs from each experiment result directory and runs
the 5 primary comparisons with Bonferroni correction.

Usage:
    conda run -n Typeformer python scripts/09_statistical_analysis.py --E 5
    conda run -n Typeformer python scripts/09_statistical_analysis.py --E 5 --E 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config
from src.utils.logging import setup_logger
from src.statistics.wilcoxon import compare_configurations, DEFAULT_COMPARISONS

RESULTS_DIR = ROOT / "results"

EXP_DIRS = {
    "b0_baseline": "b0_baseline",
    "b1_centroid_eucl": "b1_centroid_eucl",
    "b2_centroid_cosine": "b2_centroid_cosine",
    "n1_znorm": "n1_znorm",
    "n2_tnorm": "n2_tnorm",
    "n3_snorm": "n3_snorm",
    "n4_asnorm": "n4_asnorm",
    "f1_film_only": "f1_film_only",
    "fn_full_system": "fn_full_system",
}


def load_per_user_eers(exp_dir_name: str, E: int) -> np.ndarray | None:
    path = RESULTS_DIR / exp_dir_name / f"per_user_eers_E{E}.csv"
    if not path.exists():
        return None
    return np.loadtxt(str(path), delimiter=",")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Statistical analysis of experiment results.")
    p.add_argument("--E", nargs="+", type=int, default=[5],
                   help="E values to analyse (default: 5)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Family-wise error rate (default: 0.05)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    out_dir = RESULTS_DIR / "statistical_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("statistical_analysis", out_dir)
    logger.info("=== Statistical Analysis ===")
    logger.info("E values: %s  alpha: %.2f", args.E, args.alpha)

    all_tables = {}

    for E in args.E:
        logger.info("\n--- E=%d ---", E)

        eer_dict = {}
        missing = []
        for name, dir_name in EXP_DIRS.items():
            arr = load_per_user_eers(dir_name, E)
            if arr is None:
                missing.append(name)
            else:
                eer_dict[name] = arr
                logger.info("  %s: n=%d  mean=%.2f%%", name, len(arr), arr.mean()*100)

        if missing:
            logger.warning("Missing results for E=%d: %s — skipping some comparisons.", E, missing)

        # Filter comparisons to those with both configs available
        valid_comparisons = [
            (a, b) for (a, b) in DEFAULT_COMPARISONS
            if a in eer_dict and b in eer_dict
        ]
        if not valid_comparisons:
            logger.warning("No valid comparisons for E=%d.", E)
            continue

        df = compare_configurations(eer_dict, valid_comparisons, alpha=args.alpha)
        logger.info("\nWilcoxon results (E=%d):", E)
        logger.info("\n%s", df.to_string(index=False))

        csv_path = out_dir / f"wilcoxon_E{E}.csv"
        df.to_csv(str(csv_path), index=False)
        all_tables[f"E_{E}"] = df.to_dict(orient="records")

    # Save summary JSON
    summary_path = out_dir / "statistical_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_tables, f, indent=2)

    logger.info("Results saved to %s", out_dir)
    logger.info("=== Statistical analysis complete ===")


if __name__ == "__main__":
    main()
