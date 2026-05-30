"""
10_generate_tables.py — Generate result tables for thesis.

Reads metrics.json from each experiment directory and builds:
  - results_table.csv: mean per-subject EER and global EER for all configs × E
  - results_table.tex: LaTeX tabular (optional)

Usage:
    conda run -n Typeformer python scripts/10_generate_tables.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logging import setup_logger

RESULTS_DIR = ROOT / "results"
E_VALUES = [1, 2, 5, 7, 10]

EXP_ORDER = [
    "b0_baseline", "b1_centroid_eucl", "b2_centroid_cosine",
    "n1_znorm", "n2_tnorm", "n3_snorm", "n4_asnorm",
    "f1_film_only", "fn_full_system",
]

DISPLAY_NAMES = {
    "b0_baseline": "B0 (Euclidean)",
    "b1_centroid_eucl": "B1 (Centroid+Eucl)",
    "b2_centroid_cosine": "B2 (Centroid+Cos)",
    "n1_znorm": "N1 (Z-Norm)",
    "n2_tnorm": "N2 (T-Norm)",
    "n3_snorm": "N3 (S-Norm)",
    "n4_asnorm": "N4 (AS-Norm)",
    "f1_film_only": "F1 (FiLM only)",
    "fn_full_system": "FN (FiLM+AS-Norm)",
}


def load_metrics(exp_dir_name: str) -> dict | None:
    path = RESULTS_DIR / exp_dir_name / "metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    out_dir = RESULTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("generate_tables", out_dir)
    logger.info("=== Generating result tables ===")

    rows = []
    for exp_id in EXP_ORDER:
        metrics = load_metrics(exp_id)
        if metrics is None:
            logger.warning("No metrics.json found for %s — skipping.", exp_id)
            continue

        results = metrics.get("results", {})
        row = {"Config": DISPLAY_NAMES.get(exp_id, exp_id)}

        for E in E_VALUES:
            key = f"E_{E}"
            if key in results:
                r = results[key]
                # Per-subject EER
                row[f"EER_ps_E{E}"] = round(r["mean_per_subject_eer"] * 100, 2)
                # Global EER
                row[f"EER_gl_E{E}"] = round(r["global_eer"] * 100, 2)
                row[f"CI_E{E}"] = (
                    f"[{r['global_eer_ci_lower']*100:.2f}, {r['global_eer_ci_upper']*100:.2f}]"
                    if "global_eer_ci_lower" in r else "N/A"
                )
            else:
                row[f"EER_ps_E{E}"] = None
                row[f"EER_gl_E{E}"] = None
                row[f"CI_E{E}"] = "N/A"

        rows.append(row)

    if not rows:
        logger.warning("No results found. Run experiment scripts first.")
        return

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = out_dir / "results_table.csv"
    df.to_csv(str(csv_path), index=False)
    logger.info("CSV table saved to %s", csv_path)

    # Print summary for E=5
    logger.info("\n=== Results at E=5 ===")
    e5_cols = ["Config", "EER_ps_E5", "EER_gl_E5", "CI_E5"]
    existing_cols = [c for c in e5_cols if c in df.columns]
    logger.info("\n%s", df[existing_cols].to_string(index=False))

    logger.info("=== Table generation complete ===")


if __name__ == "__main__":
    main()
