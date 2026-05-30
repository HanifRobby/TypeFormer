"""
04_diagnostic_d2.py — Diagnostic D2: Stabilitas s_u.

Computes s_u(E=5) and s_u(E=15) for validation users and measures
Pearson correlation per dimension.

Decision rule:
  mean_corr > 0.8: FiLM viable without shrinkage
  0.6-0.8:         FiLM viable with shrinkage estimator
  < 0.6:           FiLM only as supporting evidence

Usage:
    conda run -n Typeformer python scripts/04_diagnostic_d2.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config
from src.utils.logging import setup_logger
from src.utils.seeds import set_global_seed
from src.data.aalto_loader import AaltoDataset
from src.statistics.user_stats import compute_user_stats, FEATURE_NAMES


def main() -> None:
    cfg = load_config("config/diagnostics/d2_stability.yaml")
    set_global_seed(cfg.seed)

    out_dir = ROOT / cfg.paths.results_dir / "diagnostics" / "d2_stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("d2_stability", out_dir)
    logger.info("=== Diagnostic D2: Stabilitas s_u ===")

    d = cfg.diagnostics
    n_val = d.n_val_users
    E_low = d.E_low
    E_high = d.E_high

    val_npz = ROOT / cfg.paths.processed_dir / "val_sessions.npz"
    if not val_npz.exists():
        logger.error("Val data not found — run 01_preprocess_data.py first.")
        sys.exit(1)

    val_ds = AaltoDataset(val_npz)
    n_val = min(n_val, val_ds.n_users)
    logger.info("Using %d validation users (E_low=%d, E_high=%d)", n_val, E_low, E_high)

    s_u_low_list, s_u_high_list = [], []
    for user_idx in range(n_val):
        sessions = val_ds.get_user_sessions(user_idx)   # (15, 50, 5)
        s_low = compute_user_stats(sessions[:E_low])
        s_high = compute_user_stats(sessions[:E_high])
        s_u_low_list.append(s_low)
        s_u_high_list.append(s_high)

    s_low = np.stack(s_u_low_list)    # (n_val, 20)
    s_high = np.stack(s_u_high_list)  # (n_val, 20)

    # Pearson correlation per dimension
    n_dims = s_low.shape[1]
    corrs = np.zeros(n_dims, dtype=np.float64)
    for dim in range(n_dims):
        r, _ = pearsonr(s_low[:, dim], s_high[:, dim])
        corrs[dim] = r

    mean_corr = float(np.mean(corrs))
    std_corr = float(np.std(corrs))
    logger.info("Pearson correlation (E=%d vs E=%d): mean=%.3f  std=%.3f",
                E_low, E_high, mean_corr, std_corr)
    logger.info("Per-dimension correlations: %s", np.round(corrs, 3))

    if mean_corr > d.stability_threshold_high:
        decision = "FiLM VIABLE without shrinkage (mean_corr > 0.8)"
    elif mean_corr > d.stability_threshold_low:
        decision = "FiLM viable with shrinkage estimator (0.6 < mean_corr <= 0.8)"
    else:
        decision = "FiLM only as supporting evidence (mean_corr <= 0.6)"
    logger.info("Decision: %s", decision)

    # Plot: correlation per dimension
    stat_labels = []
    stat_names = ["mean", "std", "p25", "p50", "p75"]
    for stat in stat_names:
        for feat in FEATURE_NAMES:
            stat_labels.append(f"{stat}_{feat[:3]}")

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(n_dims), corrs, color=['green' if c > 0.8 else 'orange' if c > 0.6 else 'red' for c in corrs])
    ax.axhline(0.8, color='green', linestyle='--', label='0.8 threshold')
    ax.axhline(0.6, color='orange', linestyle='--', label='0.6 threshold')
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels(stat_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Pearson Correlation")
    ax.set_title(f"D2: s_u Stability — Pearson(E={E_low}, E={E_high}) per dimension")
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_dir / "s_u_stability.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    report = {
        "experiment_id": "d2_stability",
        "n_val_users": n_val,
        "E_low": E_low,
        "E_high": E_high,
        "mean_pearson_corr": mean_corr,
        "std_pearson_corr": std_corr,
        "per_dim_correlations": corrs.tolist(),
        "decision": decision,
    }
    with open(out_dir / "d2_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=== D2 complete ===")
    logger.info("DECISION: %s", decision)


if __name__ == "__main__":
    main()
