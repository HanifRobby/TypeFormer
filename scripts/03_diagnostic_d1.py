"""
03_diagnostic_d1.py — Diagnostic D1: Heterogenitas sigma_u.

Computes the per-user cohort score standard deviation (sigma_u) for a sample
of validation users and plots its histogram.

Decision rule:
  max(sigma_u) / min(sigma_u) > 2x → AS-Norm viable
  else                              → revisit approach

Usage:
    conda run -n Typeformer python scripts/03_diagnostic_d1.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config
from src.utils.logging import setup_logger
from src.utils.seeds import set_global_seed
from src.utils.caching import get_cached_embeddings
from src.models.typeformer_wrapper import TypeFormerWrapper
from src.data.aalto_loader import AaltoDataset
from src.data.cohort_sampler import sample_cohort, load_cohort_sessions


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def main() -> None:
    cfg = load_config("config/diagnostics/d1_heterogeneity.yaml")
    set_global_seed(cfg.seed)

    out_dir = ROOT / cfg.paths.results_dir / "diagnostics" / "d1_heterogeneity"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("d1_heterogeneity", out_dir)
    logger.info("=== Diagnostic D1: Heterogenitas sigma_u ===")

    d = cfg.diagnostics
    n_val = d.n_val_users
    n_cohort = d.cohort_size
    E = d.E_for_sigma

    # Load validation embeddings
    val_npz = ROOT / cfg.paths.processed_dir / "val_sessions.npz"
    if not val_npz.exists():
        logger.error("Val data not found — run 01_preprocess_data.py first.")
        sys.exit(1)

    val_ds = AaltoDataset(val_npz)
    n_val = min(n_val, val_ds.n_users)
    logger.info("Using %d validation users (E=%d)", n_val, E)

    wrapper = TypeFormerWrapper(checkpoint_path=ROOT / cfg.paths.typeformer_checkpoint)

    # Encode validation set
    val_cache = ROOT / cfg.paths.results_dir / "embeddings_cache" / "val" / "val_embeddings.npy"
    val_embs = get_cached_embeddings(
        val_cache,
        lambda: wrapper.encode_dataset(val_ds.sessions, desc="Encoding val set"),
    )  # (N_val, 15, 64)

    # Sample cohort from cohort_pool
    cohort_pool_npz = ROOT / cfg.paths.processed_dir / "cohort_pool_sessions.npz"
    if not cohort_pool_npz.exists():
        logger.error("Cohort pool not found — run 01_preprocess_data.py first.")
        sys.exit(1)

    cohort_idx = sample_cohort(cohort_pool_npz, n_cohort=n_cohort, seed=cfg.seed)
    cohort_seqs = load_cohort_sessions(cohort_pool_npz, cohort_idx, session_idx=-1)
    cohort_embs = wrapper.encode_numpy(cohort_seqs)   # (n_cohort, 64)
    logger.info("Cohort embeddings: %s", cohort_embs.shape)

    # Compute sigma_u for each validation user
    sigma_u_list = []
    for user_idx in range(n_val):
        e_u = val_embs[user_idx, :E, :].mean(axis=0)   # centroid of E sessions
        e_u_n = e_u / (np.linalg.norm(e_u) + 1e-12)
        cohort_n = cohort_embs / (np.linalg.norm(cohort_embs, axis=1, keepdims=True) + 1e-12)
        scores_u = cohort_n @ e_u_n    # (n_cohort,)
        sigma_u_list.append(float(scores_u.std()))

    sigma_u = np.array(sigma_u_list)
    ratio = sigma_u.max() / (sigma_u.min() + 1e-12)

    logger.info("sigma_u: mean=%.4f  std=%.4f  min=%.4f  max=%.4f  ratio=%.2f",
                sigma_u.mean(), sigma_u.std(), sigma_u.min(), sigma_u.max(), ratio)

    decision = "AS-Norm VIABLE" if ratio > d.decision_threshold else "AS-Norm questionable — discuss with supervisor"
    logger.info("Decision (ratio > %.1f): %s", d.decision_threshold, decision)

    # Plot histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sigma_u, bins=20, edgecolor='black', alpha=0.8)
    ax.set_xlabel("sigma_u (std of cohort scores)")
    ax.set_ylabel("Count")
    ax.set_title(f"D1: Distribution of sigma_u across {n_val} validation users (E={E})")
    ax.axvline(sigma_u.mean(), color='red', linestyle='--', label=f'mean={sigma_u.mean():.4f}')
    ax.legend()
    fig_path = out_dir / "sigma_u_histogram.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Histogram saved to %s", fig_path)

    # Save results
    report = {
        "experiment_id": "d1_heterogeneity",
        "n_val_users": n_val,
        "n_cohort": n_cohort,
        "E": E,
        "sigma_u_mean": float(sigma_u.mean()),
        "sigma_u_std": float(sigma_u.std()),
        "sigma_u_min": float(sigma_u.min()),
        "sigma_u_max": float(sigma_u.max()),
        "max_min_ratio": float(ratio),
        "decision_threshold": float(d.decision_threshold),
        "decision": decision,
        "sigma_u_values": sigma_u.tolist(),
    }
    with open(out_dir / "d1_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=== D1 complete ===")
    logger.info("DECISION: %s", decision)


if __name__ == "__main__":
    main()
