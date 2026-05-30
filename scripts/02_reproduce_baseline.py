"""
02_reproduce_baseline.py — Reproduce TypeFormer paper baseline (B0).

Protocol (paper-faithful):
  - Scoring: mean pairwise Euclidean distance (negated to higher=genuine).
  - Template per probe: mean score across E individual enrolment embeddings.
  - Genuine probes: last 5 sessions per user.
  - Impostor probe: last session of each other user.
  - Metrics: mean per-subject EER + global EER with 95% bootstrap CI.

Target (E=5): mean per-subject EER ≈ 3.25% (±0.5% for pipeline validation).

Usage:
    conda run -n Typeformer python scripts/02_reproduce_baseline.py
    conda run -n Typeformer python scripts/02_reproduce_baseline.py --E 5
    conda run -n Typeformer python scripts/02_reproduce_baseline.py --all-E
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config
from src.utils.logging import setup_logger
from src.utils.seeds import set_global_seed
from src.utils.caching import get_cached_embeddings
from src.models.typeformer_wrapper import TypeFormerWrapper
from src.data.aalto_loader import AaltoDataset
from src.evaluation.metrics import compute_eer
from src.evaluation.per_subject_eer import evaluate_per_subject_mean_pairwise
from src.evaluation.global_eer import evaluate_global
from src.evaluation.bootstrap_ci import bootstrap_global_eer_ci


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce TypeFormer B0 baseline.")
    p.add_argument("--E", type=int, default=5, help="Enrolment sessions (default: 5)")
    p.add_argument("--all-E", action="store_true",
                   help="Run all E values: {1, 2, 5, 7, 10}")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Encoding batch size (default: 64)")
    p.add_argument("--no-cache", action="store_true",
                   help="Recompute embeddings even if cache exists")
    return p.parse_args()


def mean_pairwise_euclidean(enrol_embs: np.ndarray, probe_emb: np.ndarray) -> float:
    """B0 scoring: mean Euclidean distance (negated) from E enrolment embeddings.

    Args:
        enrol_embs: (E, D) enrolment embeddings.
        probe_emb: (D,) probe embedding.

    Returns:
        score: negative mean distance. Higher = more similar.
    """
    dists = np.linalg.norm(enrol_embs - probe_emb[np.newaxis, :], axis=1)  # (E,)
    return float(-dists.mean())


def run_for_E(
    embeddings: np.ndarray,
    E: int,
    cfg,
    logger,
    out_dir: Path,
) -> dict:
    """Run full evaluation pipeline for one value of E."""
    logger.info("--- E=%d ---", E)

    genuine_sessions = cfg.evaluation.genuine_sessions
    impostor_session_idx = cfg.evaluation.impostor_session_idx

    # Per-subject EER (mean pairwise protocol)
    mean_eer, std_eer, per_user_eers = evaluate_per_subject_mean_pairwise(
        embeddings,
        E=E,
        scoring_fn=mean_pairwise_euclidean,
        genuine_sessions=genuine_sessions,
        impostor_session_idx=impostor_session_idx,
    )
    logger.info(
        "E=%d  mean_per_subject_EER=%.2f%%  std=%.2f%%",
        E, mean_eer * 100, std_eer * 100,
    )

    # Global EER with centroid for the global evaluation loop
    # For global EER we use centroid (mean of E embeddings) for efficiency
    n_users = embeddings.shape[0]
    impostor_probes = embeddings[:, impostor_session_idx, :]

    all_genuine, all_impostor = [], []
    user_data_for_ci = []

    for user_idx in range(n_users):
        e_u = embeddings[user_idx, :E, :].mean(axis=0)
        genuine_embs = embeddings[user_idx, -genuine_sessions:, :]

        gen_sc = [mean_pairwise_euclidean(
            embeddings[user_idx, :E, :], ep) for ep in genuine_embs]
        imp_sc = [mean_pairwise_euclidean(
            embeddings[user_idx, :E, :], impostor_probes[j])
            for j in range(n_users) if j != user_idx]

        all_genuine.extend(gen_sc)
        all_impostor.extend(imp_sc)
        user_data_for_ci.append({
            "genuine_scores": np.array(gen_sc),
            "impostor_scores": np.array(imp_sc),
        })

    global_eer, global_thresh = compute_eer(
        np.array(all_genuine), np.array(all_impostor)
    )
    eer_pt, ci_lo, ci_hi = bootstrap_global_eer_ci(
        user_data_for_ci,
        n_iterations=cfg.evaluation.bootstrap_iterations,
        ci_level=cfg.evaluation.bootstrap_ci,
        seed=cfg.seed,
    )
    logger.info(
        "E=%d  global_EER=%.2f%%  95%%CI=[%.2f%%, %.2f%%]",
        E, global_eer * 100, ci_lo * 100, ci_hi * 100,
    )

    result = {
        "E": E,
        "mean_per_subject_eer": float(mean_eer),
        "std_per_subject_eer": float(std_eer),
        "global_eer": float(global_eer),
        "global_eer_ci_lower": float(ci_lo),
        "global_eer_ci_upper": float(ci_hi),
        "n_users": int(n_users),
    }

    # Save per-user EERs for Wilcoxon
    np.savetxt(
        str(out_dir / f"per_user_eers_E{E}.csv"),
        per_user_eers, delimiter=",", fmt="%.6f",
    )
    return result


def main() -> None:
    args = parse_args()
    cfg = load_config("config/experiments/b0_baseline.yaml")
    set_global_seed(cfg.seed)

    out_dir = ROOT / cfg.paths.results_dir / "b0_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("b0_baseline", out_dir)

    logger.info("=== B0 Baseline: TypeFormer paper-faithful Euclidean distance ===")

    # Load test dataset
    test_npz = ROOT / cfg.paths.processed_dir / "test_sessions.npz"
    if not test_npz.exists():
        logger.error("Test data not found: %s", test_npz)
        logger.error("Run scripts/01_preprocess_data.py first.")
        sys.exit(1)

    dataset = AaltoDataset(test_npz)
    logger.info("Test users: %d  sessions: %d", dataset.n_users, dataset.n_sessions)

    # Encode or load cached embeddings
    cache_path = ROOT / cfg.paths.results_dir / "embeddings_cache" / "test" / "test_embeddings.npy"
    if args.no_cache and cache_path.exists():
        cache_path.unlink()

    wrapper = TypeFormerWrapper(
        checkpoint_path=ROOT / cfg.paths.typeformer_checkpoint
    )

    def _compute():
        return wrapper.encode_dataset(
            dataset.sessions,
            batch_size=args.batch_size,
            desc="Encoding test set",
        )

    embeddings = get_cached_embeddings(cache_path, _compute)
    logger.info("Embeddings shape: %s", embeddings.shape)

    # Determine E values to run
    E_values = list(cfg.evaluation.E_values) if args.all_E else [args.E]

    all_results = {}
    for E in E_values:
        result = run_for_E(embeddings, E, cfg, logger, out_dir)
        all_results[f"E_{E}"] = result

    # Sanity check for E=5
    if 5 in E_values:
        eer_5 = all_results["E_5"]["mean_per_subject_eer"] * 100
        if abs(eer_5 - 3.25) > 0.5:
            logger.warning(
                "SANITY CHECK: E=5 per-subject EER=%.2f%% deviates from expected ~3.25%%",
                eer_5,
            )
        else:
            logger.info(
                "SANITY CHECK PASSED: E=5 per-subject EER=%.2f%% (expected ~3.25%%)", eer_5
            )

    # Save metrics
    metrics = {
        "experiment_id": cfg.experiment.id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scoring": vars(cfg.scoring) if hasattr(cfg.scoring, '__dict__') else str(cfg.scoring),
        "results": all_results,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Results saved to %s", out_dir)
    logger.info("=== B0 complete ===")


if __name__ == "__main__":
    main()
