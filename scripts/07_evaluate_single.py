"""
07_evaluate_single.py — Evaluate one experiment configuration for all E values.

Supports B1, B2, N1, N2, N3, N4, F1, FN configurations.
B0 is handled separately by 02_reproduce_baseline.py (mean-pairwise protocol).

Usage:
    conda run -n Typeformer python scripts/07_evaluate_single.py --config config/experiments/n4_asnorm.yaml
    conda run -n Typeformer python scripts/07_evaluate_single.py --config config/experiments/b2_centroid_cosine.yaml
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
from src.data.cohort_sampler import sample_cohort, load_cohort_sessions
from src.scoring.cosine_raw import RawCosineScorer
from src.scoring.znorm import ZNormScorer
from src.scoring.tnorm import TNormScorer
from src.scoring.snorm import SNormScorer
from src.scoring.asnorm import ASNormScorer
from src.evaluation.metrics import compute_eer
from src.evaluation.per_subject_eer import evaluate_per_subject
from src.evaluation.bootstrap_ci import bootstrap_global_eer_ci


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate one experiment config.")
    p.add_argument("--config", required=True,
                   help="Path to experiment YAML (e.g. config/experiments/n4_asnorm.yaml)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--split", default="test",
                   choices=["test", "val"],
                   help="Dataset split to evaluate on (default: test)")
    return p.parse_args()


def _neg_euclidean(e_u: np.ndarray, e_p: np.ndarray) -> float:
    return float(-np.linalg.norm(e_u - e_p))


def build_scorer(method: str, cohort: np.ndarray | None, K: int, epsilon: float):
    """Instantiate scorer based on config method string."""
    if method == "cosine":
        return RawCosineScorer()
    if method == "euclidean":
        return None   # handled separately via centroid
    if method == "znorm":
        return ZNormScorer(cohort, epsilon=epsilon)
    if method == "tnorm":
        return TNormScorer(cohort, epsilon=epsilon)
    if method == "snorm":
        return SNormScorer(cohort, epsilon=epsilon)
    if method == "asnorm":
        return ASNormScorer(cohort, K=K, epsilon=epsilon)
    raise ValueError(f"Unknown scoring method: {method}")


def run_for_E(
    embeddings: np.ndarray,
    scorer,
    method: str,
    E: int,
    cfg,
    logger,
    out_dir: Path,
) -> dict:
    logger.info("--- E=%d ---", E)
    n_users = embeddings.shape[0]
    genuine_sessions = cfg.evaluation.genuine_sessions
    impostor_session_idx = cfg.evaluation.impostor_session_idx
    impostor_probes = embeddings[:, impostor_session_idx, :]

    all_genuine, all_impostor = [], []
    per_user_eers = []
    user_data_ci = []

    for user_idx in range(n_users):
        e_u = embeddings[user_idx, :E, :].mean(axis=0)     # centroid
        genuine_embs = embeddings[user_idx, -genuine_sessions:, :]

        if method == "euclidean":
            gen_sc = [_neg_euclidean(e_u, ep) for ep in genuine_embs]
            imp_sc = [_neg_euclidean(e_u, impostor_probes[j])
                      for j in range(n_users) if j != user_idx]
        else:
            gen_sc = [scorer.score(e_u, ep) for ep in genuine_embs]
            imp_sc = scorer.score_batch(e_u, impostor_probes[
                [j for j in range(n_users) if j != user_idx]
            ]).tolist()

        gen_arr = np.array(gen_sc)
        imp_arr = np.array(imp_sc)

        eer, _ = compute_eer(gen_arr, imp_arr)
        per_user_eers.append(eer)
        all_genuine.extend(gen_sc)
        all_impostor.extend(imp_sc)
        user_data_ci.append({"genuine_scores": gen_arr, "impostor_scores": imp_arr})

    per_user_eers = np.array(per_user_eers)
    mean_eer = float(per_user_eers.mean())
    std_eer = float(per_user_eers.std())

    global_eer, _ = compute_eer(np.array(all_genuine), np.array(all_impostor))
    eer_pt, ci_lo, ci_hi = bootstrap_global_eer_ci(
        user_data_ci,
        n_iterations=cfg.evaluation.bootstrap_iterations,
        ci_level=cfg.evaluation.bootstrap_ci,
        seed=cfg.seed,
    )

    logger.info(
        "E=%d | per_subj=%.2f%%(±%.2f%%) | global=%.2f%%[%.2f%%,%.2f%%]",
        E, mean_eer*100, std_eer*100, global_eer*100, ci_lo*100, ci_hi*100,
    )

    np.savetxt(str(out_dir / f"per_user_eers_E{E}.csv"), per_user_eers,
               delimiter=",", fmt="%.6f")

    return {
        "E": E,
        "mean_per_subject_eer": mean_eer,
        "std_per_subject_eer": std_eer,
        "global_eer": global_eer,
        "global_eer_ci_lower": ci_lo,
        "global_eer_ci_upper": ci_hi,
        "n_users": n_users,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.seed)

    exp_id = cfg.experiment.id
    out_dir = ROOT / cfg.paths.results_dir / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(exp_id, out_dir)
    logger.info("=== Evaluating: %s ===", exp_id)

    # Load dataset
    npz_path = ROOT / cfg.paths.processed_dir / f"{args.split}_sessions.npz"
    if not npz_path.exists():
        logger.error("Data not found: %s — run 01_preprocess_data.py first.", npz_path)
        sys.exit(1)

    dataset = AaltoDataset(npz_path)
    logger.info("Users: %d  Sessions: %d  Split: %s", dataset.n_users, dataset.n_sessions, args.split)

    # Load/compute embeddings
    wrapper = TypeFormerWrapper(checkpoint_path=ROOT / cfg.paths.typeformer_checkpoint)
    cache_path = (ROOT / cfg.paths.results_dir / "embeddings_cache" / args.split /
                  f"{args.split}_embeddings.npy")
    if args.no_cache and cache_path.exists():
        cache_path.unlink()

    embeddings = get_cached_embeddings(
        cache_path,
        lambda: wrapper.encode_dataset(dataset.sessions, batch_size=args.batch_size,
                                       desc=f"Encoding {args.split} set"),
    )
    logger.info("Embeddings: %s", embeddings.shape)

    # Build scorer
    method = cfg.scoring.method
    scorer = None
    cohort_embs = None

    if method in ("znorm", "tnorm", "snorm", "asnorm"):
        # Load cohort embeddings
        pool_npz = ROOT / cfg.paths.processed_dir / "cohort_pool_sessions.npz"
        if not pool_npz.exists():
            logger.error("Cohort pool not found — run 01_preprocess_data.py.")
            sys.exit(1)

        cohort_cache = ROOT / cfg.paths.results_dir / "cohort" / "cohort_embeddings.npy"
        cohort_idx_path = ROOT / cfg.paths.results_dir / "cohort" / "cohort_indices.npy"

        if not cohort_cache.exists():
            logger.info("Computing cohort embeddings...")
            cohort_idx = sample_cohort(
                pool_npz, n_cohort=cfg.evaluation.cohort_size, seed=cfg.seed,
                save_path=cohort_idx_path,
            )
            cohort_seqs = load_cohort_sessions(pool_npz, cohort_idx, session_idx=-1)
            cohort_embs_raw = wrapper.encode_numpy(cohort_seqs)
            cohort_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cohort_cache), cohort_embs_raw)
            cohort_embs = cohort_embs_raw
        else:
            logger.info("Loading cached cohort embeddings from %s", cohort_cache)
            cohort_embs = np.load(str(cohort_cache))

        logger.info("Cohort embeddings: %s", cohort_embs.shape)

        K = int(cfg.scoring.K)
        epsilon = float(cfg.scoring.epsilon)
        scorer = build_scorer(method, cohort_embs, K=K, epsilon=epsilon)
        logger.info("Scorer: %s  K=%d  epsilon=%g", method, K, epsilon)
    else:
        scorer = build_scorer(method, None, K=0, epsilon=1e-3)
        logger.info("Scorer: %s", method)

    # Run evaluation for all E values
    all_results = {}
    for E in list(cfg.evaluation.E_values):
        result = run_for_E(embeddings, scorer, method, E, cfg, logger, out_dir)
        all_results[f"E_{E}"] = result

    # Save metrics
    scoring_dict = {k: getattr(cfg.scoring, k) for k in vars(cfg.scoring)} \
        if hasattr(cfg.scoring, '__dict__') else {}

    metrics = {
        "experiment_id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "scoring": scoring_dict,
        "results": all_results,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Results saved to %s", out_dir)
    logger.info("=== %s complete ===", exp_id)


if __name__ == "__main__":
    main()
