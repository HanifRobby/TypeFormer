"""
07b_evaluate_film.py — Evaluate trained FiLM head (F1 and FN configurations).

For each test user and each E value:
  1. Compute s_u from the first E enrolment sessions (raw session data).
  2. Apply FiLM: e' = film_head(e, s_u) for all sessions of that user.
  3. Evaluate per-subject and global EER using FiLM-modulated embeddings.

Also runs a sanity check: evaluate with RANDOM s_u to detect if FiLM
has collapsed to s_u-based discrimination (not keystroke discrimination).

Usage:
    # Best seed (lowest val EER from training report)
    conda run -n Typeformer python scripts/07b_evaluate_film.py --exp f1_film_only

    # Specific seed checkpoint
    conda run -n Typeformer python scripts/07b_evaluate_film.py \\
        --exp f1_film_only --checkpoint results/f1_film_only/checkpoints/film_head_seed42.pt

    # With AS-Norm scoring (FN)
    conda run -n Typeformer python scripts/07b_evaluate_film.py --exp fn_full_system
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config
from src.utils.logging import setup_logger
from src.utils.seeds import set_global_seed
from src.utils.caching import get_cached_embeddings
from src.models.typeformer_wrapper import TypeFormerWrapper
from src.models.film_head import FiLMHead
from src.data.aalto_loader import AaltoDataset
from src.data.cohort_sampler import load_cohort_sessions, get_cohort_cache_paths, sample_cohort
from src.scoring.cosine_raw import RawCosineScorer
from src.scoring.asnorm import ASNormScorer
from src.statistics.user_stats import compute_user_stats
from src.evaluation.metrics import compute_eer
from src.evaluation.per_subject_eer import evaluate_per_subject
from src.evaluation.bootstrap_ci import bootstrap_global_eer_ci


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained FiLM head on test set.")
    p.add_argument("--exp", default="f1_film_only",
                   choices=["f1_film_only", "fn_full_system"],
                   help="Experiment ID (default: f1_film_only)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Explicit checkpoint path (default: best seed from training report)")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cohort-size", type=int, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--sanity-check", action="store_true",
                   help="Also evaluate with random s_u to detect s_u-based collapse")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core: apply FiLM to all sessions of all users
# ---------------------------------------------------------------------------

def apply_film_to_embeddings(
    film_head: FiLMHead,
    embeddings: np.ndarray,       # (N_users, N_sessions, D)
    sessions: np.ndarray,         # (N_users, N_sessions, L, 5) raw sessions
    E: int,
    device: str = "cuda",
    random_su: bool = False,
) -> np.ndarray:
    """Apply FiLM modulation to all user embeddings.

    s_u is computed from the first E sessions of each user (enrolment sessions).
    The SAME s_u is applied to all sessions of that user.

    Args:
        film_head: Trained FiLM module.
        embeddings: Pre-computed TypeFormer embeddings.
        sessions: Raw session arrays for s_u computation.
        E: Number of enrolment sessions used for s_u.
        device: FiLM computation device.
        random_su: If True, use random s_u (sanity check for collapse detection).

    Returns:
        (N_users, N_sessions, D) FiLM-modulated embeddings.
    """
    film_head.eval()
    n_users, n_sessions, D = embeddings.shape
    film_embs = np.zeros_like(embeddings)
    rng_su = np.random.RandomState(0) if random_su else None

    with torch.no_grad():
        for u in range(n_users):
            if random_su:
                s_u = rng_su.randn(film_head.fc1.in_features).astype(np.float32)
            else:
                s_u = compute_user_stats(sessions[u, :E])   # (20,)

            s_u_t = torch.from_numpy(s_u).float().to(device)
            s_u_batch = s_u_t.unsqueeze(0).expand(n_sessions, -1)   # (N_sess, 20)

            embs_u = torch.from_numpy(embeddings[u]).float().to(device)  # (N_sess, D)
            film_embs[u] = film_head(embs_u, s_u_batch).cpu().numpy()

    return film_embs


# ---------------------------------------------------------------------------
# Evaluation for one E value
# ---------------------------------------------------------------------------

def eval_one_e(
    film_embs: np.ndarray,
    scorer,
    E: int,
    cfg,
    logger,
    out_dir: Path,
    suffix: str = "",
) -> dict:
    """Evaluate EER for one E value using FiLM-modulated embeddings."""
    n_users = film_embs.shape[0]
    genuine_sessions = cfg.evaluation.genuine_sessions
    impostor_session_idx = cfg.evaluation.impostor_session_idx
    impostor_probes = film_embs[:, impostor_session_idx, :]

    all_genuine, all_impostor = [], []
    per_user_eers = []
    user_data_ci = []

    for user_idx in range(n_users):
        e_u = film_embs[user_idx, :E, :].mean(axis=0)
        genuine_embs = film_embs[user_idx, -genuine_sessions:, :]

        gen_sc = [scorer.score(e_u, ep) for ep in genuine_embs]
        imp_idx_list = [j for j in range(n_users) if j != user_idx]
        imp_sc = scorer.score_batch(e_u, impostor_probes[imp_idx_list]).tolist()

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
        user_data_ci, n_iterations=cfg.evaluation.bootstrap_iterations,
        ci_level=cfg.evaluation.bootstrap_ci, seed=cfg.seed,
    )

    logger.info(
        "E=%d%s | per_subj=%.2f%%(±%.2f%%) | global=%.2f%%[%.2f%%,%.2f%%]",
        E, suffix, mean_eer*100, std_eer*100, global_eer*100, ci_lo*100, ci_hi*100,
    )

    tag = f"E{E}{suffix}"
    np.savetxt(str(out_dir / f"per_user_eers_{tag}.csv"), per_user_eers,
               delimiter=",", fmt="%.6f")

    return {
        "E": E,
        "suffix": suffix,
        "mean_per_subject_eer": mean_eer,
        "std_per_subject_eer": std_eer,
        "global_eer": global_eer,
        "global_eer_ci_lower": ci_lo,
        "global_eer_ci_upper": ci_hi,
        "n_users": n_users,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(f"config/experiments/{args.exp}.yaml")
    set_global_seed(cfg.seed)

    out_dir = ROOT / cfg.paths.results_dir / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"{args.exp}_eval", out_dir)
    logger.info("=== Evaluating FiLM: %s on %s set ===", args.exp, args.split)

    # ------------------------------------------------------------------
    # Select checkpoint
    # ------------------------------------------------------------------
    ckpt_dir = out_dir / "checkpoints"
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        # Auto-select best seed from training report
        report_path = out_dir / "training_report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            best = min(report["seed_results"], key=lambda r: r["best_val_eer"])
            ckpt_path = Path(best["checkpoint"])
            logger.info("Auto-selected best seed: seed=%d  val_EER=%.2f%%",
                        best["seed"], best["best_val_eer"] * 100)
        else:
            # Fallback: seed 42
            ckpt_path = ckpt_dir / "film_head_seed42.pt"
            logger.warning("No training report found, using seed42 checkpoint")

    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)
    logger.info("Checkpoint: %s", ckpt_path)

    # ------------------------------------------------------------------
    # Load dataset and embeddings
    # ------------------------------------------------------------------
    npz_path = ROOT / cfg.paths.processed_dir / f"{args.split}_sessions.npz"
    if not npz_path.exists():
        logger.error("Data not found: %s", npz_path)
        sys.exit(1)

    dataset = AaltoDataset(npz_path)
    logger.info("Users: %d  Sessions: %d", dataset.n_users, dataset.n_sessions)

    wrapper = TypeFormerWrapper(checkpoint_path=ROOT / cfg.paths.typeformer_checkpoint)
    emb_cache = (ROOT / cfg.paths.results_dir / "embeddings_cache" / args.split /
                 f"{args.split}_embeddings.npy")
    embeddings = get_cached_embeddings(
        emb_cache,
        lambda: wrapper.encode_dataset(dataset.sessions, batch_size=args.batch_size,
                                       desc=f"Encoding {args.split} set"),
    )
    logger.info("Embeddings: %s", embeddings.shape)

    # ------------------------------------------------------------------
    # Load FiLM head
    # ------------------------------------------------------------------
    film_cfg = cfg.film
    film_head = FiLMHead(
        stats_dim=film_cfg.stats_dim,
        embedding_dim=film_cfg.embedding_dim,
        hidden_dim=film_cfg.hidden_dim,
        dropout=film_cfg.dropout,
    )
    film_head.load_state_dict(
        torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    )
    film_head.to("cuda").eval()
    logger.info("FiLM head loaded from %s", ckpt_path.name)

    # ------------------------------------------------------------------
    # Build scorer (cosine for F1, AS-Norm for FN)
    # ------------------------------------------------------------------
    scoring_method = cfg.scoring.method
    scorer = None

    if scoring_method == "asnorm":
        n_cohort = args.cohort_size or int(cfg.evaluation.cohort_size)
        K = args.K or int(cfg.scoring.K)
        pool_npz = ROOT / cfg.paths.processed_dir / "cohort_pool_sessions.npz"
        emb_p, _ = get_cohort_cache_paths(ROOT / cfg.paths.results_dir, n_cohort, cfg.seed)

        if emb_p.exists():
            cohort_embs = np.load(str(emb_p))
        else:
            logger.info("Computing cohort embeddings (N=%d)...", n_cohort)
            idx = sample_cohort(pool_npz, n_cohort=n_cohort, seed=cfg.seed)
            seqs = load_cohort_sessions(pool_npz, idx, session_idx=-1)
            cohort_embs = wrapper.encode_numpy(seqs)
            emb_p.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(emb_p), cohort_embs)

        scorer = ASNormScorer(cohort_embs, K=K, epsilon=float(cfg.scoring.epsilon))
        logger.info("AS-Norm scorer: K=%d  N_cohort=%d", K, n_cohort)
    else:
        scorer = RawCosineScorer()
        logger.info("Cosine scorer")

    # ------------------------------------------------------------------
    # Evaluate for all E values
    # ------------------------------------------------------------------
    all_results = {}
    E_values = list(cfg.evaluation.E_values)

    for E in E_values:
        logger.info("--- E=%d (FiLM-modulated) ---", E)

        film_embs = apply_film_to_embeddings(
            film_head, embeddings, dataset.sessions, E=E, device="cuda",
            random_su=False,
        )
        result = eval_one_e(film_embs, scorer, E, cfg, logger, out_dir)
        all_results[f"E_{E}"] = result

        # Sanity check: random s_u
        if args.sanity_check:
            logger.info("  [Sanity] Evaluating with RANDOM s_u ...")
            film_embs_rand = apply_film_to_embeddings(
                film_head, embeddings, dataset.sessions, E=E, device="cuda",
                random_su=True,
            )
            result_rand = eval_one_e(
                film_embs_rand, scorer, E, cfg, logger, out_dir, suffix="_random_su"
            )
            all_results[f"E_{E}_random_su"] = result_rand
            logger.info(
                "  [Sanity] E=%d | proper_s_u=%.2f%%  random_s_u=%.2f%%  "
                "(large gap → s_u matters; small gap → potential collapse)",
                E,
                result["mean_per_subject_eer"] * 100,
                result_rand["mean_per_subject_eer"] * 100,
            )

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    metrics = {
        "experiment_id": args.exp,
        "split": args.split,
        "checkpoint": str(ckpt_path),
        "scoring_method": scoring_method,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": all_results,
    }
    out_metrics = out_dir / f"metrics_{args.split}.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Results saved to %s", out_metrics)
    logger.info("=== F1 evaluation complete ===")


if __name__ == "__main__":
    main()
