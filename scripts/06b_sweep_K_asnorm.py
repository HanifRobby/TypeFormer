"""
06b_sweep_K_asnorm.py — Sweep K for AS-Norm on the validation set.

Evaluates AS-Norm with K ∈ {10, 30, 50, 100, 200, 500} on the validation
split (never the test split) and selects K_best using two criteria:
  1. Primary: lowest global EER on validation set.
  2. Constraint: per-subject EER must not rise > MAX_PERSUBJ_DELTA above
     the cosine (B2) baseline on the same validation set.

Output:
  - results/k_sweep/k_sweep_results.json  — full numeric results
  - results/k_sweep/k_vs_eer.png          — K vs EER plot (both metrics)
  - results/k_sweep/k_best.txt            — single line: optimal K value

Usage:
    conda run -n Typeformer python scripts/06b_sweep_K_asnorm.py
    conda run -n Typeformer python scripts/06b_sweep_K_asnorm.py --cohort-size 2000 --E 5
    conda run -n Typeformer python scripts/06b_sweep_K_asnorm.py --K-values 10 30 50 100 200 500
"""

import argparse
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
from src.data.cohort_sampler import sample_cohort, load_cohort_sessions, get_cohort_cache_paths
from src.scoring.asnorm import ASNormScorer
from src.scoring.cosine_raw import RawCosineScorer
from src.evaluation.metrics import compute_eer

# Maximum allowed per-subject EER degradation relative to cosine baseline (absolute)
MAX_PERSUBJ_DELTA = 0.002   # 0.2 % absolute


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep K for AS-Norm on validation set.")
    p.add_argument("--K-values", nargs="+", type=int,
                   default=[10, 30, 50, 100, 200, 500],
                   help="K values to sweep (default: 10 30 50 100 200 500)")
    p.add_argument("--cohort-size", type=int, default=None,
                   help="Override cohort_size from config (default: 2000)")
    p.add_argument("--E", type=int, default=5,
                   help="Enrolment sessions for sweep evaluation (default: 5)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_one_k(
    embeddings: np.ndarray,
    scorer,
    E: int,
    genuine_sessions: int,
    impostor_session_idx: int,
) -> tuple[float, float, np.ndarray]:
    """Return (global_eer, mean_per_subj_eer, per_user_eers) for given scorer.

    Uses centroid template (mean of E enrolment embeddings).
    """
    n_users = embeddings.shape[0]
    impostor_probes = embeddings[:, impostor_session_idx, :]

    all_genuine, all_impostor = [], []
    per_user_eers = []

    for user_idx in range(n_users):
        e_u = embeddings[user_idx, :E, :].mean(axis=0)
        genuine_embs = embeddings[user_idx, -genuine_sessions:, :]

        gen_sc = [scorer.score(e_u, ep) for ep in genuine_embs]
        imp_sc = scorer.score_batch(
            e_u,
            impostor_probes[[j for j in range(n_users) if j != user_idx]]
        ).tolist()

        gen_arr = np.array(gen_sc)
        imp_arr = np.array(imp_sc)

        eer, _ = compute_eer(gen_arr, imp_arr)
        per_user_eers.append(eer)
        all_genuine.extend(gen_sc)
        all_impostor.extend(imp_sc)

    global_eer, _ = compute_eer(np.array(all_genuine), np.array(all_impostor))
    per_user_eers = np.array(per_user_eers)
    return global_eer, float(per_user_eers.mean()), per_user_eers


def main() -> None:
    args = parse_args()
    cfg = load_config("config/experiments/n4_asnorm.yaml")
    set_global_seed(cfg.seed)

    out_dir = ROOT / cfg.paths.results_dir / "k_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("k_sweep", out_dir)
    logger.info("=== AS-Norm K Sweep ===")
    logger.info("K values: %s", args.K_values)
    logger.info("E=%d  split=val", args.E)

    n_cohort = args.cohort_size or int(cfg.evaluation.cohort_size)
    E = args.E
    genuine_sessions = cfg.evaluation.genuine_sessions
    impostor_session_idx = cfg.evaluation.impostor_session_idx

    # ------------------------------------------------------------------
    # Load validation embeddings
    # ------------------------------------------------------------------
    val_npz = ROOT / cfg.paths.processed_dir / "val_sessions.npz"
    if not val_npz.exists():
        logger.error("Val data not found — run 01_preprocess_data.py first.")
        sys.exit(1)

    val_ds = AaltoDataset(val_npz)
    logger.info("Val users: %d", val_ds.n_users)

    wrapper = TypeFormerWrapper(checkpoint_path=ROOT / cfg.paths.typeformer_checkpoint)

    val_cache = (ROOT / cfg.paths.results_dir / "embeddings_cache" / "val" /
                 "val_embeddings.npy")
    if args.no_cache and val_cache.exists():
        val_cache.unlink()

    val_embs = get_cached_embeddings(
        val_cache,
        lambda: wrapper.encode_dataset(val_ds.sessions, batch_size=args.batch_size,
                                       desc="Encoding val set"),
    )
    logger.info("Val embeddings: %s", val_embs.shape)

    # ------------------------------------------------------------------
    # Compute / load cohort embeddings
    # ------------------------------------------------------------------
    pool_npz = ROOT / cfg.paths.processed_dir / "cohort_pool_sessions.npz"
    if not pool_npz.exists():
        logger.error("Cohort pool not found — run 01_preprocess_data.py.")
        sys.exit(1)

    emb_path, idx_path = get_cohort_cache_paths(
        ROOT / cfg.paths.results_dir, n_cohort, cfg.seed
    )
    if emb_path.exists() and not args.no_cache:
        logger.info("Loading cached cohort: %s", emb_path)
        cohort_embs = np.load(str(emb_path))
    else:
        logger.info("Computing cohort embeddings (N=%d) ...", n_cohort)
        cohort_idx = sample_cohort(pool_npz, n_cohort=n_cohort, seed=cfg.seed,
                                   save_path=idx_path)
        cohort_seqs = load_cohort_sessions(pool_npz, cohort_idx, session_idx=-1)
        cohort_embs = wrapper.encode_numpy(cohort_seqs)
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(emb_path), cohort_embs)
        logger.info("Cohort saved: %s  shape=%s", emb_path, cohort_embs.shape)

    logger.info("Cohort embeddings: %s", cohort_embs.shape)

    # ------------------------------------------------------------------
    # Cosine baseline (B2 equivalent, no normalization) on val set
    # Required for the per-subject EER constraint.
    # ------------------------------------------------------------------
    logger.info("Computing cosine baseline on val set ...")
    cos_scorer = RawCosineScorer()
    cos_global_eer, cos_persubj_eer, _ = _eval_one_k(
        val_embs, cos_scorer, E, genuine_sessions, impostor_session_idx
    )
    logger.info(
        "Cosine baseline (val): global=%.2f%%  per_subj=%.2f%%",
        cos_global_eer * 100, cos_persubj_eer * 100,
    )

    # ------------------------------------------------------------------
    # Sweep K
    # ------------------------------------------------------------------
    sweep_results = []

    for K in sorted(args.K_values):
        if K >= len(cohort_embs):
            logger.warning("K=%d >= cohort size=%d — skipping.", K, len(cohort_embs))
            continue

        logger.info("Evaluating K=%d ...", K)
        scorer = ASNormScorer(cohort_embs, K=K, epsilon=float(cfg.scoring.epsilon))
        global_eer, persubj_eer, per_user_eers = _eval_one_k(
            val_embs, scorer, E, genuine_sessions, impostor_session_idx
        )

        delta_persubj = persubj_eer - cos_persubj_eer   # positive = worse
        constraint_ok = delta_persubj <= MAX_PERSUBJ_DELTA

        logger.info(
            "  K=%d | global=%.2f%% | per_subj=%.2f%% | Δper_subj=%+.2f%% | constraint=%s",
            K, global_eer*100, persubj_eer*100, delta_persubj*100,
            "OK" if constraint_ok else "VIOLATED",
        )

        np.savetxt(
            str(out_dir / f"per_user_eers_K{K}_E{E}.csv"),
            per_user_eers, delimiter=",", fmt="%.6f",
        )

        sweep_results.append({
            "K": K,
            "global_eer": float(global_eer),
            "persubj_eer": float(persubj_eer),
            "delta_persubj_from_cosine": float(delta_persubj),
            "constraint_ok": constraint_ok,
        })

    # ------------------------------------------------------------------
    # Select K_best
    # ------------------------------------------------------------------
    valid = [r for r in sweep_results if r["constraint_ok"]]
    if not valid:
        logger.warning(
            "No K satisfies the per-subject constraint (Δ ≤ %.1f%%). "
            "Using K with minimum global EER regardless.",
            MAX_PERSUBJ_DELTA * 100,
        )
        valid = sweep_results

    k_best_row = min(valid, key=lambda r: r["global_eer"])
    K_best = k_best_row["K"]

    logger.info("")
    logger.info("=== K SWEEP SUMMARY (E=%d, N_cohort=%d) ===", E, n_cohort)
    logger.info("Cosine baseline:  global=%.2f%%  per_subj=%.2f%%",
                cos_global_eer*100, cos_persubj_eer*100)
    for r in sweep_results:
        marker = " ← BEST" if r["K"] == K_best else ""
        logger.info(
            "  K=%3d | global=%5.2f%% | per_subj=%5.2f%% | Δ=%+.2f%% | %s%s",
            r["K"], r["global_eer"]*100, r["persubj_eer"]*100,
            r["delta_persubj_from_cosine"]*100,
            "OK" if r["constraint_ok"] else "CONSTRAINT VIOLATED",
            marker,
        )
    logger.info("K_best = %d  (global_eer=%.2f%%)", K_best, k_best_row["global_eer"]*100)

    # Save K_best
    (out_dir / "k_best.txt").write_text(str(K_best), encoding="utf-8")

    # ------------------------------------------------------------------
    # Plot K vs EER
    # ------------------------------------------------------------------
    K_vals = [r["K"] for r in sweep_results]
    global_eers = [r["global_eer"] * 100 for r in sweep_results]
    persubj_eers = [r["persubj_eer"] * 100 for r in sweep_results]
    colors = ["green" if r["constraint_ok"] else "red" for r in sweep_results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(K_vals, global_eers, "o-", color="steelblue", label="Global EER (primary)")
    ax.plot(K_vals, persubj_eers, "s--", color="darkorange", label="Mean per-subject EER")

    # Highlight K_best
    ax.axvline(K_best, color="green", linestyle=":", alpha=0.7, label=f"K_best={K_best}")

    # Cosine baseline reference lines
    ax.axhline(cos_global_eer * 100, color="steelblue", linestyle=":", alpha=0.5,
               label=f"Cosine global baseline ({cos_global_eer*100:.2f}%)")
    ax.axhline(cos_persubj_eer * 100, color="darkorange", linestyle=":", alpha=0.5,
               label=f"Cosine per-subj baseline ({cos_persubj_eer*100:.2f}%)")

    # Constraint bound
    ax.axhline((cos_persubj_eer + MAX_PERSUBJ_DELTA) * 100,
               color="red", linestyle="--", linewidth=0.8, alpha=0.6,
               label=f"Per-subj constraint (+{MAX_PERSUBJ_DELTA*100:.1f}%)")

    # Mark violated points
    for r, c in zip(sweep_results, colors):
        if c == "red":
            ax.scatter(r["K"], r["persubj_eer"] * 100, color="red", s=80, zorder=5)

    ax.set_xscale("log")
    ax.set_xticks(K_vals)
    ax.set_xticklabels([str(k) for k in K_vals])
    ax.set_xlabel("K (top-K cohort size)")
    ax.set_ylabel("EER (%)")
    ax.set_title(f"AS-Norm K Sweep  (val set, E={E}, N_cohort={n_cohort})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "k_vs_eer.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved to %s", out_dir / "k_vs_eer.png")

    # ------------------------------------------------------------------
    # Save JSON report
    # ------------------------------------------------------------------
    report = {
        "E": E,
        "n_cohort": n_cohort,
        "seed": cfg.seed,
        "max_persubj_delta_abs": MAX_PERSUBJ_DELTA,
        "cosine_baseline": {
            "global_eer": float(cos_global_eer),
            "persubj_eer": float(cos_persubj_eer),
        },
        "sweep": sweep_results,
        "K_best": K_best,
        "K_best_global_eer": float(k_best_row["global_eer"]),
        "K_best_persubj_eer": float(k_best_row["persubj_eer"]),
    }
    with open(out_dir / "k_sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Full report: %s", out_dir / "k_sweep_results.json")
    logger.info("=== K sweep complete. K_best=%d ===", K_best)


if __name__ == "__main__":
    main()
