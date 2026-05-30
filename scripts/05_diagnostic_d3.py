"""
05_diagnostic_d3.py — Diagnostic D3: Global EER baseline TypeFormer.

Same as B0 but focused on global EER (pooled) as the target to beat.
Also verifies that per-subject EER ≈ 3.25% at E=5.

Usage:
    conda run -n Typeformer python scripts/05_diagnostic_d3.py
"""

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
from src.evaluation.bootstrap_ci import bootstrap_global_eer_ci


def mean_pairwise_euclidean(enrol_embs: np.ndarray, probe_emb: np.ndarray) -> float:
    dists = np.linalg.norm(enrol_embs - probe_emb[np.newaxis, :], axis=1)
    return float(-dists.mean())


def main() -> None:
    cfg = load_config("config/diagnostics/d3_baseline_global.yaml")
    set_global_seed(cfg.seed)

    out_dir = ROOT / cfg.paths.results_dir / "diagnostics" / "d3_baseline_global"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("d3_baseline_global", out_dir)
    logger.info("=== Diagnostic D3: Global EER Baseline ===")

    test_npz = ROOT / cfg.paths.processed_dir / "test_sessions.npz"
    if not test_npz.exists():
        logger.error("Test data not found — run 01_preprocess_data.py first.")
        sys.exit(1)

    dataset = AaltoDataset(test_npz)
    logger.info("Test users: %d", dataset.n_users)

    wrapper = TypeFormerWrapper(checkpoint_path=ROOT / cfg.paths.typeformer_checkpoint)
    cache_path = ROOT / cfg.paths.results_dir / "embeddings_cache" / "test" / "test_embeddings.npy"
    embeddings = get_cached_embeddings(
        cache_path,
        lambda: wrapper.encode_dataset(dataset.sessions, desc="Encoding test set"),
    )

    results = {}
    for E in list(cfg.evaluation.E_values):
        n_users = embeddings.shape[0]
        genuine_sessions = cfg.evaluation.genuine_sessions
        impostor_session_idx = cfg.evaluation.impostor_session_idx
        impostor_probes = embeddings[:, impostor_session_idx, :]

        # Per-subject EER
        mean_eer, std_eer, per_user_eers = evaluate_per_subject_mean_pairwise(
            embeddings, E=E, scoring_fn=mean_pairwise_euclidean,
            genuine_sessions=genuine_sessions,
            impostor_session_idx=impostor_session_idx,
        )

        # Global EER
        all_genuine, all_impostor = [], []
        user_data = []
        for user_idx in range(n_users):
            gen_sc = [mean_pairwise_euclidean(embeddings[user_idx, :E], ep)
                      for ep in embeddings[user_idx, -genuine_sessions:]]
            imp_sc = [mean_pairwise_euclidean(embeddings[user_idx, :E], impostor_probes[j])
                      for j in range(n_users) if j != user_idx]
            all_genuine.extend(gen_sc)
            all_impostor.extend(imp_sc)
            user_data.append({"genuine_scores": np.array(gen_sc),
                               "impostor_scores": np.array(imp_sc)})

        global_eer, _ = compute_eer(np.array(all_genuine), np.array(all_impostor))
        eer_pt, ci_lo, ci_hi = bootstrap_global_eer_ci(
            user_data, n_iterations=cfg.evaluation.bootstrap_iterations,
            ci_level=cfg.evaluation.bootstrap_ci, seed=cfg.seed,
        )

        logger.info(
            "E=%d | per_subj_EER=%.2f%% (±%.2f%%) | global_EER=%.2f%% [%.2f%%, %.2f%%]",
            E, mean_eer*100, std_eer*100, global_eer*100, ci_lo*100, ci_hi*100,
        )
        results[f"E_{E}"] = {
            "E": E,
            "mean_per_subject_eer": float(mean_eer),
            "std_per_subject_eer": float(std_eer),
            "global_eer": float(global_eer),
            "global_eer_ci_lower": float(ci_lo),
            "global_eer_ci_upper": float(ci_hi),
        }

        if E == 5:
            logger.info("** TARGET TO BEAT: global EER = %.2f%% at E=5 **", global_eer*100)
            if abs(mean_eer*100 - 3.25) > 0.5:
                logger.warning(
                    "Pipeline sanity check: E=5 per-subject EER=%.2f%% deviates from ~3.25%%",
                    mean_eer*100,
                )
            else:
                logger.info("Pipeline sanity check PASSED (E=5 EER=%.2f%%)", mean_eer*100)

    report = {
        "experiment_id": "d3_baseline_global",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    with open(out_dir / "d3_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=== D3 complete ===")


if __name__ == "__main__":
    main()
