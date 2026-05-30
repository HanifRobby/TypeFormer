"""Global EER computation: pool all genuine and impostor scores."""

from typing import Callable, Dict, List, Tuple

import numpy as np

from .metrics import compute_eer, compute_far_at_frr, compute_frr_at_far


def evaluate_global(
    embeddings: np.ndarray,
    E: int,
    scoring_fn: Callable[[np.ndarray, np.ndarray], float],
    genuine_sessions: int = 5,
    impostor_session_idx: int = -1,
) -> Tuple[Dict, List[Dict]]:
    """Pool all genuine and impostor scores across users, then compute EER.

    Args:
        embeddings: (N_users, N_sessions, D) array.
        E: Enrolment sessions count.
        scoring_fn: (e_u: (D,), e_p: (D,)) → float.  Higher = genuine.
        genuine_sessions: Number of genuine probe sessions (last N).
        impostor_session_idx: Impostor session index.

    Returns:
        metrics: dict with global_eer, threshold, far_at_1frr, frr_at_1far.
        user_data: list of dicts [{genuine_scores, impostor_scores}]
                   for bootstrap CI computation.
    """
    n_users = embeddings.shape[0]
    impostor_probes = embeddings[:, impostor_session_idx, :]

    all_genuine: List[float] = []
    all_impostor: List[float] = []
    user_data: List[Dict] = []

    for user_idx in range(n_users):
        e_u = embeddings[user_idx, :E, :].mean(axis=0)
        genuine_embs = embeddings[user_idx, -genuine_sessions:, :]

        gen_scores = [scoring_fn(e_u, ep) for ep in genuine_embs]
        imp_scores = [
            scoring_fn(e_u, impostor_probes[j])
            for j in range(n_users) if j != user_idx
        ]

        all_genuine.extend(gen_scores)
        all_impostor.extend(imp_scores)
        user_data.append({
            "genuine_scores": np.array(gen_scores),
            "impostor_scores": np.array(imp_scores),
        })

    gen_arr = np.array(all_genuine)
    imp_arr = np.array(all_impostor)
    eer, threshold = compute_eer(gen_arr, imp_arr)

    metrics = {
        "global_eer": eer,
        "threshold": threshold,
        "far_at_1frr": compute_far_at_frr(gen_arr, imp_arr, 0.01),
        "frr_at_1far": compute_frr_at_far(gen_arr, imp_arr, 0.01),
        "n_genuine": int(len(gen_arr)),
        "n_impostor": int(len(imp_arr)),
    }
    return metrics, user_data
