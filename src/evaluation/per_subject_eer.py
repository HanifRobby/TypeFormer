"""Per-subject EER computation (TypeFormer paper convention).

For each test user:
  - Enrolment template: sessions [0 : E]
  - Genuine probes   : last `genuine_sessions` sessions
  - Impostor probes  : one session per every other test user

Returns the mean EER across all users, and the per-user EER array
(needed for Wilcoxon signed-rank tests).
"""

from typing import Callable, Tuple

import numpy as np

from .metrics import compute_eer


def evaluate_per_subject(
    embeddings: np.ndarray,
    E: int,
    scoring_fn: Callable[[np.ndarray, np.ndarray], float],
    genuine_sessions: int = 5,
    impostor_session_idx: int = -1,
) -> Tuple[float, float, np.ndarray]:
    """Compute mean per-subject EER over all test users.

    Args:
        embeddings: (N_users, N_sessions, D) float32 embedding array.
            sessions 0..E-1  → enrolment
            sessions [-genuine_sessions:] → genuine probes
            sessions[impostor_session_idx] → impostor probe
        E: Number of enrolment sessions (1, 2, 5, 7, or 10).
        scoring_fn: Callable(e_u: (D,), e_p: (D,)) → float.
            Higher value = more likely genuine.
        genuine_sessions: How many sessions at the end are genuine probes.
        impostor_session_idx: Which session index to use for impostors.

    Returns:
        mean_eer: Mean per-subject EER across all users [0, 1].
        std_eer: Standard deviation of per-subject EERs.
        per_user_eers: (N_users,) array of per-user EERs (for Wilcoxon).
    """
    n_users = embeddings.shape[0]
    per_user_eers = np.zeros(n_users, dtype=np.float64)

    # Impostor probe embeddings: one per user (session at impostor_session_idx)
    impostor_probes = embeddings[:, impostor_session_idx, :]   # (N_users, D)

    for user_idx in range(n_users):
        enrol_embs = embeddings[user_idx, :E, :]            # (E, D)
        genuine_embs = embeddings[user_idx, -genuine_sessions:, :]  # (G, D)

        # Enrolment template: centroid (mean of E embeddings)
        # Exception for B0 (mean_pairwise): scoring_fn handles it internally
        e_u = enrol_embs.mean(axis=0)   # (D,)

        # Genuine scores
        genuine_scores = np.array([
            scoring_fn(e_u, e_p) for e_p in genuine_embs
        ])

        # Impostor scores: all other users
        impostor_scores = np.array([
            scoring_fn(e_u, impostor_probes[j])
            for j in range(n_users) if j != user_idx
        ])

        eer, _ = compute_eer(genuine_scores, impostor_scores)
        per_user_eers[user_idx] = eer

    mean_eer = float(np.mean(per_user_eers))
    std_eer = float(np.std(per_user_eers))
    return mean_eer, std_eer, per_user_eers


def evaluate_per_subject_mean_pairwise(
    embeddings: np.ndarray,
    E: int,
    scoring_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    genuine_sessions: int = 5,
    impostor_session_idx: int = -1,
) -> Tuple[float, float, np.ndarray]:
    """Per-subject EER for B0 mean-pairwise protocol.

    For B0, the score between a template and a probe is computed as the mean
    score across all E individual enrolment embeddings (no centroid pooling).

    scoring_fn signature: (enrol_embs: (E,D), probe_emb: (D,)) → float
    """
    n_users = embeddings.shape[0]
    per_user_eers = np.zeros(n_users, dtype=np.float64)
    impostor_probes = embeddings[:, impostor_session_idx, :]

    for user_idx in range(n_users):
        enrol_embs = embeddings[user_idx, :E, :]            # (E, D)
        genuine_embs = embeddings[user_idx, -genuine_sessions:, :]

        genuine_scores = np.array([
            scoring_fn(enrol_embs, e_p) for e_p in genuine_embs
        ])
        impostor_scores = np.array([
            scoring_fn(enrol_embs, impostor_probes[j])
            for j in range(n_users) if j != user_idx
        ])

        eer, _ = compute_eer(genuine_scores, impostor_scores)
        per_user_eers[user_idx] = eer

    mean_eer = float(np.mean(per_user_eers))
    std_eer = float(np.std(per_user_eers))
    return mean_eer, std_eer, per_user_eers
