"""N4: AS-Norm (Adaptive Symmetric Score Normalization).

Key difference from S-Norm: uses only the top-K most similar cohort
imposters for normalisation, making it adaptive to each test pair.

Formula:
    z_u = (s_raw - mu_topK_u) / sigma_topK_u
    z_p = (s_raw - mu_topK_p) / sigma_topK_p
    s_asnorm = 0.5 * (z_u + z_p)
"""

import numpy as np

from .base_scorer import BaseScorer
from .distance import cosine_sim_batch


class ASNormScorer(BaseScorer):
    """Adaptive Symmetric Score Normalization.

    Args:
        cohort: (N_c, D) cohort embedding matrix.
        K: Top-K cohort subset size for adaptive selection.
        epsilon: Floor for sigma to prevent division by zero.

    Raises:
        ValueError: If K > N_c.
    """

    def __init__(
        self,
        cohort: np.ndarray,
        K: int,
        epsilon: float = 1e-3,
    ) -> None:
        n_c = len(cohort)
        if K > n_c:
            raise ValueError(f"K={K} exceeds cohort size N_c={n_c}.")
        if K <= 1:
            raise ValueError(f"K={K} must be > 1 for meaningful std computation.")

        self.K = K
        self.epsilon = epsilon

        norms = np.linalg.norm(cohort, axis=1, keepdims=True) + 1e-12
        self._cohort_n = (cohort / norms).astype(np.float32)   # (N_c, D)

    def _topk_stats(self, e: np.ndarray) -> tuple[float, float]:
        """Compute top-K cohort mean and std for embedding e."""
        e_n = (e / (np.linalg.norm(e) + 1e-12)).astype(np.float32)
        scores = self._cohort_n @ e_n   # (N_c,)
        # Partial sort — faster than full sort for large N_c
        top_k = np.partition(scores, -self.K)[-self.K:]
        return float(top_k.mean()), float(max(top_k.std(), self.epsilon))

    def score(self, e_u: np.ndarray, e_p: np.ndarray) -> float:
        """Score a single (enrolment, probe) pair."""
        e_u_n = e_u / (np.linalg.norm(e_u) + 1e-12)
        e_p_n = e_p / (np.linalg.norm(e_p) + 1e-12)
        s_raw = float(np.dot(e_u_n, e_p_n))

        mu_u, sigma_u = self._topk_stats(e_u)
        mu_p, sigma_p = self._topk_stats(e_p)

        z_u = (s_raw - mu_u) / sigma_u
        z_p = (s_raw - mu_p) / sigma_p
        return 0.5 * (z_u + z_p)

    def score_batch(self, e_u: np.ndarray, probes: np.ndarray) -> np.ndarray:
        """Score one enrolment template against many probes (vectorised).

        Enrolment-side stats are computed once; probe-side stats are
        batched using matrix operations.

        Args:
            e_u: (D,) enrolment template.
            probes: (N, D) probe embeddings.

        Returns:
            (N,) AS-Norm scores.
        """
        # Enrolment side (computed once)
        mu_u, sigma_u = self._topk_stats(e_u)

        # Normalise inputs
        e_u_n = (e_u / (np.linalg.norm(e_u) + 1e-12)).astype(np.float32)
        norms_p = np.linalg.norm(probes, axis=1, keepdims=True) + 1e-12
        probes_n = (probes / norms_p).astype(np.float32)   # (N, D)

        # Raw cosine similarity
        s_raw = probes_n @ e_u_n   # (N,)

        # Probe-side: scores of all probes against all cohort members
        scores_p_matrix = probes_n @ self._cohort_n.T   # (N, N_c)
        # Top-K per row
        top_k_p = np.partition(scores_p_matrix, -self.K, axis=1)[:, -self.K:]  # (N, K)
        mu_p = top_k_p.mean(axis=1)
        sigma_p = np.maximum(top_k_p.std(axis=1), self.epsilon)

        z_u = (s_raw - mu_u) / sigma_u
        z_p = (s_raw - mu_p) / sigma_p
        return 0.5 * (z_u + z_p)


def compute_asnorm_score(
    e_u: np.ndarray,
    e_p: np.ndarray,
    cohort: np.ndarray,
    K: int,
    epsilon: float = 1e-3,
) -> float:
    """Functional interface for single-pair AS-Norm (used in unit tests).

    Args:
        e_u: (D,) enrolment template.
        e_p: (D,) probe embedding.
        cohort: (N_c, D) cohort embeddings.
        K: Top-K cohort size.
        epsilon: Sigma floor.

    Returns:
        float: AS-Norm calibrated score. Higher = more genuine.

    Raises:
        ValueError: If K > N_c.
    """
    scorer = ASNormScorer(cohort, K=K, epsilon=epsilon)
    return scorer.score(e_u, e_p)
