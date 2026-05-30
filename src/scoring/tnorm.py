"""N2: T-Norm scorer.

Normalises the raw cosine score using probe-side cohort statistics:
    z = (s_raw - mu_cohort_p) / sigma_cohort_p
"""

import numpy as np

from .base_scorer import BaseScorer
from .distance import cosine_sim, cosine_sim_batch


class TNormScorer(BaseScorer):
    """T-Norm: probe-side normalisation using full cohort.

    Args:
        cohort: (N_c, D) cohort embedding matrix.
        epsilon: Floor for sigma.
    """

    def __init__(self, cohort: np.ndarray, epsilon: float = 1e-3) -> None:
        self.cohort = cohort
        self.epsilon = epsilon
        norms = np.linalg.norm(cohort, axis=1, keepdims=True) + 1e-12
        self._cohort_n = cohort / norms

    def _cohort_stats_p(self, e_p: np.ndarray) -> tuple[float, float]:
        e_p_n = e_p / (np.linalg.norm(e_p) + 1e-12)
        scores = self._cohort_n @ e_p_n
        return float(scores.mean()), float(max(scores.std(), self.epsilon))

    def score(self, e_u: np.ndarray, e_p: np.ndarray) -> float:
        s_raw = cosine_sim(e_u, e_p)
        mu_p, sigma_p = self._cohort_stats_p(e_p)
        return (s_raw - mu_p) / sigma_p

    def score_batch(self, e_u: np.ndarray, probes: np.ndarray) -> np.ndarray:
        # Must compute per-probe stats
        s_raw = cosine_sim_batch(e_u, probes)   # (N,)
        scores_p = probes / (np.linalg.norm(probes, axis=1, keepdims=True) + 1e-12) \
                   @ self._cohort_n.T             # (N, N_c)
        mu_p = scores_p.mean(axis=1)
        sigma_p = np.maximum(scores_p.std(axis=1), self.epsilon)
        return (s_raw - mu_p) / sigma_p
