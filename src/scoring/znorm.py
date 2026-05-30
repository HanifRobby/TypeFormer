"""N1: Z-Norm scorer.

Normalises the raw cosine score using cohort statistics computed from the
enrolment side only:
    z = (s_raw - mu_cohort_u) / sigma_cohort_u
"""

import numpy as np

from .base_scorer import BaseScorer
from .distance import cosine_sim, cosine_sim_batch


class ZNormScorer(BaseScorer):
    """Z-Norm: enrolment-side normalisation using full cohort.

    Args:
        cohort: (N_c, D) cohort embedding matrix.
        epsilon: Floor for sigma to prevent division by zero.
    """

    def __init__(self, cohort: np.ndarray, epsilon: float = 1e-3) -> None:
        self.cohort = cohort
        self.epsilon = epsilon
        # Pre-normalise cohort for efficiency
        norms = np.linalg.norm(cohort, axis=1, keepdims=True) + 1e-12
        self._cohort_n = cohort / norms   # (N_c, D)

    def _cohort_stats_u(self, e_u: np.ndarray) -> tuple[float, float]:
        e_u_n = e_u / (np.linalg.norm(e_u) + 1e-12)
        scores = self._cohort_n @ e_u_n    # (N_c,)
        return float(scores.mean()), float(max(scores.std(), self.epsilon))

    def score(self, e_u: np.ndarray, e_p: np.ndarray) -> float:
        s_raw = cosine_sim(e_u, e_p)
        mu_u, sigma_u = self._cohort_stats_u(e_u)
        return (s_raw - mu_u) / sigma_u

    def score_batch(self, e_u: np.ndarray, probes: np.ndarray) -> np.ndarray:
        s_raw = cosine_sim_batch(e_u, probes)   # (N,)
        mu_u, sigma_u = self._cohort_stats_u(e_u)
        return (s_raw - mu_u) / sigma_u
