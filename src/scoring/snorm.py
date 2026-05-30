"""N3: S-Norm scorer.

Symmetric normalisation: average of Z-Norm and T-Norm.
    s_snorm = 0.5 * (z_u + z_p)
where z_u = Z-Norm (enrolment side) and z_p = T-Norm (probe side).
"""

import numpy as np

from .base_scorer import BaseScorer
from .distance import cosine_sim, cosine_sim_batch


class SNormScorer(BaseScorer):
    """S-Norm: symmetric cohort normalisation using full cohort.

    Args:
        cohort: (N_c, D) cohort embedding matrix.
        epsilon: Floor for sigma.
    """

    def __init__(self, cohort: np.ndarray, epsilon: float = 1e-3) -> None:
        self.cohort = cohort
        self.epsilon = epsilon
        norms = np.linalg.norm(cohort, axis=1, keepdims=True) + 1e-12
        self._cohort_n = cohort / norms

    def _stats(self, e: np.ndarray) -> tuple[float, float]:
        e_n = e / (np.linalg.norm(e) + 1e-12)
        scores = self._cohort_n @ e_n
        return float(scores.mean()), float(max(scores.std(), self.epsilon))

    def score(self, e_u: np.ndarray, e_p: np.ndarray) -> float:
        s_raw = cosine_sim(e_u, e_p)
        mu_u, sigma_u = self._stats(e_u)
        mu_p, sigma_p = self._stats(e_p)
        z_u = (s_raw - mu_u) / sigma_u
        z_p = (s_raw - mu_p) / sigma_p
        return 0.5 * (z_u + z_p)

    def score_batch(self, e_u: np.ndarray, probes: np.ndarray) -> np.ndarray:
        s_raw = cosine_sim_batch(e_u, probes)   # (N,)
        mu_u, sigma_u = self._stats(e_u)
        # Probe-side stats
        probes_n = probes / (np.linalg.norm(probes, axis=1, keepdims=True) + 1e-12)
        scores_p = probes_n @ self._cohort_n.T   # (N, N_c)
        mu_p = scores_p.mean(axis=1)
        sigma_p = np.maximum(scores_p.std(axis=1), self.epsilon)
        z_u = (s_raw - mu_u) / sigma_u
        z_p = (s_raw - mu_p) / sigma_p
        return 0.5 * (z_u + z_p)
