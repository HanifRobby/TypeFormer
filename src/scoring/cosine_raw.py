"""B2: Raw cosine similarity scorer (centroid template)."""

import numpy as np

from .base_scorer import BaseScorer
from .distance import cosine_sim, cosine_sim_batch


class RawCosineScorer(BaseScorer):
    """Raw cosine similarity — no normalization."""

    def score(self, e_u: np.ndarray, e_p: np.ndarray) -> float:
        return cosine_sim(e_u, e_p)

    def score_batch(self, e_u: np.ndarray, probes: np.ndarray) -> np.ndarray:
        return cosine_sim_batch(e_u, probes)
