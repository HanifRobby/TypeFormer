"""Abstract base class for all scorers."""

from abc import ABC, abstractmethod

import numpy as np


class BaseScorer(ABC):
    """All scorers return a real-valued score where higher = more genuine."""

    @abstractmethod
    def score(self, e_u: np.ndarray, e_p: np.ndarray) -> float:
        """Score one (enrolment_template, probe) pair.

        Args:
            e_u: (D,) enrolment template embedding.
            e_p: (D,) probe embedding.

        Returns:
            float: Higher → more likely genuine match.
        """

    def score_batch(self, e_u: np.ndarray, probes: np.ndarray) -> np.ndarray:
        """Score one enrolment template against many probes.

        Default: vectorised loop over score().  Subclasses should override
        for efficiency.

        Args:
            e_u: (D,) enrolment template.
            probes: (N, D) probe embeddings.

        Returns:
            (N,) float array of scores.
        """
        return np.array([self.score(e_u, p) for p in probes])
