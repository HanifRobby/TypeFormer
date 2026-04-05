import numpy as np
from typing import List, Tuple

class GlobalThresholdEstimator:
    """
    Global EER-optimal threshold, same for all users.
    Used for experiments E1 (baseline) and E2 (preprocessing only).
    """
    
    def __init__(self):
        self.threshold: float = None
    
    def fit(self, all_genuine_scores: List[float], all_impostor_scores: List[float]) -> float:
        """Find global EER threshold from validation set scores."""
        from src.evaluation.metrics import find_eer_threshold
        self.threshold = find_eer_threshold(all_genuine_scores, all_impostor_scores)
        return self.threshold
    
    def predict(self, distance: float) -> Tuple[str, float]:
        if self.threshold is None:
            raise RuntimeError("Must call fit() first")
        decision = 'accept' if distance <= self.threshold else 'reject'
        return decision, distance
