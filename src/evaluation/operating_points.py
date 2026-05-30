"""Operating point metrics: FAR @ x% FRR and FRR @ x% FAR."""

from typing import Dict

import numpy as np

from .metrics import compute_far_at_frr, compute_frr_at_far


def compute_operating_points(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute standard biometric operating point metrics.

    Args:
        genuine_scores: Genuine pair scores (higher = genuine).
        impostor_scores: Impostor pair scores.

    Returns:
        Dict with far_at_1frr, far_at_5frr, frr_at_1far, frr_at_5far.
    """
    return {
        "far_at_1frr":  compute_far_at_frr(genuine_scores, impostor_scores, 0.01),
        "far_at_5frr":  compute_far_at_frr(genuine_scores, impostor_scores, 0.05),
        "frr_at_1far":  compute_frr_at_far(genuine_scores, impostor_scores, 0.01),
        "frr_at_5far":  compute_frr_at_far(genuine_scores, impostor_scores, 0.05),
    }
