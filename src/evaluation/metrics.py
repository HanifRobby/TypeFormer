"""Core biometric evaluation metrics.

Convention throughout this module:
    - Higher score → more likely genuine match.
    - labels: 1 = genuine, 0 = impostor.
"""

from typing import Tuple

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) from two score arrays.

    Args:
        genuine_scores: 1-D array of scores for genuine (same-user) pairs.
        impostor_scores: 1-D array of scores for impostor (cross-user) pairs.

    Returns:
        eer: Equal Error Rate in [0, 1].
        threshold: Decision threshold at the EER operating point.
    """
    genuine_scores = np.asarray(genuine_scores, dtype=np.float64)
    impostor_scores = np.asarray(impostor_scores, dtype=np.float64)

    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return float("nan"), float("nan")

    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores)),
    ])
    scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold


def compute_far_at_frr(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    target_frr: float,
) -> float:
    """Compute FAR at a given FRR operating point.

    Args:
        genuine_scores: Genuine pair scores.
        impostor_scores: Impostor pair scores.
        target_frr: Target False Reject Rate (e.g. 0.01 for 1 % FRR).

    Returns:
        FAR at the threshold that achieves the closest FRR <= target_frr.
    """
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    frr = 1.0 - tpr

    # Find the highest threshold where FRR <= target_frr
    # (minimum FAR while satisfying the FRR constraint)
    valid = frr <= target_frr + 1e-9
    if not np.any(valid):
        return 1.0
    idx = np.where(valid)[0][0]   # first occurrence = highest threshold in sorted array
    return float(fpr[idx])


def compute_frr_at_far(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    target_far: float,
) -> float:
    """Compute FRR at a given FAR operating point.

    Args:
        genuine_scores: Genuine pair scores.
        impostor_scores: Impostor pair scores.
        target_far: Target False Accept Rate (e.g. 0.01 for 1 % FAR).

    Returns:
        FRR at the threshold that achieves the closest FAR <= target_far.
    """
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    frr = 1.0 - tpr

    valid = fpr <= target_far + 1e-9
    if not np.any(valid):
        return 1.0
    idx = np.where(valid)[0][-1]
    return float(frr[idx])
