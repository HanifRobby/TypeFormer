import numpy as np
from sklearn.metrics import roc_curve
from typing import List, Tuple, Dict

def compute_eer(genuine_scores: List[float], impostor_scores: List[float]) -> float:
    """Compute EER from distance scores."""
    if not genuine_scores or not impostor_scores:
        return 0.5
        
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.zeros(len(genuine_scores)),   # genuine = 0
        np.ones(len(impostor_scores))    # impostor = 1
    ])
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    return eer

def find_eer_threshold(genuine_scores: List[float], impostor_scores: List[float]) -> float:
    """Find the threshold at which FAR ≈ FRR."""
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.zeros(len(genuine_scores)),
        np.ones(len(impostor_scores))
    ])
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    return float(thresholds[eer_idx])

def compute_far_at_frr(genuine_scores: List[float], impostor_scores: List[float], target_frr: float = 0.01) -> Tuple[float, float]:
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.zeros(len(genuine_scores)),
        np.ones(len(impostor_scores))
    ])
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fnr - target_frr))
    return float(fpr[idx]), float(fnr[idx])

def compute_user_metrics(genuine_scores: List[float], impostor_scores: List[float]) -> Dict:
    eer = compute_eer(genuine_scores, impostor_scores)
    far_1, frr_1 = compute_far_at_frr(genuine_scores, impostor_scores, 0.01)
    far_10, frr_10 = compute_far_at_frr(genuine_scores, impostor_scores, 0.10)
    
    return {
        'eer': eer,
        'far_at_1frr': far_1,
        'frr_at_1frr': frr_1,
        'far_at_10frr': far_10,
        'frr_at_10frr': frr_10,
        'n_genuine': len(genuine_scores),
        'n_impostor': len(impostor_scores)
    }

def aggregate_results(user_metrics: List[Dict]) -> Dict:
    eers = [m['eer'] for m in user_metrics]
    
    return {
        'avg_eer': float(np.mean(eers)),
        'std_eer': float(np.std(eers)),
        'median_eer': float(np.median(eers)),
        'worst_eer': float(np.max(eers)),
        'best_eer': float(np.min(eers)),
        'avg_far_1': float(np.mean([m['far_at_1frr'] for m in user_metrics])),
        'avg_frr_1': float(np.mean([m['frr_at_1frr'] for m in user_metrics])),
        'n_users': len(user_metrics)
    }
