"""Bootstrap confidence interval for global EER.

Resampling is done at the user level (not trial level) because scores within
one user are not independent.
"""

from typing import Dict, List, Tuple

import numpy as np

from .metrics import compute_eer


def bootstrap_global_eer_ci(
    user_data: List[Dict],
    n_iterations: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for global EER.

    Args:
        user_data: List of dicts with keys 'genuine_scores' and
                   'impostor_scores' (one dict per test user).
        n_iterations: Number of bootstrap resamples.
        ci_level: Confidence level (e.g. 0.95 for 95 % CI).
        seed: Random seed for the bootstrap RNG.

    Returns:
        eer_point: EER on the full (non-resampled) data.
        ci_lower: Lower CI bound.
        ci_upper: Upper CI bound.
    """
    rng = np.random.RandomState(seed)
    n_users = len(user_data)

    # Point estimate
    all_gen = np.concatenate([u["genuine_scores"] for u in user_data])
    all_imp = np.concatenate([u["impostor_scores"] for u in user_data])
    eer_point, _ = compute_eer(all_gen, all_imp)

    # Bootstrap distribution
    eer_bootstrap = np.zeros(n_iterations, dtype=np.float64)
    for i in range(n_iterations):
        idx = rng.choice(n_users, size=n_users, replace=True)
        gen_b = np.concatenate([user_data[j]["genuine_scores"] for j in idx])
        imp_b = np.concatenate([user_data[j]["impostor_scores"] for j in idx])
        eer_b, _ = compute_eer(gen_b, imp_b)
        eer_bootstrap[i] = eer_b

    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(eer_bootstrap, alpha * 100))
    ci_upper = float(np.percentile(eer_bootstrap, (1.0 - alpha) * 100))
    return eer_point, ci_lower, ci_upper
