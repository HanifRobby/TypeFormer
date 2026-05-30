"""Wilcoxon signed-rank test with Bonferroni correction for paired EER comparison."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def compare_configurations(
    eer_dict: Dict[str, np.ndarray],
    comparisons: List[Tuple[str, str]],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Wilcoxon signed-rank paired comparison with Bonferroni correction.

    Args:
        eer_dict: {config_name: (N_users,) per-user EER array}.
        comparisons: List of (config_A, config_B) pairs to compare.
                     Hypothesis: A and B have different distributions.
        alpha: Family-wise error rate (default 0.05).
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        DataFrame with columns:
            comparison, statistic, p_raw, p_corrected, significant, n_users.
    """
    results = []
    for a, b in comparisons:
        if a not in eer_dict:
            raise KeyError(f"Config '{a}' not in eer_dict.")
        if b not in eer_dict:
            raise KeyError(f"Config '{b}' not in eer_dict.")

        arr_a = np.asarray(eer_dict[a], dtype=np.float64)
        arr_b = np.asarray(eer_dict[b], dtype=np.float64)

        if len(arr_a) != len(arr_b):
            raise ValueError(
                f"EER arrays for '{a}' and '{b}' have different lengths "
                f"({len(arr_a)} vs {len(arr_b)})."
            )

        stat, p = wilcoxon(arr_a, arr_b, alternative=alternative)
        results.append({
            "comparison": f"{a} vs {b}",
            "config_a": a,
            "config_b": b,
            "statistic": float(stat),
            "p_raw": float(p),
            "n_users": len(arr_a),
            "mean_eer_a": float(arr_a.mean()),
            "mean_eer_b": float(arr_b.mean()),
            "delta_mean_eer": float(arr_b.mean() - arr_a.mean()),
        })

    # Bonferroni correction
    p_raws = [r["p_raw"] for r in results]
    reject, p_corrected, _, _ = multipletests(
        p_raws, alpha=alpha, method="bonferroni"
    )

    for r, sig, p_c in zip(results, reject, p_corrected):
        r["p_bonferroni"] = float(p_c)
        r["significant"] = bool(sig)
        r["alpha"] = alpha

    return pd.DataFrame(results)


DEFAULT_COMPARISONS = [
    ("b0_baseline", "n4_asnorm"),    # AS-Norm primary contribution
    ("b0_baseline", "fn_full_system"),  # Full system
    ("n4_asnorm", "fn_full_system"),    # FiLM on top of AS-Norm
    ("b2_centroid_cosine", "n4_asnorm"),  # Value of normalization
    ("f1_film_only", "fn_full_system"),   # Value of AS-Norm on top of FiLM
]
