"""Compute user statistics vector s_u from enrolment sessions.

s_u is a 20-dimensional vector capturing the typing style distribution:
    [mean(4), std(4), p25(4), p50(4), p75(4)]
for features HL, IL, IRL, IKT (columns 0-3, excluding ASCII column 4).
"""

from typing import List

import numpy as np


FEATURE_COLUMNS = [0, 1, 2, 3]   # HL, IL, IRL, IKT — exclude ASCII
FEATURE_NAMES = ["hold_time", "inter_press", "inter_release", "inter_key"]
STATS_DIM = 20   # 4 features × 5 statistics (mean, std, p25, p50, p75)


def compute_user_stats(
    sessions: np.ndarray,
    feature_columns: List[int] = FEATURE_COLUMNS,
    use_percentiles: bool = True,
) -> np.ndarray:
    """Compute user statistics vector from enrolment sessions.

    Args:
        sessions: (E, L, 5) float32 array — E enrolment sessions,
                  each with L=50 keystrokes and 5 feature channels.
        feature_columns: Which feature columns to include (default: 0-3).
        use_percentiles: If True, append p25, p50, p75; else only mean+std.

    Returns:
        s_u: (20,) float32 array if use_percentiles else (8,) float32.
             NaN-free: if not enough valid keystrokes, returns zeros
             with a warning log (does not crash).
    """
    sessions = np.asarray(sessions, dtype=np.float32)
    if sessions.ndim != 3:
        raise ValueError(f"Expected sessions of shape (E, L, 5), got {sessions.shape}")

    # Flatten: (E*L, 5)
    all_keystrokes = sessions.reshape(-1, sessions.shape[-1])

    # Filter zero-padding: rows where all timing features are 0
    padding_mask = np.all(all_keystrokes[:, :4] == 0.0, axis=1)
    valid = all_keystrokes[~padding_mask][:, feature_columns]   # (T_valid, n_feat)

    n_feat = len(feature_columns)
    n_stats = 5 if use_percentiles else 2
    s_u = np.zeros(n_feat * n_stats, dtype=np.float32)

    if len(valid) == 0:
        return s_u   # all zeros — edge case

    if len(valid) < 4 and use_percentiles:
        # Too few samples for stable percentiles; fall back to zeros for percentiles
        s_u[:n_feat] = valid.mean(axis=0).astype(np.float32)
        s_u[n_feat:2*n_feat] = valid.std(axis=0).astype(np.float32)
        return s_u

    stats = [
        valid.mean(axis=0),
        valid.std(axis=0),
    ]
    if use_percentiles:
        stats += [
            np.percentile(valid, 25, axis=0),
            np.percentile(valid, 50, axis=0),
            np.percentile(valid, 75, axis=0),
        ]

    s_u = np.concatenate(stats).astype(np.float32)
    return s_u


def compute_user_stats_multi_e(
    sessions: np.ndarray,
    E: int,
    **kwargs,
) -> np.ndarray:
    """Convenience wrapper: compute s_u from the first E sessions."""
    return compute_user_stats(sessions[:E], **kwargs)
