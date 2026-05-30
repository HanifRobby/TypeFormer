import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def sample_cohort(
    cohort_pool_npz: str | Path,
    n_cohort: int,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Sample a fixed cohort from the cohort pool and return their embeddings.

    This function samples user *indices* from the cohort pool, then extracts
    one session per sampled user (last session, index -1).  The returned array
    is ready to be passed to scorer constructors.

    NOTE: Call this once and cache the result.  Changing n_cohort or seed
    invalidates the cache.

    Args:
        cohort_pool_npz: Path to cohort_pool.npz (N_pool, N_sess, L, 5).
        n_cohort: Number of users to sample from the pool.
        seed: Random seed for reproducible cohort selection.
        save_path: If provided, save cohort user indices to this .npy file.

    Returns:
        cohort_user_indices: (n_cohort,) int array — indices into the pool.
    """
    cohort_pool_npz = Path(cohort_pool_npz)
    data = np.load(str(cohort_pool_npz))
    n_pool = len(data["user_ids"])

    if n_cohort > n_pool:
        raise ValueError(
            f"n_cohort={n_cohort} exceeds cohort pool size={n_pool}."
        )

    rng = np.random.RandomState(seed)
    indices = rng.choice(n_pool, size=n_cohort, replace=False)
    indices.sort()

    logger.info(
        "Sampled %d cohort users from pool of %d  (seed=%d)",
        n_cohort, n_pool, seed,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), indices)
        logger.info("Cohort indices saved to %s", save_path)

    return indices


def load_cohort_sessions(
    cohort_pool_npz: str | Path,
    cohort_indices: np.ndarray,
    session_idx: int = -1,
) -> np.ndarray:
    """Extract one session per cohort user.

    Args:
        cohort_pool_npz: Path to cohort_pool.npz.
        cohort_indices: (n_cohort,) int array from sample_cohort().
        session_idx: Which session to extract per user (default: last = -1).

    Returns:
        (n_cohort, L, 5) float32 array — raw keystroke sequences.
    """
    data = np.load(str(cohort_pool_npz))
    sessions = data["sessions"]   # (N_pool, N_sess, L, 5)
    return sessions[cohort_indices, session_idx]   # (n_cohort, L, 5)
