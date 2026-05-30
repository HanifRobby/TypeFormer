import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def cohort_tag(n_cohort: int, seed: int) -> str:
    """Return a short deterministic tag encoding cohort parameters.

    Used to build filename-safe identifiers for cohort cache files so that
    changing N or seed automatically invalidates the old cache.

    Example: cohort_tag(2000, 42) → "N2000_s42"
    """
    return f"N{n_cohort}_s{seed}"


def sample_cohort(
    cohort_pool_npz: str | Path,
    n_cohort: int,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Sample a fixed cohort from the cohort pool.

    NOTE: Changing n_cohort or seed produces a different cohort.
    Use cohort_tag() to build cache filenames that encode these parameters.

    Args:
        cohort_pool_npz: Path to cohort_pool_sessions.npz.
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
        "Sampled %d cohort users from pool of %d  (seed=%d)  tag=%s",
        n_cohort, n_pool, seed, cohort_tag(n_cohort, seed),
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
        cohort_pool_npz: Path to cohort_pool_sessions.npz.
        cohort_indices: (n_cohort,) int array from sample_cohort().
        session_idx: Which session to extract per user (default: last = -1).

    Returns:
        (n_cohort, L, 5) float32 array — raw keystroke sequences.
    """
    data = np.load(str(cohort_pool_npz))
    sessions = data["sessions"]   # (N_pool, N_sess, L, 5)
    return sessions[cohort_indices, session_idx]   # (n_cohort, L, 5)


def get_cohort_cache_paths(
    results_dir: Path,
    n_cohort: int,
    seed: int,
) -> tuple[Path, Path]:
    """Return the canonical paths for cohort embeddings and indices files.

    Both filenames encode (n_cohort, seed) so stale caches are never reused.

    Returns:
        emb_path: Path for .npy file with (n_cohort, D) embeddings.
        idx_path: Path for .npy file with (n_cohort,) pool indices.
    """
    tag = cohort_tag(n_cohort, seed)
    cohort_dir = results_dir / "cohort"
    return (
        cohort_dir / f"cohort_embeddings_{tag}.npy",
        cohort_dir / f"cohort_indices_{tag}.npy",
    )
