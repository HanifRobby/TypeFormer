import logging
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def get_cached_embeddings(cache_path: Path, compute_fn: Callable[[], np.ndarray]) -> np.ndarray:
    """Load embeddings from cache if available, else compute and save.

    Args:
        cache_path: Path to .npy cache file.
        compute_fn: Zero-argument callable that returns the embedding array.

    Returns:
        Embedding array loaded from cache or freshly computed.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        logger.info("Loading cached embeddings from %s", cache_path)
        return np.load(str(cache_path))

    logger.info("Cache not found — computing embeddings...")
    embeddings = compute_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), embeddings)
    logger.info("Embeddings saved to %s  shape=%s", cache_path, embeddings.shape)
    return embeddings


def invalidate_cache(cache_path: Path) -> None:
    """Delete a cached embedding file."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        cache_path.unlink()
        logger.info("Cache invalidated: %s", cache_path)
