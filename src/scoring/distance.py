"""Low-level distance and similarity helpers (vectorised, numpy)."""

import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def cosine_sim_batch(e_u: np.ndarray, probes: np.ndarray) -> np.ndarray:
    """Cosine similarity of one vector against many. Returns (N,) array."""
    e_u_n = e_u / (np.linalg.norm(e_u) + 1e-12)
    norms = np.linalg.norm(probes, axis=1, keepdims=True) + 1e-12
    probes_n = probes / norms
    return probes_n @ e_u_n   # (N,)


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 1-D vectors."""
    return float(np.linalg.norm(a - b))


def euclidean_dist_batch(e_u: np.ndarray, probes: np.ndarray) -> np.ndarray:
    """Euclidean distance of one vector against many. Returns (N,) array."""
    return np.linalg.norm(probes - e_u[np.newaxis, :], axis=1)
