"""Unit tests for AS-Norm scorer (asnorm.py).

Covers the three test cases specified in implementation_guide.md §7.6.
"""

import numpy as np
import pytest
from src.scoring.asnorm import ASNormScorer, compute_asnorm_score


@pytest.fixture
def cohort():
    rng = np.random.RandomState(0)
    return rng.randn(100, 64).astype(np.float32)


class TestASNormSinglePair:
    def test_identical_embeddings_positive_score(self, cohort):
        """Score between two identical embeddings should be positive."""
        e = np.random.RandomState(1).randn(64).astype(np.float32)
        score = compute_asnorm_score(e, e, cohort, K=10)
        assert score > 0, f"Identical embeddings should give positive score, got {score}"

    def test_K_larger_than_cohort_raises(self):
        """K > N_c must raise ValueError."""
        cohort_small = np.random.randn(50, 64).astype(np.float32)
        e_u = np.random.randn(64).astype(np.float32)
        e_p = np.random.randn(64).astype(np.float32)
        with pytest.raises(ValueError):
            compute_asnorm_score(e_u, e_p, cohort_small, K=100)

    def test_epsilon_floor_no_nan(self, cohort):
        """Sigma = 0 (uniform cohort) must not produce NaN."""
        rng = np.random.RandomState(2)
        e_u = rng.randn(64).astype(np.float32)
        e_p = rng.randn(64).astype(np.float32)
        # Near-uniform cohort → sigma ≈ 0
        base = e_u / (np.linalg.norm(e_u) + 1e-12)
        uniform_cohort = np.tile(base, (100, 1)) + rng.randn(100, 64).astype(np.float32) * 1e-10
        score = compute_asnorm_score(e_u, e_p, uniform_cohort, K=10, epsilon=1e-3)
        assert not np.isnan(score), f"Should not produce NaN with epsilon floor, got {score}"
        assert np.isfinite(score)

    def test_score_is_finite(self, cohort):
        rng = np.random.RandomState(42)
        e_u = rng.randn(64).astype(np.float32)
        e_p = rng.randn(64).astype(np.float32)
        score = compute_asnorm_score(e_u, e_p, cohort, K=20)
        assert np.isfinite(score)


class TestASNormBatch:
    def test_batch_matches_scalar(self, cohort):
        rng = np.random.RandomState(5)
        scorer = ASNormScorer(cohort, K=30, epsilon=1e-3)
        e_u = rng.randn(64).astype(np.float32)
        probes = rng.randn(8, 64).astype(np.float32)

        batch = scorer.score_batch(e_u, probes)
        for i in range(8):
            scalar = scorer.score(e_u, probes[i])
            assert abs(batch[i] - scalar) < 1e-4, \
                f"Batch vs scalar mismatch at i={i}: {batch[i]:.6f} vs {scalar:.6f}"

    def test_genuine_higher_than_impostor(self, cohort):
        """Genuine pairs should on average score higher than impostor pairs."""
        rng = np.random.RandomState(99)
        scorer = ASNormScorer(cohort, K=30)

        n = 20
        centers = rng.randn(n, 64).astype(np.float32)
        genuine_scores = []
        impostor_scores = []
        for i in range(n):
            e_u = centers[i] + rng.randn(64).astype(np.float32) * 0.1
            e_gen = centers[i] + rng.randn(64).astype(np.float32) * 0.1
            e_imp = centers[(i + 1) % n] + rng.randn(64).astype(np.float32) * 0.1
            genuine_scores.append(scorer.score(e_u, e_gen))
            impostor_scores.append(scorer.score(e_u, e_imp))

        assert np.mean(genuine_scores) > np.mean(impostor_scores)
