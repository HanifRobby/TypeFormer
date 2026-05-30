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


class TestASNormMonotonicity:
    def test_orthogonal_lower_than_identical(self, cohort):
        """Score(e, e⊥) must be strictly lower than score(e, e).

        With random Gaussian embeddings:
          - cosine(e, e) = 1.0  → z-score >> 0
          - cosine(e, e⊥) = 0.0 → z-score ≈ -(mu_topK / sigma_topK) < 0
        """
        rng = np.random.RandomState(10)
        e = rng.randn(64).astype(np.float32)

        # Gram-Schmidt: project out e-component from a random vector
        v = rng.randn(64).astype(np.float32)
        e_perp = v - (np.dot(v, e) / np.dot(e, e)) * e

        # Verify orthogonality
        e_n = e / np.linalg.norm(e)
        e_perp_n = e_perp / np.linalg.norm(e_perp)
        assert abs(np.dot(e_n, e_perp_n)) < 1e-5, "e and e_perp are not orthogonal"

        score_identical = compute_asnorm_score(e, e, cohort, K=20)
        score_orthogonal = compute_asnorm_score(e, e_perp, cohort, K=20)

        assert score_identical > score_orthogonal, (
            f"score(e,e)={score_identical:.4f} should > score(e,e⊥)={score_orthogonal:.4f}"
        )

    def test_negative_lower_than_orthogonal(self, cohort):
        """score(e, -e) < score(e, e⊥) < score(e, e): three-way ordering."""
        rng = np.random.RandomState(11)
        e = rng.randn(64).astype(np.float32)

        v = rng.randn(64).astype(np.float32)
        e_perp = v - (np.dot(v, e) / np.dot(e, e)) * e

        score_self = compute_asnorm_score(e, e, cohort, K=20)
        score_perp = compute_asnorm_score(e, e_perp, cohort, K=20)
        score_anti = compute_asnorm_score(e, -e, cohort, K=20)

        assert score_self > score_perp > score_anti, (
            f"Expected score(e,e) > score(e,e⊥) > score(e,-e), "
            f"got {score_self:.3f} > {score_perp:.3f} > {score_anti:.3f}"
        )

    def test_output_deterministic(self, cohort):
        """Same inputs must always produce exactly the same output (no hidden RNG)."""
        rng = np.random.RandomState(42)
        e_u = rng.randn(64).astype(np.float32)
        e_p = rng.randn(64).astype(np.float32)

        score1 = compute_asnorm_score(e_u, e_p, cohort, K=20)
        score2 = compute_asnorm_score(e_u, e_p, cohort, K=20)
        score3 = compute_asnorm_score(e_u, e_p, cohort, K=20)

        assert score1 == score2 == score3, (
            f"Non-deterministic output: {score1}, {score2}, {score3}"
        )

    def test_batch_deterministic(self, cohort):
        """score_batch must be deterministic across repeated calls."""
        rng = np.random.RandomState(7)
        scorer = ASNormScorer(cohort, K=20)
        e_u = rng.randn(64).astype(np.float32)
        probes = rng.randn(5, 64).astype(np.float32)

        out1 = scorer.score_batch(e_u, probes)
        out2 = scorer.score_batch(e_u, probes)

        np.testing.assert_array_equal(out1, out2, err_msg="score_batch is not deterministic")


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
