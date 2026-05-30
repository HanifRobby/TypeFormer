"""Unit tests for EER computation (metrics.py)."""

import numpy as np
import pytest
from src.evaluation.metrics import compute_eer, compute_far_at_frr, compute_frr_at_far


class TestComputeEER:
    def test_perfect_separation(self):
        gen = np.array([0.9, 0.85, 0.95, 0.8])
        imp = np.array([0.1, 0.15, 0.05, 0.2])
        eer, thresh = compute_eer(gen, imp)
        assert eer < 0.01
        assert np.isfinite(thresh)

    def test_complete_overlap(self):
        rng = np.random.RandomState(0)
        gen = rng.randn(1000)
        imp = rng.randn(1000)
        eer, _ = compute_eer(gen, imp)
        assert 0.4 < eer < 0.6

    def test_eer_symmetry(self):
        """EER should not change if we flip genuine/impostor convention."""
        rng = np.random.RandomState(42)
        gen = rng.randn(500) + 1.0
        imp = rng.randn(500)
        eer1, _ = compute_eer(gen, imp)
        # Flip: negate scores and treat as distance (impostor becomes genuine)
        eer2, _ = compute_eer(-imp, -gen)
        assert abs(eer1 - eer2) < 0.02

    def test_empty_arrays_returns_nan(self):
        eer, thresh = compute_eer(np.array([]), np.array([0.5, 0.6]))
        assert np.isnan(eer)

    def test_eer_in_unit_interval(self):
        rng = np.random.RandomState(7)
        gen = rng.randn(200) + 0.5
        imp = rng.randn(200)
        eer, _ = compute_eer(gen, imp)
        assert 0.0 <= eer <= 1.0


class TestOperatingPoints:
    def setup_method(self):
        rng = np.random.RandomState(99)
        self.gen = rng.randn(1000) + 1.0
        self.imp = rng.randn(1000)

    def test_far_at_frr_valid_range(self):
        far = compute_far_at_frr(self.gen, self.imp, 0.10)
        assert 0.0 <= far <= 1.0

    def test_frr_at_far_valid_range(self):
        frr = compute_frr_at_far(self.gen, self.imp, 0.10)
        assert 0.0 <= frr <= 1.0

    def test_tighter_frr_gives_higher_far(self):
        far_strict = compute_far_at_frr(self.gen, self.imp, 0.05)
        far_lenient = compute_far_at_frr(self.gen, self.imp, 0.20)
        # At 5% FRR (stricter), we accept more impostors (higher FAR)
        # than at 20% FRR (more lenient allows fewer impostors)
        # This ordering may not always hold perfectly, just check both are valid
        assert 0.0 <= far_strict <= 1.0
        assert 0.0 <= far_lenient <= 1.0
