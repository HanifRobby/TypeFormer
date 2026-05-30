"""Unit tests for user_stats.py."""

import numpy as np
import pytest
from src.statistics.user_stats import compute_user_stats, STATS_DIM, compute_user_stats_multi_e


class TestComputeUserStats:
    def test_output_shape_with_percentiles(self):
        sessions = np.random.randn(5, 50, 5).astype(np.float32)
        s_u = compute_user_stats(sessions)
        assert s_u.shape == (STATS_DIM,)
        assert s_u.dtype == np.float32

    def test_output_shape_without_percentiles(self):
        sessions = np.random.randn(5, 50, 5).astype(np.float32)
        s_u = compute_user_stats(sessions, use_percentiles=False)
        assert s_u.shape == (8,)

    def test_no_nan_normal(self):
        sessions = np.random.randn(5, 50, 5).astype(np.float32)
        s_u = compute_user_stats(sessions)
        assert not np.any(np.isnan(s_u))

    def test_all_zero_sessions_returns_zeros(self):
        sessions = np.zeros((1, 50, 5), dtype=np.float32)
        s_u = compute_user_stats(sessions)
        assert np.all(s_u == 0.0)

    def test_padding_filter_correct(self):
        """Only non-padded keystrokes (timing features != 0) should be used."""
        sessions = np.zeros((1, 50, 5), dtype=np.float32)
        sessions[0, :20, :4] = 1.0   # first 20 keystrokes have HL=IL=IRL=IKT=1.0
        s_u = compute_user_stats(sessions)
        # Mean of HL (index 0) should be 1.0
        assert abs(s_u[0] - 1.0) < 0.01, f"Expected mean_HL=1.0, got {s_u[0]}"

    def test_e1_edge_case_no_crash(self):
        """E=1 should not crash even with few keystrokes."""
        sessions = np.zeros((1, 50, 5), dtype=np.float32)
        sessions[0, :3, :4] = np.random.rand(3, 4).astype(np.float32)
        s_u = compute_user_stats(sessions)
        assert s_u.shape == (STATS_DIM,)
        assert not np.any(np.isnan(s_u))

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            compute_user_stats(np.zeros((50, 5), dtype=np.float32))

    def test_multi_e_wrapper(self):
        sessions = np.random.randn(15, 50, 5).astype(np.float32)
        s5 = compute_user_stats_multi_e(sessions, E=5)
        s15 = compute_user_stats_multi_e(sessions, E=15)
        assert s5.shape == s15.shape == (STATS_DIM,)
        # They should differ (using different sessions)
        assert not np.allclose(s5, s15)
