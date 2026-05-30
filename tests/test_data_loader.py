"""Unit tests for AaltoDataset and sequence_processor."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.aalto_loader import AaltoDataset
from src.data.sequence_processor import process_session, process_user_sessions


@pytest.fixture
def tmp_npz(tmp_path):
    """Create a small test .npz file."""
    n_users, n_sess, L, C = 5, 15, 50, 5
    user_ids = np.arange(n_users, dtype=np.int64)
    sessions = np.random.randn(n_users, n_sess, L, C).astype(np.float32)
    path = tmp_path / "test.npz"
    np.savez(str(path), user_ids=user_ids, sessions=sessions)
    return path, n_users, n_sess, L, C


class TestSequenceProcessor:
    def test_shape_normal(self):
        raw = np.random.rand(30, 6).astype(np.float32)
        out = process_session(raw, seq_len=50)
        assert out.shape == (50, 5)
        assert out.dtype == np.float32

    def test_padding_at_end(self):
        raw = np.random.rand(10, 6).astype(np.float32)
        out = process_session(raw, seq_len=50)
        assert np.all(out[10:] == 0.0)
        assert np.allclose(out[:10], raw[:10, :5].astype(np.float32))

    def test_truncation(self):
        raw = np.random.rand(80, 6).astype(np.float32)
        out = process_session(raw, seq_len=50)
        assert out.shape == (50, 5)

    def test_user_sessions_shape(self):
        sessions = [np.random.rand(30, 6).astype(np.float32) for _ in range(20)]
        out = process_user_sessions(sessions, seq_len=50, num_sessions=15)
        assert out.shape == (15, 50, 5)


class TestAaltoDataset:
    def test_len(self, tmp_npz):
        path, n_users, n_sess, L, C = tmp_npz
        ds = AaltoDataset(path)
        assert len(ds) == n_users * n_sess

    def test_getitem_shape(self, tmp_npz):
        path, n_users, n_sess, L, C = tmp_npz
        ds = AaltoDataset(path)
        item = ds[0]
        assert item["sequence"].shape == torch.Size([L, C])
        assert item["sequence"].dtype == torch.float32
        assert isinstance(item["user_id"], int)

    def test_get_user_sessions_shape(self, tmp_npz):
        path, n_users, n_sess, L, C = tmp_npz
        ds = AaltoDataset(path)
        sess = ds.get_user_sessions(0)
        assert sess.shape == (n_sess, L, C)

    def test_properties(self, tmp_npz):
        path, n_users, n_sess, L, C = tmp_npz
        ds = AaltoDataset(path)
        assert ds.n_users == n_users
        assert ds.n_sessions == n_sess
        assert ds.seq_len == L

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AaltoDataset(tmp_path / "nonexistent.npz")
