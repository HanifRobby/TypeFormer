from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class AaltoDataset(Dataset):
    """Flat dataset: each item is one (user, session) keystroke sequence.

    Loads pre-processed data from a .npz file produced by
    scripts/01_preprocess_data.py.  The npz must contain:
        user_ids: (N_users,) int64 — original participant IDs
        sessions: (N_users, N_sessions, L, 5) float32

    Two access modes are provided:
        - Flat  (__getitem__): yields one session at a time (for training).
        - Grouped (get_user_sessions): yields all sessions for a user (for eval).
    """

    def __init__(self, npz_path: str | Path) -> None:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")

        data = np.load(str(npz_path))
        self.user_ids: np.ndarray = data["user_ids"]      # (N_users,)
        self.sessions: np.ndarray = data["sessions"]      # (N_users, N_sess, L, 5)

        self._n_users, self._n_sessions, self._seq_len, self._n_features = self.sessions.shape
        # Flat index: (user_idx, session_idx) for __getitem__
        self._index = [
            (u, s)
            for u in range(self._n_users)
            for s in range(self._n_sessions)
        ]

    # ------------------------------------------------------------------
    # Standard Dataset interface (flat, for DataLoader)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int]:
        u, s = self._index[idx]
        return {
            "sequence": torch.from_numpy(self.sessions[u, s]).float(),  # (L, 5)
            "user_id": int(self.user_ids[u]),
            "user_idx": u,
            "session_id": s,
        }

    # ------------------------------------------------------------------
    # Grouped access (for evaluation)
    # ------------------------------------------------------------------

    def get_user_sessions(self, user_idx: int) -> np.ndarray:
        """Return all sessions for user_idx. Shape: (N_sessions, L, 5)."""
        return self.sessions[user_idx]

    def get_user_sessions_tensor(self, user_idx: int) -> torch.Tensor:
        """Return all sessions for user_idx as float32 Tensor."""
        return torch.from_numpy(self.sessions[user_idx]).float()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_users(self) -> int:
        return self._n_users

    @property
    def n_sessions(self) -> int:
        return self._n_sessions

    @property
    def seq_len(self) -> int:
        return self._seq_len
