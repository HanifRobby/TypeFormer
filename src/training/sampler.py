"""Multi-E triplet sampler for FiLM training.

Each iteration samples:
  - anchor_user, E (random from E_range)
  - E enrolment sessions → s_u (user stats)
  - anchor and positive sessions (different from enrolment)
  - negative from a different user

The multi-E approach trains FiLM to be robust across different
enrolment sizes, matching the distribution at evaluation time.
"""

import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import IterableDataset

from src.statistics.user_stats import compute_user_stats


class MultiETripletSampler(IterableDataset):
    """Infinite triplet dataset with multi-E sampling.

    Args:
        sessions: (N_users, N_sessions, L, 5) float32 array.
        E_range: List of E values to sample from.
        length: Number of triplets per epoch (controls DataLoader epoch length).
        seed: Random seed.
    """

    def __init__(
        self,
        sessions: np.ndarray,
        E_range: List[int] = (2, 5, 7, 10),
        length: int = 10000,
        seed: int = 42,
    ) -> None:
        self.sessions = sessions
        self.E_range = list(E_range)
        self.length = length
        self.n_users, self.n_sessions = sessions.shape[:2]
        self._rng = random.Random(seed)

    def _sample_one(self) -> Dict[str, torch.Tensor]:
        rng = self._rng
        n_users = self.n_users
        n_sessions = self.n_sessions

        # Anchor user and E
        anchor_user = rng.randint(0, n_users - 1)
        E = rng.choice(self.E_range)

        # Pick E distinct enrolment indices
        enrol_idx = rng.sample(range(n_sessions), min(E, n_sessions))

        # Compute s_u from enrolment sessions
        enrol_sessions = self.sessions[anchor_user, enrol_idx]   # (E, L, 5)
        s_u = compute_user_stats(enrol_sessions)                  # (20,)

        # Anchor and positive: from remaining sessions
        remaining = [i for i in range(n_sessions) if i not in set(enrol_idx)]
        if len(remaining) < 2:
            remaining = list(range(n_sessions))   # fallback: allow overlap
        anchor_idx, positive_idx = rng.sample(remaining, 2)

        # Negative: different user, random session
        neg_user = rng.randint(0, n_users - 1)
        while neg_user == anchor_user:
            neg_user = rng.randint(0, n_users - 1)
        neg_sess_idx = rng.randint(0, n_sessions - 1)

        # Compute s_u for negative user
        neg_E = rng.choice(self.E_range)
        neg_enrol_idx = rng.sample(range(n_sessions), min(neg_E, n_sessions))
        neg_enrol = self.sessions[neg_user, neg_enrol_idx]
        s_u_neg = compute_user_stats(neg_enrol)

        return {
            "anchor_seq": torch.from_numpy(
                self.sessions[anchor_user, anchor_idx]).float(),
            "positive_seq": torch.from_numpy(
                self.sessions[anchor_user, positive_idx]).float(),
            "negative_seq": torch.from_numpy(
                self.sessions[neg_user, neg_sess_idx]).float(),
            "s_u_anchor": torch.from_numpy(s_u).float(),
            "s_u_positive": torch.from_numpy(s_u).float(),   # same user
            "s_u_negative": torch.from_numpy(s_u_neg).float(),
        }

    def __iter__(self):
        for _ in range(self.length):
            yield self._sample_one()

    def __len__(self) -> int:
        return self.length
