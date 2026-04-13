import numpy as np
from typing import List, Dict, Optional


class HoldTimePreprocessor:
    """
    Hold-time-only z-score preprocessing for TypeFormer input sequences.

    Only standardizes column 0 (hold_latency = release_time - press_time).
    All other features are left in their original raw form:
      [0] hold_latency   → z-scored per user
      [1] inter_press    → raw (unchanged)
      [2] inter_release  → raw (unchanged)
      [3] inter_key      → raw (unchanged)
      [4] ascii_code     → raw (unchanged)

    Rationale:
      Hold time is the most user-distinctive biometric feature — it reflects
      individual finger mechanics (how long a user presses each key).
      The inter-key features depend more on text content and cognitive
      typing patterns, which may not benefit from per-user standardization.

    Buffer aggregation (KDPrint Eq. 1) is still applied to all features
    for noise smoothing.
    """

    def __init__(
        self, buffer_size: int = 5, seq_len: int = 50
    ):
        self.B = buffer_size
        self.seq_len = seq_len

        # Statistics for hold_latency (column 0) only
        self.mu: Optional[float] = None       # scalar
        self.sigma: Optional[float] = None    # scalar
        self._is_fitted: bool = False

    def fit(self, enrolment_sessions: List[np.ndarray]) -> "HoldTimePreprocessor":
        """Compute μ and σ of hold_latency from enrolment sessions."""
        assert len(enrolment_sessions) >= 1, "Need at least 1 enrolment session"

        # Collect hold_latency (col 0) from all enrolment sessions
        all_hold = np.concatenate(
            [s[:, 0].astype(np.float64) for s in enrolment_sessions]
        )

        self.mu = float(all_hold.mean())
        self.sigma = float(all_hold.std())

        # Numerical stability
        if self.sigma < 1e-4:
            self.sigma = 1e-4

        self._is_fitted = True
        return self

    def standardize(self, session: np.ndarray) -> np.ndarray:
        """
        Z-score only column 0 (hold_latency). All other columns pass through.
        No clipping — the fine-tuned model learns to handle the full z-score range.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before standardize().")

        result = session[:, :5].copy().astype(np.float64)

        # Z-score only hold_latency (column 0)
        result[:, 0] = (result[:, 0] - self.mu) / self.sigma

        # Columns 1-4 remain unchanged
        return result

    def apply_buffer(self, sessions: List[np.ndarray]) -> List[np.ndarray]:
        """Apply weighted moving average smoothing (KDPrint Eq 1) to all features."""
        if self.B <= 1:
            return sessions

        buffered = []
        for session in sessions:
            N, n_feats = session.shape
            b = np.zeros_like(session, dtype=np.float64)
            denom = 2 * (self.B - 1)

            for t in range(N):
                start = max(0, t - self.B + 1)
                window = session[start : t + 1]
                w_len = len(window)

                if w_len == 1:
                    b[t] = session[t]
                else:
                    b[t] = (window.sum(axis=0) + (self.B - 1) * session[t]) / denom

            buffered.append(b)

        return buffered

    def _pad_or_truncate(self, session: np.ndarray) -> np.ndarray:
        N = len(session)
        if N >= self.seq_len:
            return session[: self.seq_len]
        else:
            pad = np.zeros((self.seq_len - N, 5), dtype=np.float64)
            return np.vstack([session, pad])

    def fit_transform(
        self, enrolment_sessions: List[np.ndarray], use_buffer: bool = True
    ) -> List[np.ndarray]:
        """Fit on enrolment sessions and transform them."""
        self.fit(enrolment_sessions)
        standardized = [self.standardize(s) for s in enrolment_sessions]
        if use_buffer:
            standardized = self.apply_buffer(standardized)
        return [self._pad_or_truncate(s) for s in standardized]

    def transform(self, session: np.ndarray, use_buffer: bool = False) -> np.ndarray:
        """Transform a single session using stored statistics."""
        standardized = self.standardize(session)
        if use_buffer:
            standardized = self.apply_buffer([standardized])[0]
        return self._pad_or_truncate(standardized)

    def get_template_stats(self) -> Dict:
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted yet.")
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "buffer_size": self.B,
            "seq_len": self.seq_len,
        }

    @classmethod
    def from_template_stats(cls, stats: Dict) -> "HoldTimePreprocessor":
        instance = cls(buffer_size=stats["buffer_size"], seq_len=stats["seq_len"])
        instance.mu = float(stats["mu"])
        instance.sigma = float(stats["sigma"])
        instance._is_fitted = True
        return instance
