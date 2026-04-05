import numpy as np
from typing import List, Dict, Optional


class KDPrintPreprocessor:
    """
    KDPrint-inspired preprocessing for TypeFormer input sequences.

    Implements two components from KDPrint (Kim et al., 2025):
      1. Z-score standardization (Section 3.3) — formula identical to paper
      2. Buffer aggregation (Section 3.4, Equation 1) — formula identical to paper
    """

    def __init__(
        self, buffer_size: int = 5, seq_len: int = 50, clip_sigma: float = 5.0
    ):
        self.B = buffer_size
        self.seq_len = seq_len
        self.clip_sigma = clip_sigma

        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def fit(self, enrolment_sessions: List[np.ndarray]) -> "KDPrintPreprocessor":
        """Compute μ and σ from enrolment sessions (timing columns 0-3 only)."""
        assert len(enrolment_sessions) >= 1, "Need at least 1 enrolment session"

        # Collect all timing features from all enrolment sessions
        all_timing = np.vstack(
            [s[:, :4].astype(np.float64) for s in enrolment_sessions]
        )

        self.mu = all_timing.mean(axis=0)  # shape (4,)
        self.sigma = all_timing.std(axis=0)  # shape (4,)

        # Numerical stability: use slightly larger epsilon if sigma is tiny
        self.sigma = np.where(self.sigma < 1e-4, 1e-4, self.sigma)

        self._is_fitted = True
        return self

    def standardize(self, session: np.ndarray, clip_sigma: float = 5.0) -> np.ndarray:
        """
        Apply z-score normalization using fitted statistics.

        clip_sigma: Maximum absolute z-score value allowed.
                    Values beyond this are clipped to ±clip_sigma.

                    Mengapa diperlukan:
                    Model dilatih dengan data range [-0.12, 0.23] (sigma ~0.03).
                    Impostor dengan pola sangat berbeda bisa menghasilkan z-score
                    ekstrem (±20σ), menyebabkan saturasi sigmoid di model.
                    Clipping ke ±5σ cukup untuk menandai impostor sebagai "anomali"
                    tanpa merusak representasi embedding.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before standardize().")

        result = session[:, :5].copy().astype(np.float64)

        # Z-score normalization untuk timing features (cols 0–3)
        result[:, :4] = (result[:, :4] - self.mu) / self.sigma

        # Clip ke range aman untuk mencegah saturasi sigmoid
        result[:, :4] = np.clip(result[:, :4], -clip_sigma, clip_sigma)

        return result

    def apply_buffer(self, sessions: List[np.ndarray]) -> List[np.ndarray]:
        """Apply weighted moving average smoothing (Eq 1)."""
        if self.B <= 1:
            return sessions  # no buffering

        buffered = []
        for session in sessions:
            N, n_feats = session.shape
            b = np.zeros_like(session, dtype=np.float64)
            denom = 2 * (self.B - 1)

            for t in range(N):
                start = max(0, t - self.B + 1)
                window = session[start : t + 1]  # shape (min(t+1, B), n_feats)
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
        self.fit(enrolment_sessions)
        standardized = [
            self.standardize(s, clip_sigma=self.clip_sigma) for s in enrolment_sessions
        ]
        if use_buffer:
            standardized = self.apply_buffer(standardized)
        return [self._pad_or_truncate(s) for s in standardized]

    def transform(self, session: np.ndarray, use_buffer: bool = False) -> np.ndarray:
        standardized = self.standardize(session, clip_sigma=self.clip_sigma)
        if use_buffer:
            standardized = self.apply_buffer([standardized])[0]
        return self._pad_or_truncate(standardized)

    def get_template_stats(self) -> Dict:
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted yet.")
        return {
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "buffer_size": self.B,
            "seq_len": self.seq_len,
        }

    @classmethod
    def from_template_stats(cls, stats: Dict) -> "KDPrintPreprocessor":
        instance = cls(buffer_size=stats["buffer_size"], seq_len=stats["seq_len"])
        instance.mu = np.array(stats["mu"], dtype=np.float64)
        instance.sigma = np.array(stats["sigma"], dtype=np.float64)
        instance._is_fitted = True
        return instance
