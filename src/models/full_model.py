"""TypeFormerWithFiLM: compose frozen backbone + optional FiLM head."""

import numpy as np
import torch

from .typeformer_wrapper import TypeFormerWrapper
from .film_head import FiLMHead


class TypeFormerWithFiLM:
    """Frozen TypeFormer backbone with a trainable FiLM head.

    The backbone always runs on CUDA (required by HARTrans).
    The FiLM head is a nn.Module and can be placed on any device.

    Usage:
        model = TypeFormerWithFiLM(film_head, wrapper)
        e_modulated = model.encode_with_film(sequences, user_stats)
    """

    def __init__(
        self,
        film_head: FiLMHead,
        backbone: TypeFormerWrapper,
    ) -> None:
        self.film_head = film_head
        self.backbone = backbone

    @torch.no_grad()
    def encode_with_film(
        self,
        sequences: torch.Tensor,
        user_stats: torch.Tensor,
        film_device: str = "cuda",
    ) -> torch.Tensor:
        """Encode sequences and apply FiLM modulation.

        Args:
            sequences: (B, L, 5) float32 Tensor on any device.
            user_stats: (B, S) float32 Tensor.
            film_device: Device for FiLM computation.

        Returns:
            (B, D) modulated embeddings on CPU.
        """
        embs = self.backbone.encode(sequences)           # (B, D) on CPU
        embs = embs.to(film_device)
        user_stats = user_stats.to(film_device)
        modulated = self.film_head(embs, user_stats)
        return modulated.cpu()

    def encode_dataset_with_film(
        self,
        sessions: np.ndarray,
        user_stats_arr: np.ndarray,
        batch_size: int = 64,
        film_device: str = "cuda",
    ) -> np.ndarray:
        """Encode all sessions with FiLM modulation.

        Args:
            sessions: (N_users, N_sessions, L, 5) float32.
            user_stats_arr: (N_users, S) float32 — one s_u per user.
            batch_size: Sequences per batch.
            film_device: FiLM device.

        Returns:
            (N_users, N_sessions, D) modulated embeddings.
        """
        n_users, n_sess, L, C = sessions.shape
        out = np.zeros((n_users, n_sess, self.backbone.EMBEDDING_DIM), dtype=np.float32)

        self.film_head.eval()
        with torch.no_grad():
            for u in range(n_users):
                s_u = torch.from_numpy(user_stats_arr[u]).float().unsqueeze(0)  # (1, S)
                for s in range(0, n_sess, batch_size):
                    batch_seq = torch.from_numpy(sessions[u, s: s + batch_size]).float()  # (B, L, 5)
                    B = batch_seq.shape[0]
                    s_u_batch = s_u.expand(B, -1)
                    embs = self.encode_with_film(batch_seq, s_u_batch, film_device)
                    out[u, s: s + batch_size] = embs.numpy()

        return out
