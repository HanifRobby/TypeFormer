"""FiLM Head: Feature-wise Linear Modulation for keystroke embeddings.

Conditions TypeFormer embeddings on user statistics s_u:
    e' = γ(s_u) * e + β(s_u)

Identity initialisation ensures e' = e at the start of training
(γ=1, β=0), so FiLM degrades gracefully to the backbone baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMHead(nn.Module):
    """Feature-wise Linear Modulation head.

    Architecture:
        fc1: Linear(stats_dim → hidden_dim) + ReLU
        fc2: Linear(hidden_dim → 2 * embedding_dim)
            first half  → γ (scale)
            second half → β (shift)

    Identity init:
        fc2.weight = 0  → γ and β are constant (ignore s_u initially)
        fc2.bias[:D] = 1  → γ = 1
        fc2.bias[D:] = 0  → β = 0

    Args:
        stats_dim: Dimensionality of s_u (default 20).
        embedding_dim: TypeFormer embedding size (default 64).
        hidden_dim: Hidden layer size (default 64).
        dropout: Dropout rate after fc1 (default 0.0 = disabled).
    """

    def __init__(
        self,
        stats_dim: int = 20,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.fc1 = nn.Linear(stats_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * embedding_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_identity()

    def _init_identity(self) -> None:
        """Initialise FiLM to the identity transform (γ=1, β=0)."""
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)

        nn.init.zeros_(self.fc2.weight)
        with torch.no_grad():
            self.fc2.bias[: self.embedding_dim] = 1.0   # γ bias = 1
            self.fc2.bias[self.embedding_dim :] = 0.0   # β bias = 0

    def forward(
        self,
        embedding: torch.Tensor,
        user_stats: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            embedding: (B, D) TypeFormer embeddings.
            user_stats: (B, S) user statistics vectors.

        Returns:
            (B, D) modulated embeddings.
        """
        h = F.relu(self.fc1(user_stats))   # (B, hidden_dim)
        h = self.drop(h)
        gamma_beta = self.fc2(h)           # (B, 2D)
        gamma = gamma_beta[:, : self.embedding_dim]
        beta = gamma_beta[:, self.embedding_dim :]
        return gamma * embedding + beta

    def get_gamma_beta_norms(
        self, user_stats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ||γ - 1|| and ||β|| for monitoring during training."""
        with torch.no_grad():
            h = F.relu(self.fc1(user_stats))
            gb = self.fc2(h)
            gamma = gb[:, : self.embedding_dim]
            beta = gb[:, self.embedding_dim :]
            return (gamma - 1).norm(dim=1).mean(), beta.norm(dim=1).mean()
