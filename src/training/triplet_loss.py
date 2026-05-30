"""Triplet loss for FiLM training."""

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Triplet margin loss with Euclidean distance.

    Args:
        margin: Margin for triplet loss (default 1.0, matches TypeFormer paper).
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin
        self._loss = nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        return self._loss(anchor, positive, negative)
