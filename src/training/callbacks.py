"""Training callbacks: early stopping and model checkpointing."""

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when monitored metric stops improving.

    Args:
        patience: Epochs to wait before stopping.
        min_delta: Minimum improvement to count as progress.
        mode: 'min' for loss/EER, 'max' for accuracy.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._counter = 0
        self._best = float("inf") if mode == "min" else float("-inf")
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Update state. Returns True if this is the best epoch so far."""
        if self.mode == "min":
            improved = value < self._best - self.min_delta
        else:
            improved = value > self._best + self.min_delta

        if improved:
            self._best = value
            self._counter = 0
            return True
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "EarlyStopping: no improvement for %d epochs — stopping.",
                    self.patience,
                )
            return False

    @property
    def best(self) -> float:
        return self._best


class ModelCheckpoint:
    """Save the best model checkpoint.

    Args:
        path: File path for the checkpoint .pt file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, model: nn.Module) -> None:
        torch.save(model.state_dict(), str(self.path))
        logger.debug("Checkpoint saved to %s", self.path)

    def load(self, model: nn.Module) -> None:
        model.load_state_dict(
            torch.load(str(self.path), map_location="cpu", weights_only=True)
        )
        logger.info("Checkpoint loaded from %s", self.path)
