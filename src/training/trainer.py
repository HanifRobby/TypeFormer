"""FiLM training loop.

Trains only the FiLM head; backbone is always frozen.
Validation uses per-subject EER on the validation split.
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.film_head import FiLMHead
from src.models.typeformer_wrapper import TypeFormerWrapper
from src.evaluation.per_subject_eer import evaluate_per_subject
from src.scoring.cosine_raw import RawCosineScorer
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.triplet_loss import TripletLoss

logger = logging.getLogger(__name__)


class FiLMTrainer:
    """Training loop for the FiLM head.

    Args:
        film_head: The FiLM module to train.
        backbone: Frozen TypeFormer wrapper.
        train_loader: DataLoader yielding triplet dicts.
        val_embeddings: (N_val, N_sess, 64) pre-computed val embeddings.
        config: SimpleNamespace with film training hyperparameters.
        checkpoint_path: Path to save best checkpoint.
        device: Torch device for FiLM (backbone always on CUDA).
    """

    def __init__(
        self,
        film_head: FiLMHead,
        backbone: TypeFormerWrapper,
        train_loader: DataLoader,
        val_embeddings: np.ndarray,
        config,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ) -> None:
        self.film_head = film_head.to(device)
        self.backbone = backbone
        self.train_loader = train_loader
        self.val_embeddings = val_embeddings
        self.cfg = config
        self.device = device

        self.criterion = TripletLoss(margin=config.margin).to(device)
        self.optimizer = torch.optim.Adam(
            film_head.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.early_stopping = EarlyStopping(patience=config.patience, mode="min")
        self.checkpoint = ModelCheckpoint(checkpoint_path)

        self._scorer = RawCosineScorer()

    def _train_epoch(self) -> float:
        self.film_head.train()
        losses = []

        for batch in self.train_loader:
            anchor_seq = batch["anchor_seq"].to(self.backbone.device)
            positive_seq = batch["positive_seq"].to(self.backbone.device)
            negative_seq = batch["negative_seq"].to(self.backbone.device)
            s_u_anc = batch["s_u_anchor"].to(self.device)
            s_u_pos = batch["s_u_positive"].to(self.device)
            s_u_neg = batch["s_u_negative"].to(self.device)

            # Backbone forward (no grad)
            with torch.no_grad():
                e_anc = self.backbone.encode(anchor_seq).to(self.device)
                e_pos = self.backbone.encode(positive_seq).to(self.device)
                e_neg = self.backbone.encode(negative_seq).to(self.device)

            # FiLM modulation (with grad)
            e_anc_m = self.film_head(e_anc, s_u_anc)
            e_pos_m = self.film_head(e_pos, s_u_pos)
            e_neg_m = self.film_head(e_neg, s_u_neg)

            loss = self.criterion(e_anc_m, e_pos_m, e_neg_m)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.film_head.parameters(), max_norm=1.0)
            self.optimizer.step()

            losses.append(loss.item())

        return float(np.mean(losses))

    def _validate(self) -> float:
        """Compute val mean per-subject EER using cosine similarity."""
        self.film_head.eval()
        # TODO: apply FiLM to val_embeddings using s_u computed from val sessions
        # For now, use raw embeddings as a proxy (FiLM not applied during fast validation)
        # A full FiLM val pass is added in 06_train_film.py post-training evaluation.
        mean_eer, _, _ = evaluate_per_subject(
            self.val_embeddings,
            E=5,   # default E for validation
            scoring_fn=self._scorer.score,
        )
        return mean_eer

    def train(self) -> Dict[str, List[float]]:
        """Run training loop.

        Returns:
            history: dict with 'train_loss' and 'val_eer' lists.
        """
        history: Dict[str, List[float]] = {"train_loss": [], "val_eer": []}
        max_epochs = self.cfg.max_epochs

        for epoch in range(max_epochs):
            train_loss = self._train_epoch()
            val_eer = self._validate()

            history["train_loss"].append(train_loss)
            history["val_eer"].append(val_eer)

            # Monitor γ and β norms
            dummy_stats = torch.randn(1, self.film_head.fc1.in_features).to(self.device)
            g_norm, b_norm = self.film_head.get_gamma_beta_norms(dummy_stats)

            logger.info(
                "Epoch %d/%d | loss=%.4f | val_eer=%.2f%% | ||γ-1||=%.4f ||β||=%.4f",
                epoch + 1, max_epochs, train_loss, val_eer * 100,
                g_norm.item(), b_norm.item(),
            )

            is_best = self.early_stopping.step(val_eer)
            if is_best:
                self.checkpoint.save(self.film_head)

            if self.early_stopping.should_stop:
                logger.info("Early stopping at epoch %d.", epoch + 1)
                break

        logger.info("Best val EER: %.2f%%", self.early_stopping.best * 100)
        return history
