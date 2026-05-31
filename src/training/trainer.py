"""FiLM training loop.

Trains only the FiLM head; backbone is always frozen.
Validation applies FiLM to val embeddings using per-user s_u computed
from enrolment sessions, then measures per-subject EER with cosine scoring.
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.film_head import FiLMHead
from src.models.typeformer_wrapper import TypeFormerWrapper
from src.evaluation.per_subject_eer import evaluate_per_subject
from src.scoring.cosine_raw import RawCosineScorer
from src.statistics.user_stats import compute_user_stats
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.triplet_loss import TripletLoss

logger = logging.getLogger(__name__)


class FiLMTrainer:
    """Training loop for the FiLM head.

    Args:
        film_head: The FiLM module to train.
        backbone: Frozen TypeFormer wrapper (always on CUDA).
        train_loader: DataLoader yielding triplet dicts.
        val_embeddings: (N_val, N_sess, 64) pre-computed val embeddings.
        val_sessions: (N_val, N_sess, L, 5) raw val sessions for s_u computation.
        config: SimpleNamespace with film training hyperparameters.
        checkpoint_path: Path to save best checkpoint.
        device: Torch device for FiLM (backbone always on CUDA).
        val_E: Number of enrolment sessions used during validation (default: 5).
    """

    def __init__(
        self,
        film_head: FiLMHead,
        backbone: TypeFormerWrapper,
        train_loader: DataLoader,
        val_embeddings: np.ndarray,
        val_sessions: np.ndarray,
        config,
        checkpoint_path: str | Path,
        device: str = "cuda",
        val_E: int = 5,
    ) -> None:
        self.film_head = film_head.to(device)
        self.backbone = backbone
        self.train_loader = train_loader
        self.val_embeddings = val_embeddings   # (N_val, N_sess, D)
        self.val_sessions = val_sessions       # (N_val, N_sess, L, 5)
        self.cfg = config
        self.device = device
        self.val_E = val_E

        self.criterion = TripletLoss(margin=config.margin).to(device)
        self.optimizer = torch.optim.Adam(
            film_head.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        # Identity regularisation strength (0 = disabled, 0.1 = recommended).
        self._lambda_reg = float(getattr(config, "film_reg_lambda", 0.0))

        self.early_stopping = EarlyStopping(patience=config.patience, mode="min")
        self.checkpoint = ModelCheckpoint(checkpoint_path)
        self._scorer = RawCosineScorer()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

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

            loss_triplet = self.criterion(e_anc_m, e_pos_m, e_neg_m)

            # Identity regularisation: penalise large deviation from γ=1, β=0.
            # This prevents the degenerate "β dominates, γ≈0" collapse where FiLM
            # routes all embeddings of user u to a fixed point β(s_u_u) and the
            # backbone embedding is suppressed entirely.
            loss_reg = torch.tensor(0.0, device=self.device)
            if self._lambda_reg > 0.0:
                gb_anc = self.film_head.fc2(
                    torch.relu(self.film_head.fc1(s_u_anc))
                )
                gamma_anc = gb_anc[:, : self.film_head.embedding_dim]
                beta_anc  = gb_anc[:, self.film_head.embedding_dim :]
                loss_reg = self._lambda_reg * (
                    ((gamma_anc - 1.0) ** 2).mean() + (beta_anc ** 2).mean()
                )

            loss = loss_triplet + loss_reg

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.film_head.parameters(), max_norm=1.0)
            self.optimizer.step()

            losses.append(loss_triplet.item())

        return float(np.mean(losses))

    # ------------------------------------------------------------------
    # Validation — FiLM applied to val embeddings using per-user s_u
    # ------------------------------------------------------------------

    def _validate(self) -> float:
        """Compute val mean per-subject EER with FiLM-modulated embeddings.

        For each val user:
            1. Compute s_u from the first val_E enrolment sessions.
            2. Apply FiLM: e' = film_head(e, s_u) for all sessions.
            3. Evaluate per-subject EER using cosine similarity on e'.
        """
        self.film_head.eval()
        n_users, n_sessions, D = self.val_embeddings.shape

        film_val_embs = np.zeros_like(self.val_embeddings)   # (N_val, N_sess, D)

        with torch.no_grad():
            for u in range(n_users):
                # s_u from E enrolment sessions
                s_u = compute_user_stats(self.val_sessions[u, : self.val_E])   # (20,)
                s_u_t = torch.from_numpy(s_u).float().to(self.device)          # (20,)
                s_u_batch = s_u_t.unsqueeze(0).expand(n_sessions, -1)          # (N_sess, 20)

                # Apply FiLM to all sessions for this user in one batch
                embs_u = torch.from_numpy(
                    self.val_embeddings[u]
                ).float().to(self.device)                                        # (N_sess, D)

                film_embs_u = self.film_head(embs_u, s_u_batch)                 # (N_sess, D)
                film_val_embs[u] = film_embs_u.cpu().numpy()

        mean_eer, _, _ = evaluate_per_subject(
            film_val_embs,
            E=self.val_E,
            scoring_fn=self._scorer.score,
        )
        return mean_eer

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

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

            # Monitor γ and β deviation from identity
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
