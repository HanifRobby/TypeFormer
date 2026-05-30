"""
06_train_film.py — Train FiLM head (F1: FiLM only).

Trains 3 random seeds and reports mean ± std EER.
Best checkpoint per seed is saved.

Usage:
    conda run -n Typeformer python scripts/06_train_film.py
    conda run -n Typeformer python scripts/06_train_film.py --config config/experiments/fn_full_system.yaml
    conda run -n Typeformer python scripts/06_train_film.py --seeds 0 1 2
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config
from src.utils.logging import setup_logger
from src.utils.seeds import set_global_seed
from src.utils.caching import get_cached_embeddings
from src.models.typeformer_wrapper import TypeFormerWrapper
from src.models.film_head import FiLMHead
from src.data.aalto_loader import AaltoDataset
from src.training.sampler import MultiETripletSampler
from src.training.trainer import FiLMTrainer
from src.evaluation.per_subject_eer import evaluate_per_subject
from src.evaluation.bootstrap_ci import bootstrap_global_eer_ci
from src.evaluation.metrics import compute_eer
from src.scoring.cosine_raw import RawCosineScorer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FiLM head.")
    p.add_argument("--config", default="config/experiments/f1_film_only.yaml")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123],
                   help="Random seeds to run (default: 42 7 123)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size from config")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def train_one_seed(
    seed: int,
    cfg,
    wrapper: TypeFormerWrapper,
    train_sessions: np.ndarray,
    val_embeddings: np.ndarray,
    out_dir: Path,
    logger,
    batch_size: int,
) -> dict:
    set_global_seed(seed)
    logger.info("=== Seed %d ===", seed)

    # FiLM head
    film_cfg = cfg.film
    film_head = FiLMHead(
        stats_dim=film_cfg.stats_dim,
        embedding_dim=film_cfg.embedding_dim,
        hidden_dim=film_cfg.hidden_dim,
        dropout=film_cfg.dropout,
    )

    # Sampler
    sampler = MultiETripletSampler(
        sessions=train_sessions,
        E_range=list(film_cfg.E_range),
        length=batch_size * 30,  # batches per epoch
        seed=seed,
    )
    loader = DataLoader(sampler, batch_size=batch_size, num_workers=0)

    # Checkpoint path per seed
    ckpt_path = out_dir / f"film_head_seed{seed}.pt"

    trainer = FiLMTrainer(
        film_head=film_head,
        backbone=wrapper,
        train_loader=loader,
        val_embeddings=val_embeddings,
        config=film_cfg,
        checkpoint_path=ckpt_path,
    )
    history = trainer.train()

    # Load best checkpoint
    film_head.load_state_dict(
        torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    )
    film_head.eval()

    best_val_eer = trainer.early_stopping.best
    logger.info("Seed %d — best val EER: %.2f%%", seed, best_val_eer * 100)

    return {
        "seed": seed,
        "best_val_eer": float(best_val_eer),
        "n_epochs": len(history["train_loss"]),
        "checkpoint": str(ckpt_path),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.seed)

    exp_id = cfg.experiment.id
    out_dir = ROOT / cfg.paths.results_dir / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(exp_id, out_dir)
    logger.info("=== Training FiLM: %s ===", exp_id)

    film_cfg = cfg.film
    batch_size = args.batch_size or film_cfg.batch_size

    # Load datasets
    train_npz = ROOT / cfg.paths.processed_dir / "train_sessions.npz"
    val_npz = ROOT / cfg.paths.processed_dir / "val_sessions.npz"
    for path in [train_npz, val_npz]:
        if not path.exists():
            logger.error("Data not found: %s — run 01_preprocess_data.py first.", path)
            sys.exit(1)

    train_ds = AaltoDataset(train_npz)
    val_ds = AaltoDataset(val_npz)
    logger.info("Train users: %d  Val users: %d", train_ds.n_users, val_ds.n_users)

    # TypeFormer backbone (frozen)
    wrapper = TypeFormerWrapper(checkpoint_path=ROOT / cfg.paths.typeformer_checkpoint)

    # Pre-compute val embeddings (cached)
    val_cache = ROOT / cfg.paths.results_dir / "embeddings_cache" / "val" / "val_embeddings.npy"
    if args.no_cache and val_cache.exists():
        val_cache.unlink()

    val_embeddings = get_cached_embeddings(
        val_cache,
        lambda: wrapper.encode_dataset(val_ds.sessions, desc="Encoding val set"),
    )
    logger.info("Val embeddings: %s", val_embeddings.shape)

    # Train with multiple seeds
    seed_results = []
    for seed in args.seeds:
        result = train_one_seed(
            seed=seed,
            cfg=cfg,
            wrapper=wrapper,
            train_sessions=train_ds.sessions,
            val_embeddings=val_embeddings,
            out_dir=out_dir / "checkpoints",
            logger=logger,
            batch_size=batch_size,
        )
        seed_results.append(result)

    val_eers = [r["best_val_eer"] for r in seed_results]
    logger.info(
        "Multi-seed summary: val_EER = %.2f%% ± %.2f%%",
        np.mean(val_eers) * 100, np.std(val_eers) * 100,
    )

    report = {
        "experiment_id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seeds": args.seeds,
        "val_eer_mean": float(np.mean(val_eers)),
        "val_eer_std": float(np.std(val_eers)),
        "seed_results": seed_results,
    }
    with open(out_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Training complete. Report saved to %s", out_dir)
    logger.info("Best checkpoint per seed in %s/checkpoints/", out_dir)


if __name__ == "__main__":
    main()
