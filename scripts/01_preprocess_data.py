"""
01_preprocess_data.py — Split keystroke_all_list.npy into processed splits.

Reads the preprocessed dataset produced by TypeFormer/preprocess_Aalto.py and
creates four .npz files (train, val, test, cohort_pool) with uniform shape:
    (N_users, N_sessions, L=50, C=5)

Usage:
    conda run -n Typeformer python scripts/01_preprocess_data.py
    conda run -n Typeformer python scripts/01_preprocess_data.py --npy-path data/preprocessed/keystroke_all_list.npy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config_loader import load_config
from src.utils.logging import setup_logger
from src.utils.seeds import set_global_seed
from src.data.sequence_processor import process_user_sessions

ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split and preprocess Aalto dataset.")
    p.add_argument("--npy-path", type=str, default=None,
                   help="Path to keystroke_all_list.npy (overrides config)")
    p.add_argument("--seq-len", type=int, default=50,
                   help="Target sequence length (default: 50)")
    p.add_argument("--num-sessions", type=int, default=15,
                   help="Sessions per user to keep (default: 15)")
    return p.parse_args()


def process_split(
    raw_dataset: list,
    indices: range,
    seq_len: int,
    num_sessions: int,
    split_name: str,
    logger,
) -> tuple[np.ndarray, np.ndarray]:
    """Process a slice of the raw dataset into uniform arrays.

    Returns:
        user_ids: (N,) int64 array of original user indices
        sessions: (N, num_sessions, seq_len, 5) float32 array
    """
    user_ids = []
    sessions_list = []

    skipped = 0
    for orig_idx in tqdm(indices, desc=f"Processing {split_name}", unit="user"):
        user_data = raw_dataset[orig_idx]

        if len(user_data) < num_sessions:
            logger.warning(
                "User %d has only %d sessions (need %d) — skipping.",
                orig_idx, len(user_data), num_sessions,
            )
            skipped += 1
            continue

        try:
            processed = process_user_sessions(user_data, seq_len=seq_len,
                                              num_sessions=num_sessions)
        except Exception as exc:
            logger.warning("User %d failed processing (%s) — skipping.", orig_idx, exc)
            skipped += 1
            continue

        user_ids.append(orig_idx)
        sessions_list.append(processed)

    if not sessions_list:
        raise RuntimeError(f"No valid users found in split '{split_name}'.")

    logger.info(
        "%s: %d users OK, %d skipped.",
        split_name, len(sessions_list), skipped,
    )

    return (
        np.array(user_ids, dtype=np.int64),
        np.stack(sessions_list, axis=0),   # (N, num_sessions, seq_len, 5)
    )


def main() -> None:
    args = parse_args()
    cfg = load_config()
    set_global_seed(cfg.seed)

    log_dir = ROOT / cfg.paths.results_dir / "logs"
    logger = setup_logger("01_preprocess_data", log_dir)

    npy_path = Path(args.npy_path) if args.npy_path else ROOT / cfg.paths.preprocessed_npy
    if not npy_path.exists():
        logger.error("Dataset not found: %s", npy_path)
        logger.error("Place keystroke_all_list.npy at %s and re-run.", npy_path)
        sys.exit(1)

    logger.info("Loading raw dataset from %s ...", npy_path)
    raw_dataset = list(np.load(str(npy_path), allow_pickle=True))
    n_total = len(raw_dataset)
    logger.info("Total users in raw dataset: %d", n_total)

    split = cfg.dataset.split
    seq_len = args.seq_len
    num_sessions = args.num_sessions

    splits_def = {
        "val":         range(split.val_start,    split.val_end),
        "test":        range(split.test_start,   split.test_end),
        "train":       range(split.train_start,  split.train_end),
        "cohort_pool": range(split.cohort_start, split.cohort_end),
    }

    out_dir = ROOT / cfg.paths.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = ROOT / cfg.paths.splits_dir
    splits_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "seq_len": seq_len,
        "num_sessions": num_sessions,
        "source_npy": str(npy_path),
        "splits": {},
    }

    for split_name, indices in splits_def.items():
        logger.info("--- Processing split: %s (indices %d–%d) ---",
                    split_name, indices.start, indices.stop - 1)

        user_ids, sessions = process_split(
            raw_dataset, indices, seq_len, num_sessions, split_name, logger
        )

        out_path = out_dir / f"{split_name}_sessions.npz"
        np.savez_compressed(
            str(out_path),
            user_ids=user_ids,
            sessions=sessions,
        )
        logger.info("Saved %s  shape=%s  → %s", split_name, sessions.shape, out_path)

        # Save user ID list as text for quick reference
        ids_path = splits_dir / f"{split_name}_user_ids.txt"
        np.savetxt(str(ids_path), user_ids, fmt="%d")

        metadata["splits"][split_name] = {
            "n_users": int(len(user_ids)),
            "shape": list(sessions.shape),
            "npz": str(out_path),
        }

    meta_path = out_dir / "preprocessing_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
