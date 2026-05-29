import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit duplicate sessions in preprocessed keystroke dataset (.npy).")
    parser.add_argument(
        "--dataset",
        type=str,
        default="../data/preprocessed/keystroke_all_dict.npy",
        help="Path to generated dataset file (.npy). Supports dict and list formats.",
    )
    parser.add_argument(
        "--feature-columns",
        type=int,
        default=5,
        help="Number of leading numeric feature columns used to compare sessions.",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=6,
        help="Decimal rounding before hashing (set < 0 to disable rounding).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=30,
        help="Maximum number of duplicate groups to include in report examples.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="./results/check_duplicates.json",
        help="Optional path to save full duplicate audit report as JSON.",
    )
    return parser.parse_args()


def _to_py_number(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _resolve_existing_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    search_roots = [Path.cwd(), SCRIPT_DIR, SCRIPT_DIR.parent]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return None


def load_dataset(path: Path):
    try:
        raw = np.load(str(path), allow_pickle=True)
    except EOFError as exc:
        raise RuntimeError(
            f"Dataset file is incomplete/corrupted: {path}. "
            "This usually means preprocessing was interrupted before np.save finished."
        ) from exc
    if isinstance(raw, np.ndarray) and raw.shape == ():
        return raw.item()
    return raw


def _iter_user_sessions(dataset) -> Iterable[Tuple[str, str, np.ndarray]]:
    if isinstance(dataset, dict):
        for user_id, sessions in dataset.items():
            if isinstance(sessions, dict):
                for session_id, session in sessions.items():
                    yield str(user_id), str(session_id), session
            else:
                for idx, session in enumerate(list(sessions)):
                    yield str(user_id), str(idx), session
        return

    if isinstance(dataset, np.ndarray):
        dataset = dataset.tolist()

    if not isinstance(dataset, (list, tuple)):
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    for user_idx, sessions in enumerate(dataset):
        if isinstance(sessions, dict):
            for session_id, session in sessions.items():
                yield str(user_idx), str(session_id), session
        else:
            for idx, session in enumerate(list(sessions)):
                yield str(user_idx), str(idx), session


def _make_signature(numeric: np.ndarray, round_decimals: int) -> str:
    arr = np.ascontiguousarray(numeric, dtype=np.float64)
    if round_decimals >= 0:
        arr = np.round(arr, decimals=round_decimals)
    hasher = hashlib.sha1()
    hasher.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    hasher.update(arr.tobytes())
    return hasher.hexdigest()


def audit_duplicates(dataset, feature_columns: int = 5, round_decimals: int = 6, max_examples: int = 30) -> Dict:
    if feature_columns < 1:
        raise ValueError("feature_columns must be >= 1")

    total_sessions = 0
    valid_sessions = 0
    invalid_sessions = []

    signature_counts: Dict[str, int] = {}
    signature_first_entry: Dict[str, Dict] = {}
    duplicate_groups: Dict[str, list] = {}

    for user_id, session_id, session in _iter_user_sessions(dataset):
        total_sessions += 1
        arr = np.asarray(session)
        if arr.ndim != 2 or arr.shape[1] < feature_columns:
            invalid_sessions.append(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "reason": "malformed_shape",
                    "shape": list(arr.shape),
                }
            )
            continue

        numeric = np.asarray(arr[:, :feature_columns], dtype=np.float64)
        if not np.isfinite(numeric).all():
            invalid_sessions.append(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "reason": "non_finite_values",
                    "shape": [int(numeric.shape[0]), int(numeric.shape[1])],
                }
            )
            continue

        valid_sessions += 1
        entry = {
            "user_id": user_id,
            "session_id": session_id,
            "shape": [int(numeric.shape[0]), int(numeric.shape[1])],
        }
        signature = _make_signature(numeric, round_decimals)

        prev_count = signature_counts.get(signature, 0)
        if prev_count == 0:
            signature_counts[signature] = 1
            signature_first_entry[signature] = entry
            continue
        if prev_count == 1:
            duplicate_groups[signature] = [signature_first_entry[signature], entry]
            signature_counts[signature] = 2
            continue
        duplicate_groups[signature].append(entry)
        signature_counts[signature] = prev_count + 1

    within_user_groups = []
    cross_user_groups = []
    users_with_within_duplicates = set()
    users_with_cross_duplicates = set()
    extra_duplicate_sessions = 0

    for signature, entries in duplicate_groups.items():
        extra_duplicate_sessions += len(entries) - 1
        grouped_by_user = defaultdict(list)
        for entry in entries:
            grouped_by_user[entry["user_id"]].append(entry["session_id"])
        user_ids = sorted(grouped_by_user.keys())
        group_info = {
            "signature": signature,
            "signature_prefix": signature[:12],
            "occurrences": len(entries),
            "users": user_ids,
            "sessions_by_user": {uid: sorted(sids) for uid, sids in grouped_by_user.items()},
            "entries": entries,
        }

        if any(len(sids) > 1 for sids in grouped_by_user.values()):
            within_user_groups.append(group_info)
            users_with_within_duplicates.update(uid for uid, sids in grouped_by_user.items() if len(sids) > 1)
        if len(user_ids) > 1:
            cross_user_groups.append(group_info)
            users_with_cross_duplicates.update(user_ids)

    report = {
        "summary": {
            "total_sessions_scanned": int(total_sessions),
            "valid_sessions_scanned": int(valid_sessions),
            "invalid_sessions_skipped": int(len(invalid_sessions)),
            "feature_columns_compared": int(feature_columns),
            "round_decimals": int(round_decimals),
            "unique_session_signatures": int(valid_sessions - extra_duplicate_sessions),
            "duplicate_signature_groups": int(len(duplicate_groups)),
            "duplicate_sessions_extra": int(extra_duplicate_sessions),
            "within_user_duplicate_groups": int(len(within_user_groups)),
            "cross_user_duplicate_groups": int(len(cross_user_groups)),
            "users_with_within_duplicates": int(len(users_with_within_duplicates)),
            "users_with_cross_duplicates": int(len(users_with_cross_duplicates)),
        },
        "invalid_sessions": invalid_sessions[:max_examples],
        "within_user_duplicate_examples": within_user_groups[:max_examples],
        "cross_user_duplicate_examples": cross_user_groups[:max_examples],
    }
    return report


def print_summary(report: Dict) -> None:
    summary = report["summary"]
    print("=== Dataset Duplicate Audit Summary ===")
    print(f"Total sessions scanned: {summary['total_sessions_scanned']}")
    print(f"Valid sessions scanned: {summary['valid_sessions_scanned']}")
    print(f"Invalid sessions skipped: {summary['invalid_sessions_skipped']}")
    print(f"Feature columns compared: {summary['feature_columns_compared']}")
    print(f"Rounding decimals: {summary['round_decimals']}")
    print(f"Unique session signatures: {summary['unique_session_signatures']}")
    print(f"Duplicate signature groups: {summary['duplicate_signature_groups']}")
    print(f"Duplicate sessions (extra copies): {summary['duplicate_sessions_extra']}")
    print(f"Within-user duplicate groups: {summary['within_user_duplicate_groups']}")
    print(f"Cross-user duplicate groups: {summary['cross_user_duplicate_groups']}")
    print(f"Users with within-user duplicates: {summary['users_with_within_duplicates']}")
    print(f"Users with cross-user duplicates: {summary['users_with_cross_duplicates']}")


def main():
    args = parse_args()
    dataset_path = _resolve_existing_path(args.dataset)
    if dataset_path is None:
        raise FileNotFoundError(
            f"Dataset file not found: {args.dataset}. "
            f"Tried relative to: {Path.cwd()}, {SCRIPT_DIR}, and {SCRIPT_DIR.parent}"
        )

    dataset = load_dataset(dataset_path)
    report = audit_duplicates(
        dataset,
        feature_columns=args.feature_columns,
        round_decimals=args.round_decimals,
        max_examples=args.max_examples,
    )
    print_summary(report)

    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = (SCRIPT_DIR / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=_to_py_number)
        print(f"\nSaved duplicate audit report to: {out_path}")


if __name__ == "__main__":
    main()
