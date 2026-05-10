import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit generated Aalto preprocessed dataset (.npy).")
    parser.add_argument(
        "--dataset",
        type=str,
        default="../data/preprocessed/keystroke_all_dict.npy",
        help="Path to generated dataset file (.npy). Supports dict and list formats.",
    )
    parser.add_argument(
        "--outlier-z-threshold",
        type=float,
        default=3.5,
        help="Modified z-score threshold for user-level outlier detection.",
    )
    parser.add_argument(
        "--max-problem-users",
        type=int,
        default=30,
        help="Maximum number of problematic users to print.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="./results/check_dataset.json",
        help="Optional path to save full audit report as JSON.",
    )
    return parser.parse_args()


def _to_py_number(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _safe_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def _iter_users(dataset) -> Iterable[Tuple[str, List[np.ndarray]]]:
    if isinstance(dataset, dict):
        for user_id, sessions in dataset.items():
            if isinstance(sessions, dict):
                yield str(user_id), list(sessions.values())
            else:
                yield str(user_id), list(sessions)
        return

    if isinstance(dataset, np.ndarray):
        dataset = dataset.tolist()

    if not isinstance(dataset, (list, tuple)):
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    for idx, sessions in enumerate(dataset):
        if isinstance(sessions, dict):
            yield str(idx), list(sessions.values())
        else:
            yield str(idx), list(sessions)


def _modified_z_scores(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values, dtype=np.float64)
    return 0.6745 * (values - median) / mad


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


def audit_dataset(dataset, outlier_z_threshold: float = 3.5) -> Dict:
    user_reports = []
    global_session_lengths: List[int] = []
    feature_names = ["hold_time", "inter_press_time", "inter_release_time", "inter_key_time", "key_code"]

    for user_id, sessions in _iter_users(dataset):
        session_lengths: List[int] = []
        sum_features = np.zeros(5, dtype=np.float64)
        count_features = np.zeros(5, dtype=np.int64)
        nan_count = 0
        inf_count = 0
        malformed_sessions = 0

        for session in sessions:
            arr = np.asarray(session)
            if arr.ndim != 2 or arr.shape[1] < 5:
                malformed_sessions += 1
                continue

            numeric = np.asarray(arr[:, :5], dtype=np.float64)
            session_lengths.append(int(numeric.shape[0]))
            global_session_lengths.append(int(numeric.shape[0]))

            nan_mask = np.isnan(numeric)
            inf_mask = np.isinf(numeric)
            nan_count += int(np.sum(nan_mask))
            inf_count += int(np.sum(inf_mask))

            finite_mask = np.isfinite(numeric)
            sum_features += np.where(finite_mask, numeric, 0.0).sum(axis=0)
            count_features += finite_mask.sum(axis=0)

        means = np.divide(
            sum_features,
            np.maximum(count_features, 1),
            out=np.zeros_like(sum_features),
            where=np.maximum(count_features, 1) > 0,
        )

        user_reports.append(
            {
                "user_id": str(user_id),
                "session_count": int(len(sessions)),
                "valid_session_count": int(len(session_lengths)),
                "malformed_session_count": int(malformed_sessions),
                "total_keystrokes": int(np.sum(session_lengths) if session_lengths else 0),
                "median_session_len": float(np.median(session_lengths) if session_lengths else 0.0),
                "mean_session_len": float(np.mean(session_lengths) if session_lengths else 0.0),
                "nan_count": int(nan_count),
                "inf_count": int(inf_count),
                "feature_means": {feature_names[i]: float(means[i]) for i in range(5)},
            }
        )

    session_counts = np.asarray([u["session_count"] for u in user_reports], dtype=np.float64)
    med_session_lens = np.asarray([u["median_session_len"] for u in user_reports], dtype=np.float64)
    hold_means = np.asarray([u["feature_means"]["hold_time"] for u in user_reports], dtype=np.float64)
    inter_press_means = np.asarray([u["feature_means"]["inter_press_time"] for u in user_reports], dtype=np.float64)
    inter_release_means = np.asarray([u["feature_means"]["inter_release_time"] for u in user_reports], dtype=np.float64)
    inter_key_means = np.asarray([u["feature_means"]["inter_key_time"] for u in user_reports], dtype=np.float64)

    metrics = {
        "session_count": session_counts,
        "median_session_len": med_session_lens,
        "mean_hold_time": hold_means,
        "mean_inter_press_time": inter_press_means,
        "mean_inter_release_time": inter_release_means,
        "mean_inter_key_time": inter_key_means,
    }

    outlier_by_metric: Dict[str, set] = {name: set() for name in metrics}
    for metric_name, values in metrics.items():
        if len(values) == 0:
            continue
        mz = _modified_z_scores(values)
        idxs = np.where(np.abs(mz) > outlier_z_threshold)[0]
        for idx in idxs:
            outlier_by_metric[metric_name].add(int(idx))

    problematic_users = []
    for idx, report in enumerate(user_reports):
        reasons = []
        if report["nan_count"] > 0:
            reasons.append("has_nan")
        if report["inf_count"] > 0:
            reasons.append("has_inf")
        if report["malformed_session_count"] > 0:
            reasons.append("malformed_session")

        user_outlier_metrics = [name for name, idxs in outlier_by_metric.items() if idx in idxs]
        if user_outlier_metrics:
            reasons.append(f"outlier:{','.join(user_outlier_metrics)}")

        if reasons:
            problematic_users.append(
                {
                    "user_id": report["user_id"],
                    "reasons": reasons,
                    "session_count": report["session_count"],
                    "median_session_len": report["median_session_len"],
                    "nan_count": report["nan_count"],
                    "inf_count": report["inf_count"],
                }
            )

    report = {
        "summary": {
            "total_users": int(len(user_reports)),
            "total_sessions": int(sum(u["session_count"] for u in user_reports)),
            "total_valid_sessions": int(sum(u["valid_session_count"] for u in user_reports)),
            "total_keystrokes": int(sum(u["total_keystrokes"] for u in user_reports)),
            "users_with_nan": int(sum(1 for u in user_reports if u["nan_count"] > 0)),
            "users_with_inf": int(sum(1 for u in user_reports if u["inf_count"] > 0)),
            "users_with_malformed_sessions": int(sum(1 for u in user_reports if u["malformed_session_count"] > 0)),
            "users_flagged_outlier_or_invalid": int(len(problematic_users)),
            "session_length_stats_global": _safe_stats(global_session_lengths),
            "session_count_per_user_stats": _safe_stats([u["session_count"] for u in user_reports]),
        },
        "problematic_users": problematic_users,
        "user_reports": user_reports,
    }
    return report


def print_summary(report: Dict, max_problem_users: int) -> None:
    summary = report["summary"]
    print("=== Dataset Audit Summary ===")
    print(f"Total users: {summary['total_users']}")
    print(f"Total sessions: {summary['total_sessions']}")
    print(f"Total valid sessions: {summary['total_valid_sessions']}")
    print(f"Total keystrokes: {summary['total_keystrokes']}")
    print(f"Users with NaN: {summary['users_with_nan']}")
    print(f"Users with Inf: {summary['users_with_inf']}")
    print(f"Users with malformed sessions: {summary['users_with_malformed_sessions']}")
    print(f"Users flagged (outlier/invalid): {summary['users_flagged_outlier_or_invalid']}")

    print("\nSession length stats (global):")
    for k, v in summary["session_length_stats_global"].items():
        print(f"  {k}: {v:.4f}")

    print("\nSessions per user stats:")
    for k, v in summary["session_count_per_user_stats"].items():
        print(f"  {k}: {v:.4f}")

    problematic = report["problematic_users"]
    if not problematic:
        print("\nNo problematic users found based on current rules.")
        return

    print(f"\nProblematic users (showing up to {max_problem_users}):")
    for user in problematic[:max_problem_users]:
        print(
            f"  user_id={user['user_id']}, reasons={user['reasons']}, "
            f"sessions={user['session_count']}, median_len={user['median_session_len']:.2f}, "
            f"nan={user['nan_count']}, inf={user['inf_count']}"
        )


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset = load_dataset(dataset_path)
    report = audit_dataset(dataset, outlier_z_threshold=args.outlier_z_threshold)
    print_summary(report, args.max_problem_users)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=_to_py_number)
        print(f"\nSaved audit report to: {out_path}")


if __name__ == "__main__":
    main()
