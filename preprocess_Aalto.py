import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd


KEYS_COLUMNS = ['KEYSTROKE_ID', 'PRESS_TIME', 'RELEASE_TIME', 'LETTER', 'TEST_SECTION_ID', 'KEYCODE', 'IKI']
USERS_COLUMNS = ['TEST_SECTION_ID', 'SENTENCE_ID', 'PARTICIPANT_ID', 'USER_INPUT', 'INPUT_TIME', 'EDIT_DISTANCE',
                 'ERROR_RATE', 'WPM', 'INPUT_LENGTH', 'ERROR_LEN', 'POTENTIAL_WPM', 'POTENTIAL_LENGTH', 'DEVICE']


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Aalto keystroke dataset.")
    parser.add_argument("--file-raw", default="./data/raw/keystrokes.csv", help="Path to raw keystrokes CSV.")
    parser.add_argument("--file-users", default="./data/raw/test_sections.csv", help="Path to user/session CSV.")
    parser.add_argument("--output-list", default="data/preprocessed/keystroke_all_list.npy",
                        help="Output path for list-based preprocessed dataset.")
    parser.add_argument("--output-dict", default="data/preprocessed/keystroke_all_dict.npy",
                        help="Output path for dict-based preprocessed dataset.")
    parser.add_argument("--min-sessions", type=int, default=15,
                        help="Minimum number of sessions required per participant.")
    parser.add_argument("--clip-min", type=float, default=-10.0,
                        help="Minimum clipping bound (seconds) for timing features.")
    parser.add_argument("--clip-max", type=float, default=10.0,
                        help="Maximum clipping bound (seconds) for timing features.")
    parser.add_argument(
        "--deduplicate-scope",
        choices=["none", "within-user", "global"],
        default="global",
        help="Session deduplication scope after feature extraction.",
    )
    parser.add_argument(
        "--dedup-feature-columns",
        type=int,
        default=5,
        help="Number of leading numeric feature columns used for deduplication signature.",
    )
    parser.add_argument(
        "--dedup-round-decimals",
        type=int,
        default=6,
        help="Rounding decimals before deduplication hashing (set < 0 to disable rounding).",
    )
    return parser.parse_args()


def read_csv_with_bad_line_fallback(path, **kwargs):
    # Fast path: C engine with malformed-line skipping (newer pandas).
    try:
        return pd.read_csv(path, on_bad_lines='skip', **kwargs)
    except (TypeError, ValueError):
        pass
    # Compatibility path: python engine with on_bad_lines (mid pandas versions).
    try:
        return pd.read_csv(path, engine='python', on_bad_lines='skip', **kwargs)
    except TypeError:
        # Legacy compatibility path for older pandas versions.
        return pd.read_csv(path, engine='python', error_bad_lines=False, warn_bad_lines=False, **kwargs)


def extract_keys_features(session_key, clip_min, clip_max):
    press = np.asarray(session_key.PRESS_TIME)
    release = np.asarray(session_key.RELEASE_TIME)
    key_code = np.asarray(session_key.KEYCODE) / 255
    key_name = np.asarray(session_key.LETTER)
    hold_time = np.clip((release - press) / 1000, clip_min, clip_max)
    inter_press_time = np.clip(np.append(0, np.diff(press) / 1000), clip_min, clip_max)
    inter_release_time = np.clip(np.append(0, np.diff(release) / 1000), clip_min, clip_max)
    inter_key_time = np.clip(np.append(0, (release[:-1] - press[1:]) / 1000), clip_min, clip_max)
    keys_features = np.array(
        [hold_time.astype(np.float32), inter_press_time.astype(np.float32), inter_release_time.astype(np.float32),
         inter_key_time.astype(np.float32), key_code.astype(np.float32), key_name])
    return keys_features.T


def session_signature(keys_features, feature_columns, round_decimals):
    numeric = np.ascontiguousarray(np.asarray(keys_features[:, :feature_columns], dtype=np.float64))
    if round_decimals >= 0:
        numeric = np.round(numeric, decimals=round_decimals)
    hasher = hashlib.sha1()
    hasher.update(np.asarray(numeric.shape, dtype=np.int64).tobytes())
    hasher.update(numeric.tobytes())
    return hasher.hexdigest()


def main():
    args = parse_args()
    if args.clip_min > args.clip_max:
        raise ValueError("--clip-min must be <= --clip-max")
    if args.min_sessions < 1:
        raise ValueError("--min-sessions must be >= 1")
    if args.dedup_feature_columns < 1:
        raise ValueError("--dedup-feature-columns must be >= 1")

    start = time.time()

    keys_db = read_csv_with_bad_line_fallback(
        args.file_raw,
        sep=",",
        index_col=False,
        header=None,
        encoding_errors='replace',
        names=KEYS_COLUMNS,
        usecols=['KEYSTROKE_ID', 'PRESS_TIME', 'RELEASE_TIME', 'LETTER', 'TEST_SECTION_ID', 'KEYCODE']
    )

    other_db = read_csv_with_bad_line_fallback(
        args.file_users,
        sep=",",
        index_col=False,
        header=None,
        encoding_errors='replace',
        names=USERS_COLUMNS,
        usecols=['TEST_SECTION_ID', 'PARTICIPANT_ID']
    )

    # Normalize potentially mixed-type IDs/timestamps from malformed rows.
    keys_db['KEYSTROKE_ID'] = pd.to_numeric(keys_db['KEYSTROKE_ID'], errors='coerce')
    keys_db['PRESS_TIME'] = pd.to_numeric(keys_db['PRESS_TIME'], errors='coerce')
    keys_db['RELEASE_TIME'] = pd.to_numeric(keys_db['RELEASE_TIME'], errors='coerce')
    keys_db['TEST_SECTION_ID'] = pd.to_numeric(keys_db['TEST_SECTION_ID'], errors='coerce')
    keys_db['KEYCODE'] = pd.to_numeric(keys_db['KEYCODE'], errors='coerce')
    keys_db = keys_db.dropna(subset=['KEYSTROKE_ID', 'PRESS_TIME', 'RELEASE_TIME', 'TEST_SECTION_ID', 'KEYCODE']).copy()
    keys_db['KEYSTROKE_ID'] = keys_db['KEYSTROKE_ID'].astype(np.int64)
    keys_db['PRESS_TIME'] = keys_db['PRESS_TIME'].astype(np.int64)
    keys_db['RELEASE_TIME'] = keys_db['RELEASE_TIME'].astype(np.int64)
    keys_db['TEST_SECTION_ID'] = keys_db['TEST_SECTION_ID'].astype(np.int64)
    keys_db['KEYCODE'] = keys_db['KEYCODE'].astype(np.int64)

    other_db['TEST_SECTION_ID'] = pd.to_numeric(other_db['TEST_SECTION_ID'], errors='coerce')
    other_db['PARTICIPANT_ID'] = pd.to_numeric(other_db['PARTICIPANT_ID'], errors='coerce')
    other_db = other_db.dropna(subset=['TEST_SECTION_ID', 'PARTICIPANT_ID']).copy()
    other_db['TEST_SECTION_ID'] = other_db['TEST_SECTION_ID'].astype(np.int64)
    other_db['PARTICIPANT_ID'] = other_db['PARTICIPANT_ID'].astype(np.int64)

    participant_map = other_db[['TEST_SECTION_ID', 'PARTICIPANT_ID']].drop_duplicates(subset=['TEST_SECTION_ID'])
    keys_db = keys_db.merge(participant_map, on='TEST_SECTION_ID', how='left', sort=False)
    keys_db = keys_db[keys_db['PARTICIPANT_ID'].notna()].copy()
    keys_db['PARTICIPANT_ID'] = keys_db['PARTICIPANT_ID'].astype(int)

    participant_sessions = keys_db.groupby('PARTICIPANT_ID')['TEST_SECTION_ID'].nunique()
    valid_participants = participant_sessions[participant_sessions >= args.min_sessions].index
    keys_db = keys_db[keys_db['PARTICIPANT_ID'].isin(valid_participants)].copy()
    keys_db = keys_db.sort_values(['PARTICIPANT_ID', 'TEST_SECTION_ID', 'KEYSTROKE_ID'], kind='mergesort')
    users_before_dedup = int(keys_db['PARTICIPANT_ID'].nunique())

    keys_feature_session = []
    keys_feature_session_dict = {}
    keys_features_db = []
    keys_features_db_dict = {}
    current_user = None
    current_user_signatures = set()
    global_signatures = set()
    duplicates_removed_total = 0
    duplicates_removed_within_user = 0
    duplicates_removed_cross_user = 0
    users_dropped_post_dedup = 0
    sessions_dropped_post_dedup = 0

    def flush_current_user():
        nonlocal keys_feature_session, keys_feature_session_dict
        nonlocal users_dropped_post_dedup, sessions_dropped_post_dedup
        if current_user is None:
            return
        if len(keys_feature_session) < args.min_sessions:
            users_dropped_post_dedup += 1
            sessions_dropped_post_dedup += len(keys_feature_session)
            return
        keys_features_db.append(keys_feature_session)
        keys_features_db_dict[str(current_user)] = keys_feature_session_dict

    for (participant_id, test_section_id), session_key in keys_db.groupby(['PARTICIPANT_ID', 'TEST_SECTION_ID'], sort=True):
        if current_user is None:
            current_user = participant_id
            current_user_signatures = set()
        elif current_user != participant_id:
            flush_current_user()
            keys_feature_session = []
            keys_feature_session_dict = {}
            current_user = participant_id
            current_user_signatures = set()
        keys_features = extract_keys_features(session_key, args.clip_min, args.clip_max)

        if args.deduplicate_scope != "none":
            signature = session_signature(
                keys_features,
                feature_columns=args.dedup_feature_columns,
                round_decimals=args.dedup_round_decimals,
            )
            duplicate_within_user = signature in current_user_signatures
            duplicate_global = signature in global_signatures
            if args.deduplicate_scope == "within-user":
                is_duplicate = duplicate_within_user
            else:
                is_duplicate = duplicate_global
            if is_duplicate:
                duplicates_removed_total += 1
                if duplicate_within_user:
                    duplicates_removed_within_user += 1
                else:
                    duplicates_removed_cross_user += 1
                continue
            current_user_signatures.add(signature)
            if args.deduplicate_scope == "global":
                global_signatures.add(signature)

        keys_feature_session.append(keys_features)
        keys_feature_session_dict[str(test_section_id)] = keys_features

    flush_current_user()

    output_list = Path(args.output_list)
    output_dict = Path(args.output_dict)
    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_dict.parent.mkdir(parents=True, exist_ok=True)

    # Ragged nested sessions/users require object dtype for stable serialization.
    np.save(str(output_list), np.asarray(keys_features_db, dtype=object), allow_pickle=True)
    np.save(str(output_dict), keys_features_db_dict)

    end = time.time()
    time_elapsed = (end - start) / 60
    print(f"{time_elapsed:.2f} minutes")
    print(f"Saved: {output_list}")
    print(f"Saved: {output_dict}")
    print(f"Timing clipping range (seconds): [{args.clip_min}, {args.clip_max}]")
    print(f"Minimum sessions per participant: {args.min_sessions} (enforced pre and post dedup)")
    print(
        f"Deduplication: scope={args.deduplicate_scope}, "
        f"feature_columns={args.dedup_feature_columns}, "
        f"round_decimals={args.dedup_round_decimals}"
    )
    print(
        f"Duplicates removed: total={duplicates_removed_total}, "
        f"within_user={duplicates_removed_within_user}, "
        f"cross_user={duplicates_removed_cross_user}"
    )
    print(
        f"Users retained after dedup: {len(keys_features_db_dict)} / {users_before_dedup}; "
        f"users dropped post-dedup (< min-sessions): {users_dropped_post_dedup}; "
        f"sessions dropped with those users: {sessions_dropped_post_dedup}"
    )


if __name__ == "__main__":
    main()
