"""
Bulk preprocess the Aalto dataset using hold-time-only z-score normalization.

This variant only standardizes column 0 (hold_latency), leaving the other
timing features (inter_press, inter_release, inter_key) and ASCII in their
original raw form.

Rationale:
  Hold time is the most user-distinctive biometric feature — it reflects
  individual finger mechanics. The inter-key features are more dependent
  on text content and cognitive typing patterns, which may not benefit
  from per-user standardization.

Output: src/data/processed/Mobile_keys_db_holdonly.npy
"""

import numpy as np
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.preprocessing.holdtime_preprocess import HoldTimePreprocessor


def main():
    print("=" * 60)
    print("Bulk Preprocessing — Hold-Time Only Z-Score")
    print("=" * 60)

    print("\nLoading original dataset...")
    start_time = time.time()
    raw_data_path = r"data\Mobile_keys_db_6_features.npy"

    data = np.load(raw_data_path, allow_pickle=True)
    n_users = len(data)
    print(f"Loaded {n_users} users in {time.time() - start_time:.2f} seconds.")

    # Show raw data stats for context
    sample_raw = np.array(data[0][0][:, :5], dtype=np.float64)
    print(f"\nRaw data sample (user 0, session 0):")
    print(f"  Shape:        {sample_raw.shape}")
    print(f"  hold_lat [0]: [{sample_raw[:, 0].min():.4f}, {sample_raw[:, 0].max():.4f}]")
    print(f"  inter_pr [1]: [{sample_raw[:, 1].min():.4f}, {sample_raw[:, 1].max():.4f}]")
    print(f"  inter_re [2]: [{sample_raw[:, 2].min():.4f}, {sample_raw[:, 2].max():.4f}]")
    print(f"  inter_ke [3]: [{sample_raw[:, 3].min():.4f}, {sample_raw[:, 3].max():.4f}]")
    print(f"  ascii    [4]: [{sample_raw[:, 4].min():.4f}, {sample_raw[:, 4].max():.4f}]")

    processed_data = []

    buffer_size = 5
    seq_len = 50

    start_time = time.time()
    print(f"\nStarting bulk preprocessing...")
    print(f"  Standardized:     column 0 (hold_latency) only")
    print(f"  buffer_size:      {buffer_size}")
    print(f"  No clipping — fine-tuning will adapt to the full z-score range")
    print(f"  Option B:         fit on all 15 sessions per user")

    for i in range(n_users):
        if i % 5000 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (n_users - i) / rate / 60
            print(f"  Processed {i}/{n_users} users... ({elapsed:.0f}s elapsed, ~{eta:.1f}min remaining)")

        user_sessions = data[i]

        prep = HoldTimePreprocessor(buffer_size=buffer_size, seq_len=seq_len)

        enrolment_sessions = [s for s in user_sessions]

        # Fit on ALL sessions (Option B)
        prep.fit(enrolment_sessions)

        # Transform all sessions
        processed_sessions = []
        for s in enrolment_sessions:
            proc_s = prep.transform(s, use_buffer=True)
            processed_sessions.append(proc_s)

        processed_data.append(processed_sessions)

    print(f"Finished processing all users in {time.time() - start_time:.2f} seconds.")

    # Save output
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../src/data/processed")
    )
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "Mobile_keys_db_holdonly.npy")

    print(f"\nSaving to {out_path} ...")
    start_time = time.time()

    processed_array = np.array(processed_data, dtype=object)
    np.save(out_path, processed_array)
    print(f"Saved successfully in {time.time() - start_time:.2f} seconds.")

    # Verify output
    print("\n--- Verification ---")
    verify = np.load(out_path, allow_pickle=True)
    sample = np.array(verify[0][0], dtype=np.float64)
    print(f"Output shape:     {sample.shape}")
    print(f"hold_lat [0]:     [{sample[:, 0].min():.4f}, {sample[:, 0].max():.4f}]  <- z-scored")
    print(f"inter_pr [1]:     [{sample[:, 1].min():.4f}, {sample[:, 1].max():.4f}]  <- raw (unchanged)")
    print(f"inter_re [2]:     [{sample[:, 2].min():.4f}, {sample[:, 2].max():.4f}]  <- raw (unchanged)")
    print(f"inter_ke [3]:     [{sample[:, 3].min():.4f}, {sample[:, 3].max():.4f}]  <- raw (unchanged)")
    print(f"ascii    [4]:     [{sample[:, 4].min():.4f}, {sample[:, 4].max():.4f}]  <- raw (unchanged)")
    print("\nDone!")


if __name__ == "__main__":
    main()
