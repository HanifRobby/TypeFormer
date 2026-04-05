import numpy as np
import os
import sys
import time

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.preprocessing.kdprint_preprocess import KDPrintPreprocessor


def main():
    print("Loading original dataset...")
    start_time = time.time()
    raw_data_path = r"data\Mobile_keys_db_6_features.npy"

    # allow_pickle=True is required for arrays of objects/lists
    data = np.load(raw_data_path, allow_pickle=True)
    n_users = len(data)
    print(f"Loaded {n_users} users in {time.time() - start_time:.2f} seconds.")

    processed_data = []

    # Configure KDPrint Preprocessor
    buffer_size = 5
    seq_len = 50

    start_time = time.time()
    print(
        "Starting bulk preprocessing using Option B (fitting on all sessions per user)..."
    )

    for i in range(n_users):
        if i % 5000 == 0 and i > 0:
            print(f"Processed {i}/{n_users} users...")

        user_sessions = data[i]

        prep = KDPrintPreprocessor(buffer_size=buffer_size, seq_len=seq_len)

        # We don't cast to float here, KDPrintPreprocessor internal logic
        # safely slices `s[:, :5]` to discard strings, and casts to float.
        enrolment_sessions = [s for s in user_sessions]

        # Fit on ALL sessions for the user to establish optimal statistics (Option B)
        prep.fit(enrolment_sessions)

        # Transform all sessions for the user
        processed_sessions = []
        for s in enrolment_sessions:
            # We use buffer=True as specified by KDPrint implementation
            proc_s = prep.transform(s, use_buffer=True)
            processed_sessions.append(proc_s)

        processed_data.append(processed_sessions)

    print(f"Finished processing all users in {time.time() - start_time:.2f} seconds.")

    # Save the output
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../src/data/processed")
    )
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "Mobile_keys_db_new.npy")

    print(f"Saving to {out_path} ...")
    start_time = time.time()

    # Using python list to object dtype numpy array
    processed_array = np.array(processed_data, dtype=object)
    np.save(out_path, processed_array)
    print(f"Saved successfully in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
