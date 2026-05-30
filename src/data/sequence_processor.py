import numpy as np


def process_session(session: np.ndarray, seq_len: int = 50) -> np.ndarray:
    """Convert one raw keystroke session to a fixed-length float32 array.

    The raw session from preprocess_Aalto.py has shape (N_keystrokes, 6):
        col 0: hold_time (HL)
        col 1: inter_press_time (IL)
        col 2: inter_release_time (IRL)
        col 3: inter_key_time (IKT)
        col 4: keycode / 255  (ASCII normalised)
        col 5: key_name / letter  (string — dropped here)

    Processing steps:
        1. Keep only numeric feature columns 0-4.
        2. Zero-pad at the end to seq_len, or truncate if longer.
        3. Cast to float32.

    Args:
        session: (N_keystrokes, >=5) array, possibly object dtype.
        seq_len: Target sequence length L (default 50).

    Returns:
        (seq_len, 5) float32 array.
    """
    # Cast numeric columns to float32, dropping the letter column
    try:
        features = np.array(session[:, :5], dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Cannot convert session of shape {session.shape} to float32. "
            "Ensure columns 0-4 are numeric."
        ) from exc

    result = np.zeros((seq_len, 5), dtype=np.float32)
    n = min(len(features), seq_len)
    result[:n] = features[:n]
    return result


def process_user_sessions(
    user_sessions: list,
    seq_len: int = 50,
    num_sessions: int = 15,
) -> np.ndarray:
    """Process all sessions for one user.

    Args:
        user_sessions: List of session arrays (length may be > num_sessions).
        seq_len: Target keystroke sequence length.
        num_sessions: Number of sessions to keep (first N).

    Returns:
        (num_sessions, seq_len, 5) float32 array.
    """
    sessions = user_sessions[:num_sessions]
    processed = np.stack([process_session(s, seq_len) for s in sessions], axis=0)
    return processed
