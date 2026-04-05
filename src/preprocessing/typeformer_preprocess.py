import numpy as np
from typing import List

def typeformer_preprocess(sessions: List[np.ndarray], seq_len: int = 50) -> List[np.ndarray]:
    """
    TypeFormer's original preprocessing.
    
    Steps:
      1. Truncate or zero-pad to fixed length seq_len
      2. Assume features are: [hold, inter_press, inter_release, inter_key, ascii_normalized]
      (The ASCII value is already normalized by /255 in the raw data loader)
    
    Args:
        sessions: List of raw feature arrays, each shape (N, 5) or (N, 6).
                  If 6, the last column is the letter name and will be dropped.
        seq_len:  Target sequence length (50 in TypeFormer paper)
    
    Returns:
        List of processed arrays, each shape (seq_len, 5)
    """
    processed = []
    for session in sessions:
        # Keep only the first 5 columns and convert to float64 for the model
        result = session[:, :5].copy().astype(np.float64)
        N = len(result)
        
        # Step 1: Truncate or zero-pad
        if N >= seq_len:
            result = result[:seq_len]
        else:
            pad = np.zeros((seq_len - N, 5), dtype=np.float64)
            result = np.vstack([result, pad])
            
        processed.append(result)
    
    return processed
