# Implementation Guide: Keystroke Dynamics Authentication
## KDPrint Preprocessing + TypeFormer + Adaptive Per-User Threshold

> **Thesis S2 — Mobile Keystroke Biometrics**  
> Based on: TypeFormer (Stragapede et al., 2024) + KDPrint (Kim et al., 2025)  
> Dataset: Aalto Mobile Keystroke Database  

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Environment Setup](#2-environment-setup)
3. [Project Structure](#3-project-structure)
4. [Data Loading & Understanding](#4-data-loading--understanding)
5. [Preprocessing Module (KDPrint)](#5-preprocessing-module-kdprint)
6. [TypeFormer Integration](#6-typeformer-integration)
7. [Adaptive Threshold Module](#7-adaptive-threshold-module)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Experiment Runner (E1–E4)](#9-experiment-runner-e1e4)
10. [Analysis & Visualization](#10-analysis--visualization)
11. [Optimization & Hyperparameter Tuning](#11-optimization--hyperparameter-tuning)
12. [Troubleshooting Guide](#12-troubleshooting-guide)
13. [Checklist Before You Start](#13-checklist-before-you-start)

---

## 1. System Overview

### 1.1 What You Are Building

You are extending the **TypeFormer** keystroke authentication system with two independent contributions:

1. **KDPrint-inspired Preprocessing** — replacing TypeFormer's minimal preprocessing with z-score standardization and buffer aggregation
2. **Adaptive Per-User Threshold** — replacing the single global EER threshold with a per-user threshold derived from the user's own enrollment embedding distribution

The TypeFormer model itself is **not retrained**. You use the publicly available pretrained weights.

### 1.2 Full Pipeline

```
ENROLLMENT PHASE
================
Raw Keystroke Sessions (E sessions, each L×5)
            │
            ▼
┌─────────────────────────────────────┐
│       KDPrint Preprocessing         │
│  1. fit():  compute μ, σ from all   │
│             enrolment sessions      │
│  2. standardize(): x' = (x-μ)/σ    │
│     (timing cols 0–3 only)          │
│  3. apply_buffer(): weighted        │
│     moving average, B=5             │
└─────────────────────────────────────┘
            │
            ▼  E processed sessions, each shape (L, 5)
┌─────────────────────────────────────┐
│         TypeFormer (frozen)         │
│  Temporal Module + Channel Module   │
│  Output: embedding z ∈ ℝ⁶⁴         │
└─────────────────────────────────────┘
            │
            ▼  E embeddings, shape (E, 64)
┌─────────────────────────────────────┐
│    Adaptive Threshold Estimation    │
│  z_mean  = mean(z_1 … z_E)         │
│  d_i     = ‖z_i - z_mean‖₂         │
│  μ_user  = mean(d_i)               │
│  σ_user  = std(d_i)                │
│  T_user  = μ_user + k·σ_user       │
└─────────────────────────────────────┘
            │
            ▼
SAVE TEMPLATE: {μ_feat, σ_feat, z_mean, μ_user, σ_user, T_user}


VERIFICATION PHASE
==================
Raw Keystroke Session (1 test session, L×5)
            │
            ▼
  KDPrint Preprocessing
  (LOAD μ, σ from template — do NOT recompute)
            │
            ▼
  TypeFormer → z_test ∈ ℝ⁶⁴
            │
            ▼
  d_test = ‖z_test - z_mean‖₂
            │
      ┌─────┴─────┐
      │           │
  d ≤ T_user   d > T_user
  ACCEPT ✓     REJECT ✗
```

### 1.3 The 2×2 Factorial Experiment Design

| Config | Label | Preprocessing | Threshold | Purpose |
|--------|-------|---------------|-----------|---------|
| **E1** | Baseline | Raw (TypeFormer original) | Global EER | Reproduce paper result |
| **E2** | Preprocessing only | KDPrint standardization | Global EER | Test Contribution 1 alone |
| **E3** | Threshold only | Raw (TypeFormer original) | Adaptive per-user | Test Contribution 2 alone |
| **E4** | Full system | KDPrint standardization | Adaptive per-user | Combined system |

Run each config with **E = {1, 2, 5, 7, 10}** enrolment sessions and **L = 50** (fixed).  
Total: 4 configs × 5 E values = **20 main experiments**.

### 1.4 Key Numbers to Know

| Parameter | Value | Source |
|-----------|-------|--------|
| Embedding dimension | 64 | TypeFormer architecture |
| Sequence length L | 50 | TypeFormer optimal |
| Triplet loss margin α | 1.0 | TypeFormer training |
| Number of features | 5 | TypeFormer feature extraction |
| Training users | 30,000 | Aalto (TypeFormer protocol) |
| Validation users | 400 | Aalto (TypeFormer protocol) |
| Evaluation users | 1,000 | Aalto (TypeFormer protocol) |
| Genuine scores per user | 5 | 5 verification sessions |
| Impostor scores per user | 999 | All other evaluation users |
| TypeFormer baseline EER | 3.25% | L=50, E=5 |
| TypeFormer inference time | ~46ms | Per sample, RTX 3070 Ti |

---

## 2. Environment Setup

### 2.1 Requirements

```bash
# Python version
python --version  # needs 3.8+

# Core dependencies
pip install torch torchvision          # PyTorch — check cuda version first
pip install numpy scipy scikit-learn
pip install matplotlib seaborn
pip install tqdm                       # progress bars
pip install pandas                     # data manipulation
pip install h5py                       # for saving/loading large arrays
```

**GPU check (important):**
```python
import torch
print(torch.cuda.is_available())       # should print True
print(torch.cuda.get_device_name(0))   # shows your GPU name
print(torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
```

If no GPU is available, inference will work but will be slow (~10× slower). For 1000 users × multiple sessions, expect several hours on CPU.

### 2.2 Clone TypeFormer

```bash
git clone https://github.com/BiDAlab/TypeFormer
cd TypeFormer
```

Verify the repository structure:
```
TypeFormer/
├── model/
│   ├── Model.py          ← main architecture file, READ THIS
│   └── Preliminary.py    ← earlier version
├── pretrained/
│   └── TypeFormer_pretrained.pt   ← weights file (download or contact authors)
├── train.py
├── test.py
├── utils/
│   └── ...
└── README.md
```

> **Important:** The preprocessed Aalto data used in the paper is NOT publicly available.  
> Contact the authors: `giuseppe.stragapede@estudiante.uam.es`  
> If no response, download raw Aalto data and preprocess yourself (see Section 4).

### 2.3 Download Aalto Database

```bash
# Dataset page:
# https://userinterfaces.aalto.fi/typing37k/

# After downloading, your data directory should look like:
data/raw_aalto/
├── sentences.csv         # the text sentences users typed
└── keystrokes.csv        # the actual keystroke data (large file, ~several GB)
```

The raw Aalto data columns you need:
- `PARTICIPANT_ID` — user identifier
- `TEST_SECTION_ID` — session identifier  
- `KEYSTROKE_ID` — position in sequence
- `PRESS_TIME` — timestamp of key press (ms)
- `RELEASE_TIME` — timestamp of key release (ms)
- `LETTER` — character typed (for ASCII code)

---

## 3. Project Structure

Create this directory structure **before writing any code**:

```
thesis_project/
│
├── data/
│   ├── raw_aalto/              # raw downloaded Aalto files
│   ├── processed/              # TypeFormer-style preprocessed features
│   │   ├── features_train/     # 30,000 training users
│   │   ├── features_val/       # 400 validation users
│   │   └── features_eval/      # 1,000 evaluation users
│   └── splits/
│       ├── train_users.txt     # user IDs for training split
│       ├── val_users.txt       # user IDs for validation split
│       └── eval_users.txt      # user IDs for evaluation split
│
├── models/
│   └── TypeFormer/             # cloned repository
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── typeformer_preprocess.py   # original TypeFormer preprocessing
│   │   ├── kdprint_preprocess.py      # YOUR KDPrint preprocessing
│   │   └── aalto_loader.py            # data loading utilities
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   └── typeformer_wrapper.py      # wrapper for pretrained TypeFormer
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # EER, FAR, FRR computation
│   │   ├── adaptive_threshold.py      # YOUR adaptive threshold module
│   │   └── global_threshold.py        # original global threshold
│   │
│   └── experiments/
│       ├── __init__.py
│       ├── run_experiment.py          # generic experiment runner
│       ├── E1_baseline.py
│       ├── E2_preprocessing.py
│       ├── E3_adaptive.py
│       └── E4_full_system.py
│
├── analysis/
│   ├── visualize_embeddings.py        # t-SNE plots
│   ├── per_user_analysis.py           # per-user EER breakdown
│   ├── threshold_distribution.py      # T_user distribution plots
│   └── correlation_analysis.py        # σ_user vs EER correlation
│
├── results/
│   ├── E1/
│   ├── E2/
│   ├── E3/
│   └── E4/
│
├── configs/
│   └── experiment_config.yaml         # all hyperparameters in one place
│
├── tests/
│   ├── test_preprocessing.py          # unit tests
│   ├── test_threshold.py
│   └── test_metrics.py
│
└── notebooks/
    └── exploration.ipynb              # for quick experiments
```

---

## 4. Data Loading & Understanding

### 4.1 Understanding the Raw Aalto Format

```python
# src/preprocessing/aalto_loader.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class AaltoLoader:
    """
    Loads and preprocesses raw Aalto Mobile Keystroke Database.
    
    The raw data contains timestamps (press/release) per keystroke.
    We need to compute the 5 features TypeFormer expects:
      [0] hold_latency       = release_time - press_time
      [1] inter_key_latency  = press_time[t+1] - release_time[t]   (flight time)
      [2] press_latency      = press_time[t+1] - press_time[t]
      [3] release_latency    = release_time[t+1] - release_time[t]
      [4] ascii_code         = ord(character) / 127.0  (normalized)
    
    Note: Features at index t are DIGRAPH features — they require
    consecutive keystrokes t and t+1. So a sequence of N keystrokes
    produces N-1 feature vectors.
    
    TypeFormer uses L=50 features, so you need at least 51 keystrokes
    per session.
    """
    
    def __init__(self, data_dir: str, seq_len: int = 50):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        
    def load_user_sessions(self, user_id: str, 
                            n_sessions: int = 15) -> List[np.ndarray]:
        """
        Load all sessions for one user.
        
        Returns:
            List of arrays, each shape (seq_len, 5)
            May return fewer than n_sessions if user has insufficient data.
        """
        # Load keystroke data for this user
        user_data = self._load_raw_keystrokes(user_id)
        
        sessions = []
        for session_id in user_data['TEST_SECTION_ID'].unique():
            session_df = user_data[
                user_data['TEST_SECTION_ID'] == session_id
            ].sort_values('KEYSTROKE_ID')
            
            features = self._extract_features(session_df)
            if features is not None:
                sessions.append(features)
            
            if len(sessions) >= n_sessions:
                break
        
        return sessions
    
    def _extract_features(self, session_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extract 5 TypeFormer features from one session.
        
        Returns None if session has insufficient keystrokes.
        """
        press_times   = session_df['PRESS_TIME'].values.astype(float)
        release_times = session_df['RELEASE_TIME'].values.astype(float)
        letters       = session_df['LETTER'].values
        
        N = len(press_times)
        if N < self.seq_len + 1:
            return None  # not enough keystrokes
        
        # Compute digraph features (length N-1)
        hold_lat    = release_times[:-1] - press_times[:-1]      # hold duration
        inter_key   = press_times[1:]   - release_times[:-1]     # flight time
        press_lat   = press_times[1:]   - press_times[:-1]       # press-to-press
        release_lat = release_times[1:] - release_times[:-1]     # release-to-release
        
        # ASCII codes (use character at position t, normalized to [0,1])
        ascii_codes = np.array([
            ord(c) / 127.0 if isinstance(c, str) and len(c) == 1 else 0.0
            for c in letters[:-1]
        ])
        
        # Stack into (N-1, 5) array
        features = np.column_stack([
            hold_lat, inter_key, press_lat, release_lat, ascii_codes
        ])
        
        # Clip extreme outliers (occasional data errors in Aalto)
        features[:, :4] = np.clip(features[:, :4], 0, 2000)  # max 2 seconds
        
        # Slice or pad to seq_len
        return self._pad_or_truncate(features)
    
    def _pad_or_truncate(self, features: np.ndarray) -> np.ndarray:
        """
        Ensures output shape is exactly (seq_len, 5).
        TypeFormer original: truncate if too long, zero-pad if too short.
        """
        N = len(features)
        if N >= self.seq_len:
            return features[:self.seq_len]
        else:
            # Zero-pad at the end
            pad = np.zeros((self.seq_len - N, 5))
            return np.vstack([features, pad])
    
    def _load_raw_keystrokes(self, user_id: str) -> pd.DataFrame:
        """Load raw CSV data for one user. Override this based on your file format."""
        # Adjust path based on your actual Aalto file structure
        raise NotImplementedError("Implement based on your Aalto file structure")
    
    def load_split(self, split_file: str) -> List[str]:
        """Load user IDs from a split file."""
        with open(split_file) as f:
            return [line.strip() for line in f if line.strip()]
```

### 4.2 Verifying Your Data Loading

Always verify before running experiments:

```python
# tests/test_data_loading.py
import numpy as np

def verify_data_loading(loader, user_id, expected_shape=(50, 5)):
    """Run this before ANY experiment to verify data integrity."""
    sessions = loader.load_user_sessions(user_id, n_sessions=15)
    
    print(f"User {user_id}: {len(sessions)} sessions loaded")
    assert len(sessions) >= 5, f"Need at least 5 sessions, got {len(sessions)}"
    
    for i, sess in enumerate(sessions):
        assert sess.shape == expected_shape, \
            f"Session {i} shape mismatch: {sess.shape} != {expected_shape}"
        
        # Check for NaN/Inf
        assert not np.any(np.isnan(sess)), f"NaN found in session {i}"
        assert not np.any(np.isinf(sess)), f"Inf found in session {i}"
        
        # Check timing features are positive
        assert np.all(sess[:, :4] >= 0), f"Negative timing in session {i}"
        
        # Check ASCII is in [0, 1]
        assert np.all(sess[:, 4] >= 0) and np.all(sess[:, 4] <= 1), \
            f"ASCII out of range in session {i}"
    
    print(f"  Shape: {sessions[0].shape}")
    print(f"  hold_lat range: [{sessions[0][:,0].min():.1f}, {sessions[0][:,0].max():.1f}] ms")
    print(f"  inter_key range: [{sessions[0][:,1].min():.1f}, {sessions[0][:,1].max():.1f}] ms")
    print(f"  ascii range: [{sessions[0][:,4].min():.3f}, {sessions[0][:,4].max():.3f}]")
    print("  ✓ Data verification passed")
    return True
```

### 4.3 Working with Pre-Processed Data (if authors provide it)

If the TypeFormer authors share their preprocessed data:

```python
import numpy as np

# The preprocessed data is typically stored as numpy arrays
# Shape: (n_users, n_sessions, seq_len, n_features)
# = (62454, 15, 50, 5)

# Loading example (adjust path):
data = np.load('path/to/preprocessed_data.npz')
# data['X']       → shape (n_users, n_sessions, seq_len, n_features)
# data['user_ids'] → array of user IDs

# Access one user's sessions:
user_idx = 0
user_sessions = data['X'][user_idx]  # shape (15, 50, 5)
print(f"Sessions: {user_sessions.shape}")
# (15, 50, 5) → 15 sessions, 50 keystrokes each, 5 features
```

---

## 5. Preprocessing Module (KDPrint)

### 5.1 TypeFormer Original Preprocessing (Baseline — E1, E3)

Implement this **exactly** as TypeFormer does it, so E1 reproduces the paper result:

```python
# src/preprocessing/typeformer_preprocess.py

import numpy as np
from typing import List

def typeformer_preprocess(sessions: List[np.ndarray], 
                           seq_len: int = 50) -> List[np.ndarray]:
    """
    TypeFormer's original preprocessing (Section 3.1 of TypeFormer paper).
    
    Steps:
      1. Truncate or zero-pad to fixed length seq_len
      2. Normalize ASCII code to [0, 1] by dividing by 127
      3. Timing features (cols 0-3) are left as-is (raw milliseconds)
    
    This is what TypeFormer uses. Note: timing features are NOT normalized.
    
    Args:
        sessions: List of raw feature arrays, each shape (N, 5) where N may vary
        seq_len:  Target sequence length (50 in TypeFormer paper)
    
    Returns:
        List of processed arrays, each shape (seq_len, 5)
    """
    processed = []
    for session in sessions:
        result = session.copy().astype(np.float32)
        N = len(result)
        
        # Step 1: Truncate or zero-pad
        if N >= seq_len:
            result = result[:seq_len]
        else:
            pad = np.zeros((seq_len - N, 5), dtype=np.float32)
            result = np.vstack([result, pad])
        
        # Step 2: Normalize ASCII to [0, 1]
        # TypeFormer divides by 127 (max ASCII value it considers)
        result[:, 4] = result[:, 4] / 127.0
        
        # Step 3: Timing features remain in raw milliseconds
        # (no normalization applied)
        
        processed.append(result)
    
    return processed
```

### 5.2 KDPrint Preprocessing (Contributions — E2, E4)

```python
# src/preprocessing/kdprint_preprocess.py

import numpy as np
import json
from typing import List, Dict, Optional, Tuple

class KDPrintPreprocessor:
    """
    KDPrint-inspired preprocessing for TypeFormer input sequences.
    
    Implements two components from KDPrint (Kim et al., 2025):
      1. Z-score standardization (Section 3.3) — formula identical to paper
      2. Buffer aggregation (Section 3.4, Equation 1) — formula identical to paper
    
    WHAT IS ADAPTED vs ORIGINAL:
    ─────────────────────────────────────────────────────────────────────────
    Component               KDPrint Original         This Implementation
    ─────────────────────────────────────────────────────────────────────────
    Standardization formula  x' = (x - μ) / σ        ✅ Identical
    Standardization scope    timing + touch + force   ⚡ Timing only*
    Buffer formula           Equation 1               ✅ Identical  
    Buffer purpose           Training augmentation    ⚡ Inference smoothing
    Hash permutation         PIN-based reordering     ❌ Not applicable**
    Image encoding           Graph-based visual       ❌ Not applicable
    Deep SVDD classifier     One-class hypersphere    ❌ Replaced by TypeFormer
    ─────────────────────────────────────────────────────────────────────────
    * Aalto database does not contain touch location (x,y) or pressure data.
      KDPrint reports that removing touch location increases EER by ~4%.
      This is a dataset limitation, not a design choice. Mention in thesis.
    
    ** Hash permutation reorders keystroke sequence using PIN hash.
       This is incompatible with TypeFormer because LSTM processes sequences
       in temporal order. Permuting destroys timing relationships.
    
    IMPORTANT — WHEN TO CALL WHAT:
    ─────────────────────────────────────────────────────────────────────────
    fit()           → Call ONCE during enrollment, on enrolment sessions only
    transform()     → Call on EVERY session (enrolment AND test) after fit()
    fit_transform() → Convenience: fit() + transform() in one call
    apply_buffer()  → Optional: apply after standardize(), on any sessions
    ─────────────────────────────────────────────────────────────────────────
    """
    
    def __init__(self, buffer_size: int = 5, seq_len: int = 50):
        """
        Args:
            buffer_size: B in KDPrint paper. Controls smoothing strength.
                         B=5  → moderate smoothing (default)
                         B=10 → stronger smoothing (KDPrint paper uses 10)
                         B=1  → no smoothing (equivalent to no buffer)
            seq_len:    Target sequence length (must match TypeFormer config)
        """
        self.B = buffer_size
        self.seq_len = seq_len
        
        # These are set by fit() and must be saved in user template
        self.mu: Optional[np.ndarray] = None     # shape (4,)
        self.sigma: Optional[np.ndarray] = None  # shape (4,)
        self._is_fitted: bool = False
    
    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: FIT — compute statistics from enrolment data
    # ──────────────────────────────────────────────────────────────────────
    
    def fit(self, enrolment_sessions: List[np.ndarray]) -> 'KDPrintPreprocessor':
        """
        Compute μ and σ from enrolment sessions.
        
        Called ONCE per user during enrollment. The resulting mu and sigma
        represent that user's baseline timing behavior and are stored in
        their biometric template.
        
        Args:
            enrolment_sessions: List of raw feature arrays, each shape (L, 5).
                                 L can be any length (we use all timing data).
                                 These are the sessions used for enrollment.
        
        Returns:
            self (for method chaining)
        
        Why we only use columns 0–3 (timing) and skip column 4 (ASCII):
            Column 4 is the character being typed — it depends on TEXT CONTENT,
            not on the user's biometric behavior. Including it in μ/σ would
            make normalization content-dependent, which is wrong.
            TypeFormer already handles ASCII with its own [0,1] normalization.
        
        Why vstack all sessions before computing statistics:
            With E=5 sessions × L=50 keystrokes = 250 data points per feature.
            This gives a stable estimate of μ and σ. Using only 1 session
            (50 points) risks unstable estimates, especially for σ.
        
        Why sigma += 1e-8 (epsilon):
            Prevents division by zero in standardize() if a feature has
            zero variance in enrolment data (theoretically possible if user
            is perfectly consistent in one feature, or if data is corrupted).
        """
        assert len(enrolment_sessions) >= 1, "Need at least 1 enrolment session"
        
        # Collect all timing features from all enrolment sessions
        # Shape: (E * L, 4) where E = n_sessions, L = seq_len
        all_timing = np.vstack([s[:, :4] for s in enrolment_sessions])
        
        self.mu    = all_timing.mean(axis=0)  # shape (4,)
        self.sigma = all_timing.std(axis=0)   # shape (4,)
        
        # Numerical stability: avoid division by zero
        self.sigma = np.where(self.sigma < 1e-8, 1e-8, self.sigma)
        
        self._is_fitted = True
        return self
    
    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: STANDARDIZE — apply z-score normalization
    # ──────────────────────────────────────────────────────────────────────
    
    def standardize(self, session: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to one session.
        
        Formula (identical to KDPrint paper Section 3.3):
            x'[t, i] = (x[t, i] - μ[i]) / σ[i]   for i in {0, 1, 2, 3}
            x'[t, 4] = x[t, 4] / 127.0            (ASCII, same as TypeFormer)
        
        CRITICAL: μ and σ come from ENROLMENT DATA, not from this session.
        This is what enables impostor detection:
            - Genuine user: x ≈ μ → x' ≈ 0 (within a few σ of normal)
            - Impostor:     x ≠ μ → x' far from 0 (anomalous in user's context)
        
        Args:
            session: Raw feature array, shape (L, 5) or (seq_len, 5)
        
        Returns:
            Standardized array, same shape as input.
            Cols 0–3: z-scores (mean≈0, std≈1 for genuine user)
            Col 4:    ASCII normalized to [0, 1] (same as TypeFormer)
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Must call fit() before standardize(). "
                "Did you forget to call fit() on enrolment sessions?"
            )
        
        result = session.copy().astype(np.float32)
        
        # Z-score normalization for timing features (cols 0–3)
        # Broadcasting: self.mu has shape (4,), session[:, :4] has shape (L, 4)
        result[:, :4] = (session[:, :4] - self.mu) / self.sigma
        
        # ASCII normalization (same as TypeFormer original, unchanged)
        result[:, 4] = session[:, 4] / 127.0
        
        return result
    
    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: BUFFER — weighted moving average smoothing
    # ──────────────────────────────────────────────────────────────────────
    
    def apply_buffer(self, sessions: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply buffer aggregation to smooth sessions.
        
        Formula (identical to KDPrint paper Section 3.4, Equation 1):
        
            b_t = 1/(2(B-1)) × [ Σ_{i=t-B+1}^{t-1} kd_i  +  (B-1)·kd_t ]
        
        Equivalently (what the code computes):
            window  = session[max(0, t-B+1) : t+1]   # up to B elements
            b[t]    = (sum(window) + (B-1) × session[t]) / (2(B-1))
        
        This gives kd_t (the most recent keystroke) extra weight:
            Weight of kd_t    = 1/(2(B-1)) + (B-1)/(2(B-1)) = B/(2(B-1))
            Weight of kd_{t-j} = 1/(2(B-1))  for j = 1..B-1
        
        For B=5:
            kd_t gets weight   5/8 = 0.625   (62.5%)
            Each older kd gets 1/8 = 0.125   (12.5% each, 4 of them = 50%)
            Total = 0.625 + 4×0.125 = 1.0 ✓
        
        TWO PURPOSES:
            1. Smoothing: reduces spike noise from accidental keystrokes
            2. Augmentation (for training): different orderings of B-sized
               windows from N keystrokes create C(N,B) virtual samples.
               NOTE: in our case we only use purpose 1 (smoothing), since
               we are not training the model.
        
        WARM-UP PERIOD: For t < B-1, the window is shorter than B.
            t=0: window=[kd_0], b[0] = kd_0  (no smoothing possible)
            t=1: window=[kd_0, kd_1], partial smoothing
            t≥B-1: full B-size window, formula works as designed
        
        Args:
            sessions: List of standardized arrays, each shape (seq_len, 5)
        
        Returns:
            List of buffered arrays, same shapes as input
        """
        if self.B <= 1:
            return sessions  # no buffering
        
        buffered = []
        for session in sessions:
            N, n_feats = session.shape
            b = np.zeros_like(session, dtype=np.float32)
            denom = 2 * (self.B - 1)
            
            for t in range(N):
                start  = max(0, t - self.B + 1)
                window = session[start : t + 1]   # shape (min(t+1, B), n_feats)
                w_len  = len(window)
                
                if w_len == 1:
                    # Not enough history: just copy the value
                    b[t] = session[t]
                else:
                    # Apply weighted formula
                    # window.sum(axis=0) includes kd_t once from the sum
                    # (B-1)*session[t] adds extra weight to kd_t
                    b[t] = (window.sum(axis=0) + (self.B - 1) * session[t]) / denom
            
            buffered.append(b)
        
        return buffered
    
    # ──────────────────────────────────────────────────────────────────────
    # CONVENIENCE METHODS
    # ──────────────────────────────────────────────────────────────────────
    
    def fit_transform(self, enrolment_sessions: List[np.ndarray],
                       use_buffer: bool = True) -> List[np.ndarray]:
        """
        Fit on enrolment sessions and transform them.
        Called during ENROLLMENT.
        
        Equivalent to: fit(sessions) then [transform(s) for s in sessions]
        
        Args:
            enrolment_sessions: Raw enrolment feature arrays
            use_buffer:         Whether to apply buffer smoothing
        
        Returns:
            Processed enrolment sessions ready for TypeFormer
        """
        self.fit(enrolment_sessions)
        
        # First standardize all sessions
        standardized = [self.standardize(s) for s in enrolment_sessions]
        
        # Optionally apply buffer smoothing
        if use_buffer:
            return self.apply_buffer(standardized)
        return standardized
    
    def transform(self, session: np.ndarray, 
                   use_buffer: bool = False) -> np.ndarray:
        """
        Transform a single verification session using stored statistics.
        Called during VERIFICATION.
        
        IMPORTANT: Uses μ and σ from enrollment — does NOT recompute.
        This is what enables genuine vs impostor discrimination.
        
        Args:
            session:    Raw test session, shape (L, 5)
            use_buffer: Apply buffer to single session (smoothing only,
                        no augmentation benefit for single session)
        
        Returns:
            Processed session, shape (seq_len, 5)
        """
        standardized = self.standardize(session)
        
        if use_buffer:
            return self.apply_buffer([standardized])[0]
        return standardized
    
    # ──────────────────────────────────────────────────────────────────────
    # TEMPLATE PERSISTENCE
    # ──────────────────────────────────────────────────────────────────────
    
    def get_template_stats(self) -> Dict:
        """
        Return preprocessing statistics for saving in user template.
        
        These MUST be saved alongside the embedding template so that
        verification sessions can be preprocessed consistently.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted yet.")
        
        return {
            "mu":          self.mu.tolist(),
            "sigma":       self.sigma.tolist(),
            "buffer_size": self.B,
            "seq_len":     self.seq_len
        }
    
    @classmethod
    def from_template_stats(cls, stats: Dict) -> 'KDPrintPreprocessor':
        """
        Reconstruct preprocessor from saved template statistics.
        Used during verification to restore the enrolled user's preprocessing.
        
        Usage:
            # During enrollment:
            prep = KDPrintPreprocessor()
            prep.fit(enrolment_sessions)
            template['preprocessing'] = prep.get_template_stats()
            
            # During verification:
            prep = KDPrintPreprocessor.from_template_stats(template['preprocessing'])
            processed = prep.transform(test_session)
        """
        instance = cls(
            buffer_size = stats['buffer_size'],
            seq_len     = stats['seq_len']
        )
        instance.mu       = np.array(stats['mu'])
        instance.sigma    = np.array(stats['sigma'])
        instance._is_fitted = True
        return instance
```

### 5.3 Unit Tests for Preprocessing

**Run these before any experiment. If any test fails, do not proceed.**

```python
# tests/test_preprocessing.py

import numpy as np
import pytest
from src.preprocessing.kdprint_preprocess import KDPrintPreprocessor
from src.preprocessing.typeformer_preprocess import typeformer_preprocess

def make_fake_session(hold_mu=100, hold_sd=15, inter_mu=160, inter_sd=30,
                       seq_len=50, seed=0):
    """Create a realistic fake keystroke session."""
    rng = np.random.default_rng(seed)
    hold    = np.clip(rng.normal(hold_mu, hold_sd, seq_len), 20, 500)
    inter   = np.clip(rng.normal(inter_mu, inter_sd, seq_len), 5, 800)
    press   = hold + inter + rng.normal(0, 8, seq_len)
    release = press + rng.normal(0, 5, seq_len)
    ascii_  = rng.integers(65, 123, seq_len).astype(float) / 127.0
    return np.column_stack([hold, inter, press, release, ascii_])


class TestKDPrintPreprocessor:
    
    def test_fit_produces_correct_shapes(self):
        """μ and σ should have shape (4,) — one per timing feature."""
        prep = KDPrintPreprocessor()
        sessions = [make_fake_session(seed=i) for i in range(5)]
        prep.fit(sessions)
        
        assert prep.mu.shape    == (4,), f"mu shape: {prep.mu.shape}"
        assert prep.sigma.shape == (4,), f"sigma shape: {prep.sigma.shape}"
    
    def test_standardize_centers_genuine_data(self):
        """After fitting on user A's data, transforming user A's sessions
        should produce values with mean≈0 and std≈1 per feature."""
        prep = KDPrintPreprocessor()
        enrol = [make_fake_session(hold_mu=100, seed=i) for i in range(5)]
        prep.fit(enrol)
        
        test_session = make_fake_session(hold_mu=100, seed=99)
        result = prep.standardize(test_session)
        
        # Timing features should be approximately standardized
        for i in range(4):
            mean = result[:, i].mean()
            std  = result[:, i].std()
            assert abs(mean) < 0.5, f"Feature {i} mean not near 0: {mean:.3f}"
            assert 0.5 < std < 2.0, f"Feature {i} std unexpected: {std:.3f}"
    
    def test_standardize_detects_impostor(self):
        """An impostor (different typing speed) should produce
        values far from 0 after standardization with user A's stats."""
        prep = KDPrintPreprocessor()
        enrol_A = [make_fake_session(hold_mu=70, inter_mu=90, seed=i) for i in range(5)]
        prep.fit(enrol_A)
        
        # Impostor is a slow typer (very different from user A)
        impostor_session = make_fake_session(hold_mu=145, inter_mu=280, seed=99)
        result = prep.standardize(impostor_session)
        
        # Impostor's hold_lat should be many sigmas above user A's mean
        impostor_hold_mean = result[:, 0].mean()
        assert impostor_hold_mean > 3.0, \
            f"Impostor should be >3σ away, got {impostor_hold_mean:.2f}σ"
    
    def test_buffer_preserves_mean(self):
        """Buffer aggregation should not significantly change the mean."""
        prep = KDPrintPreprocessor(buffer_size=5)
        sessions = [make_fake_session(seed=i) for i in range(3)]
        processed = prep.fit_transform(sessions, use_buffer=False)
        buffered  = prep.apply_buffer(processed)
        
        for i, (proc, buff) in enumerate(zip(processed, buffered)):
            for feat_idx in range(4):
                mean_diff = abs(proc[:, feat_idx].mean() - buff[:, feat_idx].mean())
                assert mean_diff < 0.5, \
                    f"Session {i}, feature {feat_idx}: mean changed by {mean_diff:.4f}"
    
    def test_buffer_reduces_variance(self):
        """Buffer should reduce std deviation (smoothing effect)."""
        prep = KDPrintPreprocessor(buffer_size=5)
        sessions = [make_fake_session(seed=i) for i in range(3)]
        processed = prep.fit_transform(sessions, use_buffer=False)
        buffered  = prep.apply_buffer(processed)
        
        for i, (proc, buff) in enumerate(zip(processed, buffered)):
            for feat_idx in range(4):
                std_before = proc[:, feat_idx].std()
                std_after  = buff[:, feat_idx].std()
                assert std_after <= std_before * 1.05, \
                    f"Buffer did not reduce variance for session {i}, feature {feat_idx}"
    
    def test_transform_uses_enrolment_stats(self):
        """transform() must use μ,σ from enrolment, not from test session."""
        prep = KDPrintPreprocessor()
        enrol = [make_fake_session(hold_mu=70, seed=i) for i in range(3)]
        prep.fit(enrol)
        
        mu_before = prep.mu.copy()
        
        # Transform a test session with very different timing
        test = make_fake_session(hold_mu=200, seed=99)
        _ = prep.transform(test)
        
        # μ and σ must not change after transform()
        np.testing.assert_array_equal(prep.mu, mu_before, 
            err_msg="transform() must not modify mu")
    
    def test_template_save_load(self):
        """Saving and loading template stats must produce identical results."""
        prep1 = KDPrintPreprocessor(buffer_size=7)
        enrol = [make_fake_session(seed=i) for i in range(5)]
        prep1.fit(enrol)
        
        # Save stats
        stats = prep1.get_template_stats()
        
        # Reconstruct from stats
        prep2 = KDPrintPreprocessor.from_template_stats(stats)
        
        # Both should produce identical output
        test_session = make_fake_session(seed=99)
        out1 = prep1.transform(test_session)
        out2 = prep2.transform(test_session)
        np.testing.assert_array_almost_equal(out1, out2, decimal=6)
    
    def test_ascii_normalization(self):
        """ASCII column (index 4) should be in [0, 1] after preprocessing."""
        prep = KDPrintPreprocessor()
        sessions = [make_fake_session(seed=i) for i in range(3)]
        processed = prep.fit_transform(sessions)
        
        for s in processed:
            assert np.all(s[:, 4] >= 0) and np.all(s[:, 4] <= 1), \
                "ASCII values out of [0, 1] range"
    
    def test_no_nan_or_inf(self):
        """Processed output must not contain NaN or Inf."""
        prep = KDPrintPreprocessor()
        sessions = [make_fake_session(seed=i) for i in range(5)]
        processed = prep.fit_transform(sessions)
        
        for i, s in enumerate(processed):
            assert not np.any(np.isnan(s)), f"NaN in processed session {i}"
            assert not np.any(np.isinf(s)), f"Inf in processed session {i}"


# Run: pytest tests/test_preprocessing.py -v
```

---

## 6. TypeFormer Integration

### 6.1 Understanding TypeFormer's Model Interface

Before wrapping TypeFormer, read `TypeFormer/model/Model.py` carefully. Key things to understand:

```python
# What you need from Model.py (read the actual file):
# 1. The forward() method signature
# 2. How the model expects input tensor shape
# 3. Where the 64-dim embedding is produced
# 4. Whether L2 normalization is applied to output

# Typical TypeFormer forward pass (verify against actual code):
# Input:  tensor of shape (batch_size, seq_len, n_features)
#         = (batch_size, 50, 5)
# Output: embedding tensor of shape (batch_size, 64)
#         (L2-normalized, so ‖z‖₂ = 1 for each embedding)
```

### 6.2 TypeFormer Wrapper

```python
# src/model/typeformer_wrapper.py

import torch
import numpy as np
from pathlib import Path
from typing import List, Union
import sys

# Add TypeFormer to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models' / 'TypeFormer'))

class TypeFormerWrapper:
    """
    Wrapper around pretrained TypeFormer for embedding extraction.
    
    Handles:
    - Loading pretrained weights
    - Converting numpy arrays to tensors
    - Batched inference for efficiency
    - Consistent embedding extraction across E1-E4 experiments
    
    Usage:
        model = TypeFormerWrapper('path/to/TypeFormer_pretrained.pt')
        
        # Extract embeddings for a list of sessions
        sessions = [array1, array2, ...]  # each shape (50, 5)
        embeddings = model.extract_embeddings(sessions)  # shape (N, 64)
    """
    
    def __init__(self, weights_path: str, device: str = 'auto',
                  batch_size: int = 64):
        """
        Args:
            weights_path: Path to TypeFormer_pretrained.pt
            device:       'cuda', 'cpu', or 'auto' (auto-detect)
            batch_size:   Number of sessions to process at once.
                         Larger = faster but more GPU memory.
                         64 is safe for most GPUs with 4GB+ VRAM.
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.batch_size = batch_size
        
        # Load model
        self.model = self._load_model(weights_path)
        self.model.eval()  # IMPORTANT: set to eval mode (disables dropout, etc.)
        
        print(f"TypeFormer loaded on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self, weights_path: str):
        """Load the TypeFormer model architecture and weights."""
        from model.Model import TypeFormer  # adjust import based on actual file
        
        # Initialize model — check Model.py for correct arguments
        model = TypeFormer()  # you may need to pass config arguments
        
        # Load pretrained weights
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model.to(self.device)
    
    def extract_embeddings(self, sessions: List[np.ndarray]) -> np.ndarray:
        """
        Extract 64-dimensional embeddings for a list of sessions.
        
        Args:
            sessions: List of preprocessed feature arrays, each shape (50, 5)
                      All arrays must have the same shape.
        
        Returns:
            numpy array of shape (len(sessions), 64)
            Each row is one L2-normalized embedding.
        
        Note on L2 normalization:
            TypeFormer outputs L2-normalized embeddings (‖z‖₂ = 1).
            This means Euclidean distance between embeddings is in [0, 2].
            (d=0: identical, d=2: maximally different, d=√2≈1.41: orthogonal)
            Keep this in mind when interpreting threshold values.
        """
        if len(sessions) == 0:
            return np.empty((0, 64))
        
        all_embeddings = []
        
        # Process in batches for efficiency
        for start in range(0, len(sessions), self.batch_size):
            batch = sessions[start : start + self.batch_size]
            
            # Convert to tensor: (batch_size, seq_len, n_features)
            batch_tensor = torch.FloatTensor(
                np.stack(batch, axis=0)
            ).to(self.device)
            
            with torch.no_grad():  # IMPORTANT: disable gradient computation
                embeddings = self.model(batch_tensor)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)  # shape (N, 64)
    
    def extract_single(self, session: np.ndarray) -> np.ndarray:
        """Extract embedding for a single session. Shape (64,)."""
        return self.extract_embeddings([session])[0]
```

### 6.3 Verifying TypeFormer Loads Correctly

```python
# tests/test_typeformer.py

import numpy as np
from src.model.typeformer_wrapper import TypeFormerWrapper

def test_typeformer_basics(weights_path):
    model = TypeFormerWrapper(weights_path)
    
    # Test 1: output shape
    fake_session = np.random.randn(50, 5).astype(np.float32)
    embedding = model.extract_single(fake_session)
    assert embedding.shape == (64,), f"Expected (64,), got {embedding.shape}"
    
    # Test 2: L2 normalization (TypeFormer outputs unit-norm embeddings)
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 0.01, f"Expected unit norm, got {norm:.4f}"
    
    # Test 3: same input → same output (deterministic in eval mode)
    embedding2 = model.extract_single(fake_session)
    np.testing.assert_array_almost_equal(embedding, embedding2, decimal=5)
    
    # Test 4: different inputs → different outputs
    different_session = np.random.randn(50, 5).astype(np.float32)
    different_embedding = model.extract_single(different_session)
    assert not np.allclose(embedding, different_embedding), \
        "Different inputs should produce different embeddings"
    
    print("✓ TypeFormer basic tests passed")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"  Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
```

---

## 7. Adaptive Threshold Module

### 7.1 Core Implementation

```python
# src/evaluation/adaptive_threshold.py

import numpy as np
from typing import Dict, List, Tuple, Optional

class AdaptiveThresholdEstimator:
    """
    Per-user adaptive authentication threshold.
    
    The core insight: different users have different consistency in their
    typing. A single global threshold is a poor fit because:
      - Consistent users (low intra-class variance) → threshold too loose
      - Inconsistent users (high intra-class variance) → threshold too tight
    
    This module computes T_user = μ_user + k·σ_user where:
      - μ_user = mean distance from enrolment embeddings to their centroid
      - σ_user = std of those distances
      - k      = single hyperparameter, same for all users, optimized on val set
    
    This means:
      - Consistent users get smaller T_user (tighter security)
      - Inconsistent users get larger T_user (more tolerant, fewer false rejects)
    
    The value of k controls the overall FAR/FRR trade-off:
      - Small k (e.g., 0.5) → stricter for everyone → lower FAR, higher FRR
      - Large k (e.g., 3.0) → more lenient → higher FAR, lower FRR
      - k ≈ 2.0 is a reasonable starting point (corresponds to ~95% confidence
        interval under normality assumption)
    """
    
    def __init__(self, k: float = 2.0):
        """
        Args:
            k: Threshold multiplier. Optimized on validation set.
               Start with k=2.0, then run optimize_k() on validation data.
        """
        self.k = k
    
    def estimate_user_threshold(self, 
                                  enrolment_embeddings: np.ndarray
                                  ) -> Dict:
        """
        Compute per-user threshold from enrolment embeddings.
        
        This is called ONCE per user during enrollment.
        
        Args:
            enrolment_embeddings: shape (E, 64) where E = number of enrolment sessions
        
        Returns:
            Template dict containing everything needed for verification:
            {
                'z_mean':     centroid embedding, shape (64,)
                'mu_user':    mean distance to centroid (scalar)
                'sigma_user': std of distances (scalar)
                'T_user':     decision threshold (scalar)
                'E':          number of enrolment sessions used
            }
        
        Edge case — E=1:
            With only 1 enrolment session, sigma_user = 0.
            T_user = mu_user + k×0 = mu_user.
            This is valid but the threshold will be poorly calibrated.
            Consider using a minimum sigma from the population:
                sigma_user = max(sigma_user, global_sigma_floor)
            where global_sigma_floor is the median σ_user across all users.
        """
        E = enrolment_embeddings.shape[0]
        assert E >= 1, "Need at least 1 enrolment session"
        
        # Compute centroid (mean embedding)
        z_mean = enrolment_embeddings.mean(axis=0)   # shape (64,)
        
        # Compute distance from each enrolment embedding to centroid
        dists = np.linalg.norm(
            enrolment_embeddings - z_mean, axis=1
        )  # shape (E,)
        
        mu_user    = float(dists.mean())
        sigma_user = float(dists.std()) if E > 1 else 0.0
        
        # Adaptive threshold
        T_user = mu_user + self.k * sigma_user
        
        return {
            'z_mean':     z_mean,
            'mu_user':    mu_user,
            'sigma_user': sigma_user,
            'T_user':     T_user,
            'E':          E
        }
    
    def predict(self, z_test: np.ndarray, 
                 template: Dict) -> Tuple[str, float]:
        """
        Make authentication decision for a test embedding.
        
        Args:
            z_test:   Test embedding, shape (64,)
            template: Dict from estimate_user_threshold()
        
        Returns:
            (decision, distance) where:
                decision = 'accept' or 'reject'
                distance = Euclidean distance from z_test to z_mean
        """
        d = float(np.linalg.norm(z_test - template['z_mean']))
        decision = 'accept' if d <= template['T_user'] else 'reject'
        return decision, d
    
    def optimize_k(self, 
                    val_data: Dict[str, Dict],
                    k_range: np.ndarray = None) -> Tuple[float, float]:
        """
        Find optimal k on validation set using grid search.
        
        This is a critical step — run this BEFORE evaluating on the test set.
        The optimal k minimizes average EER across all validation users.
        
        Args:
            val_data: {user_id: {'enrol': array(E,64), 'genuine': array(5,64),
                                  'impostor': array(999,64)}}
            k_range:  Array of k values to try.
                      Default: np.arange(0.5, 4.0, 0.25)
        
        Returns:
            (best_k, best_eer) — the k value and corresponding average EER
        
        Usage:
            val_data = build_val_data(model, preprocessor, val_users)
            estimator = AdaptiveThresholdEstimator()
            best_k, best_eer = estimator.optimize_k(val_data)
            print(f"Optimal k: {best_k:.2f}, Val EER: {best_eer:.4f}")
        """
        if k_range is None:
            k_range = np.arange(0.5, 4.25, 0.25)
        
        best_k   = self.k
        best_eer = 1.0
        
        results = []
        for k_cand in k_range:
            self.k = k_cand
            avg_eer = self._compute_avg_eer(val_data)
            results.append((k_cand, avg_eer))
            
            if avg_eer < best_eer:
                best_eer = avg_eer
                best_k   = k_cand
        
        self.k = best_k
        
        # Print search results for debugging
        print(f"k optimization results (best k={best_k:.2f}, EER={best_eer:.4f}):")
        for k_val, eer in results:
            marker = " ←" if k_val == best_k else ""
            print(f"  k={k_val:.2f}: EER={eer:.4f}{marker}")
        
        return best_k, best_eer
    
    def _compute_avg_eer(self, val_data: Dict) -> float:
        """Compute average per-user EER across all validation users."""
        from src.evaluation.metrics import compute_eer
        
        user_eers = []
        for user_id, data in val_data.items():
            template = self.estimate_user_threshold(data['enrol'])
            
            genuine_scores  = [
                np.linalg.norm(z - template['z_mean']) 
                for z in data['genuine']
            ]
            impostor_scores = [
                np.linalg.norm(z - template['z_mean']) 
                for z in data['impostor']
            ]
            
            user_eer = compute_eer(genuine_scores, impostor_scores)
            user_eers.append(user_eer)
        
        return float(np.mean(user_eers))
```

### 7.2 Global Threshold (for E1 and E2)

```python
# src/evaluation/global_threshold.py

import numpy as np
from typing import List, Dict, Tuple

class GlobalThresholdEstimator:
    """
    Global EER-optimal threshold, same for all users.
    This is the TypeFormer original approach.
    Used for experiments E1 (baseline) and E2 (preprocessing only).
    
    Finding the global threshold:
    1. Collect ALL genuine scores and ALL impostor scores across all val users
    2. Find the threshold T where FAR ≈ FRR
    3. Apply that same T to all evaluation users
    """
    
    def __init__(self):
        self.threshold: float = None
    
    def fit(self, all_genuine_scores: List[float], 
             all_impostor_scores: List[float]) -> float:
        """
        Find global EER threshold from validation set scores.
        
        Args:
            all_genuine_scores:  All genuine scores from validation users
            all_impostor_scores: All impostor scores from validation users
        
        Returns:
            The optimal threshold value
        """
        from src.evaluation.metrics import find_eer_threshold
        
        self.threshold = find_eer_threshold(
            all_genuine_scores, all_impostor_scores
        )
        return self.threshold
    
    def predict(self, distance: float) -> Tuple[str, float]:
        """
        Make decision for a single distance value.
        Same threshold for ALL users.
        """
        if self.threshold is None:
            raise RuntimeError("Must call fit() first")
        decision = 'accept' if distance <= self.threshold else 'reject'
        return decision, distance
```

---

## 8. Evaluation Metrics

```python
# src/evaluation/metrics.py

import numpy as np
from sklearn.metrics import roc_curve
from typing import List, Tuple, Dict

def compute_eer(genuine_scores: List[float], 
                 impostor_scores: List[float]) -> float:
    """
    Compute Equal Error Rate (EER) for one user.
    
    EER is the point where FAR = FRR. Lower EER = better system.
    TypeFormer baseline achieves EER = 3.25% at E=5, L=50.
    
    IMPORTANT — Score interpretation:
        We use Euclidean DISTANCE as score (lower = more similar).
        A genuine user should have LOW distance to their template.
        An impostor should have HIGH distance.
        
        For roc_curve, we label:
            genuine  = 0 (low distance → likely same user)
            impostor = 1 (high distance → likely different user)
        
        This means FAR (False Accept Rate) = FPR (False Positive Rate)
        when the "positive" class is "impostor".
    
    Args:
        genuine_scores:  List of distances for genuine test sessions (should be low)
        impostor_scores: List of distances for impostor test sessions (should be high)
    
    Returns:
        EER as a fraction in [0, 1]. Multiply by 100 for percentage.
        Example: 0.0325 = 3.25% EER (TypeFormer baseline)
    """
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.zeros(len(genuine_scores)),   # genuine = 0
        np.ones(len(impostor_scores))    # impostor = 1
    ])
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    
    # Find the point where FPR (FAR) ≈ FNR (FRR)
    # Use linear interpolation for smoother estimate
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    
    return eer


def find_eer_threshold(genuine_scores: List[float],
                        impostor_scores: List[float]) -> float:
    """
    Find the distance threshold at which FAR ≈ FRR.
    Used for fitting the global threshold on validation data.
    """
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.zeros(len(genuine_scores)),
        np.ones(len(impostor_scores))
    ])
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    
    return float(thresholds[eer_idx])


def compute_far_at_frr(genuine_scores: List[float],
                        impostor_scores: List[float],
                        target_frr: float = 0.01) -> Tuple[float, float]:
    """
    Compute FAR at a specific FRR operating point.
    Commonly reported as FAR@FRR=1% or FAR@FRR=10%.
    
    Args:
        target_frr: FRR operating point (e.g., 0.01 for 1%)
    
    Returns:
        (far, actual_frr) at the threshold closest to target_frr
    """
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.zeros(len(genuine_scores)),
        np.ones(len(impostor_scores))
    ])
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    
    idx = np.argmin(np.abs(fnr - target_frr))
    return float(fpr[idx]), float(fnr[idx])


def compute_user_metrics(genuine_scores: List[float],
                          impostor_scores: List[float]) -> Dict:
    """
    Compute all metrics for one user.
    
    Returns:
        Dict with keys: eer, far_at_1frr, frr_at_1frr, far_at_10frr, frr_at_10frr
    """
    eer = compute_eer(genuine_scores, impostor_scores)
    far_1, frr_1   = compute_far_at_frr(genuine_scores, impostor_scores, 0.01)
    far_10, frr_10 = compute_far_at_frr(genuine_scores, impostor_scores, 0.10)
    
    return {
        'eer':          eer,
        'far_at_1frr':  far_1,
        'frr_at_1frr':  frr_1,
        'far_at_10frr': far_10,
        'frr_at_10frr': frr_10,
        'n_genuine':    len(genuine_scores),
        'n_impostor':   len(impostor_scores)
    }


def aggregate_results(user_metrics: List[Dict]) -> Dict:
    """
    Aggregate per-user metrics into global statistics.
    
    TypeFormer reports per-user average EER (average of per-user EERs),
    which is different from global EER (computed on pooled score distributions).
    We report both.
    
    Args:
        user_metrics: List of dicts from compute_user_metrics()
    
    Returns:
        Dict with mean, std, and percentile statistics.
    """
    eers = [m['eer'] for m in user_metrics]
    
    return {
        # Primary metric (matches TypeFormer paper reporting)
        'avg_eer':       float(np.mean(eers)),
        'std_eer':       float(np.std(eers)),
        
        # Distribution of per-user EERs
        'median_eer':    float(np.median(eers)),
        'p25_eer':       float(np.percentile(eers, 25)),
        'p75_eer':       float(np.percentile(eers, 75)),
        'worst_eer':     float(np.max(eers)),
        'best_eer':      float(np.min(eers)),
        
        # Other metrics (averaged across users)
        'avg_far_1':     float(np.mean([m['far_at_1frr'] for m in user_metrics])),
        'avg_frr_1':     float(np.mean([m['frr_at_1frr'] for m in user_metrics])),
        
        'n_users':       len(user_metrics)
    }
```

---

## 9. Experiment Runner (E1–E4)

### 9.1 Generic Experiment Function

```python
# src/experiments/run_experiment.py

import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from src.preprocessing.typeformer_preprocess import typeformer_preprocess
from src.preprocessing.kdprint_preprocess import KDPrintPreprocessor
from src.model.typeformer_wrapper import TypeFormerWrapper
from src.evaluation.metrics import compute_user_metrics, aggregate_results
from src.evaluation.adaptive_threshold import AdaptiveThresholdEstimator
from src.evaluation.global_threshold import GlobalThresholdEstimator


def run_single_experiment(
    config_name:        str,
    model:              TypeFormerWrapper,
    eval_users:         List[Dict],    # list of user data dicts
    val_users:          List[Dict],    # for threshold fitting
    E:                  int,           # number of enrolment sessions
    L:                  int = 50,      # sequence length
    use_kdprint:        bool = False,  # whether to apply KDPrint preprocessing
    use_adaptive:       bool = False,  # whether to use adaptive threshold
    buffer_size:        int = 5,
    output_dir:         str = 'results/',
    verbose:            bool = True
) -> Dict:
    """
    Run one experiment configuration (E1, E2, E3, or E4).
    
    Args:
        config_name:   Label for this experiment (e.g., 'E1_baseline')
        model:         Pretrained TypeFormer model
        eval_users:    1000 evaluation users with their raw sessions
        val_users:     400 validation users for threshold optimization
        E:             Number of enrolment sessions (1, 2, 5, 7, or 10)
        L:             Sequence length (50, fixed for all experiments)
        use_kdprint:   True for E2 and E4; False for E1 and E3
        use_adaptive:  True for E3 and E4; False for E1 and E2
        buffer_size:   B parameter for KDPrint buffer (default 5)
        output_dir:    Directory to save results
        verbose:       Print progress
    
    Returns:
        Dict with aggregated metrics for this configuration
    
    Protocol (matches TypeFormer paper Section 5.3):
        For each evaluation user:
        - E enrolment sessions (sessions 0..E-1)
        - 5 genuine verification sessions (sessions E..E+4)
        - 999 impostor sessions (one session from each other eval user)
        
        Genuine scores:  5 distances (enrolment centroid vs each genuine session)
        Impostor scores: 999 distances (enrolment centroid vs each impostor session)
    """
    start_time = time.time()
    output_path = Path(output_dir) / f"{config_name}_E{E}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {config_name} | E={E} | L={L}")
        print(f"KDPrint: {use_kdprint} | Adaptive: {use_adaptive}")
        print(f"{'='*60}")
    
    # ──────────────────────────────────────────────────────────────
    # STEP 1: Fit threshold on validation set
    # ──────────────────────────────────────────────────────────────
    if verbose:
        print("\n[1/3] Fitting threshold on validation set...")
    
    if use_adaptive:
        threshold_estimator = AdaptiveThresholdEstimator(k=2.0)
        val_data = _extract_val_embeddings(
            val_users, model, E, use_kdprint, buffer_size, L
        )
        best_k, val_eer = threshold_estimator.optimize_k(val_data)
        if verbose:
            print(f"  Optimal k={best_k:.2f}, Val EER={val_eer:.4f}")
    else:
        # Global threshold: collect all val scores, find EER threshold
        threshold_estimator = GlobalThresholdEstimator()
        all_gen, all_imp = _collect_val_scores(
            val_users, model, E, use_kdprint, buffer_size, L
        )
        global_T = threshold_estimator.fit(all_gen, all_imp)
        if verbose:
            print(f"  Global threshold={global_T:.4f}")
    
    # ──────────────────────────────────────────────────────────────
    # STEP 2: Evaluate on test set (1000 users)
    # ──────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n[2/3] Evaluating on {len(eval_users)} test users...")
    
    all_user_metrics = []
    
    for user in tqdm(eval_users, desc="Users", disable=not verbose):
        user_id = user['id']
        raw_sessions = user['sessions']  # shape: list of (L, 5) arrays
        
        # Split into enrolment and verification
        enrol_raw  = raw_sessions[:E]
        verify_raw = raw_sessions[E : E + 5]       # 5 genuine test sessions
        
        # Get one session from all OTHER evaluation users as impostors
        other_users = [u for u in eval_users if u['id'] != user_id]
        impostor_raw = [u['sessions'][0] for u in other_users]  # 999 sessions
        
        # ── Preprocessing ──────────────────────────────────────────
        if use_kdprint:
            prep = KDPrintPreprocessor(buffer_size=buffer_size, seq_len=L)
            enrol_proc   = prep.fit_transform(enrol_raw, use_buffer=True)
            verify_proc  = [prep.transform(s, use_buffer=False) for s in verify_raw]
            impostor_proc = [prep.transform(s, use_buffer=False) for s in impostor_raw]
        else:
            enrol_proc   = typeformer_preprocess(enrol_raw, seq_len=L)
            verify_proc  = typeformer_preprocess(verify_raw, seq_len=L)
            impostor_proc = typeformer_preprocess(impostor_raw, seq_len=L)
        
        # ── Feature Extraction ─────────────────────────────────────
        z_enrol    = model.extract_embeddings(enrol_proc)    # (E, 64)
        z_verify   = model.extract_embeddings(verify_proc)   # (5, 64)
        z_impostor = model.extract_embeddings(impostor_proc) # (999, 64)
        
        # ── Compute scores (distances to enrolment centroid) ────────
        z_mean = z_enrol.mean(axis=0)  # (64,)
        
        genuine_scores  = [np.linalg.norm(z - z_mean) for z in z_verify]
        impostor_scores = [np.linalg.norm(z - z_mean) for z in z_impostor]
        
        # ── Compute per-user EER ────────────────────────────────────
        # EER is threshold-independent — computed from score distributions
        user_metrics = compute_user_metrics(genuine_scores, impostor_scores)
        user_metrics['user_id'] = user_id
        
        # Additional: threshold-specific FAR/FRR
        if use_adaptive:
            template = threshold_estimator.estimate_user_threshold(z_enrol)
            user_metrics['T_user']     = template['T_user']
            user_metrics['mu_user']    = template['mu_user']
            user_metrics['sigma_user'] = template['sigma_user']
        
        all_user_metrics.append(user_metrics)
    
    # ──────────────────────────────────────────────────────────────
    # STEP 3: Aggregate and save results
    # ──────────────────────────────────────────────────────────────
    aggregated = aggregate_results(all_user_metrics)
    aggregated['config']  = config_name
    aggregated['E']       = E
    aggregated['L']       = L
    aggregated['runtime_seconds'] = time.time() - start_time
    
    # Save detailed per-user results
    with open(output_path / 'per_user_metrics.json', 'w') as f:
        json.dump(all_user_metrics, f, indent=2)
    
    # Save aggregated results
    with open(output_path / 'aggregated.json', 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    if verbose:
        print(f"\n[3/3] Results for {config_name} E={E}:")
        print(f"  Average EER: {aggregated['avg_eer']*100:.4f}%")
        print(f"  Std EER:     {aggregated['std_eer']*100:.4f}%")
        print(f"  Runtime:     {aggregated['runtime_seconds']:.1f}s")
        print(f"  Saved to:    {output_path}")
    
    return aggregated


def _extract_val_embeddings(val_users, model, E, use_kdprint, 
                              buffer_size, L) -> Dict:
    """Extract embeddings for all validation users (for k optimization)."""
    from src.preprocessing.kdprint_preprocess import KDPrintPreprocessor
    from src.preprocessing.typeformer_preprocess import typeformer_preprocess
    
    val_data = {}
    for user in tqdm(val_users, desc="Val users"):
        raw = user['sessions']
        enrol_raw  = raw[:E]
        verify_raw = raw[E:E+5]
        others     = [u['sessions'][0] for u in val_users if u['id'] != user['id']]
        
        if use_kdprint:
            prep = KDPrintPreprocessor(buffer_size=buffer_size, seq_len=L)
            enrol_proc  = prep.fit_transform(enrol_raw)
            verify_proc = [prep.transform(s) for s in verify_raw]
            other_proc  = [prep.transform(s) for s in others]
        else:
            enrol_proc  = typeformer_preprocess(enrol_raw, L)
            verify_proc = typeformer_preprocess(verify_raw, L)
            other_proc  = typeformer_preprocess(others, L)
        
        z_enrol  = model.extract_embeddings(enrol_proc)
        z_verify = model.extract_embeddings(verify_proc)
        z_others = model.extract_embeddings(other_proc)
        
        val_data[user['id']] = {
            'enrol':    z_enrol,
            'genuine':  z_verify,
            'impostor': z_others
        }
    
    return val_data
```

### 9.2 Running All Four Experiments

```python
# src/experiments/run_all.py
"""
Main script to run all experiments E1–E4 for all E values.
Expected runtime: 4–8 hours per configuration on GPU.
"""

from src.model.typeformer_wrapper import TypeFormerWrapper
from src.experiments.run_experiment import run_single_experiment
import numpy as np
import json

# ── Configuration ──────────────────────────────────────────────────────────
WEIGHTS_PATH = 'models/TypeFormer/pretrained/TypeFormer_pretrained.pt'
DATA_DIR     = 'data/processed/'
RESULTS_DIR  = 'results/'
E_VALUES     = [1, 2, 5, 7, 10]
L            = 50
BUFFER_SIZE  = 5

# ── Load model ─────────────────────────────────────────────────────────────
model = TypeFormerWrapper(WEIGHTS_PATH, device='auto', batch_size=64)

# ── Load data ───────────────────────────────────────────────────────────────
# Implement load_users() to return list of {'id': str, 'sessions': list}
eval_users = load_users(DATA_DIR + 'features_eval/')   # 1000 users
val_users  = load_users(DATA_DIR + 'features_val/')    # 400 users

# ── Run experiments ─────────────────────────────────────────────────────────
all_results = {}

for E in E_VALUES:
    print(f"\n{'#'*60}")
    print(f"# Running all configs for E={E}")
    print(f"{'#'*60}")
    
    # E1: Baseline (TypeFormer original)
    r1 = run_single_experiment(
        'E1_baseline', model, eval_users, val_users,
        E=E, L=L, use_kdprint=False, use_adaptive=False
    )
    
    # E2: KDPrint preprocessing only
    r2 = run_single_experiment(
        'E2_preprocessing', model, eval_users, val_users,
        E=E, L=L, use_kdprint=True, use_adaptive=False,
        buffer_size=BUFFER_SIZE
    )
    
    # E3: Adaptive threshold only
    r3 = run_single_experiment(
        'E3_adaptive', model, eval_users, val_users,
        E=E, L=L, use_kdprint=False, use_adaptive=True
    )
    
    # E4: Full system (both contributions)
    r4 = run_single_experiment(
        'E4_full', model, eval_users, val_users,
        E=E, L=L, use_kdprint=True, use_adaptive=True,
        buffer_size=BUFFER_SIZE
    )
    
    all_results[E] = {'E1': r1, 'E2': r2, 'E3': r3, 'E4': r4}

# ── Print summary table ──────────────────────────────────────────────────────
print("\n\nSUMMARY TABLE (Average EER %)")
print(f"{'E':<5} {'E1 (Baseline)':>15} {'E2 (Preproc)':>15} "
      f"{'E3 (Adaptive)':>15} {'E4 (Full)':>15}")
print("-" * 65)
for E, configs in all_results.items():
    row = f"{E:<5}"
    for config in ['E1', 'E2', 'E3', 'E4']:
        eer_pct = configs[config]['avg_eer'] * 100
        row += f"{eer_pct:>15.4f}"
    print(row)

# Save full results
with open(f'{RESULTS_DIR}/all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
```

---

## 10. Analysis & Visualization

### 10.1 t-SNE Embedding Visualization

```python
# analysis/visualize_embeddings.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne_embeddings(embeddings_per_user: dict,
                          n_users: int = 10,
                          title: str = "Embedding Space",
                          save_path: str = None):
    """
    Visualize embedding space using t-SNE (same as TypeFormer Fig. 5).
    
    Args:
        embeddings_per_user: {user_id: array(n_sessions, 64)}
        n_users:   Number of users to visualize (keep ≤ 15 for clarity)
        title:     Plot title
        save_path: If provided, save figure to this path
    """
    # Collect data
    all_embeddings = []
    all_labels     = []
    
    users = list(embeddings_per_user.keys())[:n_users]
    for i, user_id in enumerate(users):
        embs = embeddings_per_user[user_id]
        all_embeddings.append(embs)
        all_labels.extend([i] * len(embs))
    
    X = np.vstack(all_embeddings)
    y = np.array(all_labels)
    
    # t-SNE (TypeFormer uses perplexity=14)
    tsne = TSNE(n_components=2, perplexity=14, init='pca', 
                n_iter=1000, random_state=42)
    X_2d = tsne.fit_transform(X)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, n_users))
    
    for i, user_id in enumerate(users):
        mask = y == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                   c=[colors[i]], label=f'User {user_id}',
                   alpha=0.7, s=30)
    
    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_tsne_before_after(raw_embeddings: dict,
                                std_embeddings: dict,
                                n_users: int = 10,
                                save_path: str = None):
    """
    Side-by-side comparison: raw preprocessing vs KDPrint preprocessing.
    Shows whether standardization improves cluster separation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    for ax, emb_dict, title in zip(
        axes,
        [raw_embeddings, std_embeddings],
        ['TypeFormer Original (Raw Input)', 'With KDPrint Preprocessing']
    ):
        users = list(emb_dict.keys())[:n_users]
        all_embs = np.vstack([emb_dict[u] for u in users])
        labels   = np.concatenate([
            [i]*len(emb_dict[u]) for i, u in enumerate(users)
        ])
        
        tsne  = TSNE(n_components=2, perplexity=14, random_state=42)
        X_2d  = tsne.fit_transform(all_embs)
        
        colors = plt.cm.tab20(np.linspace(0, 1, n_users))
        for i, uid in enumerate(users):
            mask = labels == i
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[colors[i]], alpha=0.7, s=30)
        
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
    
    plt.suptitle("Embedding Space Comparison", fontsize=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### 10.2 Per-User Analysis

```python
# analysis/per_user_analysis.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def analyze_threshold_distribution(user_templates: list,
                                    global_threshold: float,
                                    save_path: str = None):
    """
    Show distribution of adaptive thresholds across users.
    Demonstrates WHY a global threshold is suboptimal.
    """
    T_users = [t['T_user'] for t in user_templates]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Distribution of T_user
    axes[0].hist(T_users, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    axes[0].axvline(global_threshold, color='red', linestyle='--', 
                     linewidth=2, label=f'Global T={global_threshold:.3f}')
    axes[0].axvline(np.mean(T_users), color='navy', linestyle='-',
                     linewidth=2, label=f'Mean T_user={np.mean(T_users):.3f}')
    axes[0].set_xlabel("Threshold Value")
    axes[0].set_ylabel("Number of Users")
    axes[0].set_title("Distribution of Per-User Thresholds")
    axes[0].legend()
    
    # Right: sigma_user vs T_user (consistency vs threshold)
    sigmas = [t['sigma_user'] for t in user_templates]
    mus    = [t['mu_user']    for t in user_templates]
    axes[1].scatter(sigmas, T_users, alpha=0.4, s=10, color='steelblue')
    axes[1].set_xlabel("σ_user (typing consistency measure)")
    axes[1].set_ylabel("T_user (adaptive threshold)")
    axes[1].set_title("User Consistency vs Threshold")
    axes[1].axhline(global_threshold, color='red', linestyle='--',
                     label='Global threshold')
    axes[1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Threshold statistics:")
    print(f"  Global T:         {global_threshold:.4f}")
    print(f"  Mean T_user:      {np.mean(T_users):.4f}")
    print(f"  Std T_user:       {np.std(T_users):.4f}")
    print(f"  Min T_user:       {np.min(T_users):.4f}")
    print(f"  Max T_user:       {np.max(T_users):.4f}")
    pct_tighter = sum(1 for t in T_users if t < global_threshold) / len(T_users) * 100
    print(f"  Users with tighter threshold: {pct_tighter:.1f}%")


def plot_eer_vs_sigma(user_metrics: list, save_path: str = None):
    """
    Correlation between σ_user (consistency) and per-user EER.
    
    Expected finding: higher σ_user → higher EER (consistent users easier to authenticate).
    This justifies the adaptive threshold design.
    """
    sigmas = [m['sigma_user'] for m in user_metrics if 'sigma_user' in m]
    eers   = [m['eer']        for m in user_metrics if 'sigma_user' in m]
    
    from scipy.stats import pearsonr
    r, p = pearsonr(sigmas, eers)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(sigmas, [e*100 for e in eers], alpha=0.4, s=10, color='steelblue')
    
    # Fit trendline
    z = np.polyfit(sigmas, [e*100 for e in eers], 1)
    x_line = np.linspace(min(sigmas), max(sigmas), 100)
    plt.plot(x_line, np.polyval(z, x_line), 'r-', linewidth=2,
             label=f'r={r:.3f}, p={p:.3f}')
    
    plt.xlabel("σ_user (intra-class std of enrolment distances)")
    plt.ylabel("Per-user EER (%)")
    plt.title("Typing Consistency vs Authentication Error Rate")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation σ_user vs EER: r={r:.3f}, p={p:.4f}")
    return r, p
```

---

## 11. Optimization & Hyperparameter Tuning

### 11.1 Configuration File

```yaml
# configs/experiment_config.yaml

# Dataset
data:
  raw_aalto_path: "data/raw_aalto/"
  processed_path: "data/processed/"
  n_eval_users: 1000
  n_val_users: 400
  n_train_users: 30000
  sessions_per_user: 15
  seq_len: 50              # fixed per TypeFormer paper

# Model
model:
  weights_path: "models/TypeFormer/pretrained/TypeFormer_pretrained.pt"
  device: "auto"           # "cuda", "cpu", or "auto"
  batch_size: 64           # reduce if OOM errors
  embedding_dim: 64

# Experiments
experiments:
  E_values: [1, 2, 5, 7, 10]
  configs:
    - name: E1_baseline
      use_kdprint: false
      use_adaptive: false
    - name: E2_preprocessing
      use_kdprint: true
      use_adaptive: false
    - name: E3_adaptive
      use_kdprint: false
      use_adaptive: true
    - name: E4_full
      use_kdprint: true
      use_adaptive: true

# KDPrint preprocessing
preprocessing:
  buffer_size: 5           # B parameter (try 1, 3, 5, 10)
  timing_features: [0, 1, 2, 3]   # columns to standardize
  ascii_normalization: 127         # divisor for ASCII

# Adaptive threshold
threshold:
  k_init: 2.0              # starting value for grid search
  k_min: 0.5
  k_max: 4.0
  k_step: 0.25
  sigma_floor: 0.0         # minimum sigma for E=1 edge case

# Output
output:
  results_dir: "results/"
  figures_dir: "figures/"
  save_per_user: true
```

### 11.2 Sensitivity Analysis (Optional but Recommended)

```python
# Run this to understand how sensitive results are to buffer_size B

from src.experiments.run_experiment import run_single_experiment

buffer_sizes = [1, 3, 5, 7, 10]
results_by_B = {}

for B in buffer_sizes:
    result = run_single_experiment(
        f'sensitivity_B{B}', model, eval_users, val_users,
        E=5, L=50, use_kdprint=True, use_adaptive=True,
        buffer_size=B
    )
    results_by_B[B] = result['avg_eer']
    print(f"B={B}: EER={result['avg_eer']*100:.4f}%")
```

---

## 12. Troubleshooting Guide

### Problem: E1 EER is far from 3.25%

**Most common cause:** Data loading or preprocessing issue.

```python
# Debug steps:
# 1. Check that you're using the exact same 1000 evaluation users as TypeFormer
#    The user IDs are in: TypeFormer/TypeFormer_benchmark_sessions.json

# 2. Check that session ordering matches
#    TypeFormer uses specific session IDs per user

# 3. Verify feature extraction
#    Print first 5 features from a session and compare with expected range:
session = eval_users[0]['sessions'][0]
print("hold_lat:    ", session[:5, 0])   # expect ~50-200ms
print("inter_key:   ", session[:5, 1])   # expect ~50-500ms
print("press_lat:   ", session[:5, 2])   # expect ~100-600ms
print("release_lat: ", session[:5, 3])   # expect ~100-600ms
print("ascii:       ", session[:5, 4])   # expect [0, 1]

# 4. Check embedding norms (should be ≈1.0 for TypeFormer)
z = model.extract_single(session)
print(f"Embedding norm: {np.linalg.norm(z):.4f}")  # should be ~1.0
```

### Problem: RuntimeError in TypeFormer forward pass

```python
# Likely cause: wrong input shape or dtype
# TypeFormer expects: (batch_size, seq_len, n_features) = (B, 50, 5)
# And dtype: torch.FloatTensor (float32)

session = eval_users[0]['sessions'][0]
print(f"Session shape: {session.shape}")      # must be (50, 5)
print(f"Session dtype: {session.dtype}")      # must be float32 or float64
tensor = torch.FloatTensor(session).unsqueeze(0)   # add batch dim → (1, 50, 5)
print(f"Tensor shape: {tensor.shape}")
with torch.no_grad():
    out = model.model(tensor)
print(f"Output shape: {out.shape}")           # should be (1, 64)
```

### Problem: Memory error (OOM) on GPU

```python
# Reduce batch_size in TypeFormerWrapper
model = TypeFormerWrapper(WEIGHTS_PATH, device='cuda', batch_size=16)

# Or process impostors in smaller chunks
def get_impostor_scores_batched(z_mean, impostor_sessions, model, chunk_size=100):
    scores = []
    for i in range(0, len(impostor_sessions), chunk_size):
        chunk = impostor_sessions[i:i+chunk_size]
        z_chunk = model.extract_embeddings(chunk)
        scores.extend([np.linalg.norm(z - z_mean) for z in z_chunk])
    return scores
```

### Problem: Very different results between runs

```python
# Set all random seeds for reproducibility
import torch, numpy as np, random

def set_all_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(42)
```

### Problem: NaN values appearing in preprocessing output

```python
# Cause: sigma=0 in standardization (feature has zero variance)
# This happens when all enrolment sessions have identical values for a feature
# The 1e-8 epsilon in fit() should prevent this, but check:

prep = KDPrintPreprocessor()
prep.fit(enrol_sessions)
print("mu:", prep.mu)
print("sigma:", prep.sigma)
print("min sigma:", prep.sigma.min())  # should be >= 1e-8

# If sigma is very small (e.g., 1e-8), the standardized values will be huge
# Consider a larger epsilon: 
prep.sigma = np.where(prep.sigma < 0.01, 0.01, prep.sigma)
```

---

## 13. Checklist Before You Start

### Conceptual Understanding

- [ ] I understand EER and can compute it from genuine/impostor score lists
- [ ] I understand the difference between enrollment and verification phases
- [ ] I understand triplet loss: `L = max(0, d(A,P) - d(A,N) + α)`
- [ ] I understand why z-score > min-max for keystroke timing features
- [ ] I understand why a global threshold is suboptimal for users with different consistency
- [ ] I have read TypeFormer paper Sections 3 and 5
- [ ] I have read KDPrint paper Sections 3 and 4
- [ ] I have read `TypeFormer/model/Model.py` to understand the architecture

### Technical Setup

- [ ] Python 3.8+ installed
- [ ] PyTorch installed with GPU support (`torch.cuda.is_available()` = True)
- [ ] TypeFormer repository cloned successfully
- [ ] Aalto database downloaded (or preprocessed data received from authors)
- [ ] Pretrained weights available at correct path
- [ ] `test.py` from TypeFormer repo runs without errors
- [ ] All unit tests pass: `pytest tests/ -v`

### Baseline Verification (CRITICAL — do not skip)

- [ ] E1 baseline EER at E=5, L=50 is approximately **3.25%** (±0.3%)
- [ ] If E1 differs significantly, identify and fix the issue before proceeding
- [ ] E1 results match TypeFormer paper Table 3 for at least E=5

### Before Writing Results

- [ ] All 4 configurations × 5 E values = 20 experiments completed
- [ ] Per-user metrics saved for all experiments
- [ ] Threshold optimization (k) done on validation set, not test set
- [ ] t-SNE plots generated for at least one comparison
- [ ] σ_user vs EER correlation computed

---

*Guide version 1.0 | Based on TypeFormer (Stragapede et al., 2024) and KDPrint (Kim et al., 2025)*
