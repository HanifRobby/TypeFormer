# Fine-Tuning TypeFormer with KDPrint Preprocessing
## Implementation Guide — Revised

> **Based on actual codebase analysis (Backup TypeFormer)**
>
> | File | Architecture | Role |
> |------|-------------|------|
> | `model/Model.py` | **HARTrans — Full TypeFormer** | Temporal + Channel + Block-Recurrent LSTM. This is the main contribution of the paper. Used in `test.py`, `TypeFormerWrapper`, `KVC_train.py`. Pretrained weights: `TypeFormer_pretrained.pt` (15MB). |
> | `model/Preliminary.py` | HARTrans — Preliminary Transformer | Temporal + Channel only, no Block-Recurrent. This is the **ablation/baseline** used as a comparison point in the paper. Used (incorrectly) in the current `train.py`. Pretrained weights: `preliminary_transformer_pretrained.pt` (3.6MB). |
>
> **Critical fix required before fine-tuning:** `train.py` currently imports from  
> `model/Preliminary.py` — this is wrong. Fine-tuning must use `model/Model.py`.  
> Loss: `TripletLoss` margin=1.0, random triplet sampling (`utils/misc.py`)  
> Data pipeline: `KeystrokeSessionTriplet` (`utils/misc.py`)

---

## Table of Contents

1. [Critical Fix — train.py Architecture Import](#1-critical-fix--trainpy-architecture-import)
2. [What Fine-Tuning Is and Why It Applies Here](#2-what-fine-tuning-is-and-why-it-applies-here)
3. [What Changes vs What Stays the Same](#3-what-changes-vs-what-stays-the-same)
4. [Data Preparation (bulk_preprocess.py)](#4-data-preparation)
5. [Creating the Fine-Tuning Script (finetune.py)](#5-creating-the-fine-tuning-script)
6. [Monitoring and Early Stopping](#6-monitoring-and-early-stopping)
7. [Evaluation After Fine-Tuning](#7-evaluation-after-fine-tuning)
8. [Expected Results and Interpretation](#8-expected-results-and-interpretation)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Critical Fix — train.py Architecture Import

This must be fixed **before writing finetune.py**. The current `train.py` imports the wrong
architecture. This means any previous training run with `train.py` produced weights for the
Preliminary model, not TypeFormer.

**Open `train.py` and change line 13:**

```python
# CURRENT (wrong) — trains the ablation/baseline model:
from model.Preliminary import HARTrans

# CORRECTED — trains the full TypeFormer with Block-Recurrent attention:
from model.Model import HARTrans
```

**Why this matters for fine-tuning:**

The pretrained weights in `TypeFormer_pretrained.pt` (15MB) were produced by the full
`Model.py` architecture. If you load those weights into a `Preliminary.py` instance,
PyTorch will raise a `RuntimeError: unexpected key` because the state dicts are
incompatible — `Model.py` has `transformer_pre`, `transformer_rec`, and `transformer_post`
layers that simply do not exist in `Preliminary.py`.

Fine-tuning must use the same architecture as the pretrained weights being loaded.

**Verify `train_config.py` has all required parameters for `Model.py`:**

`Model.py`'s `HARTrans.__init__` requires three parameters that `Preliminary.py` does not:
`hlayers_rec`, `hlayers_pos`. Check that `utils/train_config.py` contains them:

```python
# These must exist in train_config.py for Model.py to initialize correctly:
configs.hlayers_rec = 2   # Block-Recurrent layers in temporal module
configs.hlayers_pos = 1   # Post-recurrent layers in temporal module
```

Your current `train_config.py` already has these values. No changes needed there.

---

## 2. What Fine-Tuning Is and Why It Applies Here

### The core problem from your experiments

```
E1 (pretrained TypeFormer, raw data):     EER = 3.03%   ratio = 6.6×
E2 (pretrained TypeFormer, KDPrint data): EER = 20.2%   ratio = 2.5×
```

The pretrained model weights were optimized for input values in range `[-0.12, 0.23]`
(the normalized Aalto timing features). KDPrint preprocessing transforms this into
z-scores clipped to `[-5.0, +5.0]` — a fundamentally different distribution.

TypeFormer's sigmoid output layer saturates on out-of-distribution inputs: when timing
values jump from the familiar ~0.1 range to ±5, the sigmoid neurons push toward their
asymptotes (0 or 1), producing embeddings where all users look similar regardless of
their actual keystroke behavior. This is why the impostor/genuine distance ratio collapses
from 6.6× to 2.5×.

### What fine-tuning does

Fine-tuning continues training from **pretrained weights** (rather than random
initialization) for a small number of additional epochs on the new data distribution.
The model's already-learned keystroke patterns serve as a starting point; only the
weight calibration for the new input scale needs to be adjusted.

```
Random init ──► 1000 epochs on raw data ──► pretrained weights (TypeFormer_pretrained.pt)
                                                        │
                                           ┌────────────▼────────────┐
                                           │  Load pretrained weights │
                                           │  (model/Model.py)        │
                                           └────────────┬────────────┘
                                                        │
                                           ┌────────────▼────────────┐
                                           │  Fine-tune 30–100 epochs │
                                           │  on KDPrint-preprocessed │
                                           │  data (z-score dist.)    │
                                           └────────────┬────────────┘
                                                        │
                                           ┌────────────▼────────────┐
                                           │  Fine-tuned weights      │
                                           │  TypeFormer_finetuned.pt │
                                           └─────────────────────────┘
```

**Why fine-tune instead of retrain from scratch:**
- Full retraining: 1000 epochs × 30,000 users = days of GPU time with no guarantee
- Fine-tuning: 30–100 epochs is typically sufficient for distribution adaptation
- Pre-learned keystroke discriminability is preserved, only scale adaptation is needed

---

## 3. What Changes vs What Stays the Same

| Component | Original training (`train.py`) | Fine-Tuning (`finetune.py`) |
|-----------|-------------------------------|----------------------------|
| Architecture | `model/Model.py` HARTrans (**after fix**) | **Identical — no changes** |
| Loss function | `TripletLoss` margin=1.0 | **Identical — no changes** |
| Triplet sampling | Random | **Identical — no changes** |
| Optimizer | Adam β=(0.9, 0.999) lr=0.001 | Adam β=(0.9, 0.999) lr=**0.0001** |
| Initial weights | Random (Xavier/Kaiming) | **Loaded from TypeFormer_pretrained.pt** |
| Data | `Mobile_keys_db_6_features.npy` (raw) | `Mobile_keys_db_new.npy` (KDPrint) |
| Max epochs | 1000 | **100** (with early stopping at patience=20) |
| Dataset split | val=[0:400], train=[30000:60000] | **Identical — same indices** |
| Batch size | train=64, val=400 | **Identical** |
| Output path | `latest_experiment/TypeFormer_KDPrint_retrained.pt` | `finetuned/TypeFormer_finetuned.pt` |

**Why the learning rate is 10× smaller:**
The pretrained weights encode useful keystroke representations developed over 1000 epochs.
A high learning rate would overwrite these, forcing the model to relearn from scratch.
A 10× reduction (0.001 → 0.0001) allows gradual adaptation to the new distribution while
preserving the learned structure of the embedding space.

---

## 4. Data Preparation

Your `bulk_preprocess.py` is already correctly written with Option B (fit on all 15 sessions
per user). Before running it, add explicit `clip_sigma` to the constructor call for clarity:

**Fix in `bulk_preprocess.py`:**

```python
# Change:
prep = KDPrintPreprocessor(buffer_size=buffer_size, seq_len=seq_len)

# To (explicit clip_sigma):
prep = KDPrintPreprocessor(buffer_size=buffer_size, seq_len=seq_len, clip_sigma=5.0)
```

**Verify input data before running:**

```python
import numpy as np

data = np.load(r'data\Mobile_keys_db_6_features.npy', allow_pickle=True)
print(f"Total users:       {len(data)}")           # expect ~64,615
print(f"Sessions user 0:   {len(data[0])}")        # expect 15
print(f"Session 0 shape:   {data[0][0].shape}")    # expect (N, 6), N varies (e.g. 36)
print(f"Session 0 dtype:   {data[0][0].dtype}")    # expect object
```

**Run bulk preprocessing:**

```bash
python src/preprocessing/bulk_preprocess.py
```

Expected runtime: 30–90 minutes depending on CPU speed.

**Verify output after running:**

```python
import numpy as np

new_data = np.load('src/data/processed/Mobile_keys_db_new.npy', allow_pickle=True)
print(f"Processed users:   {len(new_data)}")           # same as input
print(f"Sessions user 0:   {len(new_data[0])}")        # expect 15
print(f"Session 0 shape:   {new_data[0][0].shape}")    # expect (50, 5) — padded/truncated
print(f"Session 0 dtype:   {new_data[0][0].dtype}")    # expect float64

# Verify value range after z-score + clipping
sample = np.array(new_data[0][0], dtype=np.float64)
print(f"Timing range:      [{sample[:, :4].min():.3f}, {sample[:, :4].max():.3f}]")
# Expected: within [-5.0, 5.0]
print(f"ASCII range:       [{sample[:, 4].min():.3f}, {sample[:, 4].max():.3f}]")
# Expected: same as raw (not standardized, already in [0, ~0.5])
```

---

## 5. Creating the Fine-Tuning Script

Create `finetune.py` in the root directory (same level as `train.py`).

```python
# finetune.py
# Fine-tune pretrained TypeFormer (model/Model.py) on KDPrint-preprocessed data.
#
# Differences from train.py:
#   1. Imports from model/Model.py  (train.py currently wrongly uses Preliminary.py)
#   2. Loads TypeFormer_pretrained.pt weights before training starts
#   3. Learning rate 0.0001 instead of 0.001 (10x smaller — preserves learned patterns)
#   4. Max 100 epochs with early stopping (instead of 1000)
#   5. Adds LR scheduler to further decay LR if validation stagnates
#   6. Saves to finetuned/ directory (separate from latest_experiment/)

import os
import warnings
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.misc import KeystrokeSessionTriplet
from utils.train_config import configs
from utils.misc import compute_eer, TripletLoss
import time

# Suppress the known UserWarning from Model.py line 142 (harmless, cosmetic)
warnings.filterwarnings('ignore', category=UserWarning)

# ── CRITICAL: Full TypeFormer from Model.py, NOT Preliminary.py ───────────
# model/Model.py  = Full TypeFormer (Temporal + Channel + Block-Recurrent LSTM)
# model/Preliminary.py = Preliminary/baseline transformer (ablation study model)
from model.Model import HARTrans

# ── Fine-tuning configuration ─────────────────────────────────────────────
PRETRAINED_WEIGHTS  = "pretrained/TypeFormer_pretrained.pt"  # 15MB — full TypeFormer
FINETUNE_OUTPUT_DIR = "finetuned/"
FINETUNE_MODEL_PATH = FINETUNE_OUTPUT_DIR + "TypeFormer_finetuned.pt"
FINETUNE_LOG_PATH   = FINETUNE_OUTPUT_DIR + "finetune_log.txt"

FINETUNE_LR             = 0.0001   # 10× smaller than original 0.001
FINETUNE_EPOCHS         = 100      # max epochs; early stopping will likely trigger sooner
EARLY_STOPPING_PATIENCE = 20       # stop if no val EER improvement for this many epochs

KDPRINT_DATA_PATH = "src/data/processed/Mobile_keys_db_new.npy"

# ── Device setup ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("TypeFormer Fine-Tuning")
print("=" * 60)
print(f"Device:             {device}")
print(f"Architecture:       model/Model.py — Full TypeFormer")
print(f"Pretrained weights: {PRETRAINED_WEIGHTS}")
print(f"Fine-tuned output:  {FINETUNE_MODEL_PATH}")
print(f"Training data:      {KDPRINT_DATA_PATH}")
print(f"Learning rate:      {FINETUNE_LR}  (original: 0.001)")
print(f"Max epochs:         {FINETUNE_EPOCHS}  (with early stopping)")
print("=" * 60)

os.makedirs(FINETUNE_OUTPUT_DIR, exist_ok=True)

# ── Load KDPrint-preprocessed data ────────────────────────────────────────
print(f"\nLoading KDPrint-preprocessed data...")
keystroke_dataset = list(np.load(KDPRINT_DATA_PATH, allow_pickle=True))
print(f"Loaded {len(keystroke_dataset)} users")

# Verify data range — should be within [-5.0, 5.0] for timing features
sample = np.array(keystroke_dataset[0][0], dtype=np.float64)
print(f"Sample session shape: {sample.shape}")  # expect (50, 5)
print(f"Timing range:  [{sample[:, :4].min():.3f}, {sample[:, :4].max():.3f}]")
print(f"ASCII range:   [{sample[:, 4].min():.3f}, {sample[:, 4].max():.3f}]")

# ── Dataset splits (identical to original training) ───────────────────────
# Validation: indices [0 : num_validation_subjects]   = [0 : 400]
# Training:   indices [num_training_subjects : 2×num] = [30000 : 60000]
print(f"\nBuilding dataset splits...")
print(f"  Training users:   [{configs.num_training_subjects} : {2*configs.num_training_subjects}]")
print(f"  Validation users: [0 : {configs.num_validation_subjects}]")

ds_t = KeystrokeSessionTriplet(
    keystroke_dataset[
        configs.num_training_subjects : 2 * configs.num_training_subjects
    ],
    data_length=configs.sequence_length,
    length=len(keystroke_dataset),
)
ds_v = KeystrokeSessionTriplet(
    keystroke_dataset[: configs.num_validation_subjects],
    data_length=configs.sequence_length,
    length=len(keystroke_dataset),
)

train_dataloader = DataLoader(ds_t, batch_size=configs.batch_size_train, shuffle=True)
val_dataloader   = DataLoader(ds_v, batch_size=configs.batch_size_val,   shuffle=True)

# ── Load pretrained weights into full TypeFormer ──────────────────────────
print(f"\nInitializing model (model/Model.py — Full TypeFormer)...")
TransformerModel = HARTrans(configs).double()

print(f"Loading pretrained weights from {PRETRAINED_WEIGHTS}...")
checkpoint = torch.load(PRETRAINED_WEIGHTS, map_location=device)

# Handle different checkpoint formats
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    TransformerModel.load_state_dict(checkpoint['model_state_dict'])
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    TransformerModel.load_state_dict(checkpoint['state_dict'])
else:
    # Direct state dict — most common format for TypeFormer_pretrained.pt
    TransformerModel.load_state_dict(checkpoint)

TransformerModel = TransformerModel.to(device)
print("Pretrained weights loaded successfully.")

# Quick sanity check — verify model produces valid output
TransformerModel.eval()
with torch.no_grad():
    fake_input = torch.zeros(
        1, configs.sequence_length, configs.dimensionality,
        dtype=torch.float64, device=device
    )
    out = TransformerModel(fake_input)
    print(f"Model output shape: {out.shape}")   # expect (1, 64)
    print(f"Model output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    # sigmoid output: values between 0 and 1

# ── Optimizer — 10× smaller LR than original ──────────────────────────────
optimizer = torch.optim.Adam(
    TransformerModel.parameters(),
    lr=FINETUNE_LR,       # 0.0001
    betas=(0.9, 0.999)    # same as original
)

# LR scheduler: reduce LR by half if val EER stops improving for 10 epochs
# This provides a second layer of adaptation after early plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # minimize EER
    factor=0.5,      # halve the LR when triggered
    patience=10,     # wait 10 epochs before triggering
    verbose=True
)

criterion = torch.jit.script(TripletLoss())

# ── Training functions (identical logic to train.py) ──────────────────────
def inner_ops(input_):
    """Single batch forward+backward pass. Returns (eer, loss)."""
    optimizer.zero_grad()
    anchor_sgm, positive_sgm, negative_sgm = (
        Variable(input_[0]).to(device),
        Variable(input_[1]).to(device),
        Variable(input_[2]).to(device),
    )
    anchor_out, positive_out, negative_out = (
        TransformerModel(anchor_sgm),
        TransformerModel(positive_sgm),
        TransformerModel(negative_sgm),
    )
    loss = criterion(anchor_out, positive_out, negative_out)
    loss.backward(retain_graph=True)
    optimizer.step()

    running_loss = np.round(loss.item(), configs.decimals)
    pred_a = np.round(anchor_out.cpu().detach().numpy(),   configs.decimals)
    pred_p = np.round(positive_out.cpu().detach().numpy(), configs.decimals)
    pred_n = np.round(negative_out.cpu().detach().numpy(), configs.decimals)

    scores_g = np.sqrt(np.add.reduce(np.square(pred_a - pred_p), 1))
    scores_i = np.sqrt(np.add.reduce(np.square(pred_a - pred_n), 1))
    labels   = np.array(
        [0] * len(scores_g) + [1] * len(scores_i)
    )
    eer = np.round(
        compute_eer(labels, np.concatenate((scores_g, scores_i)))[0],
        configs.decimals
    )
    return eer, running_loss


def train_one_epoch():
    TransformerModel.train()
    epoch_eers, total_loss = [], 0.0
    from tqdm import tqdm
    for _, (anchor_sgm, positive_sgm, negative_sgm) in enumerate(
        tqdm(train_dataloader, desc="Fine-tuning", leave=False)
    ):
        eer_, loss_ = inner_ops((anchor_sgm, positive_sgm, negative_sgm))
        epoch_eers.append(eer_)
        total_loss += loss_
    return total_loss, np.round(epoch_eers[-1], configs.decimals)


def eval_one_epoch():
    TransformerModel.eval()
    epoch_eers, total_loss = [], 0.0
    from tqdm import tqdm
    for _, (anchor_sgm, positive_sgm, negative_sgm) in enumerate(
        tqdm(val_dataloader, desc="Validation", leave=False)
    ):
        eer_, loss_ = inner_ops((anchor_sgm, positive_sgm, negative_sgm))
        epoch_eers.append(eer_)
        total_loss += loss_
    return total_loss, np.round(np.mean(epoch_eers), configs.decimals)


# ── Fine-tuning loop ───────────────────────────────────────────────────────
print(f"\nStarting fine-tuning (max {FINETUNE_EPOCHS} epochs, "
      f"early stopping patience={EARLY_STOPPING_PATIENCE})...")
print(f"{'Epoch':>6} | {'Train Loss':>12} {'Train EER':>10} | "
      f"{'Val Loss':>10} {'Val EER':>9} | {'Time':>7} | Note")
print("-" * 80)

best_eer_v        = 100.0
best_epoch        = 0
epochs_no_improve = 0

loss_t_list, eer_t_list = [], []
loss_v_list, eer_v_list = [], []

for epoch in range(FINETUNE_EPOCHS):
    start = time.time()

    loss_t, eer_t = train_one_epoch()
    loss_v, eer_v = eval_one_epoch()

    loss_t_list.append(loss_t)
    eer_t_list.append(eer_t)
    loss_v_list.append(loss_v)
    eer_v_list.append(eer_v)

    elapsed = (time.time() - start) / 60

    # Save best model
    note = ""
    if eer_v < best_eer_v:
        best_eer_v, best_epoch = eer_v, epoch
        epochs_no_improve = 0
        torch.save(TransformerModel.state_dict(), FINETUNE_MODEL_PATH)
        note = f"✓ best ({100*best_eer_v:.2f}% @ ep{best_epoch})"
    else:
        epochs_no_improve += 1
        note = f"no improve {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}"

    # Step LR scheduler based on validation EER
    scheduler.step(eer_v)

    print(
        f"{epoch:6d} | {loss_t:12.2f} {100*eer_t:9.2f}% | "
        f"{loss_v:10.2f} {100*eer_v:8.2f}% | {elapsed:6.1f}m | {note}"
    )

    # Persist log every epoch (safe to kill process at any time)
    with open(FINETUNE_LOG_PATH, "w") as f:
        f.write(str([loss_t_list, loss_v_list, eer_t_list, eer_v_list]))

    # Early stopping
    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered — "
              f"no improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs.")
        break

print(f"\nFine-tuning complete.")
print(f"Best validation EER: {100*best_eer_v:.2f}% at epoch {best_epoch}")
print(f"Fine-tuned weights saved to: {FINETUNE_MODEL_PATH}")
print(f"Compare with E1 baseline:    3.03%")
```

---

## 6. Monitoring and Early Stopping

### Reading the output

The fine-tuning script prints a structured table per epoch:

```
 Epoch | Train Loss  Train EER |   Val Loss   Val EER |   Time | Note
--------------------------------------------------------------------------------
     0 |     203.45     12.80% |      52.10    10.26% |   2.1m | ✓ best (10.26% @ ep0)
     1 |     185.22     10.50% |      44.73     8.97% |   2.1m | ✓ best (8.97% @ ep1)
     5 |     134.10      7.20% |      28.05     5.42% |   2.1m | ✓ best (5.42% @ ep5)
    15 |      98.40      4.80% |      18.22     3.98% |   2.1m | ✓ best (3.98% @ ep15)
    25 |      92.15      4.20% |      19.11     4.12% |   2.1m | no improve 10/20
    35 |      91.80      4.10% |      20.33     4.30% |   2.1m | no improve 20/20
Early stopping triggered...
```

The best model is saved automatically whenever validation EER improves.
**Safe to Ctrl+C at any time** — the best weights up to that point are already saved.

### Three scenarios and responses

**Scenario A — Val EER drops below 3.03% (E1 baseline):**
Fine-tuning worked. Let early stopping run its course, then proceed to full evaluation.

**Scenario B — Val EER plateaus at 4–6% and stops improving:**
The model has adapted partially but is stuck. Resume from the best checkpoint with
a smaller learning rate:

```python
# resume_finetune.py — run after Scenario B
import torch
from model.Model import HARTrans
from utils.train_config import configs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TransformerModel = HARTrans(configs).double().to(device)

# Load the best checkpoint from the previous run
TransformerModel.load_state_dict(
    torch.load("finetuned/TypeFormer_finetuned.pt", map_location=device)
)

# Continue with smaller LR
optimizer = torch.optim.Adam(
    TransformerModel.parameters(),
    lr=0.00003,    # further reduced from 0.0001
    betas=(0.9, 0.999)
)
# Then run the same training loop for another 50 epochs
```

**Scenario C — Val EER stays above 10% after 20 epochs:**
Fine-tuning is not sufficient. The distribution gap is too large.
In this case, consider full retraining (`train.py` with fixed import + KDPrint data),
which will take significantly longer but allows the model to adapt from the ground up.

### Visualizing training progress

```python
# read_finetune_log.py
import ast
import matplotlib.pyplot as plt

with open('finetuned/finetune_log.txt') as f:
    loss_t, loss_v, eer_t, eer_v = ast.literal_eval(f.read())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot([e * 100 for e in eer_t], label='Train EER', color='steelblue')
ax1.plot([e * 100 for e in eer_v], label='Val EER',   color='darkorange')
ax1.axhline(3.03, color='red', linestyle='--', linewidth=1.5,
            label='E1 baseline (3.03%)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('EER (%)')
ax1.set_title('Fine-Tuning EER Progress')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(loss_t, label='Train Loss', color='steelblue')
ax2.plot(loss_v, label='Val Loss',   color='darkorange')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Triplet Loss')
ax2.set_title('Fine-Tuning Loss Progress')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('finetuned/training_curve.png', dpi=150)
plt.show()
print("Saved to finetuned/training_curve.png")
```

---

## 7. Evaluation After Fine-Tuning

### Step 1 — Quick embedding quality check

Run this before the full evaluation to confirm fine-tuning improved the genuine/impostor
separation ratio:

```python
# quick_eval_finetuned.py
import numpy as np
from src.preprocessing.kdprint_preprocess import KDPrintPreprocessor
from src.preprocessing.typeformer_preprocess import typeformer_preprocess
from src.model.typeformer_wrapper import TypeFormerWrapper

data = np.load(r'data\Mobile_keys_db_6_features.npy', allow_pickle=True)
enrol_raw    = [np.array(data[0][s], dtype=np.float64)[:, :5] for s in range(5)]
genuine_raw  = np.array(data[0][5], dtype=np.float64)[:, :5]
impostor_raw = np.array(data[1][0], dtype=np.float64)[:, :5]

# KDPrint preprocessing
prep       = KDPrintPreprocessor(buffer_size=5, seq_len=50, clip_sigma=5.0)
enrol_proc = prep.fit_transform(enrol_raw, use_buffer=True)
gen_proc   = prep.transform(genuine_raw)
imp_proc   = prep.transform(impostor_raw)

# Raw preprocessing (for comparison)
enrol_raw_proc = typeformer_preprocess(enrol_raw, seq_len=50)
gen_raw_proc   = typeformer_preprocess([genuine_raw], seq_len=50)[0]
imp_raw_proc   = typeformer_preprocess([impostor_raw], seq_len=50)[0]

print(f"{'Config':<32} {'d_genuine':>10} {'d_impostor':>10} {'ratio':>8}")
print("-" * 65)

models = {
    "Pretrained  + raw  (E1)":     ("pretrained/TypeFormer_pretrained.pt",  enrol_raw_proc, gen_raw_proc, imp_raw_proc),
    "Pretrained  + KDPrint (E2)":  ("pretrained/TypeFormer_pretrained.pt",  enrol_proc,     gen_proc,     imp_proc),
    "Fine-tuned  + KDPrint (E2-FT)": ("finetuned/TypeFormer_finetuned.pt", enrol_proc,     gen_proc,     imp_proc),
    "Fine-tuned  + raw":           ("finetuned/TypeFormer_finetuned.pt",    enrol_raw_proc, gen_raw_proc, imp_raw_proc),
}

for label, (weights, enrol, gen, imp) in models.items():
    model = TypeFormerWrapper(weights, device='auto')
    model.model.eval()
    z_enrol = model.extract_embeddings(enrol)
    z_mean  = z_enrol.mean(axis=0)
    d_gen   = float(np.linalg.norm(model.extract_single(gen)  - z_mean))
    d_imp   = float(np.linalg.norm(model.extract_single(imp) - z_mean))
    print(f"{label:<32} {d_gen:>10.4f} {d_imp:>10.4f} {d_imp/d_gen:>8.2f}×")

print()
print("Target: E2-FT ratio should be higher than E2 (currently 2.83×)")
print("        Ideally approaching E1 ratio (6.60×)")
```

### Step 2 — Update TypeFormerWrapper for fine-tuned model

`TypeFormerWrapper` already imports from `model/Model.py` (correct). You only need to
point it to the fine-tuned weights:

```python
# In run_E1_E2.py (or a new run_all.py), add:
from src.model.typeformer_wrapper import TypeFormerWrapper

ft_model = TypeFormerWrapper('finetuned/TypeFormer_finetuned.pt', device='auto')

# E2-FT: Fine-tuned model + KDPrint preprocessing
r_e2_ft = run_single_experiment(
    config_name  = 'E2_finetuned',
    model        = ft_model,
    eval_users   = eval_users,
    val_users    = val_users,
    E            = 5,
    L            = 50,
    use_kdprint  = True,
    buffer_size  = 5,
    output_dir   = 'results/'
)
print(f"E2-FT (Fine-tuned + KDPrint): EER = {r_e2_ft['avg_eer']*100:.4f}%")
print(f"E1    (Pretrained + raw):     EER = 3.03%")
```

### Step 3 — Full experiment matrix with fine-tuned model

The complete comparison table for the thesis should include:

| Config | Model | Preprocessing | Expected |
|--------|-------|--------------|---------|
| E1 | Pretrained | Raw | 3.03% (baseline) |
| E2 | Pretrained | KDPrint | ~20% (distribution shift) |
| E2-FT | **Fine-tuned** | KDPrint | **Target: < E1** |
| E3 | Pretrained | Raw + Adaptive threshold | TBD |
| E4 | Fine-tuned | KDPrint + Adaptive threshold | TBD |

---

## 8. Expected Results and Interpretation

### Success thresholds

```
Quick eval ratio (E2-FT, single user pair):
  ratio > 4.0×   → fine-tuning working, run full evaluation
  ratio > 6.0×   → excellent, likely beats E1

Full evaluation EER:
  EER < 3.03%    → fine-tuning outperforms baseline (best case)
  EER 3.0–4.0%   → comparable to baseline (acceptable)
  EER > 5.0%     → fine-tuning insufficient, consider full retraining
```

### Thesis framing for each outcome

**If E2-FT EER < E1 (3.03%):**
```
"Fine-tuning TypeFormer on KDPrint-preprocessed data achieves EER of X%,
outperforming the original TypeFormer baseline (3.03%). This demonstrates
that KDPrint's z-score standardization provides a more discriminative input
representation when the model is adapted to the new distribution through
fine-tuning. The distribution shift inherent in applying standardization to
a pretrained model requires this adaptation step to realize the full benefit."
```

**If E2-FT EER ≈ E1:**
```
"Fine-tuning TypeFormer on KDPrint-preprocessed data achieves EER of X%,
comparable to the original baseline (3.03%). This suggests that TypeFormer
can effectively learn from z-score standardized inputs when given the
opportunity to adapt, and that the preprocessing provides an equivalent
but structurally different representation of keystroke dynamics."
```

**If E2-FT EER > E1 (negative finding):**
```
"Fine-tuning TypeFormer with KDPrint preprocessing did not improve upon
the baseline EER, suggesting that the original normalized Aalto features
already represent an input distribution well-suited to TypeFormer's
architecture. This finding motivates focusing the preprocessing contribution
on the decision layer, as supported by the adaptive threshold results (E3, E4)."
```

All three outcomes are scientifically valid and publishable.  
The adaptive threshold contribution (E3, E4) remains entirely independent of
this result and provides a solid second contribution regardless.

---

## 9. Troubleshooting

### RuntimeError: unexpected key(s) in state_dict

**Cause:** Trying to load `TypeFormer_pretrained.pt` (15MB, from `Model.py`) into
a `Preliminary.py` instance, or vice versa.

**Fix:** Ensure `finetune.py` imports `from model.Model import HARTrans` and that
`PRETRAINED_WEIGHTS` points to `TypeFormer_pretrained.pt` (15MB), not
`preliminary_transformer_pretrained.pt` (3.6MB).

```python
# Verify which file you're loading:
import os
size_mb = os.path.getsize('pretrained/TypeFormer_pretrained.pt') / 1e6
print(f"Checkpoint size: {size_mb:.1f} MB")
# TypeFormer_pretrained.pt        → ~15 MB  (correct for Model.py)
# preliminary_transformer_pretrained.pt → ~3.6 MB (for Preliminary.py only)
```

### CUDA out of memory during fine-tuning

Reduce batch sizes in `utils/train_config.py`:

```python
configs.batch_size_train = 32   # from 64
configs.batch_size_val   = 200  # from 400
```

### Loss explodes or oscillates from epoch 0

Learning rate too large for fine-tuning. Reduce by another 10×:

```python
FINETUNE_LR = 0.00001   # from 0.0001
```

### Validation EER does not improve after 5 epochs

This can mean one of two things:

1. **The preprocessed data is wrong.** Check timing range:
   ```python
   new_data = np.load('src/data/processed/Mobile_keys_db_new.npy', allow_pickle=True)
   sample   = np.array(new_data[0][0], dtype=np.float64)
   print(sample[:, :4].min(), sample[:, :4].max())
   # Must be within [-5.0, 5.0] — if outside, clip_sigma was not applied
   ```

2. **The LR scheduler triggered too early.** Check the console output for
   `"Epoch     X: reducing learning rate..."` messages. If LR was halved before
   epoch 5, increase scheduler patience:
   ```python
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, mode='min', factor=0.5, patience=15, verbose=True
   )
   ```

### UserWarning: Converting a tensor with requires_grad=True to a scalar

This is a known harmless warning from `Model.py` line 142. Already suppressed by
`warnings.filterwarnings('ignore', category=UserWarning)` at the top of `finetune.py`.
It does not affect training correctness or results.

---

## Summary: Execution Steps

```
Step 0: Fix train.py (if you plan to use it)
         Change: from model.Preliminary import HARTrans
         To:     from model.Model import HARTrans

Step 1: Prepare KDPrint-preprocessed data
         python src/preprocessing/bulk_preprocess.py
         → Verify: src/data/processed/Mobile_keys_db_new.npy exists
         → Verify: timing values in [-5.0, 5.0], shape (50, 5) per session

Step 2: Run fine-tuning
         python finetune.py
         → Monitor val EER per epoch
         → Early stopping triggers automatically at patience=20
         → Best weights saved to finetuned/TypeFormer_finetuned.pt

Step 3: Quick quality check
         python quick_eval_finetuned.py
         → Compare genuine/impostor distance ratios
         → E2-FT ratio should be higher than E2 (2.83×)
         → If ratio < 3.0×, revisit Scenario B in Section 6

Step 4: Full E1-E4 evaluation with fine-tuned model
         → Add E2-FT config to run_E1_E2.py
         → Run full 1000-user evaluation
         → Build complete comparison table for thesis

Step 5 (parallel, independent): Run E3 and E4
         → Adaptive threshold does not depend on fine-tuning results
         → Can run simultaneously with Steps 2-4
```

---

*Guide version 2.0 — Revised: corrected architecture (Model.py not Preliminary.py),
fixed import in finetune.py, added architecture comparison table, updated troubleshooting.*
