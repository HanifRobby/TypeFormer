# finetune.py
# Fine-tune pretrained TypeFormer (model/Model.py) on KDPrint-preprocessed data.
#
# Differences from train.py:
#   1. Imports from model/Model.py  (train.py uses Preliminary.py — wrong for fine-tuning)
#   2. Loads TypeFormer_pretrained.pt weights before training starts
#   3. Learning rate 0.0001 instead of 0.001 (10x smaller — preserves learned patterns)
#   4. Max 100 epochs with early stopping patience=20
#   5. LR scheduler halves LR if val EER stagnates for 10 epochs
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
from tqdm import tqdm

# Suppress harmless UserWarning from Model.py Gaussian_Position (line 142)
warnings.filterwarnings('ignore', category=UserWarning)

# ── CRITICAL: Full TypeFormer from Model.py, NOT Preliminary.py ───────────
# model/Model.py      = Full TypeFormer (Temporal + Channel + Block-Recurrent LSTM)
# model/Preliminary.py = Preliminary transformer (ablation model, smaller)
from model.Model import HARTrans

# ── Fine-tuning configuration ─────────────────────────────────────────────
PRETRAINED_WEIGHTS  = "pretrained/TypeFormer_pretrained.pt"  # 14.2MB — full TypeFormer
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

# ── Optimizer — 10× smaller LR than original ──────────────────────────────
optimizer = torch.optim.Adam(
    TransformerModel.parameters(),
    lr=FINETUNE_LR,       # 0.0001
    betas=(0.9, 0.999)    # same as original
)

# LR scheduler: reduce LR by half if val EER stops improving for 10 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # minimize EER
    factor=0.5,      # halve the LR when triggered
    patience=10,     # wait 10 epochs before triggering
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
    with torch.no_grad():
        for _, (anchor_sgm, positive_sgm, negative_sgm) in enumerate(
            tqdm(val_dataloader, desc="Validation", leave=False)
        ):
            anchor_sgm = anchor_sgm.to(device)
            positive_sgm = positive_sgm.to(device)
            negative_sgm = negative_sgm.to(device)

            anchor_out = TransformerModel(anchor_sgm)
            positive_out = TransformerModel(positive_sgm)
            negative_out = TransformerModel(negative_sgm)

            loss = criterion(anchor_out, positive_out, negative_out)
            running_loss = np.round(loss.item(), configs.decimals)

            pred_a = np.round(anchor_out.cpu().numpy(),   configs.decimals)
            pred_p = np.round(positive_out.cpu().numpy(), configs.decimals)
            pred_n = np.round(negative_out.cpu().numpy(), configs.decimals)

            scores_g = np.sqrt(np.add.reduce(np.square(pred_a - pred_p), 1))
            scores_i = np.sqrt(np.add.reduce(np.square(pred_a - pred_n), 1))
            labels = np.array([0] * len(scores_g) + [1] * len(scores_i))
            eer = np.round(
                compute_eer(labels, np.concatenate((scores_g, scores_i)))[0],
                configs.decimals
            )
            epoch_eers.append(eer)
            total_loss += running_loss
    return total_loss, np.round(np.mean(epoch_eers), configs.decimals)


# ── Fine-tuning loop ───────────────────────────────────────────────────────
print(f"\nStarting fine-tuning (max {FINETUNE_EPOCHS} epochs, "
      f"early stopping patience={EARLY_STOPPING_PATIENCE})...")
print(f"{'Epoch':>6} | {'Train Loss':>12} {'Train EER':>10} | "
      f"{'Val Loss':>10} {'Val EER':>9} | {'Time':>7} | {'LR':>10} | Note")
print("-" * 95)

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
    current_lr = optimizer.param_groups[0]['lr']

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
        f"{loss_v:10.2f} {100*eer_v:8.2f}% | {elapsed:6.1f}m | {current_lr:.2e} | {note}"
    )

    # Persist log every epoch (safe to Ctrl+C at any time)
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
