# finetune_holdonly.py
# Fine-tune pretrained TypeFormer on hold-time-only preprocessed data.
#
# Supports resuming: saves full checkpoint (model + optimizer + scheduler + epoch).
# To resume, simply run the script again — it auto-detects the checkpoint.
#
# Usage:
#   First run:   python finetune_holdonly.py
#   Resume:      python finetune_holdonly.py          (auto-detects checkpoint)
#   Fresh start: python finetune_holdonly.py --fresh   (ignores checkpoint)

import os
import sys
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

warnings.filterwarnings('ignore', category=UserWarning)

from model.Model import HARTrans

# ── Configuration ─────────────────────────────────────────────────────────
PRETRAINED_WEIGHTS  = "pretrained/TypeFormer_pretrained.pt"
OUTPUT_DIR          = "finetuned_holdonly/"
BEST_MODEL_PATH     = OUTPUT_DIR + "TypeFormer_finetuned_holdonly.pt"
CHECKPOINT_PATH     = OUTPUT_DIR + "checkpoint.pt"        # full state for resuming
LOG_PATH            = OUTPUT_DIR + "finetune_log.txt"

FINETUNE_LR             = 0.0001
FINETUNE_EPOCHS         = 100
EARLY_STOPPING_PATIENCE = 20

DATA_PATH = "src/data/processed/Mobile_keys_db_holdonly.npy"

# ── Device ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check for --fresh flag
FRESH_START = "--fresh" in sys.argv

print("=" * 60)
print("TypeFormer Fine-Tuning — Hold-Time Only Preprocessing")
print("=" * 60)
print(f"Device:             {device}")
print(f"Pretrained weights: {PRETRAINED_WEIGHTS}")
print(f"Training data:      {DATA_PATH}")
print(f"Best model output:  {BEST_MODEL_PATH}")
print(f"Checkpoint (resume):{CHECKPOINT_PATH}")
print(f"Learning rate:      {FINETUNE_LR}")
print(f"Max epochs:         {FINETUNE_EPOCHS}")
print(f"Resume mode:        {'FRESH START' if FRESH_START else 'auto-detect'}")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────
print(f"\nLoading data from {DATA_PATH}...")
keystroke_dataset = list(np.load(DATA_PATH, allow_pickle=True))
print(f"Loaded {len(keystroke_dataset)} users")

sample = np.array(keystroke_dataset[0][0], dtype=np.float64)
print(f"Sample shape: {sample.shape}")
print(f"hold_lat [0]: [{sample[:, 0].min():.4f}, {sample[:, 0].max():.4f}]  (z-scored)")
print(f"inter_pr [1]: [{sample[:, 1].min():.4f}, {sample[:, 1].max():.4f}]  (raw)")

# ── Dataset splits ────────────────────────────────────────────────────────
print(f"\nBuilding splits...")
print(f"  Training:   [{configs.num_training_subjects} : {2*configs.num_training_subjects}]")
print(f"  Validation: [0 : {configs.num_validation_subjects}]")

ds_t = KeystrokeSessionTriplet(
    keystroke_dataset[configs.num_training_subjects : 2 * configs.num_training_subjects],
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

# ── Model ─────────────────────────────────────────────────────────────────
print(f"\nInitializing model...")
TransformerModel = HARTrans(configs).double()

# ── Optimizer & Scheduler ─────────────────────────────────────────────────
optimizer = torch.optim.Adam(
    TransformerModel.parameters(), lr=FINETUNE_LR, betas=(0.9, 0.999)
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
criterion = torch.jit.script(TripletLoss())

# ── Load weights (pretrained or checkpoint) ───────────────────────────────
start_epoch       = 0
best_eer_v        = 100.0
best_epoch        = 0
epochs_no_improve = 0
loss_t_list, eer_t_list = [], []
loss_v_list, eer_v_list = [], []

can_resume = os.path.exists(CHECKPOINT_PATH) and not FRESH_START

if can_resume:
    print(f"\n>>> RESUMING from {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    TransformerModel.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch       = ckpt['epoch'] + 1
    best_eer_v        = ckpt['best_eer_v']
    best_epoch        = ckpt['best_epoch']
    epochs_no_improve = ckpt['epochs_no_improve']
    loss_t_list       = ckpt.get('loss_t_list', [])
    eer_t_list        = ckpt.get('eer_t_list', [])
    loss_v_list       = ckpt.get('loss_v_list', [])
    eer_v_list        = ckpt.get('eer_v_list', [])
    print(f"    Resuming at epoch {start_epoch}")
    print(f"    Best val EER so far: {100*best_eer_v:.2f}% (epoch {best_epoch})")
    print(f"    Current LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"    No-improve counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
else:
    print(f"\nLoading pretrained weights from {PRETRAINED_WEIGHTS}...")
    checkpoint = torch.load(PRETRAINED_WEIGHTS, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        TransformerModel.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        TransformerModel.load_state_dict(checkpoint['state_dict'])
    else:
        TransformerModel.load_state_dict(checkpoint)
    print("Pretrained weights loaded (starting fresh fine-tuning).")

TransformerModel = TransformerModel.to(device)

# ── Training functions ────────────────────────────────────────────────────
def train_one_epoch():
    TransformerModel.train()
    epoch_eers, total_loss = [], 0.0
    for _, (anchor_sgm, positive_sgm, negative_sgm) in enumerate(
        tqdm(train_dataloader, desc="Fine-tuning", leave=False)
    ):
        optimizer.zero_grad()
        a, p, n = (
            Variable(anchor_sgm).to(device),
            Variable(positive_sgm).to(device),
            Variable(negative_sgm).to(device),
        )
        a_out, p_out, n_out = TransformerModel(a), TransformerModel(p), TransformerModel(n)
        loss = criterion(a_out, p_out, n_out)
        loss.backward(retain_graph=True)
        optimizer.step()

        running_loss = np.round(loss.item(), configs.decimals)
        pred_a = np.round(a_out.cpu().detach().numpy(), configs.decimals)
        pred_p = np.round(p_out.cpu().detach().numpy(), configs.decimals)
        pred_n = np.round(n_out.cpu().detach().numpy(), configs.decimals)

        scores_g = np.sqrt(np.add.reduce(np.square(pred_a - pred_p), 1))
        scores_i = np.sqrt(np.add.reduce(np.square(pred_a - pred_n), 1))
        labels = np.array([0] * len(scores_g) + [1] * len(scores_i))
        eer = np.round(
            compute_eer(labels, np.concatenate((scores_g, scores_i)))[0], configs.decimals
        )
        epoch_eers.append(eer)
        total_loss += running_loss
    return total_loss, np.round(epoch_eers[-1], configs.decimals)


def eval_one_epoch():
    TransformerModel.eval()
    epoch_eers, total_loss = [], 0.0
    with torch.no_grad():
        for _, (anchor_sgm, positive_sgm, negative_sgm) in enumerate(
            tqdm(val_dataloader, desc="Validation", leave=False)
        ):
            a = anchor_sgm.to(device)
            p = positive_sgm.to(device)
            n = negative_sgm.to(device)

            a_out, p_out, n_out = TransformerModel(a), TransformerModel(p), TransformerModel(n)
            loss = criterion(a_out, p_out, n_out)
            running_loss = np.round(loss.item(), configs.decimals)

            pred_a = np.round(a_out.cpu().numpy(), configs.decimals)
            pred_p = np.round(p_out.cpu().numpy(), configs.decimals)
            pred_n = np.round(n_out.cpu().numpy(), configs.decimals)

            scores_g = np.sqrt(np.add.reduce(np.square(pred_a - pred_p), 1))
            scores_i = np.sqrt(np.add.reduce(np.square(pred_a - pred_n), 1))
            labels = np.array([0] * len(scores_g) + [1] * len(scores_i))
            eer = np.round(
                compute_eer(labels, np.concatenate((scores_g, scores_i)))[0], configs.decimals
            )
            epoch_eers.append(eer)
            total_loss += running_loss
    return total_loss, np.round(np.mean(epoch_eers), configs.decimals)


# ── Training loop ─────────────────────────────────────────────────────────
print(f"\nStarting fine-tuning (epochs {start_epoch}–{FINETUNE_EPOCHS-1}, "
      f"early stopping patience={EARLY_STOPPING_PATIENCE})...")
print(f"{'Epoch':>6} | {'Train Loss':>12} {'Train EER':>10} | "
      f"{'Val Loss':>10} {'Val EER':>9} | {'Time':>7} | {'LR':>10} | Note")
print("-" * 95)

for epoch in range(start_epoch, FINETUNE_EPOCHS):
    start = time.time()

    loss_t, eer_t = train_one_epoch()
    loss_v, eer_v = eval_one_epoch()

    loss_t_list.append(loss_t)
    eer_t_list.append(eer_t)
    loss_v_list.append(loss_v)
    eer_v_list.append(eer_v)

    elapsed = (time.time() - start) / 60
    current_lr = optimizer.param_groups[0]['lr']

    # Check for improvement
    note = ""
    if eer_v < best_eer_v:
        best_eer_v, best_epoch = eer_v, epoch
        epochs_no_improve = 0
        torch.save(TransformerModel.state_dict(), BEST_MODEL_PATH)
        note = f"✓ best ({100*best_eer_v:.2f}% @ ep{best_epoch})"
    else:
        epochs_no_improve += 1
        note = f"no improve {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}"

    scheduler.step(eer_v)

    print(
        f"{epoch:6d} | {loss_t:12.2f} {100*eer_t:9.2f}% | "
        f"{loss_v:10.2f} {100*eer_v:8.2f}% | {elapsed:6.1f}m | {current_lr:.2e} | {note}"
    )

    # Save full checkpoint for resuming (every epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': TransformerModel.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_eer_v': best_eer_v,
        'best_epoch': best_epoch,
        'epochs_no_improve': epochs_no_improve,
        'loss_t_list': loss_t_list,
        'eer_t_list': eer_t_list,
        'loss_v_list': loss_v_list,
        'eer_v_list': eer_v_list,
    }, CHECKPOINT_PATH)

    # Save training log
    with open(LOG_PATH, "w") as f:
        f.write(str([loss_t_list, loss_v_list, eer_t_list, eer_v_list]))

    # Early stopping
    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered — "
              f"no improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs.")
        break

print(f"\nFine-tuning complete.")
print(f"Best validation EER: {100*best_eer_v:.2f}% at epoch {best_epoch}")
print(f"Best model saved to: {BEST_MODEL_PATH}")
print(f"Checkpoint saved to: {CHECKPOINT_PATH}  (use to resume)")
