# train.py
# Train TypeFormer from scratch on KDPrint-preprocessed data.
#
# Fixes applied vs original:
#   1. Uses model/Model.py (full TypeFormer with Block-Recurrent LSTM)
#      instead of model/Preliminary.py (ablation baseline)
#   2. Validation is forward-only (no backward pass, no optimizer.step)
#      — original leaked training into validation via inner_ops()
#   3. Early stopping (patience=20) and LR scheduler added
#   4. Full checkpoint saving for resume support
#
# Resume: re-run the script — it auto-detects checkpoint.
# Fresh:  python train.py --fresh

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

# ── FIXED: Full TypeFormer architecture ───────────────────────────────────
# model/Model.py      = Full TypeFormer (Temporal + Channel + Block-Recurrent)
# model/Preliminary.py = Ablation baseline (NO Block-Recurrent — wrong)
from model.Model import HARTrans

# ── Configuration ─────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 20
CHECKPOINT_PATH = configs.base_dir + "checkpoint.pt"
FRESH_START = "--fresh" in sys.argv

# ── Device ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("TypeFormer Training — Full Model + KDPrint Data")
print("=" * 60)
print(f"Device:          {device}")
print(f"Architecture:    model/Model.py (Full TypeFormer)")
print(f"Data:            {configs.main_db}")
print(f"Output:          {configs.model_filename}")
print(f"Checkpoint:      {CHECKPOINT_PATH}")
print(f"LR:              0.001")
print(f"Max epochs:      {configs.epochs}")
print(f"Early stopping:  patience={EARLY_STOPPING_PATIENCE}")
print(f"Mode:            {'FRESH START' if FRESH_START else 'auto-detect checkpoint'}")
print("=" * 60)

os.makedirs(configs.base_dir, exist_ok=True)

# Save config for reproducibility
with open("utils/train_config.py") as f:
    config_text = f.read()
with open(configs.base_dir + "experimental_config.txt", mode="w") as f:
    f.write(config_text)

# ── Load data ─────────────────────────────────────────────────────────────
print(f"\nLoading data...")
keystroke_dataset = list(np.load(configs.main_db, allow_pickle=True))
print(f"Loaded {len(keystroke_dataset)} users")

# ── Dataset splits ────────────────────────────────────────────────────────
print(f"  Training:   [{configs.num_training_subjects} : {2*configs.num_training_subjects}]")
print(f"  Validation: [0 : {configs.num_validation_subjects}]")

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
val_dataloader = DataLoader(ds_v, batch_size=configs.batch_size_val, shuffle=True)

# ── Model ─────────────────────────────────────────────────────────────────
print(f"\nInitializing model (Full TypeFormer)...")
TransformerModel = HARTrans(configs).double()

# ── Optimizer & Scheduler ─────────────────────────────────────────────────
optimizer = torch.optim.Adam(
    TransformerModel.parameters(), lr=0.001, betas=(0.9, 0.999)
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
criterion = torch.jit.script(TripletLoss())

# ── Resume or fresh start ────────────────────────────────────────────────
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
    print("Starting from random initialization.")

TransformerModel = TransformerModel.to(device)

# ── Training function ─────────────────────────────────────────────────────
def train_one_epoch():
    TransformerModel.train()
    epoch_eers, total_loss = [], 0.0
    for _, (anchor_sgm, positive_sgm, negative_sgm) in enumerate(
        tqdm(train_dataloader, desc="Training", leave=False)
    ):
        optimizer.zero_grad()
        a = Variable(anchor_sgm).to(device)
        p = Variable(positive_sgm).to(device)
        n = Variable(negative_sgm).to(device)

        a_out = TransformerModel(a)
        p_out = TransformerModel(p)
        n_out = TransformerModel(n)

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
            compute_eer(labels, np.concatenate((scores_g, scores_i)))[0],
            configs.decimals
        )
        epoch_eers.append(eer)
        total_loss += running_loss

    return total_loss, np.round(epoch_eers[-1], configs.decimals)


# ── FIXED: Validation is forward-only (no backward, no optimizer.step) ───
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

            a_out = TransformerModel(a)
            p_out = TransformerModel(p)
            n_out = TransformerModel(n)

            loss = criterion(a_out, p_out, n_out)
            running_loss = np.round(loss.item(), configs.decimals)

            pred_a = np.round(a_out.cpu().numpy(), configs.decimals)
            pred_p = np.round(p_out.cpu().numpy(), configs.decimals)
            pred_n = np.round(n_out.cpu().numpy(), configs.decimals)

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


# ── Training loop ─────────────────────────────────────────────────────────
print(f"\nStarting training (epochs {start_epoch}–{configs.epochs-1}, "
      f"early stopping patience={EARLY_STOPPING_PATIENCE})...")
print(f"{'Epoch':>6} | {'Train Loss':>12} {'Train EER':>10} | "
      f"{'Val Loss':>10} {'Val EER':>9} | {'Time':>7} | {'LR':>10} | Note")
print("-" * 95)

for epoch in range(start_epoch, configs.epochs):
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
        torch.save(TransformerModel.state_dict(), configs.model_filename)
        note = f"✓ best ({100*best_eer_v:.2f}% @ ep{best_epoch})"
    else:
        epochs_no_improve += 1
        note = f"no improve {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}"

    scheduler.step(eer_v)

    print(
        f"{epoch:6d} | {loss_t:12.2f} {100*eer_t:9.2f}% | "
        f"{loss_v:10.2f} {100*eer_v:8.2f}% | {elapsed:6.1f}m | {current_lr:.2e} | {note}"
    )

    # Save checkpoint for resuming (every epoch)
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
    log_list = [loss_t_list, loss_v_list, eer_t_list, eer_v_list]
    with open(configs.log_filename, "w") as output:
        output.write(str(log_list))

    # Early stopping
    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered — "
              f"no improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs.")
        break

print(f"\nTraining complete.")
print(f"Best validation EER: {100*best_eer_v:.2f}% at epoch {best_epoch}")
print(f"Best model saved to: {configs.model_filename}")
print(f"Checkpoint saved to: {CHECKPOINT_PATH}  (re-run to resume)")
