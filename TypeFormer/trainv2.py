import os
import json
from multiprocessing import freeze_support
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.misc import KeystrokeSessionTriplet
from utils.config import configs
from utils.misc import compute_eer, TripletLoss

import time
from datetime import datetime, timezone


from model.Model import HARTrans


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

use_cuda = device.type == "cuda"
pin_memory = use_cuda

train_dataloader = None
val_dataloader = None
TransformerModel = None
optimizer = None
criterion = None


def setup_training():
    global train_dataloader, val_dataloader, TransformerModel, optimizer, criterion

    os.makedirs(configs.base_dir, exist_ok=True)
    os.makedirs(configs.bucket_root_dir, exist_ok=True)
    with open(configs.latest_run_file, "w", encoding="utf-8") as latest_file:
        latest_file.write(configs.run_id)

    # Saving specific config file for reproducibility
    with open('utils/config.yaml') as f:
        data = f.read()
        f.close()
    with open(configs.base_dir + "experimental_config.txt", mode="w") as f:
        f.write(data)
        f.close()
    run_metadata = {
        "run_id": configs.run_id,
        "output_bucket": configs.output_bucket,
        "script": "trainv2.py",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "model_filename": configs.model_filename,
        "log_filename": configs.log_filename,
        "dataset": configs.main_db,
        "batch_size_train": configs.batch_size_train,
        "batch_size_val": configs.batch_size_val,
        "batches_per_epoch": configs.batches_per_epoch,
        "val_batches_per_epoch": configs.val_batches_per_epoch,
        "epochs": configs.epochs,
    }
    with open(configs.run_metadata_filename, "w", encoding="utf-8") as metadata_file:
        json.dump(run_metadata, metadata_file, indent=2)

    keystroke_dataset = list(np.load(configs.main_db, allow_pickle=True))
    train_users = keystroke_dataset[configs.num_training_subjects:2 * configs.num_training_subjects]
    val_users = keystroke_dataset[:configs.num_validation_subjects]
    del keystroke_dataset

    ds_t = KeystrokeSessionTriplet(
        train_users,
        data_length=configs.sequence_length,
        length=max(configs.batch_size_train * configs.batches_per_epoch, 1),
    )
    ds_v = KeystrokeSessionTriplet(
        val_users,
        data_length=configs.sequence_length,
        length=max(configs.batch_size_val * configs.val_batches_per_epoch, 1),
    )

    num_workers = int(getattr(configs, "num_workers", 4))
    if os.name == "nt" and num_workers > 0:
        print("Windows detected: overriding num_workers to 0 to avoid dataset duplication across workers.")
        num_workers = 0

    dataloader_common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dataloader_common_kwargs["persistent_workers"] = True
        dataloader_common_kwargs["prefetch_factor"] = 2

    train_dataloader = DataLoader(
        ds_t,
        batch_size=configs.batch_size_train,
        shuffle=True,
        **dataloader_common_kwargs
    )
    val_dataloader = DataLoader(
        ds_v,
        batch_size=configs.batch_size_val,
        shuffle=True,
        **dataloader_common_kwargs
    )

    TransformerModel = HARTrans(configs).float().to(device)
    optimizer = torch.optim.Adam(TransformerModel.parameters(), lr=configs.lr, betas=configs.betas)
    criterion = TripletLoss().to(device)


def inner_ops(input_, mode='train'):
    if mode == 'train':
        optimizer.zero_grad(set_to_none=True)
    anchor_sgm, positive_sgm, negative_sgm = (
        input_[0].to(device, dtype=torch.float32, non_blocking=pin_memory),
        input_[1].to(device, dtype=torch.float32, non_blocking=pin_memory),
        input_[2].to(device, dtype=torch.float32, non_blocking=pin_memory),
    )
    anchor_out, positive_out, negative_out = (TransformerModel(anchor_sgm),
                                              TransformerModel(positive_sgm),
                                              TransformerModel(negative_sgm))
    loss = criterion(anchor_out, positive_out, negative_out)
    if mode == 'train':
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        scores_g = torch.norm(anchor_out - positive_out, dim=1).cpu().numpy()
        scores_i = torch.norm(anchor_out - negative_out, dim=1).cpu().numpy()

    return loss.item(), scores_g, scores_i


def compute_epoch_eer(genuine_scores, impostor_scores):
    labels = np.array([0 for _ in range(len(genuine_scores))] + [1 for _ in range(len(impostor_scores))])
    scores = np.concatenate((genuine_scores, impostor_scores))
    eer = compute_eer(labels, scores)[0]
    return np.round(eer, configs.decimals)


def train_one_epoch(epoch):
    # Make sure gradient tracking is on, and do a pass over the data
    TransformerModel.train()
    losses = []
    epoch_genuine_scores = []
    epoch_impostor_scores = []
    train_steps = min(len(train_dataloader), int(getattr(configs, "batches_per_epoch", len(train_dataloader))))
    train_steps = max(train_steps, 1)

    train_bar = tqdm(
        zip(range(train_steps), train_dataloader),
        total=train_steps,
        desc=f"Epoch {epoch + 1}/{configs.epochs} [Train]",
        leave=False,
        dynamic_ncols=True,
    )
    for _, (anchor_sgm, positive_sgm, negative_sgm) in train_bar:
        running_loss_, scores_g, scores_i = inner_ops((anchor_sgm, positive_sgm, negative_sgm), mode='train')
        losses.append(running_loss_)
        epoch_genuine_scores.append(scores_g)
        epoch_impostor_scores.append(scores_i)
        train_bar.set_postfix(loss=f"{np.mean(losses):.4f}")

    mean_loss = np.round(float(np.mean(losses)), configs.decimals)
    epoch_eer = compute_epoch_eer(np.concatenate(epoch_genuine_scores), np.concatenate(epoch_impostor_scores))
    return mean_loss, epoch_eer

def eval_one_epoch(epoch):
    losses = []
    epoch_genuine_scores = []
    epoch_impostor_scores = []
    TransformerModel.eval()
    val_steps = min(len(val_dataloader), int(getattr(configs, "val_batches_per_epoch", len(val_dataloader))))
    val_steps = max(val_steps, 1)
    val_bar = tqdm(
        zip(range(val_steps), val_dataloader),
        total=val_steps,
        desc=f"Epoch {epoch + 1}/{configs.epochs} [Val]",
        leave=False,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for _, (anchor_sgm, positive_sgm, negative_sgm) in val_bar:
            running_loss_, scores_g, scores_i = inner_ops((anchor_sgm, positive_sgm, negative_sgm), mode='eval')
            losses.append(running_loss_)
            epoch_genuine_scores.append(scores_g)
            epoch_impostor_scores.append(scores_i)
            val_bar.set_postfix(loss=f"{np.mean(losses):.4f}")
    mean_loss = np.round(float(np.mean(losses)), configs.decimals)
    mean_eer = compute_epoch_eer(np.concatenate(epoch_genuine_scores), np.concatenate(epoch_impostor_scores))
    return mean_loss, mean_eer



def run_training():
    setup_training()

    best_eer_v = 100.
    best_epoch, new_best_epoch = 0, False

    loss_t_list, eer_t_list = [], []
    loss_v_list, eer_v_list = [], []

    for epoch in range(configs.epochs):
        start = time.time()

        loss_t, eer_t = train_one_epoch(epoch)
        loss_t_list.append(loss_t)
        eer_t_list.append(eer_t)

        loss_v, eer_v = eval_one_epoch(epoch)
        loss_v_list.append(loss_v)
        eer_v_list.append(eer_v)

        end = time.time()
        if eer_v_list[-1] < best_eer_v:
            new_best_epoch, best_eer_v, best_epoch = True, eer_v_list[-1], epoch
            torch.save(TransformerModel.state_dict(), configs.model_filename)
        else:
            new_best_epoch = False
        print('Epoch: %d. Training set: Loss: %.2f, EER [%%]: %.2f%%. Validation set: Loss: %.2f, EER [%%]: %.2f%%. '
              'Time for last epoch [min]: %.2f. New best EER on val set: %.d'
              % (epoch, loss_t_list[-1], 100*eer_t_list[-1], loss_v_list[-1], 100*eer_v_list[-1],
                 np.round((end-start)/60, configs.decimals), new_best_epoch))
        log_list = [loss_t_list, loss_v_list, eer_t_list, eer_v_list]
        with open(configs.log_filename, "w") as output:
            output.write(str(log_list))

    run_metadata = {}
    if os.path.exists(configs.run_metadata_filename):
        with open(configs.run_metadata_filename, "r", encoding="utf-8") as metadata_file:
            run_metadata = json.load(metadata_file)
    run_metadata.update({
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_validation_eer": float(best_eer_v),
        "best_epoch": int(best_epoch),
        "model_filename": configs.model_filename,
        "log_filename": configs.log_filename,
        "dataset": configs.main_db,
    })
    with open(configs.run_metadata_filename, "w", encoding="utf-8") as metadata_file:
        json.dump(run_metadata, metadata_file, indent=2)

    print('\nBest Validation EER: %.2f%%, in epoch: %.d' % (best_eer_v, best_epoch))


if __name__ == "__main__":
    freeze_support()
    run_training()
