import ast
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from utils.config import configs


starting_epoch = 0

log_path = Path(configs.log_filename)
if not log_path.exists():
    latest_run_file = Path(configs.latest_run_file)
    if latest_run_file.exists():
        latest_run_id = latest_run_file.read_text(encoding="utf-8").strip()
        candidate = Path(configs.bucket_root_dir) / latest_run_id / f"{configs.model_name}_log.txt"
        if candidate.exists():
            log_path = candidate
if not log_path.exists():
    bucket_logs = sorted(
        Path(configs.bucket_root_dir).glob("*/*_log.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if bucket_logs:
        log_path = bucket_logs[0]
if not log_path.exists():
    raise FileNotFoundError(f"Training log not found in {configs.bucket_root_dir}")

with open(log_path, 'r') as f:
    res = ast.literal_eval(f.read())

num_epochs = len(res[0])
plt.plot(res[3][starting_epoch:num_epochs], label = 'Validation Set ' + configs.model_name, linewidth=1.5, color='m', linestyle = 'dotted')
plt.plot(res[2][starting_epoch:num_epochs], label = 'Training Set ' + configs.model_name, linewidth=1.5, color='m')



plt.ylabel('EER', fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.grid()
plt.ylim([0,1])
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend(fontsize=10)
plt.title('')
plt.show()


t = 0
