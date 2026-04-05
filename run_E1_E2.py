import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model.typeformer_wrapper import TypeFormerWrapper
from src.experiments.run_experiment import run_single_experiment

WEIGHTS_PATH = 'pretrained/TypeFormer_pretrained.pt'
DATA_PATH = 'data/Mobile_keys_db_6_features.npy'
RESULTS_DIR = 'results/'
# E_VALUES = [1, 2, 5, 7, 10]
E_VALUES = [5]
L = 50
BUFFER_SIZE = 5

print("Loading model...")
model = TypeFormerWrapper(WEIGHTS_PATH, device='auto', batch_size=128)

print("Loading preprocessed dataset...")
# Load the dataset. Shape is likely something like (N_users, N_sessions, L, features)
# Each user is a sequence/list of sessions.
dataset = np.load(DATA_PATH, allow_pickle=True)

# Dataset organization (as extracted from TypeFormer original config):
# 60k total users. 30k train. 400 val. 1000 test.
# The `test.py` accesses val users starting right at `dataset[ configs.num_validation_subjects : ... ]`
# Wait, train_config says `num_validation_subjects = 400`, meaning index [0:400] are val, and [400:1400] are test.
# Let's cleanly separate them as per validation.
VAL_USERS = dataset[0 : 400]
TEST_USERS = dataset[400 : 1400]

# Ensure each user has at least max(E) + 5 = 15 sessions
# We will truncate to 15 sessions just to be consistent.
val_users_clean = [u[:15] for u in VAL_USERS]
test_users_clean = [u[:15] for u in TEST_USERS]

all_results = {}

for E in E_VALUES:
    print(f"\n{'#'*60}")
    print(f"# Running all configs for E={E}")
    print(f"{'#'*60}")
    
    r1 = run_single_experiment(
        'E1_baseline', model, test_users_clean, val_users_clean,
        E=E, L=L, use_kdprint=False, output_dir=RESULTS_DIR
    )
    
    r2 = run_single_experiment(
        'E2_preprocessing', model, test_users_clean, val_users_clean,
        E=E, L=L, use_kdprint=True, buffer_size=BUFFER_SIZE, output_dir=RESULTS_DIR
    )
    
    all_results[E] = {'E1': r1, 'E2': r2}

print("\n\nSUMMARY TABLE (Average EER %)")
print(f"{'E':<5} {'E1 (Baseline)':>15} {'E2 (Preproc)':>15}")
print("-" * 40)
for E, configs in all_results.items():
    row = f"{E:<5}"
    for config in ['E1', 'E2']:
        eer_pct = configs[config]['avg_eer'] * 100
        row += f"{eer_pct:>15.4f}"
    print(row)

with open(f'{RESULTS_DIR}E1_E2_summary.json', 'w') as f:
    json.dump(all_results, f, indent=2)
