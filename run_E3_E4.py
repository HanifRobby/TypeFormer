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
# Start with E=5 for sanity check as per user instruction
E_VALUES = [5]
L = 50
BUFFER_SIZE = 5

print("Loading model...")
model = TypeFormerWrapper(WEIGHTS_PATH, device='auto', batch_size=128)

print("Loading preprocessed dataset...")
# Load the dataset
dataset = np.load(DATA_PATH, allow_pickle=True)

# Dataset organization (as extracted from TypeFormer original config):
VAL_USERS = dataset[0 : 400]
TEST_USERS = dataset[400 : 1400]

val_users_clean = [u[:15] for u in VAL_USERS]
test_users_clean = [u[:15] for u in TEST_USERS]

all_results = {}

for E in E_VALUES:
    print(f"\n{'#'*60}")
    print(f"# Running all configs for E={E}")
    print(f"{'#'*60}")
    
    r3 = run_single_experiment(
        'E3_adaptive', model, test_users_clean, val_users_clean,
        E=E, L=L, use_kdprint=False, use_adaptive=True, output_dir=RESULTS_DIR
    )
    
    
    # r4 = run_single_experiment(
    #     'E4_full_system', model, test_users_clean, val_users_clean,
    #     E=E, L=L, use_kdprint=True, use_adaptive=True, buffer_size=BUFFER_SIZE, output_dir=RESULTS_DIR
    # )
    
    all_results[E] = {'E3': r3}

print("\n\nSUMMARY TABLE (Average EER %)")
print(f"{'E':<5} {'E3 (Adaptive)':>15}")
print("-" * 40)
for E, configs in all_results.items():
    row = f"{E:<5}"
    for config in ['E3']:
        eer_pct = configs[config]['avg_eer'] * 100
        row += f"{eer_pct:>15.4f}"
    print(row)

with open(f'{RESULTS_DIR}E3_E4_summary.json', 'w') as f:
    json.dump(all_results, f, indent=2)
