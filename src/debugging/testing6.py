import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.evaluation.adaptive_threshold import AdaptiveThresholdEstimator

# Buat data dummy yang sudah pasti sensitif terhadap k
rng = np.random.default_rng(42)

# User "konsisten": genuine dekat, impostor jauh
n_val = 5
dummy_val = {}
for i in range(n_val):
    center = rng.standard_normal(64)
    enrol = center + rng.standard_normal((5, 64)) * 0.1  # tight cluster
    genuine = center + rng.standard_normal((5, 64)) * 0.1
    impostor = rng.standard_normal((20, 64))  # random, far from center
    dummy_val[str(i)] = {'enrol': enrol, 'genuine': genuine, 'impostor': impostor}

est = AdaptiveThresholdEstimator(k=2.0)
best_k, best_eer = est.optimize_k(dummy_val, k_range=np.arange(0.5, 4.25, 0.25), refine=False)

print(f"best_k = {best_k:.2f}")
# Jika fix benar: hasil berbeda untuk tiap k (tidak semua identik)
# Jika masih bug: semua nilai sama