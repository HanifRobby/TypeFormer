import numpy as np

# Load data Anda (sesuaikan path)
data = np.load(r'data\Mobile_keys_db_6_features.npy', allow_pickle=True)
# data shape: (n_users, n_sessions, seq_len, 6)

# Ambil satu session dari user 0
session = data[0][0][:, :5]  # (50, 5)

print("=== CEK RANGE DATA ===")
print(f"Timing col 0 (hold_lat):  min={session[:,0].astype(float).min():.6f}, max={session[:,0].astype(float).max():.6f}, mean={session[:,0].astype(float).mean():.6f}")
print(f"Timing col 1 (inter_key): min={session[:,1].astype(float).min():.6f}, max={session[:,1].astype(float).max():.6f}, mean={session[:,1].astype(float).mean():.6f}")
print(f"Timing col 2 (press_lat): min={session[:,2].astype(float).min():.6f}, max={session[:,2].astype(float).max():.6f}, mean={session[:,2].astype(float).mean():.6f}")
print(f"Timing col 3 (rel_lat):   min={session[:,3].astype(float).min():.6f}, max={session[:,3].astype(float).max():.6f}, mean={session[:,3].astype(float).mean():.6f}")
print(f"ASCII col 4:              min={session[:,4].min():.6f}, max={session[:,4].max():.6f}")
print()

# Simulasi fit()
enrol = [data[0][s][:, :5] for s in range(5)]
all_timing = np.vstack([s[:, :4] for s in enrol]).astype(float)
mu    = all_timing.mean(axis=0)
sigma = all_timing.std(axis=0)

print("=== STATISTIK FIT() ===")
for i, name in enumerate(['hold', 'inter', 'press', 'rel']):
    print(f"  {name}: mu={mu[i]:.6f}, sigma={sigma[i]:.6f}")
print()

# Simulasi standardize() pada genuine dan impostor
genuine_session  = data[0][5][:, :5]   # session ke-6 dari user yang sama
impostor_session = data[1][0][:, :5]   # session dari user berbeda

gen_std = (genuine_session[:, :4].astype(float) - mu) / sigma
imp_std = (impostor_session[:, :4].astype(float) - mu) / sigma

print("=== HASIL STANDARDISASI ===")
print(f"  Genuine  hold_lat mean: {gen_std[:, 0].mean():+.4f}σ")
print(f"  Impostor hold_lat mean: {imp_std[:, 0].mean():+.4f}σ")
print()
print("Interpretasi yang benar: genuine ≈ 0, impostor jauh dari 0")
print("Jika keduanya besar atau keduanya dekat 0, data sudah dinormalisasi")