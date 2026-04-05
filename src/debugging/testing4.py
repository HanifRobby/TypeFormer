import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.preprocessing.kdprint_preprocess import KDPrintPreprocessor

data = np.load(r"data\Mobile_keys_db_6_features.npy", allow_pickle=True)

enrol_raw = [np.array(data[0][s][:, :5], dtype=np.float64) for s in range(5)]
genuine_raw = np.array(data[0][5][:, :5], dtype=np.float64)
impostor_raw = np.array(data[1][0][:, :5], dtype=np.float64)

# Test dengan clip_sigma=5.0
prep = KDPrintPreprocessor(buffer_size=5, seq_len=50, clip_sigma=5.0)
enrol_proc = prep.fit_transform(enrol_raw, use_buffer=True)
gen_proc = prep.transform(genuine_raw)
imp_proc = prep.transform(impostor_raw)

print("=== SETELAH FIX (clip_sigma=5.0) ===")
print(
    f"Enrol timing range:   [{enrol_proc[0][:,:4].min():.3f}, {enrol_proc[0][:,:4].max():.3f}]"
)
print(f"Genuine timing range: [{gen_proc[:,:4].min():.3f}, {gen_proc[:,:4].max():.3f}]")
print(f"Impostor timing range:[{imp_proc[:,:4].min():.3f}, {imp_proc[:,:4].max():.3f}]")
# Yang diharapkan: semua dalam [-5.0, 5.0]

# Cek embedding distances
from src.model.typeformer_wrapper import TypeFormerWrapper

model = TypeFormerWrapper(r"pretrained\TypeFormer_pretrained.pt")
model.model.eval()

z_enrol = model.extract_embeddings(enrol_proc)
z_mean = z_enrol.mean(axis=0)
z_gen = model.extract_single(gen_proc)
z_imp = model.extract_single(imp_proc)

d_gen = np.linalg.norm(z_gen - z_mean)
d_imp = np.linalg.norm(z_imp - z_mean)

print(f"\nGenuine  dist to centroid: {d_gen:.4f}")
print(f"Impostor dist to centroid: {d_imp:.4f}")
print(f"Ratio impostor/genuine:    {d_imp/d_gen:.2f}×")
print()

# Bandingkan ratio dengan E1
print("Target: ratio mendekati E1 (6.6×)")
print("Jika ratio > 3×, fix sudah bekerja dengan baik")
