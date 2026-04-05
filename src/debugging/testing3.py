import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.preprocessing.kdprint_preprocess import KDPrintPreprocessor

data = np.load(r'data\Mobile_keys_db_6_features.npy', allow_pickle=True)

# Ambil enrolment dan test
enrol_raw  = [np.array(data[0][s][:, :5], dtype=np.float64) for s in range(5)]
genuine_raw  = np.array(data[0][5][:, :5], dtype=np.float64)
impostor_raw = np.array(data[1][0][:, :5], dtype=np.float64)

prep = KDPrintPreprocessor(buffer_size=5, seq_len=50)
enrol_proc  = prep.fit_transform(enrol_raw, use_buffer=True)
gen_proc    = prep.transform(genuine_raw, use_buffer=False)
imp_proc    = prep.transform(impostor_raw, use_buffer=False)

print("=== NILAI SETELAH PREPROCESSING ===")
print(f"Enrol[0] timing range:   [{enrol_proc[0][:, :4].min():.3f}, {enrol_proc[0][:, :4].max():.3f}]")
print(f"Genuine  timing range:   [{gen_proc[:, :4].min():.3f}, {gen_proc[:, :4].max():.3f}]")
print(f"Impostor timing range:   [{imp_proc[:, :4].min():.3f}, {imp_proc[:, :4].max():.3f}]")
print()

# Bandingkan dengan E1 (raw)
from src.preprocessing.typeformer_preprocess import typeformer_preprocess
enrol_raw_e1  = [np.array(data[0][s][:, :5], dtype=np.float64) for s in range(5)]
enrol_e1      = typeformer_preprocess(enrol_raw_e1, seq_len=50)
gen_e1        = typeformer_preprocess([genuine_raw], seq_len=50)[0]
imp_e1        = typeformer_preprocess([impostor_raw], seq_len=50)[0]

print("=== E1 (RAW) vs E2 (KDPRINT) — PERBANDINGAN INPUT KE MODEL ===")
print(f"E1 enrol[0] timing range: [{enrol_e1[0][:, :4].min():.4f}, {enrol_e1[0][:, :4].max():.4f}]")
print(f"E2 enrol[0] timing range: [{enrol_proc[0][:, :4].min():.4f}, {enrol_proc[0][:, :4].max():.4f}]")
print()
print(f"E1 genuine timing range:  [{gen_e1[:, :4].min():.4f}, {gen_e1[:, :4].max():.4f}]")
print(f"E2 genuine timing range:  [{gen_proc[:, :4].min():.4f}, {gen_proc[:, :4].max():.4f}]")
print()
print(f"E1 impostor timing range: [{imp_e1[:, :4].min():.4f}, {imp_e1[:, :4].max():.4f}]")
print(f"E2 impostor timing range: [{imp_proc[:, :4].min():.4f}, {imp_proc[:, :4].max():.4f}]")
print()

# Cek embedding - INI BAGIAN TERPENTING
from src.model.typeformer_wrapper import TypeFormerWrapper
model = TypeFormerWrapper(r'pretrained\TypeFormer_pretrained.pt', device='auto')
model.model.eval()

z_enrol_e1 = model.extract_embeddings(enrol_e1)
z_enrol_e2 = model.extract_embeddings(enrol_proc)

z_gen_e1   = model.extract_single(gen_e1)
z_gen_e2   = model.extract_single(gen_proc)
z_imp_e1   = model.extract_single(imp_e1)
z_imp_e2   = model.extract_single(imp_proc)

z_mean_e1  = z_enrol_e1.mean(axis=0)
z_mean_e2  = z_enrol_e2.mean(axis=0)

print("=== EMBEDDING DISTANCES ===")
print(f"E1 — Genuine  dist to centroid: {np.linalg.norm(z_gen_e1 - z_mean_e1):.4f}")
print(f"E1 — Impostor dist to centroid: {np.linalg.norm(z_imp_e1 - z_mean_e1):.4f}")
print()
print(f"E2 — Genuine  dist to centroid: {np.linalg.norm(z_gen_e2 - z_mean_e2):.4f}")
print(f"E2 — Impostor dist to centroid: {np.linalg.norm(z_imp_e2 - z_mean_e2):.4f}")
print()
print("Yang diharapkan: genuine < impostor untuk kedua E1 dan E2")
print("Jika E2 genuine > impostor atau keduanya sama → distribution shift terlalu ekstrem")
print()

# Cek embedding range
print("=== EMBEDDING VALUE RANGE ===")
print(f"E1 z_enrol range: [{z_enrol_e1.min():.4f}, {z_enrol_e1.max():.4f}]")
print(f"E2 z_enrol range: [{z_enrol_e2.min():.4f}, {z_enrol_e2.max():.4f}]")
print()
print(f"E1 centroid norm: {np.linalg.norm(z_mean_e1):.4f}")
print(f"E2 centroid norm: {np.linalg.norm(z_mean_e2):.4f}")