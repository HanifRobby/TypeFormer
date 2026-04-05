import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
data = np.load(r'data\Mobile_keys_db_6_features.npy', allow_pickle=True)

print(f"data type: {type(data)}")
print(f"data dtype: {data.dtype}")
print(f"data shape: {data.shape}")
print()

# Cek satu element
user0 = data[0]
print(f"data[0] type: {type(user0)}")
print(f"data[0] shape/len: {user0.shape if hasattr(user0, 'shape') else len(user0)}")
print()

session0 = data[0][0]
print(f"data[0][0] type: {type(session0)}")
print(f"data[0][0] shape: {session0.shape if hasattr(session0, 'shape') else len(session0)}")
print(f"data[0][0] dtype: {session0.dtype}")
print()

# Cek apakah slice menghasilkan list atau array
E = 5
enrol_raw = data[0][:E]
print(f"data[0][:5] type: {type(enrol_raw)}")
print(f"data[0][:5] shape: {enrol_raw.shape if hasattr(enrol_raw, 'shape') else 'no shape attr'}")
print()

# Simulasi fit() dengan enrol_raw aktual
from src.preprocessing.kdprint_preprocess import KDPrintPreprocessor
prep = KDPrintPreprocessor(buffer_size=5, seq_len=50)
prep.fit(enrol_raw)
print(f"mu:    {prep.mu}")
print(f"sigma: {prep.sigma}")
print()

# Cek genuine vs impostor setelah full pipeline
z_enrol_sessions = list(enrol_raw)
enrol_proc = prep.fit_transform(z_enrol_sessions, use_buffer=True)

genuine_test = data[0][5][:, :5]   # session ke-6, user 0
impostor_test = data[1][0][:, :5]  # session ke-1, user 1

gen_proc  = prep.transform(genuine_test, use_buffer=False)
imp_proc  = prep.transform(impostor_test, use_buffer=False)

print(f"Genuine  processed hold_lat — mean: {gen_proc[:,0].mean():+.4f}, std: {gen_proc[:,0].std():.4f}")
print(f"Impostor processed hold_lat — mean: {imp_proc[:,0].mean():+.4f}, std: {imp_proc[:,0].std():.4f}")
print()

# Cek embedding dari model (tambahkan jika model sudah bisa diload)
# Ini bagian terpenting — apakah embedding genuine dan impostor benar-benar berbeda?
from src.model.typeformer_wrapper import TypeFormerWrapper
model = TypeFormerWrapper('pretrained/TypeFormer_pretrained.pt')
z_enrol = model.extract_embeddings(enrol_proc)
z_mean  = z_enrol.mean(axis=0)
z_gen   = model.extract_single(gen_proc)
z_imp   = model.extract_single(imp_proc)
print(f"Genuine  dist to centroid: {np.linalg.norm(z_gen - z_mean):.4f}")
print(f"Impostor dist to centroid: {np.linalg.norm(z_imp - z_mean):.4f}")