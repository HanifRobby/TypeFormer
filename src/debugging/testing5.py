import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.preprocessing.kdprint_preprocess import KDPrintPreprocessor
from src.model.typeformer_wrapper import TypeFormerWrapper

data = np.load(r"data\Mobile_keys_db_6_features.npy", allow_pickle=True)
model = TypeFormerWrapper(r"pretrained\TypeFormer_pretrained.pt")
model.model.eval()

enrol_raw = [np.array(data[0][s][:, :5], dtype=np.float64) for s in range(5)]
genuine_raw = np.array(data[0][5][:, :5], dtype=np.float64)
impostor_raw = np.array(data[1][0][:, :5], dtype=np.float64)

print(f"{'clip_sigma':>12} {'d_genuine':>12} {'d_impostor':>12} {'ratio':>8}")
print("-" * 50)

for clip in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, None]:
    prep = KDPrintPreprocessor(
        buffer_size=5, seq_len=50, clip_sigma=clip if clip else 999
    )
    enrol_proc = prep.fit_transform(enrol_raw, use_buffer=True)
    gen_proc = prep.transform(genuine_raw)
    imp_proc = prep.transform(impostor_raw)

    z_enrol = model.extract_embeddings(enrol_proc)
    z_mean = z_enrol.mean(axis=0)
    d_gen = float(np.linalg.norm(model.extract_single(gen_proc) - z_mean))
    d_imp = float(np.linalg.norm(model.extract_single(imp_proc) - z_mean))

    label = f"{clip}" if clip else "no clip"
    print(f"{label:>12} {d_gen:>12.4f} {d_imp:>12.4f} {d_imp/d_gen:>8.2f}×")
