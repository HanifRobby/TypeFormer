# Implementation Guide — Tesis Mobile Keystroke Authentication

**Proyek**: Pengaruh Score Normalization dan User-Statistic Conditioning terhadap Performa TypeFormer dalam Autentikasi Pengetikan Mobile

**Versi**: 1.0 — *living document*, perbarui setiap kali keputusan desain berubah berdasarkan hasil diagnostik atau diskusi pembimbing.

**Status verifikasi**: dokumen ini mencampur (a) fakta yang terverifikasi dari literatur, (b) keputusan desain yang sudah Anda dan saya diskusikan, dan (c) rekomendasi praktis yang merupakan transfer rasional dari domain tetangga. Saya tandai eksplisit setiap kelas.

---

## Daftar Isi

1. [Tujuan dan Ruang Lingkup](#1-tujuan-dan-ruang-lingkup)
2. [Asumsi dan Batasan Implementasi](#2-asumsi-dan-batasan-implementasi)
3. [Tech Stack dan Dependency](#3-tech-stack-dan-dependency)
4. [Environment Setup](#4-environment-setup)
5. [Struktur Direktori Proyek](#5-struktur-direktori-proyek)
6. [Konvensi Penamaan](#6-konvensi-penamaan)
7. [Aturan dan Standar Pengembangan](#7-aturan-dan-standar-pengembangan)
8. [Alur Kerja Pengembangan](#8-alur-kerja-pengembangan)
9. [Detail Implementasi per Modul](#9-detail-implementasi-per-modul)
10. [Pseudocode Snippet untuk Komponen Kunci](#10-pseudocode-snippet-untuk-komponen-kunci)
11. [Best Practice](#11-best-practice)
12. [Catatan Teknis yang Belum Terverifikasi](#12-catatan-teknis-yang-belum-terverifikasi)
13. [Checklist Implementasi](#13-checklist-implementasi)

---

## 1. Tujuan dan Ruang Lingkup

### 1.1 Tujuan teknis

Membangun sistem evaluasi reproducible untuk menguji **dua kontribusi metodologis** pada arsitektur TypeFormer dengan dataset Aalto Mobile Keystroke:

1. **AS-Norm (Adaptive Symmetric Score Normalization)** — kalibrasi skor berbasis cohort sebagai pengganti adaptive threshold per-pengguna.
2. **FiLM (Feature-wise Linear Modulation) head** — conditioning embedding berdasarkan statistik pengguna, sebagai pengganti per-user z-score standardization yang sebelumnya gagal.

### 1.2 Ruang lingkup IN

- Reproduksi baseline TypeFormer pada protokol asli (1.000 subjek evaluasi, 15 sesi, E∈{1,2,5,7,10}).
- Implementasi AS-Norm pipeline dengan cohort-based score calibration.
- Implementasi FiLM head di atas TypeFormer beku (frozen backbone).
- Pelaporan **dua metrik**: mean per-subject EER (gaya TypeFormer) dan global EER (gaya KVC).
- Uji signifikansi statistik berpasangan (Wilcoxon signed-rank + bootstrap CI).
- Eksperimen diagnostik (D1, D2, D3) sebagai validasi feasibility sebelum komitmen metode utama.
- Eksperimen ablasi yang menjelaskan kontribusi setiap komponen.

### 1.3 Ruang lingkup OUT (tidak dilakukan di tesis ini)

- Modifikasi arsitektur backbone TypeFormer (tetap beku).
- Reimplementasi Set2Set loss (cadangan, hanya jika AS-Norm gagal).
- Replikasi pada KVC protokol resmi sebagai eksperimen utama (hanya sebagai validasi eksternal jika akses CodaLab/Codabench disetujui sebelum sidang).
- TAS-Norm (Trainable AS-Norm) — disebut sebagai future work.
- Eksperimen cross-database (BehavePassDB, dll).

---

## 2. Asumsi dan Batasan Implementasi

### 2.1 Asumsi tentang data

- **Dataset Aalto Mobile** sudah dipreprocess ke format yang kompatibel dengan TypeFormer (5-channel input: HL, IL, PL, RL, ASCII/255).
- Setiap sesi memiliki sekuens dengan panjang L=50 (zero-padded di akhir untuk sesi pendek, truncated untuk sesi panjang).
- Split: 30.000 subjek training, 400 validation, 1.000 test, sisanya (~28.600) tersedia untuk **cohort pool**.
- Pengguna pada cohort pool **disjoint** dari training, validation, dan test.

### 2.2 Asumsi tentang model

- **Pretrained TypeFormer weights tersedia** dari BiDAlab/TypeFormer repository. *[VERIFIKASI: konfirmasi akses dengan menghubungi penulis paper jika diperlukan.]*
- Output embedding TypeFormer berdimensi **64** (sesuai paper Stragapede et al. 2024). *[VERIFIKASI: cek di config resmi.]*
- Backbone **dibekukan** (`requires_grad=False`) selama training FiLM. Hanya FiLM head yang dapat dilatih.

### 2.3 Batasan komputasi

- Target: setiap eksperimen tunggal selesai dalam < 1 jam pada satu GPU modern (A100 / RTX 3090 / setara).
- Total eksperimen utama: ~90 runs (12 konfigurasi × 5 nilai E + ablasi). Estimasi total wall-clock: 1–2 minggu dengan parallelization sederhana.

### 2.4 Batasan epistemik

- **Beberapa hyperparameter** (K untuk AS-Norm, weight decay untuk FiLM, ukuran cohort) **harus di-sweep pada validation set** — tidak ada nilai default yang dapat diandalkan dari literatur untuk konteks keystroke.
- **Baseline global EER pada TypeFormer 1.000 subjek tidak tersedia di literatur**. Harus diukur sendiri (Diagnostik D3).
- **Target performa AS-Norm/FiLM** bersifat hipotesis kerja, harus dikalibrasi ulang setelah D3 selesai.

---

## 3. Tech Stack dan Dependency

### 3.1 Bahasa dan framework utama

| Komponen | Pilihan | Justifikasi |
|---|---|---|
| Bahasa | Python 3.10+ | Standar deep learning, kompatibel dengan kode TypeFormer asli |
| Deep learning | PyTorch 2.0+ | TypeFormer asli ditulis dalam PyTorch |
| Data processing | NumPy, pandas | Standar |
| Metrik biometrik | scikit-learn (untuk ROC/EER) | Reproducible, banyak dipakai literatur |
| Statistik | SciPy (Wilcoxon), statsmodels (multipletests) | Standar |
| Konfigurasi | YAML (pyyaml) atau Hydra | Reproducible experiment configs |
| Eksperiment tracking | Weights & Biases ATAU MLflow ATAU plain CSV+JSON | Pilih satu, jangan campur |
| Visualisasi | matplotlib, seaborn | Plot DET curve, histogram |

### 3.2 Dependency list (requirements.txt)

```text
# Core ML
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Configuration
pyyaml>=6.0
# OR if using Hydra:
# hydra-core>=1.3.0

# Statistical analysis
statsmodels>=0.14.0

# Optional: experiment tracking (pilih satu)
# wandb>=0.15.0
# mlflow>=2.5.0

# Utilities
tqdm>=4.65.0
joblib>=1.3.0  # untuk caching embedding

# Development
pytest>=7.4.0
black>=23.0.0
isort>=5.12.0
mypy>=1.5.0  # optional, untuk type checking
```

**Catatan**: jangan pakai versi terbaru tanpa cek. Tipikal masalah: PyTorch 2.x ada breaking changes untuk beberapa pattern. *[Saran: gunakan PyTorch versi yang sama dengan paper TypeFormer asli kalau memungkinkan. Cek README BiDAlab.]*

### 3.3 Dependency yang TIDAK dibutuhkan

Hindari menambahkan ini tanpa alasan kuat:
- `transformers` (Hugging Face) — TypeFormer custom, bukan dari HF.
- `pytorch-lightning` — overhead untuk proyek ini; manual training loop cukup.
- `tensorflow` — tidak relevan.

---

## 4. Environment Setup

### 4.1 Setup Python environment

```bash
# Buat virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verifikasi PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### 4.2 Verifikasi GPU

Minimum requirement:
- GPU dengan ≥ 8 GB VRAM untuk backbone forward pass dengan batch 32.
- ≥ 16 GB VRAM disarankan untuk batch lebih besar.

```bash
nvidia-smi  # cek GPU dan driver
```

### 4.3 Akses data

Aalto Mobile Keystroke Database tidak publik. Jalur akses:

1. **Via BiDAlab**: TypeFormer paper merujuk implementasi resmi di GitHub BiDAlab. README mungkin menjelaskan cara request akses data preprocessed.
2. **Via Aalto langsung**: paper Palin et al. 2019 — kontak penulis untuk raw data.

*[VERIFIKASI: konfirmasi metode akses yang valid sebelum mulai. Tanpa data, semua langkah lain tidak dapat dimulai.]*

### 4.4 Akses pretrained TypeFormer weights

*[VERIFIKASI: cek apakah weights publicly downloadable atau perlu request.]* Lokasi yang harus dicek:
- `github.com/BiDAlab/TypeFormer` — README dan releases section.
- Issue tracker untuk request access.

### 4.5 Struktur penyimpanan data

```
data/
├── raw/                          # Read-only, jangan diubah
│   ├── aalto_mobile/
│   │   └── (raw CSV / format asli)
│   └── pretrained/
│       └── typeformer_weights.pt # Pretrained weights
├── processed/                     # Hasil preprocessing
│   ├── train_sessions.npz        # 30.000 users × 15 sessions
│   ├── val_sessions.npz          # 400 users × 15 sessions
│   ├── test_sessions.npz         # 1.000 users × 15 sessions
│   └── cohort_pool.npz           # Pool ~28.600 users untuk cohort selection
└── splits/
    ├── train_user_ids.txt
    ├── val_user_ids.txt
    ├── test_user_ids.txt
    └── cohort_user_ids.txt       # Subset yang dipilih sebagai cohort tetap
```

---

## 5. Struktur Direktori Proyek

```
thesis-typeformer/
├── README.md                      # Quick start untuk reproduksi
├── implementation_guide.md        # Dokumen ini
├── requirements.txt
├── pyproject.toml                 # Optional: untuk pip install -e .
├── .gitignore
├── .env.example                   # Template untuk path data, API keys
│
├── config/                        # Konfigurasi eksperimen (YAML)
│   ├── base.yaml                  # Konfigurasi default
│   ├── experiments/
│   │   ├── b0_baseline.yaml       # B0: replikasi TypeFormer
│   │   ├── b1_centroid_eucl.yaml
│   │   ├── b2_centroid_cosine.yaml
│   │   ├── c1_threshold_param.yaml
│   │   ├── c2_threshold_perc.yaml
│   │   ├── c3_threshold_mno.yaml
│   │   ├── n1_znorm.yaml
│   │   ├── n2_tnorm.yaml
│   │   ├── n3_snorm.yaml
│   │   ├── n4_asnorm.yaml         # Konfigurasi utama AS-Norm
│   │   ├── f1_film_only.yaml
│   │   └── fn_full_system.yaml    # Konfigurasi utama FiLM + AS-Norm
│   └── diagnostics/
│       ├── d1_heterogeneity.yaml
│       ├── d2_stability.yaml
│       └── d3_baseline_global.yaml
│
├── data/                          # Lihat Section 4.5
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── aalto_loader.py        # Load preprocessed Aalto data
│   │   ├── kvc_loader.py          # KVC format (untuk fase 2 future)
│   │   ├── sequence_processor.py  # Padding, truncation, feature normalization
│   │   └── cohort_sampler.py      # Sampling cohort dari pool
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── typeformer_wrapper.py  # Wrapper untuk frozen TypeFormer
│   │   ├── film_head.py           # FiLM conditioning head
│   │   └── full_model.py          # Compose backbone + optional FiLM
│   │
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── base_scorer.py         # Abstract base class
│   │   ├── distance.py            # Euclidean, cosine helpers
│   │   ├── cosine_raw.py          # Skor cosine mentah
│   │   ├── znorm.py
│   │   ├── tnorm.py
│   │   ├── snorm.py
│   │   └── asnorm.py              # Implementasi inti
│   │
│   ├── thresholds/
│   │   ├── __init__.py
│   │   ├── global_threshold.py
│   │   ├── per_user_parametric.py
│   │   ├── per_user_percentile.py
│   │   └── per_user_min_non_outlier.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training loop untuk FiLM
│   │   ├── triplet_loss.py
│   │   ├── sampler.py             # Multi-E triplet sampler
│   │   ├── callbacks.py           # Early stopping, checkpointing
│   │   └── optimizer.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # EER, FAR, FRR, balanced accuracy
│   │   ├── per_subject_eer.py
│   │   ├── global_eer.py
│   │   ├── operating_points.py    # FAR@1%FRR, dll
│   │   └── bootstrap_ci.py        # Bootstrap confidence interval
│   │
│   ├── statistics/
│   │   ├── __init__.py
│   │   ├── wilcoxon.py            # Wilcoxon signed-rank dengan Bonferroni
│   │   └── user_stats.py          # Compute s_u dari sesi enrolment
│   │
│   └── utils/
│       ├── __init__.py
│       ├── seeds.py               # Reproducibility helpers
│       ├── logging.py             # Logger setup
│       ├── caching.py             # Embedding cache ke disk
│       └── config_loader.py       # YAML config loading
│
├── scripts/                       # Entry points untuk eksperimen
│   ├── 01_preprocess_data.py
│   ├── 02_reproduce_baseline.py   # B0
│   ├── 03_diagnostic_d1.py        # Heterogenitas σ_u
│   ├── 04_diagnostic_d2.py        # Stabilitas s_u
│   ├── 05_diagnostic_d3.py        # Baseline global EER
│   ├── 06_train_film.py           # Training FiLM head
│   ├── 07_evaluate_single.py      # Evaluasi satu konfigurasi
│   ├── 08_run_main_matrix.py      # Orchestrate semua konfigurasi
│   ├── 09_statistical_analysis.py # Wilcoxon + bootstrap
│   └── 10_generate_tables.py      # Generate tabel untuk tesis
│
├── notebooks/                     # Eksplorasi, jangan jadikan source of truth
│   ├── exploration.ipynb
│   └── visualization.ipynb
│
├── results/
│   ├── checkpoints/               # Model checkpoints
│   ├── embeddings_cache/          # Cached embedding TypeFormer
│   │   ├── cohort/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── cohort/
│   │   └── cohort_embeddings.npy  # Cohort tetap, hash di filename
│   ├── logs/
│   │   └── (timestamp_experiment_id.log)
│   ├── scores/
│   │   └── (per-experiment score files)
│   ├── tables/
│   │   └── (output tabel untuk tesis)
│   └── figures/
│       └── (DET curve, histogram, dll)
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_user_stats.py
│   ├── test_asnorm.py
│   ├── test_film_head.py
│   ├── test_eer_computation.py
│   └── test_wilcoxon.py
│
└── docs/
    ├── implementation_guide.md    # Dokumen ini
    ├── architecture_decisions.md  # ADR — keputusan desain yang penting
    └── experiment_log.md          # Log eksperimen kronologis
```

### 5.1 Penjelasan singkat per folder

- **`config/`** — sumber kebenaran tunggal untuk hyperparameter. Setiap eksperimen punya YAML sendiri. Jangan ada hyperparameter hardcoded di script.
- **`src/`** — modul Python reusable. Tidak ada `if __name__ == "__main__"` di sini.
- **`scripts/`** — entry points yang dipanggil dari command line. Setiap script melakukan satu hal jelas.
- **`tests/`** — unit tests untuk komponen kritis. Wajib untuk: AS-Norm scorer, user stats computation, EER computation. Optional untuk: data loader, trainer.
- **`results/`** — output eksperimen. Di-gitignore (terlalu besar untuk git).
- **`notebooks/`** — eksplorasi, debug, visualisasi. **Bukan** tempat menulis kode produksi.

### 5.2 File yang harus ada di `.gitignore`

```gitignore
# Data and results (terlalu besar untuk git)
data/raw/
data/processed/
data/pretrained/
results/checkpoints/
results/embeddings_cache/
results/cohort/
results/scores/

# Python
__pycache__/
*.pyc
*.pyo
venv/
.venv/
.env

# Notebooks
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Experiment tracking
wandb/
mlruns/
```

---

## 6. Konvensi Penamaan

### 6.1 File dan folder

| Tipe | Konvensi | Contoh |
|---|---|---|
| Module Python | `snake_case.py` | `asnorm_scorer.py` |
| Script entry point | `NN_description.py` (NN = urutan) | `06_train_film.py` |
| Config YAML | `snake_case.yaml` | `n4_asnorm.yaml` |
| Class | `PascalCase` | `ASNormScorer`, `FiLMHead` |
| Function | `snake_case` | `compute_asnorm_score` |
| Variable | `snake_case` | `cohort_embeddings`, `s_u` |
| Konstanta | `UPPER_SNAKE_CASE` | `EMBEDDING_DIM = 64` |
| Tests | `test_<module>.py` | `test_asnorm.py` |

### 6.2 Penamaan variabel matematis

Konsisten dengan notasi proposal/paper:

- `e_u` — template embedding pengguna enrolment
- `e_p` — probe embedding
- `s_u` — user statistics vector (untuk FiLM)
- `s_raw` — skor cosine mentah
- `s_asnorm` — skor setelah AS-Norm
- `mu_u`, `sigma_u` — statistik cohort sisi enrolment
- `mu_p`, `sigma_p` — statistik cohort sisi probe
- `E` — jumlah sesi enrolment
- `K` — top-K cohort size
- `L` — panjang sekuens (= 50)

### 6.3 Penamaan eksperimen

Format: `<kategori><nomor>_<nama_singkat>_<varian>`

Contoh:
- `b0_baseline` — Baseline replikasi
- `n4_asnorm_K100` — AS-Norm dengan K=100
- `fn_full_lambda1e-4` — Full system dengan weight decay 1e-4

Setiap experiment ID terkait dengan satu file config dan satu output folder di `results/`.

### 6.4 Penamaan checkpoint

Format: `<exp_id>_E<value>_<timestamp>_<metric>.pt`

Contoh: `f1_film_only_E5_20260601_120000_eer0.045.pt`

---

## 7. Aturan dan Standar Pengembangan

### 7.1 Code style

- **PEP 8** sebagai dasar.
- **black** untuk auto-formatting (line length 100).
- **isort** untuk import ordering.
- **Type hints** untuk fungsi publik (yang dipanggil dari script atau modul lain). Tidak wajib untuk fungsi helper internal.

```bash
# Sebelum commit
black src/ scripts/
isort src/ scripts/
```

### 7.2 Docstring

Format Google style. Wajib untuk class publik dan fungsi yang dipanggil lintas-modul.

```python
def compute_asnorm_score(
    e_u: np.ndarray,
    e_p: np.ndarray,
    cohort: np.ndarray,
    K: int,
    epsilon: float = 1e-3,
) -> float:
    """Compute AS-Norm calibrated similarity score.

    Args:
        e_u: Enrolment template embedding, shape (D,).
        e_p: Probe embedding, shape (D,).
        cohort: Cohort embeddings, shape (N_c, D).
        K: Top-K cohort size for adaptive selection.
        epsilon: Floor for std to prevent division by zero.

    Returns:
        Calibrated score (higher = more likely genuine).

    Raises:
        ValueError: If K > N_c or embeddings have wrong shape.
    """
    ...
```

### 7.3 Reproducibility — wajib

Setiap eksperimen harus reproducible. Implementasikan:

```python
def set_global_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility across all libraries."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Setiap script entry point harus memanggil `set_global_seed()` di awal.

**Catatan**: full determinism dengan CUDA sulit; akurasi reproducible biasanya konsisten hingga 3-4 angka desimal, tidak persis. Laporkan mean ± std dari 3 random seeds untuk konfigurasi yang dapat dilatih.

### 7.4 Logging

Pakai Python `logging` standard, bukan `print()`. Setiap script ke file + stdout.

```python
import logging
from pathlib import Path

def setup_logger(name: str, log_dir: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = log_dir / f"{name}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
```

### 7.5 Konfigurasi via YAML

Tidak ada hyperparameter hardcoded di kode. Semua via config:

```yaml
# config/experiments/n4_asnorm.yaml
experiment:
  id: "n4_asnorm"
  description: "AS-Norm dengan K dari sweep"

data:
  test_users_file: "data/splits/test_user_ids.txt"
  cohort_file: "results/cohort/cohort_embeddings.npy"
  
model:
  backbone: "typeformer_pretrained"
  use_film: false

scoring:
  method: "asnorm"
  metric: "cosine"
  K: 100  # akan di-sweep terpisah
  epsilon: 1e-3

evaluation:
  E_values: [1, 2, 5, 7, 10]
  metrics: ["per_subject_eer", "global_eer", "far_at_1frr", "frr_at_1far"]

seed: 42
```

### 7.6 Testing

Unit test wajib untuk komponen kritis:

```python
# tests/test_asnorm.py
def test_asnorm_with_identical_embeddings():
    """Skor antara dua embedding identik harus tinggi."""
    e = np.random.randn(64).astype(np.float32)
    cohort = np.random.randn(100, 64).astype(np.float32)
    score = compute_asnorm_score(e, e, cohort, K=10)
    assert score > 0, f"Identical embeddings should give positive score, got {score}"

def test_asnorm_K_larger_than_cohort_raises():
    """K > N_c harus raise error."""
    e_u = np.random.randn(64).astype(np.float32)
    e_p = np.random.randn(64).astype(np.float32)
    cohort = np.random.randn(50, 64).astype(np.float32)
    with pytest.raises(ValueError):
        compute_asnorm_score(e_u, e_p, cohort, K=100)

def test_asnorm_epsilon_floor():
    """Sigma = 0 harus tidak menyebabkan divide by zero."""
    e_u = np.random.randn(64).astype(np.float32)
    e_p = np.random.randn(64).astype(np.float32)
    # Cohort identik → sigma = 0 setelah top-K
    cohort = np.tile(e_u, (100, 1)) + np.random.randn(100, 64) * 1e-10
    score = compute_asnorm_score(e_u, e_p, cohort, K=10, epsilon=1e-3)
    assert not np.isnan(score), "Should not produce NaN with epsilon floor"
```

Run tests sebelum merge atau commit besar:

```bash
pytest tests/ -v
```

### 7.7 Git workflow

Single-developer project, tapi tetap disiplin:

- **Commit pesan**: format `<tipe>: <deskripsi singkat>`. Tipe: `feat`, `fix`, `refactor`, `test`, `docs`, `exp` (untuk hasil eksperimen).
  - `feat: implement AS-Norm scorer`
  - `exp: run D1 diagnostic, results in results/diagnostics/d1/`
  - `fix: epsilon floor in asnorm when sigma=0`
- **Branch**: kerja di `main` saja kalau solo. Tag rilis besar dengan `v0.1`, `v0.2` dst.
- **Commit hasil eksperimen ke git**: hanya tabel ringkasan (CSV/JSON kecil), bukan checkpoint atau embedding besar.

---

## 8. Alur Kerja Pengembangan

### 8.1 Roadmap fase

```
Fase 0: Setup & Verifikasi Akses    [Minggu 1]
   ↓
Fase 1: Reproduksi Baseline (B0)     [Minggu 1-2]
   ↓
Fase 2: Diagnostik (D1, D2, D3)      [Minggu 2-3]
   ↓
   ├──── Hasil mendukung? ────┐
   ↓                          ↓
  YA                          TIDAK → revisi arah, diskusi pembimbing
   ↓
Fase 3: Implementasi AS-Norm         [Minggu 3-5]
   ↓
Fase 4: Implementasi FiLM            [Minggu 5-8]
   ↓
Fase 5: Eksperimen Matriks Penuh     [Minggu 8-11]
   ↓
Fase 6: Analisis Statistik           [Minggu 11-12]
   ↓
Fase 7: (Opsional) Validasi KVC      [Minggu 13-14]
   ↓
Fase 8: Penulisan Tesis              [Minggu 14+]
```

### 8.2 Detail per fase

#### Fase 0 — Setup & Verifikasi Akses

**Deliverable**: Environment aktif, data tersedia, pretrained weights ter-load.

Langkah:
1. Setup Python environment (Section 4).
2. Verifikasi akses data Aalto Mobile.
3. Verifikasi akses pretrained TypeFormer weights.
4. Load weights dan jalankan forward pass dummy.
5. Setup git repository, struktur folder, gitignore.

**Stop criteria**: jika data atau weights tidak dapat diakses dalam 1 minggu, eskalasi ke pembimbing. **Jangan lanjut fase berikut.**

#### Fase 1 — Reproduksi Baseline (B0)

**Deliverable**: Mean per-subject EER pada test set 1.000 pengguna, untuk E∈{1,2,5,7,10}. Target: E=5 → ~3,25%.

Langkah:
1. Implementasi `aalto_loader.py` dan `sequence_processor.py`.
2. Implementasi `typeformer_wrapper.py` untuk forward pass.
3. Implementasi `per_subject_eer.py`.
4. Jalankan B0 dengan mean pairwise Euclidean distance + per-subject EER.
5. Verifikasi: E=5 menghasilkan EER dalam ±0,5% dari 3,25%.

**Stop criteria**: jika tidak match, debug pipeline sebelum lanjut. Possible causes: preprocessing berbeda, ASCII normalization beda, mismatched split.

#### Fase 2 — Diagnostik

Tiga eksperimen mini independen. Run paralel kalau bisa.

**D1 — Heterogenitas σ_u** (validitas AS-Norm)
```
Output: histogram σ_u lintas 100 pengguna validasi
Decision: rasio max/min σ_u > 2× → AS-Norm viable
         else → pivot ke FiLM atau Set2Set sebagai utama
```

**D2 — Stabilitas s_u** (validitas FiLM)
```
Output: Pearson correlation s_u(E=5) vs s_u(E=15), per dimensi
Decision: correlation mean > 0.8, std > 0.6 → FiLM viable
         0.6-0.8 → FiLM viable dengan shrinkage estimator
         < 0.6 → FiLM hanya sebagai pendukung
```

**D3 — Baseline global EER**
```
Output: global EER TypeFormer pretrained pada test set
Decision: tetapkan target AS-Norm (e.g., turunkan 30-50%)
```

**Deliverable**: laporan singkat (1-2 halaman per diagnostik) untuk diskusi pembimbing.

**Stop criteria**: setelah D1, D2, D3 selesai, **wajib diskusi pembimbing** sebelum lanjut. Bisa jadi arah penelitian perlu disesuaikan.

#### Fase 3 — Implementasi AS-Norm

**Deliverable**: Hasil N1, N2, N3, N4 pada test set untuk semua E.

Langkah:
1. Implementasi cohort sampling (`cohort_sampler.py`), simpan cohort tetap.
2. Cache embedding cohort.
3. Cache embedding test set.
4. Implementasi Z-Norm, T-Norm, S-Norm, AS-Norm di `scoring/`.
5. Sweep K pada validation set untuk AS-Norm.
6. Run B1, B2, C1, C2, C3, N1, N2, N3, N4 — semua dengan E=5 dulu sebagai sanity check, lalu sapu E penuh.

#### Fase 4 — Implementasi FiLM

**Deliverable**: FiLM head terlatih, hasil F1 dan FN pada test set.

Langkah:
1. Implementasi `film_head.py` dengan inisialisasi identitas.
2. Implementasi `user_stats.py` untuk compute s_u.
3. Implementasi `sampler.py` multi-E sampling.
4. Sweep weight decay pada validation set (small grid).
5. Training: 3 random seeds, simpan checkpoint terbaik per seed.
6. Evaluasi F1 (FiLM saja) dan FN (FiLM + AS-Norm).

#### Fase 5 — Eksperimen Matriks Penuh

**Deliverable**: Tabel hasil lengkap 12 konfigurasi × 5 nilai E.

Langkah:
1. Konfirmasi semua konfigurasi sudah punya hasil di test set.
2. Re-run kalau ada drift atau bug fix di antaranya.
3. Generate tabel ringkasan dalam CSV.

#### Fase 6 — Analisis Statistik

**Deliverable**: Tabel uji signifikansi, bootstrap CI, plot DET curve.

Langkah:
1. Wilcoxon signed-rank untuk per-subject EER (5 perbandingan utama, Bonferroni α=0.01).
2. Bootstrap 1000 iterations untuk global EER, CI 95%.
3. Plot DET curves untuk konfigurasi utama.
4. Plot histogram μ_u, σ_u untuk N4 (untuk diskusi mekanisme).
5. Analisis biometric menagerie: kategorisasi pengguna sheep/goat/lamb/wolf dan efek AS-Norm pada masing-masing kategori.

#### Fase 7 — Validasi KVC (opsional)

Hanya jalankan jika akses CodaLab/Codabench disetujui sebelum minggu 13. Lakukan untuk 3-4 konfigurasi top saja sebagai validasi eksternal.

#### Fase 8 — Penulisan Tesis

Di luar scope dokumen ini.

---

## 9. Detail Implementasi per Modul

### 9.1 `src/data/aalto_loader.py`

**Tanggung jawab**: load preprocessed Aalto sessions menjadi tensor.

Interface:
```python
class AaltoDataset(torch.utils.data.Dataset):
    def __init__(self, sessions_file: Path, user_ids: List[int]):
        ...
    
    def __len__(self) -> int: ...
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'sequence': (L, 5) tensor — keystroke features
                'user_id': int
                'session_id': int
            }
        """
```

**Catatan kritis**:
- Jangan apply z-score normalization per-user di sini (itu yang gagal).
- Apply hanya: ASCII / 255, zero-padding ke L=50.
- Verifikasi: urutan fitur harus sama persis dengan TypeFormer asli. *[Cek di kode resmi.]*

### 9.2 `src/models/typeformer_wrapper.py`

**Tanggung jawab**: wrapper di sekitar TypeFormer pretrained untuk inference.

```python
class TypeFormerWrapper:
    def __init__(self, checkpoint_path: Path, device: str = "cuda"):
        self.model = self._load_pretrained(checkpoint_path)
        self.model.to(device).eval()
        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
    
    @torch.no_grad()
    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequences: (B, L, 5)
        Returns:
            embeddings: (B, 64)
        """
        sequences = sequences.to(self.device)
        embeddings = self.model(sequences)
        return embeddings.cpu()
    
    def encode_batch(self, dataset: AaltoDataset, batch_size: int = 32) -> torch.Tensor:
        """Encode entire dataset, return (N, 64) tensor."""
        ...
```

**Catatan**: 
- TypeFormer backbone harus selalu `.eval()` mode untuk konsistensi (BatchNorm/Dropout behavior).
- Backbone tidak punya gradient — `@torch.no_grad()` wajib.
- Caching: hasil encode dapat disimpan ke disk (`results/embeddings_cache/`) karena tidak berubah selama backbone beku.

### 9.3 `src/models/film_head.py`

Lihat pseudocode lengkap di [Section 10.2](#102-film-head).

### 9.4 `src/scoring/asnorm.py`

Lihat pseudocode lengkap di [Section 10.1](#101-as-norm-score-calibration).

### 9.5 `src/statistics/user_stats.py`

**Tanggung jawab**: compute vektor s_u dari sesi enrolment.

```python
def compute_user_stats(
    sessions: np.ndarray,  # (E, L, 5) — E sesi enrolment
    feature_columns: List[int] = [0, 1, 2, 3],  # HL, IL, PL, RL (bukan ASCII)
    use_percentiles: bool = True,
) -> np.ndarray:
    """
    Returns:
        s_u: (D_s,) vector. D_s = 4*5 = 20 jika use_percentiles, else 4*2 = 8.
    """
    # Concatenate semua keystroke dari semua sesi enrolment
    all_keystrokes = sessions.reshape(-1, 5)  # (E*L, 5)
    
    # Filter zero-padding (asumsi: padding = baris semua nol)
    valid_mask = ~np.all(all_keystrokes[:, :4] == 0, axis=1)
    valid = all_keystrokes[valid_mask][:, feature_columns]  # (T_valid, 4)
    
    stats_list = [
        valid.mean(axis=0),
        valid.std(axis=0),
    ]
    if use_percentiles:
        stats_list += [
            np.percentile(valid, 25, axis=0),
            np.percentile(valid, 50, axis=0),
            np.percentile(valid, 75, axis=0),
        ]
    
    return np.concatenate(stats_list)
```

**Pertimbangan**:
- Pada E=1 dan T_valid rendah, persentil tidak stabil. Pertimbangkan fallback untuk E kecil.
- Saat training dengan multi-E sampling, hitung stats dari E sesi yang di-sample acak per iterasi.

### 9.6 `src/scoring/asnorm.py`

Lihat pseudocode di [Section 10.1](#101-as-norm-score-calibration).

### 9.7 `src/training/trainer.py`

**Tanggung jawab**: training loop FiLM head.

Pseudo-interface:
```python
class FiLMTrainer:
    def __init__(
        self,
        film_head: FiLMHead,
        backbone: TypeFormerWrapper,
        train_dataset: AaltoDataset,
        val_dataset: AaltoDataset,
        config: Dict,
    ):
        ...
    
    def train(self) -> Dict[str, List[float]]:
        """
        Returns:
            history: dict with keys 'train_loss', 'val_eer_persubject', 'val_eer_global'
        """
        ...
    
    def _train_epoch(self) -> float: ...
    def _validate(self) -> Tuple[float, float]: ...
```

Lihat pseudocode training loop di [Section 10.3](#103-film-training-loop).

### 9.8 `src/evaluation/metrics.py`

```python
def compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute EER from two score arrays.
    
    Returns:
        eer: Equal Error Rate
        threshold: threshold at EER point
    """
    from sklearn.metrics import roc_curve
    
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # EER point: where |fpr - fnr| is minimized
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer, thresholds[idx]


def compute_far_at_frr(genuine: np.ndarray, impostor: np.ndarray, target_frr: float) -> float:
    """Compute FAR at a specific FRR operating point."""
    ...


def compute_frr_at_far(genuine: np.ndarray, impostor: np.ndarray, target_far: float) -> float:
    """Compute FRR at a specific FAR operating point."""
    ...
```

---

## 10. Pseudocode Snippet untuk Komponen Kunci

### 10.1 AS-Norm score calibration

```python
import numpy as np

def compute_asnorm_score(
    e_u: np.ndarray,         # (D,) enrolment template
    e_p: np.ndarray,         # (D,) probe embedding
    cohort: np.ndarray,      # (N_c, D) cohort embeddings
    K: int,                  # top-K cohort selection
    epsilon: float = 1e-3,   # floor untuk std
) -> float:
    """
    AS-Norm symmetric calibration.
    Higher score = more likely genuine match.
    """
    # Normalize embeddings untuk cosine (jika belum normalized)
    e_u_n = e_u / (np.linalg.norm(e_u) + 1e-12)
    e_p_n = e_p / (np.linalg.norm(e_p) + 1e-12)
    cohort_n = cohort / (np.linalg.norm(cohort, axis=1, keepdims=True) + 1e-12)
    
    # Raw cosine similarity
    s_raw = float(np.dot(e_u_n, e_p_n))
    
    # Cohort scores untuk sisi enrolment
    scores_u = cohort_n @ e_u_n  # (N_c,)
    # Top-K paling mirip dengan e_u
    top_k_indices_u = np.argpartition(-scores_u, K)[:K]
    top_k_scores_u = scores_u[top_k_indices_u]
    mu_u = top_k_scores_u.mean()
    sigma_u = max(top_k_scores_u.std(), epsilon)
    
    # Cohort scores untuk sisi probe
    scores_p = cohort_n @ e_p_n
    top_k_indices_p = np.argpartition(-scores_p, K)[:K]
    top_k_scores_p = scores_p[top_k_indices_p]
    mu_p = top_k_scores_p.mean()
    sigma_p = max(top_k_scores_p.std(), epsilon)
    
    # Symmetric normalization
    z_u = (s_raw - mu_u) / sigma_u
    z_p = (s_raw - mu_p) / sigma_p
    s_asnorm = 0.5 * (z_u + z_p)
    
    return s_asnorm


def batch_asnorm_for_user(
    e_u: np.ndarray,
    probes: np.ndarray,      # (N_probes, D)
    cohort: np.ndarray,
    K: int,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """
    Hitung skor AS-Norm untuk satu enrolment terhadap banyak probe.
    Lebih efisien: hitung sisi enrolment SEKALI, sisi probe per probe.
    """
    # Normalize
    e_u_n = e_u / (np.linalg.norm(e_u) + 1e-12)
    probes_n = probes / (np.linalg.norm(probes, axis=1, keepdims=True) + 1e-12)
    cohort_n = cohort / (np.linalg.norm(cohort, axis=1, keepdims=True) + 1e-12)
    
    # Sisi enrolment (1x)
    scores_u = cohort_n @ e_u_n
    top_k_u = np.partition(scores_u, -K)[-K:]
    mu_u = top_k_u.mean()
    sigma_u = max(top_k_u.std(), epsilon)
    
    # Sisi probe (vectorized)
    scores_p_matrix = probes_n @ cohort_n.T  # (N_probes, N_c)
    # Top-K per row
    top_k_p_matrix = np.partition(scores_p_matrix, -K, axis=1)[:, -K:]
    mu_p = top_k_p_matrix.mean(axis=1)
    sigma_p = np.maximum(top_k_p_matrix.std(axis=1), epsilon)
    
    # Raw scores
    s_raw = probes_n @ e_u_n  # (N_probes,)
    
    # AS-Norm
    z_u = (s_raw - mu_u) / sigma_u
    z_p = (s_raw - mu_p) / sigma_p
    return 0.5 * (z_u + z_p)
```

### 10.2 FiLM head

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMHead(nn.Module):
    """
    Feature-wise Linear Modulation head untuk conditioning embedding
    pada statistik pengguna.
    
    Inisialisasi identitas: γ=1, β=0 di awal training.
    """
    
    def __init__(
        self,
        stats_dim: int = 20,        # dimensi s_u (4 fitur × 5 statistik)
        embedding_dim: int = 64,    # dimensi output TypeFormer
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.fc1 = nn.Linear(stats_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * embedding_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._init_identity()
    
    def _init_identity(self):
        """Inisialisasi sehingga awalnya γ=1, β=0 (identity transform)."""
        # fc1 default init
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        
        # fc2: weight = 0, bias = [1...1, 0...0]
        nn.init.zeros_(self.fc2.weight)
        with torch.no_grad():
            self.fc2.bias[:self.embedding_dim] = 1.0  # γ bias = 1
            self.fc2.bias[self.embedding_dim:] = 0.0  # β bias = 0
    
    def forward(
        self,
        embedding: torch.Tensor,    # (B, D)
        user_stats: torch.Tensor,   # (B, S)
    ) -> torch.Tensor:
        """
        Returns:
            modulated_embedding: (B, D)
        """
        h = F.relu(self.fc1(user_stats))
        h = self.dropout(h)
        gamma_beta = self.fc2(h)
        gamma = gamma_beta[:, :self.embedding_dim]
        beta = gamma_beta[:, self.embedding_dim:]
        
        return gamma * embedding + beta
```

### 10.3 FiLM training loop

```python
def train_film(
    film_head: FiLMHead,
    backbone: TypeFormerWrapper,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
):
    """
    Training FiLM head dengan triplet loss dan multi-E sampling.
    Backbone beku, hanya FiLM yang dilatih.
    """
    device = config['device']
    film_head.to(device)
    
    # Hanya parameter FiLM yang masuk optimizer
    optimizer = torch.optim.Adam(
        film_head.parameters(),
        lr=config['learning_rate'],         # e.g., 1e-3
        weight_decay=config['weight_decay'], # WAJIB: dari sweep, e.g., 1e-4
    )
    
    triplet_loss = nn.TripletMarginLoss(margin=config['margin'], p=2)
    
    best_val_eer = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_per_subj_eer': [], 'val_global_eer': []}
    
    for epoch in range(config['max_epochs']):
        # ===== Training =====
        film_head.train()
        train_losses = []
        
        for batch in train_loader:
            # batch berisi triplet: anchor, positive, negative
            # plus s_u untuk masing-masing
            anchor_seq = batch['anchor_seq'].to(device)
            positive_seq = batch['positive_seq'].to(device)
            negative_seq = batch['negative_seq'].to(device)
            
            s_u_anchor = batch['s_u_anchor'].to(device)
            s_u_positive = batch['s_u_positive'].to(device)  # = s_u_anchor (same user)
            s_u_negative = batch['s_u_negative'].to(device)
            
            # Forward backbone (no grad)
            with torch.no_grad():
                e_anchor = backbone.model(anchor_seq)
                e_positive = backbone.model(positive_seq)
                e_negative = backbone.model(negative_seq)
            
            # FiLM modulation (with grad)
            e_anchor_mod = film_head(e_anchor, s_u_anchor)
            e_positive_mod = film_head(e_positive, s_u_positive)
            e_negative_mod = film_head(e_negative, s_u_negative)
            
            loss = triplet_loss(e_anchor_mod, e_positive_mod, e_negative_mod)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Optional: gradient clipping
            torch.nn.utils.clip_grad_norm_(film_head.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        # ===== Validation =====
        film_head.eval()
        val_per_subj_eer, val_global_eer = evaluate_with_film(
            film_head, backbone, val_loader, config
        )
        
        history['train_loss'].append(np.mean(train_losses))
        history['val_per_subj_eer'].append(val_per_subj_eer)
        history['val_global_eer'].append(val_global_eer)
        
        logger.info(f"Epoch {epoch}: train_loss={np.mean(train_losses):.4f}, "
                    f"val_per_subj_eer={val_per_subj_eer:.4f}, "
                    f"val_global_eer={val_global_eer:.4f}")
        
        # Monitor γ dan β untuk sanity check
        with torch.no_grad():
            # Forward dengan s_u dummy untuk lihat current γ, β
            dummy_s_u = torch.randn(1, config['stats_dim']).to(device)
            gamma_beta = film_head.fc2(F.relu(film_head.fc1(dummy_s_u)))
            gamma = gamma_beta[0, :64]
            beta = gamma_beta[0, 64:]
            logger.debug(f"||gamma-1|| = {(gamma-1).norm():.4f}, ||beta|| = {beta.norm():.4f}")
        
        # Early stopping
        if val_per_subj_eer < best_val_eer:
            best_val_eer = val_per_subj_eer
            patience_counter = 0
            torch.save(film_head.state_dict(), config['checkpoint_path'])
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    return history
```

### 10.4 Multi-E triplet sampling

```python
def sample_multi_e_triplet(
    users: List[UserData],
    E_range: List[int] = [2, 5, 7, 10],
) -> Dict:
    """
    Sample satu triplet dengan E random dari E_range.
    Mensimulasikan distribution mismatch yang akan dialami saat deployment.
    """
    anchor_user = random.choice(users)
    
    # Sample E untuk iterasi ini
    E = random.choice(E_range)
    
    # Asumsi: setiap user punya 15 sesi
    n_sessions = len(anchor_user.sessions)
    
    # Pilih E sesi sebagai 'enrolment' untuk hitung s_u
    enrolment_indices = random.sample(range(n_sessions), E)
    enrolment_sessions = [anchor_user.sessions[i] for i in enrolment_indices]
    s_u_anchor = compute_user_stats(np.array(enrolment_sessions))
    
    # Pilih 2 sesi berbeda sebagai anchor dan positive
    remaining = [i for i in range(n_sessions) if i not in enrolment_indices]
    if len(remaining) < 2:
        # Edge case: E terlalu besar, sesi enrolment overlap dengan anchor/positive
        # Resample atau pakai sesi enrolment sebagai positive
        anchor_idx, positive_idx = random.sample(range(n_sessions), 2)
    else:
        anchor_idx, positive_idx = random.sample(remaining, 2)
    
    anchor_seq = anchor_user.sessions[anchor_idx]
    positive_seq = anchor_user.sessions[positive_idx]
    
    # Negative dari user berbeda
    negative_user = random.choice([u for u in users if u.user_id != anchor_user.user_id])
    negative_session_idx = random.choice(range(len(negative_user.sessions)))
    negative_seq = negative_user.sessions[negative_session_idx]
    
    # s_u untuk negative — gunakan random E sesi
    neg_E = random.choice(E_range)
    neg_enrol_indices = random.sample(range(len(negative_user.sessions)), 
                                       min(neg_E, len(negative_user.sessions)))
    neg_enrol = [negative_user.sessions[i] for i in neg_enrol_indices]
    s_u_negative = compute_user_stats(np.array(neg_enrol))
    
    return {
        'anchor_seq': anchor_seq,
        'positive_seq': positive_seq,
        'negative_seq': negative_seq,
        's_u_anchor': s_u_anchor,
        's_u_positive': s_u_anchor,  # same user
        's_u_negative': s_u_negative,
    }
```

### 10.5 Per-subject vs global EER

```python
def evaluate_full(
    scorer: Callable,
    test_users: List[UserData],
    cohort: np.ndarray,
    E: int,
    config: Dict,
) -> Dict[str, float]:
    """
    Evaluate one configuration on test set.
    Returns both per-subject and global EER.
    """
    all_genuine = []
    all_impostor = []
    per_user_eers = []
    
    for user in test_users:
        # Enrolment: sesi 0 sampai E-1
        enrolment_sessions = user.sessions[:E]
        # Genuine probes: 5 sesi terakhir
        genuine_probes = user.sessions[-5:]
        
        # Hitung template
        e_u = compute_template(enrolment_sessions)  # bisa mean atau centroid
        
        # Skor genuine
        genuine_scores = [scorer(e_u, encode(p), cohort) for p in genuine_probes]
        
        # Skor impostor: 1 sesi dari setiap user lain (999 user lain)
        impostor_scores = []
        for other_user in test_users:
            if other_user.user_id == user.user_id:
                continue
            # Sesi impostor: ambil sesi tertentu (deterministik via seed)
            impostor_session = other_user.sessions[config['impostor_session_idx']]
            score = scorer(e_u, encode(impostor_session), cohort)
            impostor_scores.append(score)
        
        # Per-subject EER
        eer_user, _ = compute_eer(np.array(genuine_scores), np.array(impostor_scores))
        per_user_eers.append(eer_user)
        
        # Akumulasi untuk global
        all_genuine.extend(genuine_scores)
        all_impostor.extend(impostor_scores)
    
    # Global EER
    global_eer, global_threshold = compute_eer(
        np.array(all_genuine),
        np.array(all_impostor),
    )
    
    return {
        'mean_per_subject_eer': np.mean(per_user_eers),
        'std_per_subject_eer': np.std(per_user_eers),
        'global_eer': global_eer,
        'global_threshold': global_threshold,
        'per_user_eers': per_user_eers,  # untuk Wilcoxon
        'far_at_1frr': compute_far_at_frr(all_genuine, all_impostor, 0.01),
        'frr_at_1far': compute_frr_at_far(all_genuine, all_impostor, 0.01),
    }
```

### 10.6 Wilcoxon signed-rank dengan Bonferroni

```python
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def compare_configurations_paired(
    eer_dict: Dict[str, np.ndarray],  # {config_name: array of per-user EERs}
    comparisons: List[Tuple[str, str]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Wilcoxon signed-rank paired comparison dengan Bonferroni correction.
    
    Args:
        eer_dict: per-user EERs untuk setiap konfigurasi
        comparisons: list of (config_A, config_B) pairs to compare
        alpha: family-wise error rate
    
    Returns:
        DataFrame dengan kolom: comparison, statistic, p_raw, p_bonferroni, significant
    """
    results = []
    for a, b in comparisons:
        stat, p = wilcoxon(eer_dict[a], eer_dict[b], alternative='two-sided')
        results.append({'comparison': f"{a} vs {b}", 'statistic': stat, 'p_raw': p})
    
    p_raws = [r['p_raw'] for r in results]
    reject, p_corrected, _, _ = multipletests(p_raws, alpha=alpha, method='bonferroni')
    
    for r, sig, p_c in zip(results, reject, p_corrected):
        r['p_bonferroni'] = p_c
        r['significant'] = sig
    
    return pd.DataFrame(results)
```

### 10.7 Bootstrap CI untuk global EER

```python
def bootstrap_global_eer_ci(
    user_data: List[Dict],  # [{user_id, genuine_scores, impostor_scores}, ...]
    n_iterations: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval untuk global EER.
    Resampling pada level user (bukan trial), karena trial dalam satu user tidak independen.
    
    Returns:
        eer_point: global EER pada full data
        ci_lower, ci_upper: bounds CI
    """
    rng = np.random.RandomState(seed)
    n_users = len(user_data)
    
    eer_bootstrap = []
    for _ in range(n_iterations):
        # Sample user dengan replacement
        sampled_indices = rng.choice(n_users, size=n_users, replace=True)
        
        all_gen = []
        all_imp = []
        for idx in sampled_indices:
            all_gen.extend(user_data[idx]['genuine_scores'])
            all_imp.extend(user_data[idx]['impostor_scores'])
        
        eer, _ = compute_eer(np.array(all_gen), np.array(all_imp))
        eer_bootstrap.append(eer)
    
    eer_bootstrap = np.array(eer_bootstrap)
    
    # Point estimate dari full data
    all_gen_full = np.concatenate([u['genuine_scores'] for u in user_data])
    all_imp_full = np.concatenate([u['impostor_scores'] for u in user_data])
    eer_point, _ = compute_eer(all_gen_full, all_imp_full)
    
    # CI bounds
    alpha = (1 - ci_level) / 2
    ci_lower = np.percentile(eer_bootstrap, alpha * 100)
    ci_upper = np.percentile(eer_bootstrap, (1 - alpha) * 100)
    
    return eer_point, ci_lower, ci_upper
```

---

## 11. Best Practice

### 11.1 Reproducibility

- Setiap script memanggil `set_global_seed()` di awal.
- Setiap config file punya field `seed` eksplisit.
- Setiap experiment output menyimpan: config yang dipakai, git hash, timestamp.
- Untuk komponen yang dilatih (FiLM), laporkan **3 seeds** dan tampilkan mean ± std.

### 11.2 Caching cerdas

Komputasi mahal yang **berulang persis** wajib di-cache:

| Apa | Kapan dihitung | Kapan invalid |
|---|---|---|
| Embedding cohort | Sekali setelah cohort dipilih | Cohort berubah |
| Embedding test set | Sekali setelah preprocessing | Backbone berubah / preprocessing berubah |
| User statistics (s_u) untuk fixed E | Sekali per pengguna per E | Definisi s_u berubah |

Pattern:
```python
def get_cached_embeddings(cache_path: Path, compute_fn: Callable) -> np.ndarray:
    if cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)
    
    logger.info(f"Computing embeddings, will cache to {cache_path}")
    embeddings = compute_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings
```

### 11.3 Experiment tracking

Setidaknya, untuk **setiap eksperimen** simpan:

```
results/<exp_id>/
├── config.yaml          # Config yang dipakai
├── metrics.json         # Hasil agregat
├── per_user_eers.csv    # EER per pengguna (untuk Wilcoxon)
├── scores.npz           # Raw scores (untuk re-analysis)
├── log.txt              # Console log
└── git_info.txt         # git hash, branch, dirty status
```

Format `metrics.json`:
```json
{
  "experiment_id": "n4_asnorm_K100",
  "timestamp": "20260601_120000",
  "git_hash": "abc1234",
  "config_hash": "def5678",
  "results": {
    "E_5": {
      "mean_per_subject_eer": 0.0285,
      "std_per_subject_eer": 0.018,
      "global_eer": 0.052,
      "global_eer_ci_lower": 0.048,
      "global_eer_ci_upper": 0.056,
      "far_at_1frr": 0.085,
      "frr_at_1far": 0.090
    },
    "E_1": { ... },
    ...
  }
}
```

### 11.4 Code review checklist (untuk diri sendiri)

Sebelum commit kode penting:
- [ ] Apakah semua hyperparameter dari config, bukan hardcoded?
- [ ] Apakah ada unit test untuk komponen ini?
- [ ] Apakah seed di-set di awal?
- [ ] Apakah ada logging yang cukup untuk debug?
- [ ] Apakah dokumentasi function (docstring) lengkap?
- [ ] Apakah edge cases dipertimbangkan (E=1, K=0, sigma=0)?

### 11.5 Sanity check setiap eksperimen

Sebelum percaya hasil, verifikasi:
- [ ] EER ada di range yang masuk akal (0-50%, biasanya 1-15% untuk metode bagus).
- [ ] Per-subject EER ≤ Global EER (umumnya — per-subject memilih threshold optimal per pengguna).
- [ ] Jika E lebih besar, EER cenderung lebih rendah (lebih banyak data enrolment = template lebih baik).
- [ ] Mean per-subject EER baseline mendekati 3,25% pada E=5 (sanity check pipeline).
- [ ] Distribusi skor genuine dan impostor terpisah (plot histogram).

Kalau salah satu di atas tidak terpenuhi, **debug sebelum lanjut**.

### 11.6 Arsitektur dan maintainability

- **Separation of concerns**: scoring tidak tahu apa-apa tentang backbone; FiLM tidak tahu apa-apa tentang scoring; trainer tidak tahu apa-apa tentang evaluation metric.
- **Dependency injection**: jangan import langsung di dalam class. Pass dependency lewat constructor. Memudahkan testing.
- **Abstract base class**: untuk Scorer, agar mudah switch AS-Norm ↔ Z-Norm ↔ S-Norm via config.

```python
from abc import ABC, abstractmethod

class BaseScorer(ABC):
    @abstractmethod
    def score(self, e_u: np.ndarray, e_p: np.ndarray) -> float: ...

class ASNormScorer(BaseScorer):
    def __init__(self, cohort: np.ndarray, K: int):
        self.cohort = cohort
        self.K = K
    
    def score(self, e_u, e_p):
        return compute_asnorm_score(e_u, e_p, self.cohort, self.K)
```

---

## 12. Catatan Teknis yang Belum Terverifikasi

Daftar ini wajib Anda verifikasi sebelum atau saat implementasi. Saya beri tanda `[VERIFY]` di kode/dokumen tempat masing-masing perlu konfirmasi.

### 12.1 Dari paper TypeFormer

1. **Dimensi embedding output**: saya pakai asumsi 64. *[Cek di kode resmi BiDAlab/TypeFormer.]*
2. **Triplet margin**: saya pakai 1.0 sebagai placeholder. *[Cek nilai di paper atau kode.]*
3. **Format input persis**: urutan 5 channel (HL, IL, PL, RL, ASCII/255) — *[konfirmasi urutan dan normalisasi ASCII.]*
4. **Zero-padding direction**: saya asumsikan padding di akhir sekuens. *[Konfirmasi.]*
5. **Cara hitung centroid vs mean pairwise distance untuk template**: paper TypeFormer menggunakan mean pairwise; konfigurasi adaptive saya pakai centroid (untuk kompatibilitas dengan AS-Norm). *[Konfirmasi ini tidak menyebabkan perubahan EER substansial; bisa dicek lewat eksperimen B1.]*

### 12.2 Dari paper Type2Branch

6. **Formula Set2Set loss persis**: saya tulis konseptual, tapi belum verifikasi verbatim. *[Cek IEEE TIFS 2025 versi final.]* — relevan hanya jika Anda akhirnya mengimplementasi Set2Set sebagai backup.

### 12.3 Dari paper FiLM (Perez et al.)

7. **Nilai weight decay yang dipakai Perez**: saya rekomendasikan sweep {1e-5, 1e-4, 1e-3}, tapi nilai eksak Perez tidak saya tarik dari kode. *[Cek `github.com/ethanjperez/film` config training kalau ingin starting point.]*

### 12.4 Pendekatan AS-Norm

8. **Optimal K range untuk keystroke**: saya transfer dari speaker verification (50-500). Belum diuji empirik pada keystroke. *[Sweep wajib pada validation.]*
9. **Optimal cohort size untuk Aalto**: saya rekomendasikan 1.000-2.000 sesi, tapi belum diuji. *[Sweep wajib.]*
10. **Apakah cohort harus distratifikasi demografis**: relevan untuk fairness. *[Diskusi dengan pembimbing apakah ini in/out scope.]*

### 12.5 Pendekatan FiLM

11. **Komposisi s_u optimal**: 20-dim (mean+std+p25/50/75 × 4 fitur) adalah pilihan rasional. *[Ablation experiment dapat menggunakan varian seperti tanpa persentil.]*
12. **Titik injeksi FiLM**: saya rekomendasikan pada embedding 64-dim akhir. Belum diuji vs injeksi internal. *[Eksperimen pendukung kalau ada waktu.]*
13. **Apakah multi-E sampling membantu**: hipotesis kerja, perlu diuji.

### 12.6 Apa yang BUKAN claim

Saya **tidak** mengklaim:
- AS-Norm pasti menurunkan global EER pada TypeFormer/Aalto. Itu pertanyaan empiris.
- FiLM pasti membantu. Itu pertanyaan empiris.
- Target numerik (EER < 5%, dll) yang saya sebut di Section 1 atau sebelumnya. Itu hipotesis kerja, harus dikalibrasi setelah D3.

---

## 13. Checklist Implementasi

### Fase 0: Setup
- [ ] Python 3.10+ environment aktif
- [ ] PyTorch terinstall dengan CUDA support
- [ ] Verifikasi GPU dengan `nvidia-smi`
- [ ] requirements.txt terinstall
- [ ] Akses Aalto Mobile data dikonfirmasi
- [ ] Pretrained TypeFormer weights ter-download
- [ ] Git repository diinisialisasi
- [ ] Struktur folder dibuat sesuai Section 5
- [ ] `.gitignore` terisi

### Fase 1: Reproduksi Baseline
- [ ] `aalto_loader.py` selesai
- [ ] `sequence_processor.py` selesai
- [ ] `typeformer_wrapper.py` selesai, forward pass berhasil
- [ ] Unit test untuk data loader passes
- [ ] `per_subject_eer.py` dan `metrics.py` selesai
- [ ] Unit test EER computation passes
- [ ] B0 dijalankan untuk E=5
- [ ] B0 EER mendekati 3,25% (±0,5%) — SANITY CHECK PIPELINE
- [ ] B0 dijalankan untuk semua E∈{1,2,5,7,10}
- [ ] Hasil tersimpan di `results/b0_baseline/`

### Fase 2: Diagnostik
- [ ] **D1** — Heterogenitas σ_u
  - [ ] Cohort sampling untuk 500 sesi
  - [ ] Hitung σ_u untuk 100 pengguna validasi
  - [ ] Plot histogram σ_u
  - [ ] **Keputusan**: AS-Norm viable? Catat hasil di `docs/experiment_log.md`
- [ ] **D2** — Stabilitas s_u
  - [ ] Hitung s_u(E=5) dan s_u(E=15) untuk 100 pengguna
  - [ ] Pearson correlation per dimensi
  - [ ] **Keputusan**: FiLM viable dengan/tanpa shrinkage? Catat.
- [ ] **D3** — Global EER baseline
  - [ ] Pretrained TypeFormer + 1.000 test users + E=5
  - [ ] Hitung mean per-subject EER (verifikasi ~3,25%)
  - [ ] Hitung global EER
  - [ ] **Catat sebagai target untuk dikalahkan**
- [ ] Diskusi pembimbing tentang hasil diagnostik
- [ ] (Jika perlu) Update implementation_guide.md dan arah penelitian

### Fase 3: AS-Norm
- [ ] `cohort_sampler.py` selesai, cohort tetap disimpan
- [ ] Embedding cohort di-cache
- [ ] Embedding test set di-cache
- [ ] `znorm.py`, `tnorm.py`, `snorm.py`, `asnorm.py` selesai
- [ ] Unit test untuk AS-Norm passes (identical embedding, epsilon floor, K validation)
- [ ] Threshold strategies (`per_user_*.py`) re-run untuk konsistensi
- [ ] B1, B2 (kontrol metode jarak)
- [ ] C1, C2, C3 (replikasi negative findings)
- [ ] N1, N2, N3 (ablasi score norm family)
- [ ] Sweep K pada validation untuk N4
- [ ] N4 (AS-Norm utama) untuk semua E
- [ ] Sweep cohort size sebagai sensitivity check

### Fase 4: FiLM
- [ ] `user_stats.py` selesai
- [ ] Unit test user_stats untuk edge case (E=1, padding handling)
- [ ] `film_head.py` selesai dengan inisialisasi identitas
- [ ] Verifikasi identity init: forward pass pertama harus `e' ≈ e`
- [ ] `sampler.py` multi-E sampling selesai
- [ ] `trainer.py` selesai
- [ ] Training F1 (FiLM saja) dengan 3 seeds
- [ ] Monitoring γ dan β norm sepanjang training
- [ ] Sweep weight decay pada validation (3 nilai)
- [ ] Training final F1 dengan weight decay terbaik
- [ ] Evaluasi F1 untuk semua E
- [ ] Eksperimen ablasi: mask mean di s_u (konfirmasi NF1)
- [ ] Training FN (FiLM + AS-Norm) dengan 3 seeds
- [ ] Evaluasi FN untuk semua E

### Fase 5: Eksperimen Matriks Penuh
- [ ] Semua 12 konfigurasi × 5 E memiliki hasil tersimpan
- [ ] Generate tabel ringkasan CSV
- [ ] Plot DET curve untuk konfigurasi utama
- [ ] Generate histogram μ_u, σ_u untuk N4
- [ ] Konsistensi sanity check semua hasil

### Fase 6: Analisis Statistik
- [ ] Wilcoxon signed-rank untuk 5 perbandingan utama
- [ ] Bonferroni correction (α=0.01)
- [ ] Bootstrap CI 95% untuk global EER setiap konfigurasi
- [ ] Plot DET curve dengan CI band
- [ ] Analisis menagerie (sheep/goat/lamb/wolf)
- [ ] Tabel final untuk tesis

### Fase 7: (Opsional) Validasi KVC
- [ ] Akses CodaLab/Codabench disetujui
- [ ] Adaptasi `kvc_loader.py`
- [ ] Re-run 3-4 konfigurasi top pada KVC
- [ ] Bandingkan ranking dengan hasil Aalto

### Fase 8: Penulisan Tesis
- [ ] Bab 2 direvisi (ICC → biometric menagerie)
- [ ] Bab 3 direvisi sesuai metodologi baru
- [ ] Bab 4 (hasil) dengan tabel dan figure
- [ ] Bab 5 (kesimpulan) dengan limitasi yang jujur

---

## Catatan Akhir

Dokumen ini adalah *living document*. Setiap kali Anda menemukan:
- Asumsi yang ternyata salah saat implementasi,
- Hyperparameter yang harus diubah berdasarkan diagnostik,
- Arah penelitian yang dipivot berdasarkan diskusi pembimbing,

**update dokumen ini** dan tandai versi baru. Reproducibility tergantung pada konsistensi antara dokumen dan kode.

Jika ada bagian dari dokumen ini yang tidak jelas, kontradiktif, atau Anda menemukan praktik yang lebih baik, **jangan ragu untuk menulis ulang**. Dokumen ini bukan kontrak; dokumen ini adalah peta. Peta yang baik diperbarui sesuai medan yang ditemukan.
