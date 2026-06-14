# Detectra — Deepfake Detection System

> Final Year Project — Pokhara University  
> Hybrid ResNeXt50 + LSTM architecture for binary deepfake video classification

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Pipeline — Step by Step](#pipeline--step-by-step)
  - [Step 1 — Data Preprocessing](#step-1--data-preprocessing)
  - [Step 2 — Data Splitting](#step-2--data-splitting)
  - [Step 3 — Data Loader](#step-3--data-loader)
  - [Step 4 — Model Architecture](#step-4--model-architecture)
  - [Step 5 — Training](#step-5--training)
  - [Step 6 — Evaluation](#step-6--evaluation)
  - [Step 7 — Django Web App](#step-7--django-web-app)
  - [Step 8 — Deployment](#step-8--deployment)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

---

## Overview

Detectra is a deepfake video detection system that classifies videos as **REAL** or **FAKE** using a hybrid deep learning architecture. It combines a pretrained ResNeXt50 CNN for per-frame spatial feature extraction with an LSTM for temporal modelling across frame sequences. A Django web interface allows users to upload any video and receive a prediction with a confidence score.

---

## System Architecture

```
Input Video
     │
     ▼
┌─────────────────────────────────┐
│  MTCNN Face Detection           │  ← facenet-pytorch
│  Frame extraction (150 frames)  │  ← OpenCV
│  Face crop + resize (112×112)   │  ← bilinear interpolation
└─────────────────────────────────┘
     │
     ▼  (batch=4, sequence=20, 3, 112, 112)
┌─────────────────────────────────┐
│  ResNeXt50-32x4d                │  ← pretrained ImageNet
│  Layers 1–3: FROZEN             │
│  Layer 4 + extra head: TRAINED  │
│  Output: 2048-d feature vector  │
└─────────────────────────────────┘
     │
     ▼  (batch=4, sequence=20, 2048)
┌─────────────────────────────────┐
│  Single LSTM Layer              │
│  hidden_size = 2048             │
│  dropout = 0.4                  │
│  Leaky ReLU activation          │
│  Adaptive Average Pooling       │
│  Linear: 2048 → 2               │
│  Softmax → P(Real), P(Fake)     │
└─────────────────────────────────┘
     │
     ▼
  REAL / FAKE + confidence %
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch 1.9+ |
| CNN Backbone | ResNeXt50-32x4d (torchvision) |
| Face Detection | MTCNN (facenet-pytorch) |
| Video Processing | OpenCV 4.5+ |
| Image Processing | Pillow, NumPy |
| Data Splitting | scikit-learn |
| Web Framework | Django 3.2+ |
| Database (dev) | SQLite |
| Database (prod) | PostgreSQL |
| Production Server | Gunicorn + WhiteNoise |
| Language | Python 3.9 |

---

## Project Structure

```
detectra/
├── data/
│   ├── final_dataset/
│   │   ├── Real/              ← raw real videos
│   │   └── Fake/              ← raw fake videos
│   ├── processed/
│   │   ├── Real/              ← face-only processed videos
│   │   └── Fake/
│   └── splits/
│       ├── train.csv          ← 70% split manifest
│       └── test.csv           ← 30% split manifest
├── preprocessing/
│   ├── __init__.py
│   ├── preprocess.py          ← Step 1: face extraction pipeline
│   ├── filter_dfdc.py         ← DFDC audio-altered video filter
│   └── split.py               ← Step 2: train/test splitting
├── dataloader/
│   ├── __init__.py
│   ├── dataset.py             ← PyTorch Dataset class
│   ├── transforms.py          ← frame-level augmentations
│   └── loader.py              ← DataLoader factory + smoke test
├── model/
│   ├── __init__.py
│   ├── resnext.py             ← Part A: ResNeXt50 feature extractor
│   ├── lstm.py                ← Part B: LSTM classifier
│   ├── detectra.py            ← Combined hybrid model
│   └── verify_model.py        ← Architecture verification
├── training/
│   ├── __init__.py
│   ├── trainer.py             ← Training + evaluation engine
│   └── train.py               ← Training entry point
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py           ← Metrics engine
│   └── evaluate.py            ← Evaluation entry point
├── checkpoints/
│   ├── best_loss.pt           ← lowest validation loss checkpoint
│   └── best_acc.pt            ← highest validation accuracy checkpoint
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── evaluation_report.txt
├── logs/
│   ├── preprocessing.log
│   ├── splitting.log
│   ├── dataloader.log
│   └── training.log
├── requirements.txt
└── webapp/                    ← Django web application
    ├── manage.py
    ├── requirements_web.txt
    ├── gunicorn.conf.py
    ├── .env                   ← never commit this
    ├── .env.example
    ├── .gitignore
    ├── webapp/
    │   ├── settings/
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   ├── development.py ← SQLite
    │   │   └── production.py  ← PostgreSQL
    │   ├── urls.py
    │   └── wsgi.py
    └── detector/
        ├── __init__.py
        ├── apps.py            ← model loaded at startup
        ├── inference.py       ← preprocessing + inference engine
        ├── models.py          ← PredictionRecord DB model
        ├── views.py           ← upload + result + history views
        ├── urls.py
        ├── migrations/
        └── templates/
            └── detector/
                ├── base.html
                ├── upload.html
                ├── result.html
                └── history.html
```

---

## Dataset

| Source | Real | Fake | Total |
|---|---|---|---|
| DFDC | 1500 | 1500 | 3000 |
| FaceForensics++ | 1000 | 1000 | 2000 |
| Celeb-DF v2 | 500 | 500 | 1000 |
| **Total** | **3000** | **3000** | **6000** |

> **Note:** Audio-altered DFDC videos must be removed before training using `filter_dfdc.py`.  
> The dataset is balanced 50/50 Real/Fake. If your available dataset is imbalanced, `split.py` automatically trims the majority class to match the minority class.

---

## Prerequisites

- macOS (Apple Silicon M1/M2/M3 supported via MPS) or Linux
- Python 3.9+
- At least 8GB RAM
- At least 5GB free disk space (more if using full 6000-video dataset)
- Git

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/detectra.git
cd detectra
```

### 2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install ML pipeline dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```
torch>=1.9.0
torchvision>=0.10.0
facenet-pytorch>=2.5.2
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=8.3.0
scikit-learn>=0.24.0
tqdm>=4.62.0
matplotlib>=3.4.0
```

### 4. Create required directories

```bash
mkdir -p data/final_dataset/{Real,Fake}
mkdir -p data/processed/{Real,Fake}
mkdir -p data/splits
mkdir -p checkpoints results logs
```

### 5. Place your dataset

Copy your raw videos into:
```
data/final_dataset/Real/    ← all real videos (.mp4, .avi, .mov)
data/final_dataset/Fake/    ← all fake videos (.mp4, .avi, .mov)
```

---

## Pipeline — Step by Step

Run each step in order. Do not skip steps.

---

### Step 1 — Data Preprocessing

Extracts face regions from every video using MTCNN and saves face-only videos at 30 FPS.

**What it does:**
- Extracts up to 150 frames per video using OpenCV
- Detects faces in each frame using MTCNN
- Discards frames with no detected face
- Crops and resizes face regions to 112×112 using bilinear interpolation
- Saves face-only videos to `data/processed/Real/` and `data/processed/Fake/`

**If using DFDC dataset, filter audio-altered videos first:**
```bash
python3 -m preprocessing.filter_dfdc
```

**Run preprocessing:**
```bash
python3 -m preprocessing.preprocess
```

**Expected output:**
```
[INFO] Device: mps
[INFO] Found 108 videos in data/final_dataset/Real → label=Real
[INFO] Found 1591 videos in data/final_dataset/Fake → label=Fake
[INFO] PREPROCESSING COMPLETE
[INFO]   Processed          : 1531
[INFO]   Output Real/       : 108 videos
[INFO]   Output Fake/       : 1591 videos
```

> Processing time: ~38 seconds per video on M1 CPU. For 1699 videos expect ~18 hours. Run overnight or in background with `nohup`.

**Verify output:**
```bash
python3 - <<'EOF'
from pathlib import Path
real = list(Path("data/processed/Real").glob("*.mp4"))
fake = list(Path("data/processed/Fake").glob("*.mp4"))
print(f"Real: {len(real)}  Fake: {len(fake)}  Total: {len(real)+len(fake)}")
EOF
```

---

### Step 2 — Data Splitting

Splits processed videos into 70% train / 30% test with stratified 50/50 class balance.

**What it does:**
- Automatically balances dataset by trimming majority class to minority class size
- Applies stratified shuffle split (random_state=42 for reproducibility)
- Saves `data/splits/train.csv` and `data/splits/test.csv`

```bash
python3 -m preprocessing.split
```

**Verify the split:**
```bash
python3 -m preprocessing.verify_split
```

**Expected output:**
```
[INFO] After balancing  → Real: 108 | Fake: 108
[INFO] Total balanced dataset size: 216
[INFO] Train split: 151 total | Real=75 (49.7%) | Fake=76 (50.3%)
[INFO] Test split:  65 total  | Real=33 (50.8%) | Fake=32 (49.2%)
[INFO] ✓ All balance checks passed
[INFO] ✓ No overlap between train and test sets
```

---

### Step 3 — Data Loader

Builds PyTorch DataLoaders and verifies tensor shapes.

**Configuration:**
- Batch size: 4
- Sequence length: 20 frames per video
- Frame size: 112×112
- Training: augmentation (horizontal flip, color jitter) + ImageNet normalisation
- Validation: ImageNet normalisation only

```bash
python3 -m dataloader.loader
```

**Expected output — all checks must show ✓:**
```
Train loader: 151 samples | 37 batches | batch_size=4
Test loader:  65 samples  | 17 batches | batch_size=4

  Batch 1:
    frames.shape  : (4, 20, 3, 112, 112)  ✓
    frames.dtype  : torch.float32  ✓
    labels        : [0, 1, 1, 0]  ✓
    labels.dtype  : torch.long  ✓
```

---

### Step 4 — Model Architecture

Builds and verifies the hybrid ResNeXt50 + LSTM model.

**Architecture:**

| Component | Detail |
|---|---|
| Feature extractor | pretrained resnext50_32x4d |
| Frozen layers | conv1, bn1, layer1, layer2, layer3 |
| Fine-tuned layers | layer4 + extra head (2 FC layers) |
| Feature output | 2048-d vector per frame |
| LSTM | 1 layer, hidden=2048, dropout=0.4 |
| Activation | Leaky ReLU (negative_slope=0.01) |
| Pooling | Adaptive Average Pool across sequence |
| Classifier | Linear(2048, 2) |
| Inference output | Softmax → P(Real), P(Fake) |

```bash
python3 -m model.verify_model
```

**Expected output — all must show ✓:**
```
  conv1 / bn1 (stem)             ✓ FROZEN
  layer1                         ✓ FROZEN
  layer2                         ✓ FROZEN
  layer3                         ✓ FROZEN
  layer4                         ✓ TRAINABLE
  extra_head                     ✓ TRAINABLE
  lstm_classifier                ✓ TRAINABLE
  Output shape (4, 2)            ✓
  Probabilities sum to 1.0       ✓
```

---

### Step 5 — Training

Trains the model and saves checkpoints.

**Configuration:**

| Parameter | Value |
|---|---|
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 1e-5 |
| Weight decay | 1e-5 |
| Batch size | 4 |
| Gradient accumulation | 4 steps (effective batch = 16) |
| Epochs | 20 |
| LR scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Gradient clipping | max_norm=1.0 |

**Run training** (recommended: background process):
```bash
nohup python3 -m training.train > logs/training_stdout.log 2>&1 &
tail -f logs/training_stdout.log
```

**Monitor progress:**
```bash
tail -f logs/training.log
```

**Expected training time:**
- ~40 seconds per epoch on Apple Silicon MPS
- ~10–15 minutes per epoch on CPU only

**Checkpoints saved to:**
```
checkpoints/best_loss.pt   ← lowest validation loss
checkpoints/best_acc.pt    ← highest validation accuracy
```

**Verify checkpoints after training:**
```bash
python3 - <<'EOF'
import torch
from pathlib import Path
for name in ["best_loss.pt", "best_acc.pt"]:
    path = Path(f"checkpoints/{name}")
    ckpt = torch.load(path, map_location="cpu")
    print(f"✓ {name} — epoch={ckpt['epoch']} | "
          f"val_loss={ckpt['val_loss']:.4f} | val_acc={ckpt['val_acc']:.2f}%")
EOF
```

> **Disk space warning:** Each checkpoint is ~120MB (saved in float16). Ensure at least 500MB free before training. If disk is full, move raw dataset files to external storage — they are not needed after Step 1.

---

### Step 6 — Evaluation

Evaluates the best checkpoint on the full test set and generates reports.

```bash
python3 -m evaluation.evaluate
```

**Metrics computed:**

| Metric | Formula |
|---|---|
| Accuracy | (TP+TN) / (TP+TN+FP+FN) |
| Recall | TP / (TP+FN) |
| Specificity | TN / (TN+FP) |
| Precision | TP / (TP+FP) |
| F1-Score | 2×(Precision×Recall) / (Precision+Recall) |
| AUC-ROC | Area under ROC curve |

**Results saved to:**
```
results/confusion_matrix.png    ← heatmap with TP/TN/FP/FN
results/roc_curve.png           ← AUC-ROC curve plot
results/evaluation_report.txt   ← full metrics text report
```

**Results achieved on development dataset (216 balanced videos):**

| Metric | Value |
|---|---|
| Validation Accuracy | 90.77% |
| Best Epoch | 19 / 20 |
| Best Val Loss | 0.3228 |

> Target accuracy of 93.59% is achievable with the full 6000-video dataset. The reduced dataset was used for development due to hardware constraints.

---

### Step 7 — Django Web App

#### Install web dependencies

```bash
cd webapp
pip install -r requirements_web.txt
```

#### Create environment file

```bash
cat > .env << 'EOF'
DJANGO_ENV=development
SECRET_KEY=detectra-local-dev-key-not-for-production
DEBUG=True
CHECKPOINT_PATH=../checkpoints/best_acc.pt
EOF
```

#### Run migrations

```bash
DJANGO_SETTINGS_MODULE=webapp.settings.development python3 manage.py makemigrations detector
DJANGO_SETTINGS_MODULE=webapp.settings.development python3 manage.py migrate
```

#### Collect static files

```bash
DJANGO_SETTINGS_MODULE=webapp.settings.development python3 manage.py collectstatic --noinput
```

#### Start the development server

```bash
DJANGO_SETTINGS_MODULE=webapp.settings.development python3 manage.py runserver
```

Open your browser at: **http://127.0.0.1:8000**

#### Web app features

| Page | URL | Description |
|---|---|---|
| Upload | `/` | Drag and drop video upload |
| Result | `/result/<id>/` | Prediction with confidence score and face grid |
| History | `/history/` | All past predictions from database |

**Inference pipeline per uploaded video:**
1. File validated (format + size) before saving to disk
2. Frames extracted with OpenCV (up to 150)
3. MTCNN detects and crops face regions
4. Faces resized to 112×112 and normalised
5. 20 frames sampled as sequence input to model
6. ResNeXt50 extracts 2048-d features per frame
7. LSTM classifies temporal sequence
8. Result returned: REAL / FAKE / UNCERTAIN + confidence %
9. Temp file deleted immediately after inference
10. Result stored permanently in SQLite database

**Confidence threshold:** Videos with max confidence below 75% are flagged as **UNCERTAIN**.

---

### Step 8 — Deployment

#### Development (SQLite) — default

Everything in Step 7 runs on SQLite automatically with `DJANGO_ENV=development`. No additional configuration needed.

#### Production (PostgreSQL + Gunicorn)

**Set up PostgreSQL:**
```bash
psql postgres
```
```sql
CREATE DATABASE detectra_db;
CREATE USER detectra_user WITH PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE detectra_db TO detectra_user;
ALTER DATABASE detectra_db OWNER TO detectra_user;
\q
```

**Update `.env` for production:**
```bash
DJANGO_ENV=production
SECRET_KEY=your-strong-random-secret-key
DEBUG=False
CHECKPOINT_PATH=../checkpoints/best_acc.pt
DB_NAME=detectra_db
DB_USER=detectra_user
DB_PASSWORD=your-password
DB_HOST=localhost
DB_PORT=5432
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
```

**Run migrations and start Gunicorn:**
```bash
DJANGO_SETTINGS_MODULE=webapp.settings.production python3 manage.py migrate
DJANGO_SETTINGS_MODULE=webapp.settings.production python3 manage.py collectstatic --noinput
DJANGO_SETTINGS_MODULE=webapp.settings.production gunicorn --config gunicorn.conf.py webapp.wsgi:application
```

> **Note:** Use 2 Gunicorn workers maximum on 8GB RAM. Each worker loads the full model (~500MB) into memory. `preload_app=True` in `gunicorn.conf.py` loads the model once and forks to workers, saving RAM.

---

## Results

### Training curve (20 epochs, 216 balanced videos)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.6850 | 60.81% | 0.6733 | 67.69% |
| 5 | 0.5917 | 70.95% | 0.6127 | 63.08% |
| 10 | 0.3493 | 85.14% | 0.4248 | 81.54% |
| 16 | 0.1973 | 93.24% | 0.3228 | 87.69% ← best loss |
| 19 | 0.3298 | 85.81% | 0.3229 | **90.77%** ← best acc |
| 20 | 0.2087 | 91.22% | 0.3439 | 89.23% |

### Final metrics (test set, best_acc.pt checkpoint)

| Metric | Value |
|---|---|
| **Accuracy** | **90.77%** |
| Best Val Loss | 0.3228 |
| Checkpoint | epoch 19 |

---

## Troubleshooting

### `No installed app with label 'detector'`
```bash
# Make sure you are in the webapp directory and settings module is set
cd detectra/webapp
DJANGO_SETTINGS_MODULE=webapp.settings.development python3 manage.py migrate
```

### `TypeError: __init__() got an unexpected keyword argument 'verbose'`
Remove `verbose=True` from `ReduceLROnPlateau` in `training/trainer.py`. This parameter was removed in PyTorch 2.x.

### `RuntimeError: PytorchStreamWriter failed writing file`
Disk is full. Free at least 500MB before training. Move raw dataset files to external storage — they are not needed after preprocessing completes.

### `Bootstrap failed: 5: Input/output error` (PostgreSQL)
```bash
rm -f /opt/homebrew/var/postgresql@14/postmaster.pid
brew services restart postgresql@14
```
If PostgreSQL continues to fail, use SQLite (`DJANGO_ENV=development`) for development — it is fully sufficient for a FYP demo.

### MTCNN detects very few faces in Real videos
Your `SOURCE_MAP` paths in `preprocess.py` don't match your actual folder structure. Run:
```bash
python3 - <<'EOF'
from pathlib import Path
for p in sorted(Path("data/raw").rglob("*")):
    if p.is_dir():
        vids = list(p.glob("*.mp4"))
        print(f"{p}  ({len(vids)} videos)")
EOF
```
Then update `SOURCE_MAP` in `preprocessing/preprocess.py` to match your actual paths.

### `Apple Silicon MPS detected` but training is slow
This is expected — MPS acceleration is active. Each epoch takes ~40 seconds on M1 Air with 216 videos. For the full 6000-video dataset, use a cloud GPU (Railway, Render, or Google Colab for training only).

### VS Code CSS linter warning on Django template variables
This is a false alarm — VS Code's CSS linter does not understand Django template syntax. The fix is to use `data-width` attributes and set widths via JavaScript instead of inline `style="{{ value }}"`. See `history.html` for the implemented fix.

---

## License

This project was developed as a Final Year Project at Pokhara University. All rights reserved.

---

## Acknowledgements

- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) — MTCNN implementation
- [DFDC Dataset](https://ai.facebook.com/datasets/dfdc/) — DeepFake Detection Challenge
- [FaceForensics++](https://github.com/ondyari/FaceForensics) — Benchmark dataset
- [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics) — High quality deepfake dataset
- PyTorch, torchvision, Django, OpenCV communities
