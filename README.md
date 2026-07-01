# WM-811K Wafer Map Failure Analysis

End-to-end pipeline: from the raw WM-811K pickle dataset to an LLM-based baseline evaluation.

---

## Project Structure

```
wm811k/
├── data/                        # excluded from git
│   ├── source/
│   │   └── LSWMD.pkl            # raw dataset (download manually)
│   ├── datasets/
│   │   └── wm811k_sample_1800.parquet
│   └── wafer_images/
│       ├── center/
│       ├── donut/
│       ├── edge-loc/
│       ├── edge-ring/
│       ├── loc/
│       ├── near-full/
│       ├── none/
│       ├── random/
│       └── scratch/
├── logs/                        # excluded from git
│   ├── eval_YYYYMMDD_HHMMSS.jsonl
│   └── eval_YYYYMMDD_HHMMSS.csv
├── scripts/
│   ├── pickle_reader.py         # standalone: pickle → parquet
│   ├── save_as_imgs.py          # standalone: parquet → images
│   ├── parquet_reader.py        # standalone: show parquet info
│   └── prompts/
│       └── baseline.yaml        # classification prompt template
├── .env                         # API keys (not committed)
├── .env.example                 # template for .env
├── .gitattributes               # enforce LF line endings
├── requirements.txt
├── build_dataset.py             # all-in-one pipeline (recommended entry point)
└── baseline_evaluation.py       # LLM evaluation with metrics
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Raw Dataset

Download `LSWMD.pkl` from [Kaggle — WM-811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) and place it at:

```
data/source/LSWMD.pkl
```

### 3. Configure API Keys

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-...
EVA_MODEL=gpt-4o-mini
```

---

## Pipeline

### Step A — Build Dataset (pickle → parquet + images)

```bash
python build_dataset.py                    # default: 200 samples per class
python build_dataset.py --n-per-class 50   # 50 samples per class
```

Runs three stages in sequence:

| Stage | Input | Output | Description |
|-------|-------|--------|-------------|
| 1 | `data/source/LSWMD.pkl` | `data/datasets/wm811k_sample_1800.parquet` | Cleans labels, samples up to 200 per class (9 classes → ≤1800 rows) |
| 2 | parquet | `data/wafer_images/{class}/*.png` | Renders each wafer map as a grayscale PNG |
| 3 | parquet | stdout | Prints head, shape, columns, and class distribution |

Sample output:

```
[1/3] Reading pickle: .../data/source/LSWMD.pkl
    Saved: .../data/datasets/wm811k_sample_1800.parquet  shape=(1800, 6)
    center       200
    donut        200
    ...

[2/3] Saving wafer map images to: .../data/wafer_images
    Done — 1800 images written.

[3/3] Parquet preview: .../data/datasets/wm811k_sample_1800.parquet

  Shape : (1800, 6)
  Columns: ['lotName', 'waferIndex', 'failureType', 'trianTestLabel', 'dieSize', 'waferMap']

  Class distribution:
  center       200
  donut        200
  edge-loc     200
  ...
```

---

### Step B — Baseline Evaluation

```bash
# evaluate all 1800 samples
python baseline_evaluation.py

# evaluate first N samples
python baseline_evaluation.py --n 20

# evaluate N samples per defect class (stratified, recommended for quick runs)
python baseline_evaluation.py --n-per-class 5
```

Sends each wafer map image to an OpenAI vision model with the prompt from `scripts/prompts/baseline.yaml`. The model returns a JSON prediction; results are compared against the ground-truth `failureType` from the parquet.

**Per-sample output:**

```
Model   : gpt-4o-mini
Samples : 45

[   1/ 45] ✓  true=center       pred=center       lot123_wafer5   (3.2 KB)
[   2/ 45] ✗  true=edge-ring    pred=random       lot456_wafer2   (4.1 KB)
          raw: {"type": "random", "reasoning": "..."}
...
```

The `(X.X KB)` confirms the image was read from disk and base64-encoded into the request.

**Metrics summary:**

```
============================================================
EVALUATION RESULTS
============================================================

Accuracy : 0.6222  (28/45)

Classification Report:
              precision    recall  f1-score   support
      center       0.80      0.80      0.80         5
       donut       0.60      0.60      0.60         5
    edge-loc       0.40      0.40      0.40         5
   edge-ring       0.80      0.80      0.80         5
         ...

Confusion Matrix (rows=true, cols=pred):
           center  donut  edge-loc  ...
center          4      0         1
donut           0      3         0
...
============================================================
```

**Run logs** are saved to `logs/` automatically:

| File | Contents |
|------|----------|
| `eval_YYYYMMDD_HHMMSS.jsonl` | One record per sample including full raw model response |
| `eval_YYYYMMDD_HHMMSS.csv` | Same without raw response, for easy inspection in Excel / pandas |

---

## Defect Classes

| Class | Description |
|-------|-------------|
| `center` | Defects concentrated in the wafer center |
| `donut` | Center normal; ring-shaped defective band |
| `edge-loc` | Localized defects near the wafer edge |
| `edge-ring` | Continuous ring defects near the edge |
| `loc` | Localized clustered defects anywhere |
| `near-full` | Almost full-wafer defect coverage |
| `none` | Clean wafer / negligible defects |
| `random` | Scattered, randomly distributed defects |
| `scratch` | Linear / streak-like defect pattern |

---

## Running Individual Scripts

Each script in `scripts/` can also be run standalone:

```bash
# pickle → parquet only
python scripts/pickle_reader.py

# parquet → images only
python scripts/save_as_imgs.py

# show parquet info only
python scripts/parquet_reader.py
```

---

## Notes

- All paths use `pathlib` resolved relative to each script's location — scripts work correctly regardless of the working directory.
- `.env` is not committed. Use `.env.example` as a template.
- `data/` and `logs/` are excluded from git (see `.gitignore`).
- `.gitattributes` enforces LF line endings across all platforms.
