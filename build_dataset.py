"""
build_dataset.py
================
All-in-one pipeline: pickle → parquet → images → dataset preview.

Steps
-----
1. Read LSWMD.pkl, clean labels, sample up to 200 per class, save as parquet.
2. Read parquet, render each wafer map as a grayscale PNG, save under data/wafer_images/{class}/.
3. Print a summary of the parquet contents.

Usage
-----
    python build_dataset.py                   # default: 200 per class
    python build_dataset.py --n-per-class 50  # 50 per class
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent

PICKLE_PATH  = ROOT / "data" / "source" / "LSWMD.pkl"
PARQUET_PATH = ROOT / "data" / "datasets" / "wm811k_sample_1800.parquet"
IMG_DIR      = ROOT / "data" / "wafer_images"

VALID_CLASSES = [
    "center", "donut", "edge-loc", "edge-ring",
    "loc", "near-full", "none", "random", "scratch",
]
NUM_PER_CLASS = 200


# ─── helpers ─────────────────────────────────────────────────────────────────

def _parse_label(x):
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, (list, np.ndarray)):
        try:
            x = x[0]
            if isinstance(x, (list, np.ndarray)):
                x = x[0]
            return str(x).lower().strip()
        except:
            return None
    return str(x).lower().strip()


def _parse_train_test(x):
    if isinstance(x, float) and np.isnan(x):
        return "unknown"
    if isinstance(x, (list, np.ndarray)):
        try:
            return str(x[0][0]).lower().strip()
        except:
            try:
                return str(x[0]).lower().strip()
            except:
                return "unknown"
    return str(x).lower().strip()


def _clean_wafer_index(x):
    try:
        return int(float(x))
    except:
        return -1


def _to_2d(arr):
    arr = np.array(arr, dtype=object)
    try:
        return np.stack(arr)
    except:
        return np.array(arr)


MIN_IMG_SIZE = 30


def _pad_to_min_size(arr, min_size=MIN_IMG_SIZE):
    h, w = arr.shape
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    if pad_h == 0 and pad_w == 0:
        return arr
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return np.pad(arr, ((top, bottom), (left, right)), mode="constant", constant_values=0)


def _save_img(arr, path: Path):
    arr = _to_2d(arr).astype(np.float32)
    arr -= arr.min()
    if arr.max() != 0:
        arr /= arr.max()
    img_arr = (arr * 255).astype(np.uint8)
    img_arr = _pad_to_min_size(img_arr)
    Image.fromarray(img_arr).save(path)


# ─── Step 1: pickle → parquet ────────────────────────────────────────────────

def pickle_to_parquet() -> pd.DataFrame:
    print(f"[1/3] Reading pickle: {PICKLE_PATH}")
    df = pd.read_pickle(PICKLE_PATH)

    df["failureType"]    = df["failureType"].apply(_parse_label)
    df["trianTestLabel"] = df["trianTestLabel"].apply(_parse_train_test)
    df["waferIndex"]     = df["waferIndex"].apply(_clean_wafer_index)

    df = df[df["failureType"].notna() & df["failureType"].isin(VALID_CLASSES)]

    samples = [
        sub.sample(n=min(len(sub), NUM_PER_CLASS), random_state=42)
        for c in VALID_CLASSES
        if len(sub := df[df["failureType"] == c]) > 0
    ]
    df = pd.concat(samples).reset_index(drop=True)

    df["failureType"] = df["failureType"].astype(str)
    df["waferMap"]    = df["waferMap"].apply(lambda x: np.array(x, dtype=object).tolist())

    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, engine="pyarrow", index=False)

    print(f"    Saved: {PARQUET_PATH}  shape={df.shape}")
    print(df["failureType"].value_counts().to_string())
    return df


# ─── Step 2: parquet → images ────────────────────────────────────────────────

def parquet_to_images(df: pd.DataFrame):
    print(f"\n[2/3] Saving wafer map images to: {IMG_DIR}")
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        label     = str(row["failureType"]).strip()
        lot       = str(row["lotName"])
        idx_raw   = row["waferIndex"]
        wafer_idx = str(int(float(idx_raw))) if str(idx_raw) not in ("", "nan", "-1") else "unknown"

        label_dir = IMG_DIR / label
        label_dir.mkdir(exist_ok=True)
        _save_img(row["waferMap"], label_dir / f"{lot}_wafer{wafer_idx}.png")

    print(f"    Done — {sum(1 for _ in IMG_DIR.rglob('*.png'))} images written.")


# ─── Step 3: parquet preview ─────────────────────────────────────────────────

def show_parquet(df: pd.DataFrame):
    print(f"\n[3/3] Parquet preview: {PARQUET_PATH}")
    print("\n  Head (5 rows, non-waferMap columns):")
    display_cols = [c for c in df.columns if c != "waferMap"]
    print(df[display_cols].head().to_string(index=False))

    print(f"\n  Shape : {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    print("\n  Class distribution:")
    print(df["failureType"].value_counts().to_string())


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build WM-811K dataset: pickle → parquet + images")
    parser.add_argument(
        "--n-per-class", type=int, default=NUM_PER_CLASS,
        help=f"Max samples per defect class (default: {NUM_PER_CLASS})",
    )
    args = parser.parse_args()

    df = pickle_to_parquet(num_per_class=args.n_per_class)
    parquet_to_images(df)
    show_parquet(df)
