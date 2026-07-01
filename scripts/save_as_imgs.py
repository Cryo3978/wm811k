from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

PARQUET_PATH = ROOT / "data" / "datasets" / "wm811k_sample_1800.parquet"
IMG_DIR      = ROOT / "data" / "wafer_images"


def to_2d(arr):
    arr = np.array(arr, dtype=object)
    try:
        return np.stack(arr)
    except:
        return np.array(arr)


MIN_SIZE = 30


def pad_to_min_size(arr, min_size=MIN_SIZE):
    h, w = arr.shape
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    if pad_h == 0 and pad_w == 0:
        return arr
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return np.pad(arr, ((top, bottom), (left, right)), mode="constant", constant_values=0)


def save_img(arr, path: Path):
    arr = to_2d(arr).astype(np.float32)
    arr -= arr.min()
    if arr.max() != 0:
        arr /= arr.max()
    img_arr = (arr * 255).astype(np.uint8)
    img_arr = pad_to_min_size(img_arr)
    Image.fromarray(img_arr).save(path)


def save_all_images(df: pd.DataFrame | None = None):
    if df is None:
        df = pd.read_parquet(PARQUET_PATH)

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        label      = str(row["failureType"]).strip()
        lot        = str(row["lotName"])
        wafer_idx  = str(int(float(row["waferIndex"]))) if str(row["waferIndex"]) not in ("", "nan") else "unknown"
        label_dir  = IMG_DIR / label
        label_dir.mkdir(exist_ok=True)
        save_img(row["waferMap"], label_dir / f"{lot}_wafer{wafer_idx}.png")

    print(f"Done — images saved to {IMG_DIR}")


if __name__ == "__main__":
    save_all_images()
