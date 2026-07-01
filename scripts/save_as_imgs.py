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


def save_img(arr, path: Path):
    arr = to_2d(arr).astype(np.float32)
    arr -= arr.min()
    if arr.max() != 0:
        arr /= arr.max()
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


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
