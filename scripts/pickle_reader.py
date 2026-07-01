from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

PICKLE_PATH  = ROOT / "data" / "source" / "LSWMD.pkl"
PARQUET_PATH = ROOT / "data" / "datasets" / "wm811k_sample_1800.parquet"


def parse_label(x):
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


def parse_train_test(x):
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


def clean_wafer_index(x):
    try:
        return int(float(x))
    except:
        return -1


def build_parquet(num_per_class: int = 200) -> pd.DataFrame:
    valid_classes = [
        "center", "donut", "edge-loc", "edge-ring",
        "loc", "near-full", "none", "random", "scratch",
    ]

    df = pd.read_pickle(PICKLE_PATH)

    df["failureType"]    = df["failureType"].apply(parse_label)
    df["trianTestLabel"] = df["trianTestLabel"].apply(parse_train_test)
    df["waferIndex"]     = df["waferIndex"].apply(clean_wafer_index)

    df = df[df["failureType"].notna() & df["failureType"].isin(valid_classes)]

    samples = [
        sub.sample(n=min(len(sub), num_per_class), random_state=42)
        for c in valid_classes
        if len(sub := df[df["failureType"] == c]) > 0
    ]
    df = pd.concat(samples).reset_index(drop=True)

    df["failureType"] = df["failureType"].astype(str)
    df["waferMap"]    = df["waferMap"].apply(lambda x: np.array(x, dtype=object).tolist())

    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, engine="pyarrow", index=False)

    print(f"Saved: {PARQUET_PATH}")
    print(f"Shape: {df.shape}")
    print(df["failureType"].value_counts())
    return df


if __name__ == "__main__":
    build_parquet()
