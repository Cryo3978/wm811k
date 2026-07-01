from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

PARQUET_PATH = ROOT / "data" / "datasets" / "wm811k_sample_1800.parquet"


def show_info(df: pd.DataFrame | None = None):
    if df is None:
        df = pd.read_parquet(PARQUET_PATH)

    print("=== Head ===")
    print(df.head())
    print("\n=== Shape ===")
    print(df.shape)
    print("\n=== Columns ===")
    print(df.columns.tolist())
    print("\n=== failureType counts ===")
    print(df["failureType"].value_counts())
    return df


if __name__ == "__main__":
    show_info()
