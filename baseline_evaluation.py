from pathlib import Path
import os
import base64
import json
import re
import time
from datetime import datetime

import pandas as pd
import yaml
from jinja2 import Template
from openai import OpenAI, APIStatusError, APIConnectionError
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

load_dotenv()

ROOT     = Path(__file__).resolve().parent
LOG_DIR  = ROOT / "logs"

PARQUET_PATH = ROOT / "data" / "datasets" / "wm811k_sample_1800.parquet"
IMG_DIR      = ROOT / "data" / "wafer_images"
PROMPT_PATH  = ROOT / "scripts" / "prompts" / "baseline.yaml"

VALID_CLASSES = [
    "center", "donut", "edge-loc", "edge-ring",
    "loc", "near-full", "none", "random", "scratch",
]

PROVIDERS = {
    "openai": {
        "api_key_env":   "OPENAI_API_KEY",
        "base_url":      None,
    },
    "siliconflow": {
        "api_key_env":   "SILICONFLOW_API_KEY",
        "base_url":      "https://api.siliconflow.cn/v1",
    },
    "ollama": {
        "api_key_env":   None,
        "base_url":      os.getenv("OPENAI_BASE_URL", "http://localhost:11435/v1"),
        "default_model": "qwen3.5:latest",
        "extra_body":    {"enable_thinking": True},
    },
}

PROVIDER  = os.getenv("PROVIDER", "openai").lower()
_pcfg     = PROVIDERS.get(PROVIDER)
if _pcfg is None:
    raise ValueError(f"Unknown PROVIDER={PROVIDER!r}. Choose from: {list(PROVIDERS)}")

client = OpenAI(
    api_key=os.getenv(_pcfg["api_key_env"], "") if _pcfg["api_key_env"] else "ollama",
    timeout=float(os.getenv("EVA_TIMEOUT", "300")),
    max_retries=0,  # we do our own retry loop below, with visible logging between attempts
    **({"base_url": _pcfg["base_url"]} if _pcfg["base_url"] else {}),
)
EVA_MODEL   = os.getenv("EVA_MODEL", _pcfg.get("default_model", "gpt-4o-mini"))
MAX_RETRIES = int(os.getenv("EVA_MAX_RETRIES", "3"))

# ─── prompt ──────────────────────────────────────────────────────────────────

cfg    = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))
prompt = Template(cfg["template"]).render(
    role=cfg["role"],
    instruction=cfg["instruction"],
    input=cfg["input"],
    classes=cfg["classes"],
)

# ─── helpers ─────────────────────────────────────────────────────────────────

def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def parse_prediction(raw: str) -> tuple[str, str]:
    """Return (predicted_class, reasoning). Falls back gracefully on bad JSON."""
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        data    = json.loads(cleaned)
        pred    = str(data.get("type", "")).lower().strip()
        reasoning = str(data.get("reasoning", ""))
        return pred, reasoning
    except Exception:
        lower = raw.lower()
        for cls in VALID_CLASSES:
            if cls in lower:
                return cls, raw
        return "unknown", raw


def query_model(img_path: Path) -> str:
    kwargs = {}
    if _pcfg.get("extra_body"):
        kwargs["extra_body"] = _pcfg["extra_body"]

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=EVA_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encode_image(img_path)}"},
                            },
                        ],
                    }
                ],
                max_tokens=2048,
                **kwargs,
            )
            return response.choices[0].message.content
        except (APIStatusError, APIConnectionError) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt  # 2s, 4s, 8s, ...
                print(f"          [retry {attempt}/{MAX_RETRIES - 1}] {type(e).__name__}: {e} — retrying in {wait}s")
                time.sleep(wait)

    raise last_err


# ─── evaluation loop ─────────────────────────────────────────────────────────

def _sample_df(df: pd.DataFrame, n_samples: int | None, n_per_class: int | None) -> pd.DataFrame:
    if n_per_class is not None:
        parts = [
            grp.sample(n=min(len(grp), n_per_class), random_state=42)
            for _, grp in df.groupby("failureType")
        ]
        return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    if n_samples is not None:
        return df.head(n_samples)
    return df


def run_evaluation(n_samples: int | None = None, n_per_class: int | None = None):
    LOG_DIR.mkdir(exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"eval_{ts}.jsonl"

    df = pd.read_parquet(PARQUET_PATH)
    df = _sample_df(df, n_samples, n_per_class)

    y_true, y_pred = [], []
    records = []

    print(f"Model   : {EVA_MODEL}")
    print(f"Samples : {len(df)}")
    print(f"Log     : {log_path}\n")

    for i, (_, case) in enumerate(df.iterrows(), 1):
        lot_name  = case["lotName"]
        wafer_idx_raw = int(case["waferIndex"])
        wafer_idx_str = "unknown" if wafer_idx_raw == -1 else str(wafer_idx_raw)
        label     = case["failureType"]

        img_path = IMG_DIR / label / f"{lot_name}_wafer{wafer_idx_str}.png"

        if not img_path.exists():
            print(f"[{i:>4}/{len(df)}] ✗  IMAGE NOT FOUND: {img_path}")
            y_true.append(label)
            y_pred.append("unknown")
            records.append({
                "i": i, "lot": lot_name, "wafer_idx": wafer_idx_str,
                "true": label, "pred": "unknown", "correct": False,
                "reasoning": "", "raw": f"IMAGE NOT FOUND: {img_path}",
            })
            continue

        img_size_kb = img_path.stat().st_size / 1024
        try:
            raw = query_model(img_path)
        except Exception as e:
            print(f"[{i:>4}/{len(df)}] ✗  QUERY FAILED after retries: {type(e).__name__}: {e}")
            y_true.append(label)
            y_pred.append("unknown")
            records.append({
                "i": i, "lot": lot_name, "wafer_idx": wafer_idx_str,
                "true": label, "pred": "unknown", "correct": False,
                "reasoning": "", "raw": f"QUERY FAILED: {type(e).__name__}: {e}",
            })
            continue

        pred, reasoning   = parse_prediction(raw)
        correct           = pred == label
        status            = "✓" if correct else "✗"

        print(f"[{i:>4}/{len(df)}] {status}  true={label:<12} pred={pred:<12}  {lot_name}_wafer{wafer_idx_str}  ({img_size_kb:.1f} KB)")
        if not correct:
            print(f"          raw: {raw[:160]}")

        record = {
            "i":          i,
            "lot":        lot_name,
            "wafer_idx":  wafer_idx_str,
            "true":       label,
            "pred":       pred,
            "correct":    correct,
            "reasoning":  reasoning,
            "raw":        raw,
        }
        records.append(record)

        # write to log immediately so partial runs are still recoverable
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        y_true.append(label)
        y_pred.append(pred)

    _print_metrics(y_true, y_pred)

    # also save a tidy CSV for easy inspection in Excel / pandas
    csv_path = log_path.with_suffix(".csv")
    pd.DataFrame(records).drop(columns=["raw"]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nDetailed log : {log_path}")
    print(f"Summary CSV  : {csv_path}")


def _print_metrics(y_true: list, y_pred: list):
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy : {acc:.4f}  ({int(acc * len(y_true))}/{len(y_true)})")

    labels = sorted(set(y_true))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    cm     = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df  = pd.DataFrame(cm, index=labels, columns=labels)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm_df.to_string())

    unknown_n = y_pred.count("unknown")
    if unknown_n:
        print(f"\n[!] {unknown_n} response(s) failed to parse → counted as 'unknown'. Check the .jsonl log.")

    print("=" * 60)


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline LLM evaluation on WM-811K")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--n", type=int, default=None, help="First N samples from the dataset")
    group.add_argument("--n-per-class", type=int, default=None, help="N samples per defect class (stratified)")
    args = parser.parse_args()

    run_evaluation(n_samples=args.n, n_per_class=args.n_per_class)
