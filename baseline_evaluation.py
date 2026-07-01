from pathlib import Path
import os
import base64

import pandas as pd
import yaml
from jinja2 import Template
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent

PARQUET_PATH  = ROOT / "data" / "datasets" / "wm811k_sample_1800.parquet"
IMG_DIR       = ROOT / "data" / "wafer_images"
PROMPT_PATH   = ROOT / "scripts" / "prompts" / "baseline.yaml"

client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EVA_MODEL = os.getenv("EVA_MODEL", "gpt-4o-mini")

df   = pd.read_parquet(PARQUET_PATH)
df_5 = df.head()

cfg = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))
tpl = Template(cfg["template"])

prompt = tpl.render(
    role=cfg["role"],
    instruction=cfg["instruction"],
    input=cfg["input"],
    classes=cfg["classes"],
)

print(prompt)


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


for _, case in df_5.iterrows():
    lot_name   = case["lotName"]
    wafer_idx  = int(case["waferIndex"])
    label      = case["failureType"]

    img_path = IMG_DIR / label / f"{lot_name}_wafer{wafer_idx}.png"

    image_b64 = encode_image(img_path)

    response = client.chat.completions.create(
        model=EVA_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        max_tokens=2048,
    )

    print(f"\n[{label}] {lot_name}_wafer{wafer_idx}")
    print(response.choices[0].message.content)
