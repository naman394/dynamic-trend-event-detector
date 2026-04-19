"""
Build df_clean.pkl — ABC News headlines restricted to 2019–2021 with cleaned text.

Also embedded in notebooks/11_topic_modeling_lda_bertopic.ipynb (preferred).

Output: data/df_clean.pkl
Columns:
  - publish_date  (datetime)
  - headline_text (raw)
  - clean_text    (lowercase, alphanumeric, min token length — same rules as baseline.py)

Run:  python src/prepare_df_clean.py
"""

import os
import re
import pandas as pd

DATA_CSV = "data/abcnews-date-text 2.csv"
OUT_PKL  = "data/df_clean.pkl"
YEAR_MIN, YEAR_MAX = 2019, 2021


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\b\w{1,2}\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)
    df = pd.read_csv(DATA_CSV)
    df["publish_date"] = pd.to_datetime(df["publish_date"].astype(str), format="%Y%m%d")
    df["clean_text"] = df["headline_text"].astype(str).apply(clean_text)
    df = df[(df["publish_date"].dt.year >= YEAR_MIN) & (df["publish_date"].dt.year <= YEAR_MAX)]
    df = df[df["clean_text"].str.len() > 5].reset_index(drop=True)
    df.to_pickle(OUT_PKL)
    print(f"Saved {len(df):,} rows → {OUT_PKL}")
    print(f"  Date range: {df['publish_date'].min().date()} → {df['publish_date'].max().date()}")


if __name__ == "__main__":
    main()
