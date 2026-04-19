"""
Spike detection from ``topics_over_time.csv`` (one rule only).

Velocity per topic: v[t] = (freq[t] - freq[t-1]) / (freq[t-1] + 1).
We flag a bin when that velocity's **z-score within the same topic's time series**
exceeds a threshold (default 2.5), so spikes are "unusually sharp jumps for that
topic alone" — comparable across topics without fixing one global velocity cutoff.

Output: ``reports/topic_modeling/spike_events.csv``
  columns: topic_id, spike_date, velocity, velocity_z, keywords

Run (from project root):
  python src/spike_events_from_topics_over_time.py
  python src/spike_events_from_topics_over_time.py --tot path/to/topics_over_time.csv
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

DEFAULT_TOT = "reports/topic_modeling/topics_over_time.csv"
DEFAULT_OUT = "reports/topic_modeling/spike_events.csv"
DEFAULT_PKL = "data/df_clean.pkl"
DEFAULT_BUNDLE = "models/sbert_topic_bundle.joblib"


def _normalize_tot_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with canonical names: topic_id, ts, freq, words (optional)."""
    cols = {c.lower(): c for c in df.columns}
    out = df.copy()

    if "topic" in cols and "topic_id" not in cols:
        out.rename(columns={cols["topic"]: "topic_id"}, inplace=True)
    if "frequency" in cols:
        out.rename(columns={cols["frequency"]: "freq"}, inplace=True)
    elif "proportion" in cols:
        out.rename(columns={cols["proportion"]: "freq"}, inplace=True)

    if "timestamp" in cols:
        out.rename(columns={cols["timestamp"]: "ts"}, inplace=True)
    elif "bin_mid" in cols:
        out.rename(columns={cols["bin_mid"]: "ts"}, inplace=True)

    if "words" not in [c.lower() for c in out.columns]:
        out["words"] = None
    else:
        wcol = [c for c in out.columns if c.lower() == "words"][0]
        out.rename(columns={wcol: "words"}, inplace=True)

    return out


def _simple_freq_keywords(texts: list[str], topn: int = 5) -> str:
    """When TF-IDF prunes all terms (degenerate / tiny corpus), use raw counts."""
    sw = ENGLISH_STOP_WORDS
    tokens: list[str] = []
    for t in texts:
        for w in re.findall(r"[a-zA-Z]{2,}", str(t).lower()):
            if w not in sw:
                tokens.append(w)
    if not tokens:
        return ""
    top = [w for w, _ in Counter(tokens).most_common(topn)]
    return ", ".join(top)


def _tfidf_top_keywords(texts: list[str], topn: int = 5) -> str:
    if not texts:
        return ""
    vec = TfidfVectorizer(
        max_features=8000,
        min_df=1,
        max_df=0.99,
        stop_words="english",
    )
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return _simple_freq_keywords(texts, topn)
    scores = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    idx = np.argsort(scores)[::-1][:topn]
    out = ", ".join(str(terms[i]) for i in idx if scores[i] > 0)
    if not out.strip():
        return _simple_freq_keywords(texts, topn)
    return out


def _fallback_keywords_from_bundle(
    cluster_topics: dict[int, list[tuple[str, float]]], topic_id: int, topn: int = 5
) -> str:
    if topic_id not in cluster_topics:
        return ""
    pairs = cluster_topics[topic_id][:topn]
    return ", ".join(w for w, _ in pairs)


def detect_spikes_for_topic(
    t: pd.DataFrame,
    z_threshold: float = 2.5,
) -> list[dict[str, Any]]:
    """Rows where per-topic z-score of growth-velocity exceeds ``z_threshold``."""
    if len(t) < 3:
        return []
    freq = t["freq"].astype(float).values
    velocity = np.diff(freq) / (freq[:-1] + 1.0)
    if np.nanstd(velocity) == 0 or np.all(np.isnan(velocity)):
        return []
    z = stats.zscore(velocity, nan_policy="omit")
    if isinstance(z, np.floating) or z.ndim == 0:
        return []
    rows_out: list[dict[str, Any]] = []
    for idx in range(len(velocity)):
        zi = z[idx]
        if np.isnan(zi) or zi <= z_threshold:
            continue
        row_next = t.iloc[idx + 1]
        ts = row_next["ts"]
        spike_date = (
            pd.Timestamp(ts).normalize()
            if hasattr(ts, "date")
            else pd.to_datetime(ts).normalize()
        )
        rows_out.append(
            {
                "spike_date": spike_date,
                "velocity": float(velocity[idx]),
                "velocity_z": float(zi),
                "row": row_next,
                "idx_next": idx + 1,
            }
        )
    return rows_out


def build_spike_table(
    tot_path: str,
    df_clean_path: str = DEFAULT_PKL,
    bundle_path: str = DEFAULT_BUNDLE,
    z_threshold: float = 2.5,
) -> pd.DataFrame:
    tot = pd.read_csv(tot_path)
    tot = _normalize_tot_columns(tot)

    if "topic_id" not in tot.columns or "freq" not in tot.columns:
        raise ValueError(
            f"Unexpected columns in {tot_path}: {list(tot.columns)}. "
            "Need topic_id + frequency or proportion, and Timestamp or bin_mid."
        )

    if "bin_start" not in tot.columns:
        tot["bin_start"] = pd.NaT
    if "bin_end" not in tot.columns:
        tot["bin_end"] = pd.NaT
    else:
        tot["bin_start"] = pd.to_datetime(tot["bin_start"], errors="coerce")
        tot["bin_end"] = pd.to_datetime(tot["bin_end"], errors="coerce")

    tot["ts"] = pd.to_datetime(tot["ts"], errors="coerce")

    bundle = None
    cluster_topics: Optional[dict] = None
    topic_labels: Optional[np.ndarray] = None
    if os.path.isfile(bundle_path):
        bundle = joblib.load(bundle_path)
        cluster_topics = bundle.get("cluster_topics_tfidf") or bundle.get(
            "cluster_topics"
        )
        topic_labels = bundle.get("topic_labels")

    df = None
    if os.path.isfile(df_clean_path) and topic_labels is not None:
        df = pd.read_pickle(df_clean_path)
        if len(topic_labels) != len(df):
            df = None
            topic_labels = None
        else:
            df = df.copy()
            df["publish_date"] = pd.to_datetime(df["publish_date"])
            df["_topic_id"] = topic_labels

    spikes: list[dict[str, Any]] = []
    for topic_id in sorted(tot["topic_id"].unique()):
        if topic_id == -1:
            continue
        t = tot[tot["topic_id"] == topic_id].sort_values("ts")
        if len(t) < 3:
            continue
        hit = detect_spikes_for_topic(t, z_threshold=z_threshold)
        for h in hit:
            row = h["row"]
            words_col = row["words"] if "words" in row.index else None
            if (
                words_col is not None
                and pd.notna(words_col)
                and str(words_col).strip()
                and str(words_col).lower() not in ("none", "nan")
            ):
                kw = str(words_col).strip()
                if len(kw) > 200:
                    kw = kw[:200] + "..."
            elif df is not None and topic_labels is not None:
                bs = row["bin_start"] if "bin_start" in row.index else pd.NaT
                be = row["bin_end"] if "bin_end" in row.index else pd.NaT
                kw = ""
                if pd.notna(bs) and pd.notna(be):
                    m = (
                        (df["_topic_id"] == topic_id)
                        & (df["publish_date"] > bs)
                        & (df["publish_date"] <= be)
                    )
                    texts = df.loc[m, "clean_text"].astype(str).tolist()
                    kw = _tfidf_top_keywords(texts, topn=5)
                    if not kw:
                        m_topic = df["_topic_id"] == topic_id
                        texts_topic = df.loc[m_topic, "clean_text"].astype(str).tolist()
                        kw = _tfidf_top_keywords(texts_topic, topn=5)
                if not kw and cluster_topics:
                    kw = _fallback_keywords_from_bundle(
                        cluster_topics, int(topic_id), topn=5
                    )
            elif cluster_topics:
                kw = _fallback_keywords_from_bundle(
                    cluster_topics, int(topic_id), topn=5
                )
            else:
                kw = ""

            spikes.append(
                {
                    "topic_id": int(topic_id),
                    "spike_date": h["spike_date"],
                    "velocity": h["velocity"],
                    "velocity_z": h["velocity_z"],
                    "keywords": kw,
                }
            )

    out = pd.DataFrame(spikes)
    if len(out):
        out = out.sort_values("velocity_z", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tot", default=DEFAULT_TOT, help="topics_over_time.csv path")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output spike_events.csv")
    ap.add_argument("--df-clean", default=DEFAULT_PKL)
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument(
        "--z-threshold",
        type=float,
        default=2.5,
        help="per-topic z-score on growth-velocity; flag when above this",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    spike_df = build_spike_table(
        args.tot,
        df_clean_path=args.df_clean,
        bundle_path=args.bundle,
        z_threshold=args.z_threshold,
    )
    spike_df.to_csv(args.out, index=False)
    print(f"Saved {len(spike_df)} rows → {args.out}")
    if len(spike_df):
        print(spike_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
