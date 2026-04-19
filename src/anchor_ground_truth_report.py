"""
Match spike_events.csv to three ground-truth news anchors (no GDELT call).

Outputs
-------
  reports/topic_modeling/12_spikes_anchors/anchor_ground_truth_detection.csv
  reports/topic_modeling/12_spikes_anchors/anchor_ground_truth_detection_detail.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

REPORTS = "reports/topic_modeling/12_spikes_anchors"


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    label: str
    start: str
    end: str
    probes: tuple[str, ...]
    gdelt_theme: str


ANCHORS: tuple[Anchor, ...] = (
    Anchor(
        "black_summer",
        "Black Summer bushfires",
        "2019-11-01",
        "2020-01-31",
        ("fire", "smoke", "evacuation", "blaze", "nsw", "bushfire", "bushfires"),
        "ENV_FIRES",
    ),
    Anchor(
        "covid_first_au",
        "First Australian COVID case",
        "2020-01-20",
        "2020-02-05",
        ("virus", "coronavirus", "china", "health", "travel", "covid"),
        "HEALTH_PANDEMIC",
    ),
    Anchor(
        "national_lockdown",
        "National lockdown",
        "2020-03-01",
        "2020-04-15",
        ("lockdown", "restrictions", "border", "quarantine"),
        "HEALTH_PANDEMIC",
    ),
)


def _probe_score(text: str, probes: tuple[str, ...]) -> int:
    if not text or not isinstance(text, str):
        return 0
    t = text.lower()
    return sum(1 for p in probes if p in t)


def _in_window(ts: Any, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if pd.isna(ts):
        return False
    x = pd.Timestamp(ts).normalize()
    return start.normalize() <= x <= end.normalize()


def _load_optional(path: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def collect_rows_in_window(
    df: pd.DataFrame,
    date_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    source: str,
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    d = df.copy()
    d["_source"] = source
    d["_date"] = pd.to_datetime(d[date_col], errors="coerce")
    m = d["_date"].apply(lambda x: _in_window(x, start, end))
    return d.loc[m].copy()


def run_report(
    out_dir: str = REPORTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)

    spike_df = _load_optional(os.path.join(out_dir, "spike_events.csv"))

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for a in ANCHORS:
        start = pd.Timestamp(a.start)
        end = pd.Timestamp(a.end)

        chunks: list[pd.DataFrame] = []
        if spike_df is not None and "spike_date" in spike_df.columns:
            c = collect_rows_in_window(
                spike_df, "spike_date", start, end, "spike_events"
            )
            if len(c):
                c["_kw"] = c["keywords"].fillna("").astype(str)
                chunks.append(c)

        if not chunks:
            summary_rows.append(
                {
                    "anchor_id": a.anchor_id,
                    "anchor_label": a.label,
                    "date_window": f"{a.start} .. {a.end}",
                    "expected_keywords_probe": ", ".join(a.probes),
                    "gdelt_theme_expected": a.gdelt_theme,
                    "detected": "no",
                    "best_source": "",
                    "best_date": "",
                    "best_topic_id": "",
                    "best_keywords": "",
                    "probe_hits": 0,
                    "score_column": "",
                }
            )
            continue

        all_c = pd.concat(chunks, ignore_index=True)
        all_c["probe_hits"] = all_c["_kw"].apply(
            lambda s: _probe_score(s, a.probes)
        )

        for _, row in all_c.iterrows():
            detail_rows.append(
                {
                    "anchor_id": a.anchor_id,
                    "anchor_label": a.label,
                    "gdelt_theme_expected": a.gdelt_theme,
                    "source": row["_source"],
                    "date": row["_date"],
                    "topic_id": row.get("topic_id", ""),
                    "keywords": row.get("_kw", ""),
                    "probe_hits": int(row["probe_hits"]),
                    "velocity": row.get("velocity", ""),
                    "velocity_z": row.get("velocity_z", ""),
                }
            )

        all_c["_rank"] = all_c["probe_hits"].astype(float) * 1000.0
        if "velocity_z" in all_c.columns:
            all_c["_rank"] += pd.to_numeric(
                all_c["velocity_z"], errors="coerce"
            ).fillna(0)
        elif "velocity" in all_c.columns:
            all_c["_rank"] += pd.to_numeric(all_c["velocity"], errors="coerce").fillna(0)
        best = all_c.sort_values("_rank", ascending=False)
        top = best.iloc[0]
        hits = int(top["probe_hits"])
        summary_rows.append(
            {
                "anchor_id": a.anchor_id,
                "anchor_label": a.label,
                "date_window": f"{a.start} .. {a.end}",
                "expected_keywords_probe": ", ".join(a.probes),
                "gdelt_theme_expected": a.gdelt_theme,
                "detected": "yes" if hits > 0 else "weak",
                "best_source": str(top["_source"]),
                "best_date": str(pd.Timestamp(top["_date"]).date()),
                "best_topic_id": top.get("topic_id", ""),
                "best_keywords": str(top.get("_kw", ""))[:500],
                "probe_hits": hits,
                "note": "GDELT verification pending — keyword overlap only",
            }
        )

    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)

    p1 = os.path.join(out_dir, "anchor_ground_truth_detection.csv")
    p2 = os.path.join(out_dir, "anchor_ground_truth_detection_detail.csv")
    summary.to_csv(p1, index=False)
    detail.to_csv(p2, index=False)
    return summary, detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=REPORTS)
    args = ap.parse_args()
    s, d = run_report(out_dir=args.out_dir)
    print(s.to_string(index=False))
    print(f"\nSaved → {args.out_dir}/anchor_ground_truth_detection.csv")
    print(f"Detail  → {args.out_dir}/anchor_ground_truth_detection_detail.csv  ({len(d)} rows)")


if __name__ == "__main__":
    main()
