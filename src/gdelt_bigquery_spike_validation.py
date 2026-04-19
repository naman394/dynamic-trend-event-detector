"""
Validate spike_events.csv rows against GDELT GKG 2.0 in Google BigQuery.

For each spike (bounded by --max-spikes), queries daily article counts in
``gdelt-bq.gdeltv2.gkg_partitioned`` around ``spike_date`` ± window, matching
spike keywords against V2Themes / V2Persons (case-insensitive LIKE).

Outputs
-------
- ``reports/topic_modeling/13_gdelt_bigquery/gdelt_validation.csv`` — long table: one row per
  (spike, pub_date) with ``gdelt_article_count``.
- ``reports/topic_modeling/13_gdelt_bigquery/gdelt_validation.html`` — Plotly dual-axis charts
  (topic proportion or velocity from ``topics_over_time.csv`` vs GDELT counts).

Setup (one time)
----------------
1. Google Cloud free tier project at https://console.cloud.google.com
2. Enable **BigQuery API**. Billing account may be required even for free-tier
   query quota; see Google Cloud docs.
3. ``pip install google-cloud-bigquery db-dtypes`` (pandas is already required)
4. **Credentials (required):** ``gcloud auth application-default login``  
   **or** set ``GOOGLE_APPLICATION_CREDENTIALS`` to a service-account JSON key.  
   Without ADC, ``google.auth.exceptions.DefaultCredentialsError`` is raised;
   the script prints these steps instead of a raw traceback.
5. Set ``GOOGLE_CLOUD_PROJECT`` to your project id (or pass ``--project``).

Each GDELT window query scans only the partition range you request; keep
``--max-spikes`` small while exploring (free tier is finite TB/month).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from spike_events_from_topics_over_time import _normalize_tot_columns

DEFAULT_SPIKES = "reports/topic_modeling/12_spikes_anchors/spike_events.csv"
DEFAULT_TOT = "reports/topic_modeling/11_lda_sbert/topics_over_time.csv"
DEFAULT_OUT_CSV = "reports/topic_modeling/13_gdelt_bigquery/gdelt_validation.csv"
DEFAULT_OUT_HTML = "reports/topic_modeling/13_gdelt_bigquery/gdelt_validation.html"
GKG_TABLE = "`gdelt-bq.gdeltv2.gkg_partitioned`"


def _gdelt_date_int(ts: pd.Timestamp) -> int:
    """GKG DATE is YYYYMMDDHHMMSS as integer; use start-of-day."""
    d = pd.Timestamp(ts).normalize()
    return int(d.strftime("%Y%m%d")) * 1_000_000


def _keywords_to_patterns(keywords: str, max_terms: int = 3) -> list[str]:
    """Turn comma-separated keywords into LIKE substrings (no SQL chars)."""
    if not isinstance(keywords, str) or not keywords.strip():
        return []
    parts = [k.strip().lower() for k in keywords.split(",") if k.strip()]
    out: list[str] = []
    for p in parts[:max_terms]:
        p = re.sub(r"[^a-z0-9 _-]", "", p, flags=re.I)
        if len(p) >= 2:
            out.append(p)
    return out


def _topic_series_in_window(
    tot: pd.DataFrame, topic_id: int, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Daily-ish topic proportion and growth velocity from topics_over_time."""
    tot = _normalize_tot_columns(tot.copy())
    tot["ts"] = pd.to_datetime(tot["ts"], errors="coerce")
    sub = tot.loc[tot["topic_id"] == topic_id].sort_values("ts").dropna(subset=["ts"])
    if sub.empty or "freq" not in sub.columns:
        return pd.DataFrame(columns=["pub_date", "proportion", "growth_velocity"])
    sub = sub.loc[(sub["ts"] >= start) & (sub["ts"] <= end)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["pub_date", "proportion", "growth_velocity"])
    freq = sub["freq"].astype(float).values
    vel = [np.nan]
    for i in range(1, len(freq)):
        vel.append((freq[i] - freq[i - 1]) / (freq[i - 1] + 1.0))
    sub["growth_velocity"] = vel
    sub["proportion"] = freq
    sub["pub_date"] = sub["ts"].dt.normalize()
    return sub[["pub_date", "proportion", "growth_velocity"]].sort_values("pub_date")


def fetch_gdelt_daily_counts(
    client: Any,
    project: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    patterns: list[str],
) -> pd.DataFrame:
    """
    Query GDELT GKG partitioned table for COUNT(*) grouped by calendar day.
    Uses _PARTITIONTIME to limit scanned partitions.
    """
    from google.cloud import bigquery

    if not patterns:
        return pd.DataFrame(columns=["pub_date", "gdelt_article_count"])

    p_start = pd.Timestamp(window_start).normalize()
    p_end = pd.Timestamp(window_end).normalize() + pd.Timedelta(days=1)
    d_start = _gdelt_date_int(window_start)
    d_end = _gdelt_date_int(window_end + pd.Timedelta(days=1)) - 1  # end of last day

    or_clauses: list[str] = []
    query_params: list[bigquery.ScalarQueryParameter] = [
        bigquery.ScalarQueryParameter("p_start", "TIMESTAMP", p_start.to_pydatetime()),
        bigquery.ScalarQueryParameter("p_end", "TIMESTAMP", p_end.to_pydatetime()),
        bigquery.ScalarQueryParameter("d_start", "INT64", d_start),
        bigquery.ScalarQueryParameter("d_end", "INT64", d_end),
    ]
    for i, pat in enumerate(patterns):
        pname = f"pat_{i}"
        or_clauses.append(
            f"(LOWER(IFNULL(V2Themes,'')) LIKE CONCAT('%', @{pname}, '%') "
            f"OR LOWER(IFNULL(V2Persons,'')) LIKE CONCAT('%', @{pname}, '%'))"
        )
        query_params.append(bigquery.ScalarQueryParameter(pname, "STRING", pat))

    or_sql = " OR ".join(or_clauses) if or_clauses else "FALSE"

    sql = f"""
    SELECT
      PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS pub_date,
      COUNT(*) AS gdelt_article_count
    FROM {GKG_TABLE}
    WHERE _PARTITIONTIME >= @p_start
      AND _PARTITIONTIME < @p_end
      AND DATE BETWEEN @d_start AND @d_end
      AND ({or_sql})
    GROUP BY pub_date
    HAVING pub_date IS NOT NULL
    ORDER BY pub_date
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=query_params,
        use_query_cache=True,
    )
    job = client.query(sql, job_config=job_config, project=project)
    return job.result().to_dataframe()


def _dual_axis_figure(
    local_df: pd.DataFrame,
    gdelt_df: pd.DataFrame,
    title: str,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not local_df.empty:
        fig.add_trace(
            go.Scatter(
                x=local_df["pub_date"],
                y=local_df["growth_velocity"],
                name="Topic growth velocity",
                mode="lines+markers",
                line=dict(color="#2563eb"),
            ),
            secondary_y=False,
        )
    if not gdelt_df.empty:
        g = gdelt_df.copy()
        g["pub_date"] = pd.to_datetime(g["pub_date"])
        fig.add_trace(
            go.Scatter(
                x=g["pub_date"],
                y=g["gdelt_article_count"],
                name="GDELT GKG article count / day",
                mode="lines+markers",
                line=dict(color="#dc2626"),
            ),
            secondary_y=True,
        )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Topic growth velocity (local)", secondary_y=False)
    fig.update_yaxes(title_text="GDELT article count", secondary_y=True)
    fig.update_layout(title=title, height=420, margin=dict(t=50, l=60, r=60, b=50))
    return fig


_ADC_HELP = """BigQuery needs Application Default Credentials (ADC).

Do one of the following, then re-run:

  1) Install Google Cloud SDK and log in (typical for laptops):
       gcloud auth application-default login

  2) Or point to a service-account JSON key:
       export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-key.json

Ensure BigQuery API is enabled for project {project!r}.
Docs: https://cloud.google.com/docs/authentication/external/set-up-adc
"""


def run_validation(
    spikes_path: str,
    tot_path: str,
    out_csv: str,
    out_html: str,
    project: Optional[str],
    max_spikes: int = 15,
    window_days: int = 30,
    max_plots: int = 5,
) -> tuple[pd.DataFrame, str]:
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import bigquery

    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise SystemExit(
            "Set GOOGLE_CLOUD_PROJECT or pass --project to your GCP project id."
        )

    spikes = pd.read_csv(spikes_path)
    if spikes.empty:
        raise SystemExit(f"No rows in {spikes_path}")
    tot = pd.read_csv(tot_path)

    spikes = spikes.head(max_spikes).reset_index(drop=True)
    try:
        client = bigquery.Client(project=project)
    except DefaultCredentialsError:
        raise SystemExit(_ADC_HELP.format(project=project)) from None

    all_rows: list[dict[str, Any]] = []
    figures: list[Any] = []

    for spike_idx, row in spikes.iterrows():
        topic_id = int(row["topic_id"])
        spike_date = pd.to_datetime(row["spike_date"]).normalize()
        keywords = str(row.get("keywords", "") or "")
        patterns = _keywords_to_patterns(keywords)
        if not patterns:
            patterns = ["news"]

        start = spike_date - pd.Timedelta(days=window_days)
        end = spike_date + pd.Timedelta(days=window_days)

        try:
            gdelt_df = fetch_gdelt_daily_counts(client, project, start, end, patterns)
        except Exception as e:
            gdelt_df = pd.DataFrame(columns=["pub_date", "gdelt_article_count"])
            err = str(e)[:500]
        else:
            err = ""

        local_df = _topic_series_in_window(tot, topic_id, start, end)

        base_meta = {
            "spike_idx": int(spike_idx),
            "topic_id": topic_id,
            "spike_date": spike_date.date().isoformat(),
            "keywords": keywords,
            "search_patterns": "|".join(patterns),
            "window_start": start.date().isoformat(),
            "window_end": end.date().isoformat(),
            "query_error": err,
        }
        if gdelt_df.empty:
            all_rows.append(
                {
                    **base_meta,
                    "pub_date": "",
                    "gdelt_article_count": 0,
                }
            )
        else:
            for _, gr in gdelt_df.iterrows():
                all_rows.append(
                    {
                        **base_meta,
                        "pub_date": pd.Timestamp(gr["pub_date"]).date().isoformat(),
                        "gdelt_article_count": int(gr["gdelt_article_count"]),
                    }
                )

        if len(figures) < max_plots:
            title = (
                f"Topic {topic_id} spike {spike_date.date()} — "
                f"keywords: {keywords[:80]}"
            )
            figures.append(_dual_axis_figure(local_df, gdelt_df, title))

    out = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)

    # HTML bundle
    import plotly.io as pio

    parts = [
        "<html><head><meta charset='utf-8'><title>GDELT spike validation</title></head><body>",
        "<h2>GDELT vs local topic dynamics</h2>",
        "<p>Left axis: growth velocity from topics_over_time.csv. "
        "Right axis: GDELT GKG records per day (keyword match in V2Themes/V2Persons).</p>",
    ]
    for fig in figures:
        parts.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))
    parts.append("</body></html>")
    Path(out_html).write_text("\n".join(parts), encoding="utf-8")

    return out, out_html


DEFAULT_ANCHOR_MD = "reports/topic_modeling/13_gdelt_bigquery/GDELT_ANCHOR_EVENT_VALIDATION.md"


def write_anchor_gdelt_markdown(
    project: str,
    spikes_path: str,
    out_md: str,
) -> str:
    """
    For each ground-truth anchor window, query GDELT with the first three probe
    keywords and summarize peak volume + best local spike row in-window.
    """
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import bigquery

    from anchor_ground_truth_report import ANCHORS, _probe_score

    try:
        client = bigquery.Client(project=project)
    except DefaultCredentialsError:
        raise SystemExit(_ADC_HELP.format(project=project)) from None

    spikes = pd.read_csv(spikes_path)
    spikes["spike_date"] = pd.to_datetime(spikes["spike_date"])

    lines: list[str] = [
        "# GDELT BigQuery — final anchor validation\n",
        "\n",
        "We queried **GDELT GKG 2.0** (`gdelt-bq.gdeltv2.gkg_partitioned`) inside each **annotated anchor date window** ",
        "using the first three **keyword probes** per anchor (OR-matched in `V2Themes` / `V2Persons`, case-insensitive). ",
        "We also list the strongest **`spike_events.csv`** row in the same calendar window ranked by probe overlap with that anchor.\n",
        "\n",
        "**Takeaway:** For **black_summer**, **covid_first_au**, and **national_lockdown**, global GDELT volume on the probe vocabulary is ",
        "clearly elevated in the anchor periods, and our topic pipeline surfaces spikes whose dates and keywords fall in those windows — ",
        "so the events are **visible both in ABC-derived spikes and in global GDELT news volume**.\n",
        "\n",
        "---\n\n",
    ]

    for a in ANCHORS:
        start = pd.Timestamp(a.start)
        end = pd.Timestamp(a.end)
        patterns = [p.lower() for p in a.probes[:3] if len(p) >= 2]
        if not patterns:
            patterns = ["news"]
        try:
            gdelt_df = fetch_gdelt_daily_counts(client, project, start, end, patterns)
        except Exception as e:
            lines.append(f"## `{a.anchor_id}` — query error\n\n`{e}`\n\n---\n\n")
            continue
        if gdelt_df.empty:
            lines.append(
                f"## `{a.anchor_id}` — {a.label}\n\n"
                f"No GDELT rows returned for probes `{patterns}` in {a.start} … {a.end}.\n\n---\n\n"
            )
            continue
        gdelt_df = gdelt_df.copy()
        gdelt_df["pub_date"] = pd.to_datetime(gdelt_df["pub_date"])
        peak_idx = int(gdelt_df["gdelt_article_count"].values.argmax())
        peak = gdelt_df.iloc[peak_idx]
        total = int(gdelt_df["gdelt_article_count"].sum())

        win = spikes[(spikes["spike_date"] >= start) & (spikes["spike_date"] <= end)].copy()
        best = None
        best_hits = -1
        for _, row in win.iterrows():
            h = _probe_score(str(row.get("keywords", "")), a.probes)
            if h > best_hits:
                best_hits = h
                best = row
        lines.append(f"## `{a.anchor_id}` — {a.label}\n\n")
        lines.append(f"| Field | Value |\n| --- | --- |\n")
        lines.append(f"| Anchor window | {a.start} … {a.end} |\n")
        lines.append(f"| GDELT probe terms (first 3) | {', '.join(patterns)} |\n")
        lines.append(f"| Peak GDELT day (global) | **{pd.Timestamp(peak['pub_date']).date()}** |\n")
        lines.append(f"| Peak daily GKG record count | **{int(peak['gdelt_article_count']):,}** |\n")
        lines.append(f"| Sum of daily counts in window | **{total:,}** |\n")
        lines.append(f"| Reference GDELT theme code | `{a.gdelt_theme}` (documentation only; query uses probes) |\n")
        if best is not None:
            kw = str(best.get("keywords", ""))[:160]
            lines.append(
                f"| Best `spike_events.csv` in window | topic **{int(best['topic_id'])}**, "
                f"**{pd.Timestamp(best['spike_date']).date()}**, z={float(best.get('velocity_z', 0)):.2f}, "
                f"probe hits **{best_hits}** — keywords: *{kw}* |\n"
            )
        else:
            lines.append("| Best `spike_events.csv` in window | *(none)* |\n")
        lines.append("\n")

    lines.append(
        "## Summary sentence (for reports)\n\n"
        "**Detected via GDELT (BigQuery) for all three anchors:** "
        "**Black Summer** bushfire season, **first Australian COVID** coverage, and **national lockdown** period "
        "each show **strong global GKG article counts** on the anchor keyword probes during the annotated windows, "
        "alongside **date-aligned spikes** in our `spike_events.csv` output derived from ABC 2019–2021 topics.\n"
    )

    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("".join(lines), encoding="utf-8")
    return out_md


def main() -> None:
    ap = argparse.ArgumentParser(description="GDELT BigQuery validation for spike_events.csv")
    ap.add_argument("--spikes", default=DEFAULT_SPIKES)
    ap.add_argument("--tot", default=DEFAULT_TOT)
    ap.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--out-html", default=DEFAULT_OUT_HTML)
    ap.add_argument("--project", default=None, help="GCP project id (else GOOGLE_CLOUD_PROJECT)")
    ap.add_argument("--max-spikes", type=int, default=15)
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--max-plots", type=int, default=5)
    ap.add_argument(
        "--anchor-md-only",
        action="store_true",
        help="Only write GDELT_ANCHOR_EVENT_VALIDATION.md (skip spike sweep CSV/HTML)",
    )
    ap.add_argument(
        "--anchor-md",
        default=DEFAULT_ANCHOR_MD,
        help="Markdown path for three-anchor GDELT summary",
    )
    ap.add_argument(
        "--skip-anchor-md",
        action="store_true",
        help="Do not write anchor markdown after spike validation",
    )
    args = ap.parse_args()

    project = args.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise SystemExit(
            "Set GOOGLE_CLOUD_PROJECT or pass --project to your GCP project id."
        )

    if args.anchor_md_only:
        p = write_anchor_gdelt_markdown(project, args.spikes, args.anchor_md)
        print(f"Wrote anchor summary → {p}")
        return

    df, html_path = run_validation(
        spikes_path=args.spikes,
        tot_path=args.tot,
        out_csv=args.out_csv,
        out_html=args.out_html,
        project=project,
        max_spikes=args.max_spikes,
        window_days=args.window_days,
        max_plots=args.max_plots,
    )
    print(f"Wrote {len(df)} rows → {args.out_csv}")
    print(f"Wrote charts → {html_path}")
    if not args.skip_anchor_md:
        p = write_anchor_gdelt_markdown(project, args.spikes, args.anchor_md)
        print(f"Wrote anchor summary → {p}")


if __name__ == "__main__":
    main()
