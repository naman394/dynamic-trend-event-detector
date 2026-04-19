"""
NER Entity Tracker — spaCy Named Entity Recognition over news headlines.

Extracts PERSON, GPE, ORG, EVENT entities, builds weekly frequency time series,
and surfaces which entities drove each semantic velocity rupture week.

Usage:
    python src/ner_entity_tracker.py
    python src/ner_entity_tracker.py --model en_core_web_lg   # higher accuracy
    python src/ner_entity_tracker.py --limit 50000            # quick test

Setup (one-time):
    pip install spacy
    python -m spacy download en_core_web_sm    # 12 MB, fast
    # or:
    python -m spacy download en_core_web_lg    # 700 MB, more accurate
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
DL = REPORTS / "deep_learning"

HEADLINES_FULL  = DATA / "abcnews-date-text 2.csv"
HEADLINES_SMALL = DATA / "news_headlines.csv"
VELOCITY_CSV    = DL / "semantic_velocity.csv"
OUT_ENTITIES    = REPORTS / "ner_entities.csv"
OUT_WEEKLY      = REPORTS / "ner_weekly_freq.csv"
OUT_RUPTURE_NER = REPORTS / "ner_rupture_context.csv"

ENTITY_TYPES = {"PERSON", "GPE", "ORG", "EVENT", "NORP"}


def load_headlines(limit=None):
    path = HEADLINES_FULL if HEADLINES_FULL.exists() else HEADLINES_SMALL
    print(f"Loading headlines from: {path.name}", flush=True)
    df = pd.read_csv(path, dtype={"publish_date": str})
    df.columns = [c.strip() for c in df.columns]
    if "headline_text" not in df.columns and len(df.columns) >= 2:
        df.columns = ["publish_date", "headline_text"]
    df["publish_date"] = pd.to_datetime(df["publish_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["publish_date", "headline_text"])
    df = df.sort_values("publish_date").reset_index(drop=True)
    if limit:
        df = df.head(limit)
    print(f"  {len(df):,} headlines loaded ({df['publish_date'].min().date()} → {df['publish_date'].max().date()})")
    return df


def run_ner(headlines_df, model_name="en_core_web_sm", batch_size=256):
    """Run spaCy NER over all headlines, return long-form entity DataFrame."""
    try:
        import spacy
    except ImportError:
        print("ERROR: spacy not installed.  pip install spacy", file=sys.stderr)
        sys.exit(1)

    # load model with only NER pipe (fastest)
    try:
        nlp = spacy.load(model_name, disable=["tok2vec", "tagger", "parser",
                                               "attribute_ruler", "lemmatizer"])
    except OSError:
        print(f"Model '{model_name}' not found. Downloading en_core_web_sm…")
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser",
                                                      "attribute_ruler", "lemmatizer"])

    texts   = headlines_df["headline_text"].fillna("").tolist()
    dates   = headlines_df["publish_date"].tolist()

    records = []
    total   = len(texts)
    print(f"Running NER on {total:,} headlines with '{model_name}' (batch={batch_size})…")

    for i, (doc, date) in enumerate(
        zip(nlp.pipe(texts, batch_size=batch_size), dates)
    ):
        for ent in doc.ents:
            if ent.label_ in ENTITY_TYPES:
                records.append({
                    "date":        date,
                    "entity":      ent.text.strip(),
                    "entity_type": ent.label_,
                })
        if (i + 1) % 10_000 == 0:
            print(f"  processed {i+1:,}/{total:,}", flush=True)

    entities_df = pd.DataFrame(records)
    print(f"  extracted {len(entities_df):,} entity mentions across {entities_df['entity'].nunique():,} unique entities")
    return entities_df


def build_weekly_freq(entities_df):
    """Aggregate entity mentions by ISO week."""
    df = entities_df.copy()
    df["week"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)
    weekly = (
        df.groupby(["week", "entity_type", "entity"])
        .size()
        .reset_index(name="count")
    )
    return weekly


def rupture_context(entities_df, velocity_csv=VELOCITY_CSV, top_n=10):
    """For each rupture week, find top entities and their % change vs prior week."""
    if not velocity_csv.exists():
        return pd.DataFrame()

    vel_df = pd.read_csv(velocity_csv)
    vel_df["week_start"] = pd.to_datetime(vel_df["week"].str.split("/").str[0])
    vel_df = vel_df.sort_values("week_start")

    threshold = vel_df["semantic_velocity"].mean() + vel_df["semantic_velocity"].std()
    rupture_weeks = vel_df[vel_df["semantic_velocity"] >= threshold].copy()

    entities_df = entities_df.copy()
    entities_df["week_start"] = entities_df["date"].dt.to_period("W").apply(lambda p: p.start_time)

    results = []
    for _, rw in rupture_weeks.iterrows():
        wstart = rw["week_start"]
        wend   = wstart + pd.Timedelta(days=6)
        prev_s = wstart - pd.Timedelta(days=7)

        cur  = entities_df[(entities_df["date"] >= wstart) & (entities_df["date"] <= wend)]
        prev = entities_df[(entities_df["date"] >= prev_s) & (entities_df["date"] < wstart)]

        cur_counts  = cur.groupby(["entity", "entity_type"]).size()
        prev_counts = prev.groupby(["entity", "entity_type"]).size()

        for (ent, etype), cnt in cur_counts.nlargest(top_n).items():
            prev_cnt = prev_counts.get((ent, etype), 0)
            pct_change = ((cnt - prev_cnt) / max(prev_cnt, 1)) * 100
            results.append({
                "rupture_week":      rw["week"],
                "velocity":          rw["semantic_velocity"],
                "entity":            ent,
                "entity_type":       etype,
                "count_this_week":   cnt,
                "count_prev_week":   prev_cnt,
                "pct_change":        round(pct_change, 1),
            })

    return pd.DataFrame(results)


def print_rupture_report(rupture_df):
    if rupture_df.empty:
        return
    print("\n" + "=" * 65)
    print("RUPTURE ENTITY REPORT")
    print("=" * 65)
    for week, grp in rupture_df.groupby("rupture_week"):
        vel = grp["velocity"].iloc[0]
        print(f"\nWeek {week}  (V_s = {vel:.4f})")
        print("-" * 55)
        top = grp.nlargest(8, "pct_change")
        for _, r in top.iterrows():
            arrow = "▲" if r["pct_change"] > 0 else "▼"
            print(f"  {arrow} {r['entity']:<25} [{r['entity_type']}]  "
                  f"+{r['pct_change']:.0f}% ({r['count_prev_week']}→{r['count_this_week']})")


def main():
    parser = argparse.ArgumentParser(description="NER entity tracker for news headlines")
    parser.add_argument("--model",  default="en_core_web_sm", help="spaCy model name")
    parser.add_argument("--limit",  type=int, default=None,   help="max headlines (for testing)")
    parser.add_argument("--batch",  type=int, default=256,    help="NER batch size")
    args = parser.parse_args()

    REPORTS.mkdir(exist_ok=True)

    headlines_df = load_headlines(limit=args.limit)
    entities_df  = run_ner(headlines_df, model_name=args.model, batch_size=args.batch)

    if entities_df.empty:
        print("No entities extracted — check your corpus and spaCy model.")
        return

    # save raw entities
    entities_df.to_csv(OUT_ENTITIES, index=False)
    print(f"Saved: {OUT_ENTITIES}")

    # weekly frequency
    weekly_df = build_weekly_freq(entities_df)
    weekly_df.to_csv(OUT_WEEKLY, index=False)
    print(f"Saved: {OUT_WEEKLY}")

    # rupture context
    rupture_df = rupture_context(entities_df)
    if not rupture_df.empty:
        rupture_df.to_csv(OUT_RUPTURE_NER, index=False)
        print(f"Saved: {OUT_RUPTURE_NER}")
        print_rupture_report(rupture_df)

    # summary stats
    print("\n── Top 10 PERSON entities ──────────────────────")
    persons = (
        entities_df[entities_df["entity_type"] == "PERSON"]
        .groupby("entity").size().nlargest(10)
    )
    for name, cnt in persons.items():
        print(f"  {name:<30} {cnt:>6,}")

    print("\n── Top 10 GPE (places) ─────────────────────────")
    places = (
        entities_df[entities_df["entity_type"] == "GPE"]
        .groupby("entity").size().nlargest(10)
    )
    for name, cnt in places.items():
        print(f"  {name:<30} {cnt:>6,}")

    print("\n── Top 10 ORG entities ─────────────────────────")
    orgs = (
        entities_df[entities_df["entity_type"] == "ORG"]
        .groupby("entity").size().nlargest(10)
    )
    for name, cnt in orgs.items():
        print(f"  {name:<30} {cnt:>6,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
