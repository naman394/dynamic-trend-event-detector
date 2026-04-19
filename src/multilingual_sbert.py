"""
Multilingual SBERT Expansion — Cross-lingual Semantic Similarity.

Projects ABC News headlines AND GDELT themes into the SAME language-agnostic
768-dim space using paraphrase-multilingual-mpnet-base-v2.

This reveals:
  1. Which GDELT events are semantically closest to the ABC narrative (cross-sim)
  2. Which language sources pick up a topic first (propagation timeline)
  3. How the current live GDELT snapshot maps onto historical ABC clusters

Usage:
    python src/multilingual_sbert.py
    python src/multilingual_sbert.py --abc-limit 5000 --gdelt-limit 500

Requirements:
    sentence-transformers is already in requirements.txt
    (model downloads ~1 GB on first run)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "data"
REPORTS = ROOT / "reports"
DL      = REPORTS / "deep_learning"

HEADLINES_FULL  = DATA / "abcnews-date-text 2.csv"
HEADLINES_SMALL = DATA / "news_headlines.csv"
GDELT_CSV       = DATA / "gdelt_processed.csv"
VELOCITY_CSV    = DL / "semantic_velocity.csv"

OUT_CSV     = REPORTS / "multilingual_signals.csv"
OUT_PROP    = REPORTS / "multilingual_propagation.csv"

MULTILINGUAL_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# rough source → language mapping (based on common GDELT outlets)
LANG_HINTS = {
    ".cn": "zh", ".fr": "fr", ".de": "de", ".es": "es",
    ".ar": "ar", ".ru": "ru", ".jp": "ja", ".it": "it",
    ".pt": "pt", ".nl": "nl", ".pl": "pl", ".in": "hi",
    "xinhua": "zh", "france": "fr", "spiegel": "de",
    "rtve": "es", "elpais": "es", "aljazeera": "ar",
    "rt.com": "ru", "nhk": "ja", "ansa": "it",
    "publico": "pt", "tass": "ru", "dw.com": "de",
    "lemonde": "fr", "lefigaro": "fr",
}


def infer_language(source_name: str) -> str:
    """Best-effort language label from source common name."""
    if not isinstance(source_name, str):
        return "en"
    s = source_name.lower()
    for hint, lang in LANG_HINTS.items():
        if hint in s:
            return lang
    return "en"


# ── data loaders ─────────────────────────────────────────────────────────────
def load_abc(limit=None):
    path = HEADLINES_FULL if HEADLINES_FULL.exists() else HEADLINES_SMALL
    print(f"Loading ABC headlines from: {path.name}")
    df = pd.read_csv(path, dtype={"publish_date": str})
    df.columns = [c.strip() for c in df.columns]
    if "headline_text" not in df.columns and len(df.columns) >= 2:
        df.columns = ["publish_date", "headline_text"]
    df["publish_date"] = pd.to_datetime(df["publish_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["publish_date", "headline_text"])
    if limit:
        df = df.sample(min(limit, len(df)), random_state=42)
    print(f"  {len(df):,} ABC headlines")
    return df


def load_gdelt(limit=None):
    if not GDELT_CSV.exists():
        print("WARNING: gdelt_processed.csv not found. Run gdelt_processor.py first.")
        return pd.DataFrame()
    df = pd.read_csv(GDELT_CSV)
    df["source_lang"] = df["SOURCECOMMONNAME"].apply(infer_language)

    # build clean theme string for embedding
    def clean_theme(tl):
        try:
            import ast
            themes = ast.literal_eval(tl) if isinstance(tl, str) else []
            return " ".join(
                t.replace("_", " ").lower()
                for t in themes
                if not re.match(r"^WB_\d", t) and not t.startswith("TAX_")
            )[:256]
        except Exception:
            return ""

    df["theme_string"] = df["theme_list"].apply(clean_theme)
    df = df[df["theme_string"].str.strip() != ""]
    if limit:
        df = df.head(limit)
    print(f"  {len(df):,} GDELT records  ({df['source_lang'].value_counts().to_dict()})")
    return df


# ── embedding ─────────────────────────────────────────────────────────────────
def embed(texts, model, batch_size=128, desc=""):
    """Encode texts, return L2-normalised float32 array."""
    print(f"Encoding {len(texts):,} {desc} texts…", flush=True)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


# ── cross-lingual similarity ──────────────────────────────────────────────────
def cross_similarity(abc_vecs, gdelt_vecs, top_k=5):
    """
    For each GDELT record, compute max cosine similarity to the nearest ABC headlines.
    Returns array of shape (n_gdelt,) with max cosine sim.
    """
    # abc_vecs already normalised → cosine = dot product
    # chunk to avoid OOM on large corpora
    chunk    = 512
    max_sims = np.zeros(len(gdelt_vecs), dtype=np.float32)
    for i in range(0, len(gdelt_vecs), chunk):
        g_chunk = gdelt_vecs[i: i + chunk]             # (chunk, D)
        sims    = g_chunk @ abc_vecs.T                  # (chunk, n_abc)
        max_sims[i: i + chunk] = sims.max(axis=1)
    return max_sims


def global_centroid(vecs):
    """Compute and return the L2-normalised mean vector."""
    c = vecs.mean(axis=0)
    norm = np.linalg.norm(c)
    return c / norm if norm > 1e-8 else c


# ── propagation timeline ──────────────────────────────────────────────────────
def build_propagation_timeline(gdelt_df, gdelt_vecs, abc_vecs, abc_df, n_clusters=8):
    """
    Cluster GDELT records by semantic similarity, then show which language
    source produced each cluster first (earliest DATE).
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return pd.DataFrame()

    print(f"Clustering GDELT into {n_clusters} groups for propagation analysis…")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(gdelt_vecs)

    gdelt_df = gdelt_df.copy().reset_index(drop=True)
    gdelt_df["cluster"] = labels

    # parse GDELT DATE field (format: YYYYMMDDHHmmSS)
    gdelt_df["gdelt_time"] = pd.to_datetime(
        gdelt_df["DATE"].astype(str).str[:12], format="%Y%m%d%H%M", errors="coerce"
    )

    rows = []
    for cl in range(n_clusters):
        sub = gdelt_df[gdelt_df["cluster"] == cl]
        if sub.empty:
            continue
        # find cluster centroid's semantic sim to ABC corpus
        cl_centroid = km.cluster_centers_[cl]
        cl_centroid /= max(np.linalg.norm(cl_centroid), 1e-8)
        abc_sims  = cl_centroid @ abc_vecs.T
        abc_match = abc_sims.max()

        # first appearance by language
        first_by_lang = (
            sub.dropna(subset=["gdelt_time"])
            .sort_values("gdelt_time")
            .groupby("source_lang")["gdelt_time"]
            .first()
            .sort_values()
        )

        # top keywords: pick shortest non-empty theme strings
        top_kws = (
            sub["theme_string"].dropna()
            .str.split().explode()
            .value_counts().head(5).index.tolist()
        )

        for lang, ts in first_by_lang.items():
            rows.append({
                "cluster":          cl,
                "language":         lang,
                "first_appearance": ts,
                "n_records":        len(sub[sub["source_lang"] == lang]),
                "abc_max_sim":      round(float(abc_match), 4),
                "keywords":         ", ".join(top_kws[:5]),
            })

    prop_df = pd.DataFrame(rows).sort_values(["cluster", "first_appearance"])
    return prop_df


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multilingual SBERT cross-lingual analysis")
    parser.add_argument("--abc-limit",   type=int, default=3000,
                        help="max ABC headlines to embed (for speed)")
    parser.add_argument("--gdelt-limit", type=int, default=None,
                        help="max GDELT records (default: all)")
    parser.add_argument("--batch",       type=int, default=128)
    parser.add_argument("--n-clusters",  type=int, default=8,
                        help="propagation timeline clusters")
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed.  pip install sentence-transformers",
              file=sys.stderr)
        sys.exit(1)

    REPORTS.mkdir(exist_ok=True)

    print(f"Loading multilingual model: {MULTILINGUAL_MODEL}")
    print("  (First run downloads ~1 GB — subsequent runs are cached)\n")
    model = SentenceTransformer(MULTILINGUAL_MODEL)

    # ── load data ──
    abc_df   = load_abc(limit=args.abc_limit)
    gdelt_df = load_gdelt(limit=args.gdelt_limit)

    if gdelt_df.empty:
        print("No GDELT data available — run gdelt_processor.py first.")
        return

    # ── embed ──
    abc_vecs   = embed(abc_df["headline_text"].tolist(), model, args.batch, "ABC")
    gdelt_vecs = embed(gdelt_df["theme_string"].tolist(), model, args.batch, "GDELT")

    # ── cross-lingual similarity ──
    print("\nComputing cross-lingual similarity (GDELT → ABC)…")
    cross_sim = cross_similarity(abc_vecs, gdelt_vecs)
    gdelt_df["cross_sim"] = cross_sim

    # semantic uniqueness vs ABC corpus centroid
    abc_centroid = global_centroid(abc_vecs)
    gdelt_df["uniqueness_vs_abc"] = 1.0 - (gdelt_vecs @ abc_centroid)
    gdelt_df["uniqueness_vs_abc"] = gdelt_df["uniqueness_vs_abc"].clip(0, 1)

    # combined signal: high uniqueness + high cross_sim = truly novel but related
    gdelt_df["novelty_score"] = (
        gdelt_df["uniqueness_vs_abc"] * 0.6 + (1.0 - gdelt_df["cross_sim"]) * 0.4
    )
    gdelt_df["similarity_rank"] = gdelt_df["cross_sim"].rank(ascending=False).astype(int)

    # save
    keep_cols = [c for c in [
        "GKGRECORDID", "DATE", "SOURCECOMMONNAME", "DOCUMENTIDENTIFIER",
        "source_lang", "theme_string", "tone_value",
        "cross_sim", "uniqueness_vs_abc", "novelty_score", "similarity_rank",
    ] if c in gdelt_df.columns]
    gdelt_df[keep_cols].to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}  ({len(gdelt_df):,} records)")

    # ── propagation timeline ──
    prop_df = build_propagation_timeline(
        gdelt_df, gdelt_vecs, abc_vecs, abc_df, n_clusters=args.n_clusters
    )
    if not prop_df.empty:
        prop_df.to_csv(OUT_PROP, index=False)
        print(f"Propagation timeline saved: {OUT_PROP}")

    # ── reports ──
    print("\n── Top 10 GDELT Events Most Similar to ABC Corpus ──────────────")
    top_sim = gdelt_df.nlargest(10, "cross_sim")
    for _, r in top_sim.iterrows():
        src   = str(r.get("SOURCECOMMONNAME", "?"))[:30]
        lang  = r.get("source_lang", "?")
        sim   = r.get("cross_sim", 0)
        kws   = str(r.get("theme_string", ""))[:60]
        print(f"  [{lang}] {src:<32} sim={sim:.3f}  {kws}")

    print("\n── Language Distribution of GDELT Sources ───────────────────────")
    lang_dist = gdelt_df["source_lang"].value_counts()
    for lang, cnt in lang_dist.items():
        bar = "█" * min(30, int(cnt / max(lang_dist) * 30))
        print(f"  {lang:<5} {cnt:>5}  {bar}")

    if not prop_df.empty:
        print("\n── Propagation Timeline (who covers each topic first) ───────────")
        for cl, grp in prop_df.groupby("cluster"):
            first = grp.iloc[0]
            print(
                f"  Cluster {cl}  [{first['keywords'][:40]}]"
                f"  → first: [{first['language']}] at {first['first_appearance']}"
            )

    print("\n── Most Novel GDELT Events (high uniqueness vs ABC corpus) ──────")
    top_novel = gdelt_df.nlargest(10, "novelty_score")
    for _, r in top_novel.iterrows():
        src   = str(r.get("SOURCECOMMONNAME", "?"))[:30]
        lang  = r.get("source_lang", "?")
        ns    = r.get("novelty_score", 0)
        kws   = str(r.get("theme_string", ""))[:60]
        print(f"  [{lang}] {src:<32} novelty={ns:.3f}  {kws}")

    print("\nDone.")


if __name__ == "__main__":
    main()
