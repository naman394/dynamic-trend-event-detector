"""
LDA (Gensim) vs SBERT semantic clusters — topic modelling comparison on df_clean.pkl (2019–2021).

Primary workflow: notebooks/11_topic_modeling_lda_bertopic.ipynb (inline pipeline; run all cells).

SBERT (all-MiniLM-L6-v2) embeddings → UMAP (5D) → HDBSCAN; topic words = TF-IDF per cluster.

Outputs
-------
  reports/topic_modeling/lda_k_selection_metrics.csv  (C_V, U_Mass, diversity, perplexity per K)
  reports/topic_modeling/lda_k_metrics_sweep.png
  reports/topic_modeling/topics_over_time.csv
  reports/topic_modeling/sbert_intertopic.html
  reports/topic_modeling/sbert_topics_over_time.html
  reports/topic_modeling/lda_vs_sbert_*.csv
  models/sbert_topic_bundle.joblib

Requires: Python 3.11 env with gensim, sentence-transformers, umap-learn, hdbscan, scikit-learn, plotly

Run:  python src/topic_modeling_lda_bertopic.py
      python src/topic_modeling_lda_bertopic.py --max-docs 5000   # fast smoke test
"""

from __future__ import annotations

import argparse
import math
import os
import time
import warnings
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gensim import corpora, models
from gensim.models import CoherenceModel
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from umap import UMAP

warnings.filterwarnings("ignore", category=UserWarning)

OUT_DIR = "reports/topic_modeling"
MODEL_DIR = "models"
PKL_PATH = "data/df_clean.pkl"

K_SWEEP = [5, 10, 15, 20, 25]
LDA_FIXED_K = 10
LDA_PASSES = 10
# Held-out perplexity + coherence on training slice (avoid test leakage)
LDA_HOLDOUT_FRACTION = 0.1
LDA_TOPN_DIVERSITY = 10
LDA_MIN_DOCS_FOR_HOLDOUT = 30


def ensure_df_clean() -> pd.DataFrame:
    if not os.path.isfile(PKL_PATH):
        raise FileNotFoundError(
            f"{PKL_PATH} not found. Run: python src/prepare_df_clean.py"
        )
    return pd.read_pickle(PKL_PATH)


def tokenize(text: str) -> List[str]:
    return [t for t in text.split() if len(t) > 2]


def lda_topic_word_diversity(lda_model: models.LdaModel, topn: int = 10) -> float:
    """
    Proportion of *distinct* words among all top-word slots (K topics × topn words).
    Higher ⇒ topics overlap less in their top terms.
    """
    seen: set[str] = set()
    nslots = 0
    for i in range(lda_model.num_topics):
        for w, _ in lda_model.show_topic(i, topn=topn):
            seen.add(w)
            nslots += 1
    return len(seen) / max(nslots, 1)


def _train_test_texts_corpus(
    texts: List[List[str]],
    corpus,
    random_state: int,
) -> tuple[
    List[List[str]],
    List[List[str]],
    list,
    list,
    bool,
]:
    """Split document indices; return train/test texts and BoW corpora."""
    n = len(corpus)
    if n < LDA_MIN_DOCS_FOR_HOLDOUT:
        return texts, texts, corpus, corpus, False
    idx = np.arange(n)
    tr, te = train_test_split(
        idx, test_size=LDA_HOLDOUT_FRACTION, random_state=random_state
    )
    train_texts = [texts[i] for i in tr]
    test_texts = [texts[i] for i in te]
    train_corpus = [corpus[i] for i in tr]
    test_corpus = [corpus[i] for i in te]
    return train_texts, test_texts, train_corpus, test_corpus, True


def run_lda_sweep(
    texts: List[List[str]],
    dictionary: corpora.Dictionary,
    corpus,
    random_state: int = 42,
) -> tuple[list[int], pd.DataFrame, models.LdaModel]:
    """
    For each K: fit on **training** slice; report C_V, U_Mass (train), topic diversity,
    held-out perplexity ``exp(-log_perplexity(test))`` (**lower** = better generalization).

    Refits **best K on the full corpus** for the saved ``lda_model_k{best}.gensim`` artifact.
    """
    np.random.seed(random_state)
    train_texts, test_texts, train_corpus, test_corpus, holdout_ok = (
        _train_test_texts_corpus(texts, corpus, random_state)
    )

    rows: list[dict] = []
    best_c_v, best_k = -1.0, K_SWEEP[0]

    for k in K_SWEEP:
        t0 = time.perf_counter()
        lda_k = models.LdaModel(
            corpus=train_corpus,
            id2word=dictionary,
            num_topics=k,
            passes=LDA_PASSES,
            random_state=random_state,
            alpha="auto",
        )
        c_v = CoherenceModel(
            model=lda_k,
            texts=train_texts,
            dictionary=dictionary,
            coherence="c_v",
        ).get_coherence()
        u_mass = CoherenceModel(
            model=lda_k,
            texts=train_texts,
            dictionary=dictionary,
            coherence="u_mass",
        ).get_coherence()
        div = lda_topic_word_diversity(lda_k, topn=LDA_TOPN_DIVERSITY)
        if holdout_ok and len(test_corpus) > 0:
            # Gensim: lower exp(-log_perplexity) ⇒ better held-out fit
            perp = math.exp(-lda_k.log_perplexity(test_corpus))
            perp_note = "held-out"
        else:
            perp = math.exp(-lda_k.log_perplexity(train_corpus))
            perp_note = "in-sample (corpus small for holdout)"
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "k": k,
                "c_v": c_v,
                "u_mass": u_mass,
                "topic_word_diversity": div,
                "perplexity_exp_neg_log_p": perp,
                "perplexity_split": perp_note,
                "train_secs": round(elapsed, 1),
            }
        )
        print(
            f"  K={k:2d}  C_V={c_v:.4f}  U_mass={u_mass:.4f}  "
            f"div={div:.3f}  perp={perp:.2f}  ({elapsed:.1f}s)"
        )
        if c_v > best_c_v:
            best_c_v, best_k = c_v, k

    metrics_df = pd.DataFrame(rows)
    print(
        f"\n► Best K by C_V (same primary rule as before): {best_k} (C_V={best_c_v:.4f})"
    )
    if holdout_ok:
        k_min_perp = int(
            metrics_df.loc[
                metrics_df["perplexity_exp_neg_log_p"].idxmin(), "k"
            ]
        )
        print(
            f"► Best K by lowest held-out perplexity: {k_min_perp} "
            f"(perp={metrics_df['perplexity_exp_neg_log_p'].min():.2f})"
        )

    lda_best = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=best_k,
        passes=LDA_PASSES,
        random_state=random_state,
        alpha="auto",
    )
    return list(K_SWEEP), metrics_df, lda_best


def run_lda_fixed(
    corpus,
    dictionary,
    texts: List[List[str]],
    num_topics: int = LDA_FIXED_K,
    passes: int = LDA_PASSES,
    random_state: int = 42,
) -> tuple[models.LdaModel, float, float]:
    t0 = time.perf_counter()
    lda = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=random_state,
        alpha="auto",
    )
    train_s = time.perf_counter() - t0
    cm = CoherenceModel(
        model=lda, texts=texts, dictionary=dictionary, coherence="c_v"
    )
    return lda, cm.get_coherence(), train_s


def build_cluster_tfidf_topics(
    docs: List[str], labels: np.ndarray, topn: int = 25
) -> Dict[int, List[Tuple[str, float]]]:
    out: Dict[int, List[Tuple[str, float]]] = {}
    for tid in sorted(set(labels.tolist())):
        if tid == -1:
            continue
        mask = labels == tid
        sub = [docs[i] for i in range(len(docs)) if mask[i]]
        if len(sub) < 2:
            continue
        vec = TfidfVectorizer(
            max_features=8000,
            min_df=2,
            max_df=0.95,
            stop_words="english",
        )
        try:
            X = vec.fit_transform(sub)
        except ValueError:
            continue
        scores = np.asarray(X.sum(axis=0)).ravel()
        terms = np.array(vec.get_feature_names_out())
        idx = np.argsort(scores)[::-1][:topn]
        out[int(tid)] = [(terms[i], float(scores[i])) for i in idx]
    return out


def sbert_cluster_coherence(
    cluster_topics: Dict[int, List[Tuple[str, float]]],
    texts: List[List[str]],
    dictionary: corpora.Dictionary,
) -> float:
    tw: List[List[str]] = []
    for tid in sorted(cluster_topics.keys()):
        tw.append([w for w, _ in cluster_topics[tid][:20]])
    if len(tw) < 2:
        return float("nan")
    cm = CoherenceModel(topics=tw, texts=texts, dictionary=dictionary, coherence="c_v")
    return cm.get_coherence()


def keyword_theme_table_sbert(
    lda: models.LdaModel,
    cluster_topics: Dict[int, List[Tuple[str, float]]],
    lda_k: int,
    keywords: List[str],
    sbert_topic_ids: List[int],
) -> pd.DataFrame:
    rows = []
    for kw in keywords:
        lda_hit = None
        for i in range(lda_k):
            words = [w.lower() for w, _ in lda.show_topic(i, topn=30)]
            if any(kw in w or w in kw for w in words):
                lda_hit = i
                break
        sb_hit = None
        for tid in sbert_topic_ids:
            if tid == -1 or tid not in cluster_topics:
                continue
            words = [w.lower() for w, _ in cluster_topics[tid]]
            if any(kw in w for w in words):
                sb_hit = tid
                break
        lda_words = (
            ", ".join([w for w, _ in lda.show_topic(lda_hit, topn=12)])
            if lda_hit is not None
            else ""
        )
        sb_words = (
            ", ".join([w for w, _ in cluster_topics[sb_hit][:12]])
            if sb_hit is not None
            else ""
        )
        rows.append(
            {
                "keyword_theme": kw,
                "lda_topic_id": lda_hit,
                "lda_top_words": lda_words,
                "sbert_topic_id": sb_hit,
                "sbert_top_words": sb_words,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Subsample for quick testing (default: full corpus)",
    )
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = ensure_df_clean()
    df["headline_text"] = df["headline_text"].astype(str)
    if args.max_docs:
        df = df.sample(min(args.max_docs, len(df)), random_state=42).reset_index(
            drop=True
        )
        print(f"[subsample] Using {len(df):,} documents")

    docs = df["clean_text"].tolist()
    headlines = df["headline_text"].tolist()
    texts = [tokenize(d) for d in docs]

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=3, no_above=0.55)
    corpus = [dictionary.doc2bow(t) for t in texts]

    print("\n=== 3A — LDA (Gensim): multi-metric K sweep ===\n")
    ks, metrics_df, lda_best = run_lda_sweep(texts, dictionary, corpus)

    metrics_path = os.path.join(OUT_DIR, "lda_k_selection_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved → {metrics_path}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes[0, 0].plot(metrics_df["k"], metrics_df["c_v"], "o-", color="#2E86AB", linewidth=2)
    axes[0, 0].set_ylabel("C_V (↑ higher)")
    axes[0, 0].set_title("C_V coherence")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(metrics_df["k"], metrics_df["u_mass"], "o-", color="#6B5B95", linewidth=2)
    axes[0, 1].set_ylabel("U_Mass (↑ higher, less negative)")
    axes[0, 1].set_title("U_Mass (in-corpus co-occurrence)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(
        metrics_df["k"], metrics_df["topic_word_diversity"], "o-", color="#2ca02c", linewidth=2
    )
    axes[1, 0].set_ylabel("unique words / (K×topn)")
    axes[1, 0].set_title("Topic word diversity")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(
        metrics_df["k"],
        metrics_df["perplexity_exp_neg_log_p"],
        "o-",
        color="#d62728",
        linewidth=2,
    )
    axes[1, 1].set_ylabel("exp(−log_perp) on hold-out (↓ lower)")
    axes[1, 1].set_title("Held-out generalization (perplexity proxy)")
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.ravel():
        ax.set_xlabel("num_topics (K)")
    plt.suptitle("LDA — K sweep: coherence, diversity, perplexity (2019–2021)", y=1.02)
    plt.tight_layout()
    p_sweep = os.path.join(OUT_DIR, "lda_k_metrics_sweep.png")
    plt.savefig(p_sweep, dpi=150)
    plt.close()
    print(f"Saved → {p_sweep}")

    best_k = int(metrics_df.loc[metrics_df["c_v"].idxmax(), "k"])
    lda_best.save(os.path.join(OUT_DIR, f"lda_model_k{best_k}.gensim"))

    print("\n=== LDA reference: fixed K=10, passes=10 ===\n")
    lda10, coh10, t_lda = run_lda_fixed(corpus, dictionary, texts)
    print(f"  LDA (K=10) C_V = {coh10:.4f}  |  train time = {t_lda:.1f}s")
    lda10.save(os.path.join(OUT_DIR, "lda_model_k10.gensim"))

    print("\n=== 3B — SBERT + UMAP + HDBSCAN ===\n")

    print("Loading SBERT …")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    t0 = time.perf_counter()
    embeddings = sbert.encode(
        headlines,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings)
    t_encode = time.perf_counter() - t0
    print(f"  SBERT encode: {t_encode:.1f}s  shape={embeddings.shape}")

    umap_model = UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    reduced = umap_model.fit_transform(embeddings)

    hdbscan_model = HDBSCAN(
        min_cluster_size=20,
        metric="euclidean",
        prediction_data=True,
    )
    t1 = time.perf_counter()
    topics = hdbscan_model.fit_predict(reduced)
    t_cluster = time.perf_counter() - t1
    print(f"  UMAP+HDBSCAN: {t_cluster:.1f}s")

    cluster_topics = build_cluster_tfidf_topics(docs, topics)
    coh_sb = sbert_cluster_coherence(cluster_topics, texts, dictionary)
    print(f"  SBERT-cluster TF-IDF words C_V: {coh_sb:.4f}")

    joblib.dump(
        {
            "sentence_model_name": "all-MiniLM-L6-v2",
            "topic_labels": topics,
            "cluster_topics_tfidf": cluster_topics,
            "embeddings_shape": embeddings.shape,
        },
        os.path.join(MODEL_DIR, "sbert_topic_bundle.joblib"),
    )
    print(f"Saved → {MODEL_DIR}/sbert_topic_bundle.joblib")

    df_t = df.copy()
    df_t["topic_id"] = topics
    df_t["publish_date"] = pd.to_datetime(df_t["publish_date"])
    bins = pd.qcut(df_t["publish_date"], q=40, duplicates="drop")
    df_t["time_bin"] = bins
    counts = (
        df_t.groupby(["time_bin", "topic_id"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    row_sum = counts.sum(axis=1).replace(0, np.nan)
    prop = counts.div(row_sum, axis=0)
    topics_time = prop.reset_index().melt(
        id_vars=["time_bin"], var_name="topic_id", value_name="proportion"
    )
    topics_time["bin_start"] = topics_time["time_bin"].apply(lambda x: x.left)
    topics_time["bin_end"] = topics_time["time_bin"].apply(lambda x: x.right)
    topics_time["bin_mid"] = topics_time["time_bin"].apply(lambda x: x.mid)
    p_tot = os.path.join(OUT_DIR, "topics_over_time.csv")
    topics_time.to_csv(p_tot, index=False)
    print(f"Saved → {p_tot}  ({len(topics_time)} rows)")

    sb_ids = sorted({int(t) for t in topics if t != -1})
    centroids = []
    for tid in sb_ids:
        mask = topics == tid
        centroids.append(embeddings[mask].mean(axis=0))
    if len(centroids) >= 2:
        centroids = np.asarray(centroids)
        nn = min(15, len(centroids) - 1)
        cent_2d = UMAP(
            n_components=2,
            n_neighbors=max(2, nn),
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        ).fit_transform(centroids)
        fig_it = go.Figure(
            data=[
                go.Scatter(
                    x=cent_2d[:, 0],
                    y=cent_2d[:, 1],
                    mode="markers+text",
                    text=[f"T{t}" for t in sb_ids],
                    textposition="top center",
                    marker=dict(
                        size=14, color=sb_ids, colorscale="Viridis", showscale=True
                    ),
                )
            ]
        )
        fig_it.update_layout(
            title="SBERT + HDBSCAN — topic centroids (UMAP-2D)",
            height=520,
        )
        fig_it.write_html(os.path.join(OUT_DIR, "sbert_intertopic.html"))
        print(f"Saved → {OUT_DIR}/sbert_intertopic.html")

    vc = pd.Series(topics).value_counts()
    top8 = [t for t in vc.index.tolist() if t != -1][:8]
    bin_mid = [ival.mid for ival in prop.index]
    fig_ot = go.Figure()
    for tid in top8:
        if tid not in prop.columns:
            continue
        fig_ot.add_trace(
            go.Scatter(
                x=bin_mid,
                y=prop[tid].values,
                mode="lines",
                name=f"Topic {tid}",
            )
        )
    fig_ot.update_layout(
        title="SBERT clusters — topic proportion over time (40 bins)",
        height=520,
    )
    fig_ot.write_html(os.path.join(OUT_DIR, "sbert_topics_over_time.html"))
    print(f"Saved → {OUT_DIR}/sbert_topics_over_time.html")

    rows_side = []
    for i in range(LDA_FIXED_K):
        lda_w = ", ".join([w for w, _ in lda10.show_topic(i, topn=10)])
        rows_side.append({"model": "LDA", "topic_id": i, "top_words": lda_w})
    for tid in sb_ids:
        cw = cluster_topics.get(tid, [])
        bw = ", ".join([w for w, _ in cw[:10]])
        rows_side.append({"model": "SBERT+HDBSCAN", "topic_id": tid, "top_words": bw})
    side = pd.DataFrame(rows_side)
    side.to_csv(os.path.join(OUT_DIR, "lda_vs_sbert_topwords_long.csv"), index=False)

    theme_kw = [
        "bushfire",
        "fire",
        "covid",
        "coronavirus",
        "climate",
        "police",
        "election",
    ]
    kw_df = keyword_theme_table_sbert(
        lda10, cluster_topics, LDA_FIXED_K, theme_kw, sb_ids
    )
    kw_df.to_csv(os.path.join(OUT_DIR, "lda_vs_sbert_theme_keywords.csv"), index=False)

    summary = pd.DataFrame(
        [
            {
                "metric": "LDA K=10 C_V (Gensim)",
                "value": f"{coh10:.4f}",
                "notes": f"passes={LDA_PASSES}, train_time_s={t_lda:.1f}",
            },
            {
                "metric": f"LDA best K sweep (K={best_k})",
                "value": f"{metrics_df['c_v'].max():.4f}",
                "notes": f"C_V primary; see lda_k_selection_metrics.csv; sweep K in {K_SWEEP}",
            },
            {
                "metric": "SBERT clusters C_V (TF-IDF words, same texts/dict)",
                "value": f"{coh_sb:.4f}",
                "notes": f"encode_s={t_encode:.1f}, umap_hdbscan_s={t_cluster:.1f}",
            },
            {
                "metric": "SBERT clusters (excl. -1)",
                "value": str(len(sb_ids)),
                "notes": "HDBSCAN min_cluster_size=20",
            },
        ]
    )
    summary.to_csv(os.path.join(OUT_DIR, "lda_vs_sbert_comparison.csv"), index=False)
    print("\n=== Summary ===\n")
    print(summary.to_string(index=False))
    print(f"\nKey deliverable: {OUT_DIR}/lda_vs_sbert_theme_keywords.csv")


if __name__ == "__main__":
    main()
