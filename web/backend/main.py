"""
Semantic Radar — FastAPI Backend
Wraps the existing model pipeline and exposes REST endpoints for the React frontend.
"""
from __future__ import annotations

import ast
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent.parent  # project root
sys.path.insert(0, str(ROOT))

# ── Auto-refresh state ────────────────────────────────────────────────────────
_last_refresh_utc: datetime | None = None   # when GDELT was last fetched
_refresh_lock = threading.Lock()            # only one fetch at a time
_refresh_in_progress = False
GDELT_REFRESH_INTERVAL = 15 * 60           # 15 minutes in seconds
GDELT_DEFAULT_WINDOWS  = 4                 # 1 hour of GDELT windows per auto-fetch


def _do_gdelt_refresh(windows: int = GDELT_DEFAULT_WINDOWS):
    """Fetch fresh GDELT data, then immediately compute + store a trend snapshot."""
    global _last_refresh_utc, _refresh_in_progress
    with _refresh_lock:
        _refresh_in_progress = True
        try:
            from src.gdelt_fetcher import fetch_last_n_gkg
            df = fetch_last_n_gkg(n=windows, save_to=str(ROOT / "data" / "gdelt_processed.csv"))
            _last_refresh_utc = datetime.now(timezone.utc)
            count = len(df)
        finally:
            _refresh_in_progress = False

    # After new data is saved, compute trend snapshot so history grows automatically
    try:
        _store_trend_snapshot()
        print(f"[auto-refresh] Fetched {count} records, trend snapshot stored "
              f"(history depth: {len(_snapshot_history)})")
    except Exception as e:
        print(f"[auto-refresh] Trend snapshot failed: {e}")

    return count


def _store_trend_snapshot():
    """Compute cluster distribution for current GDELT data and append to history."""
    global _snapshot_history
    result = _compute_snapshot_trends()
    if result is None:
        return
    _snapshot_history.append({
        "centroid":     result["centroid"],
        "cluster_dist": result["cluster_dist"],
        "timestamp":    result["timestamp"],
        "live_vs":      result["live_vs"],
    })
    if len(_snapshot_history) > _MAX_HISTORY:
        _snapshot_history.pop(0)


def _auto_refresh_loop():
    """Background thread — refreshes GDELT every 15 minutes and tracks trend evolution."""
    while True:
        try:
            _do_gdelt_refresh()
        except Exception as e:
            print(f"[auto-refresh] cycle failed: {e}")
        time.sleep(GDELT_REFRESH_INTERVAL)


def _start_auto_refresh():
    t = threading.Thread(target=_auto_refresh_loop, daemon=True, name="gdelt-auto-refresh")
    t.start()
    print("[auto-refresh] Background GDELT refresh thread started (every 15 min)")


app = FastAPI(title="Semantic Radar API", version="1.0.0", on_startup=[_start_auto_refresh])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sbert = None
_cluster_centroids: np.ndarray | None = None
_cluster_labels: list[str] = []
_cluster_meta: dict[str, dict] = {}

# Rolling history of snapshot centroids for live velocity computation
# Each entry: {timestamp, centroid (np.ndarray), cluster_dist (dict), theme_counts (dict)}
_snapshot_history: list[dict] = []
_MAX_HISTORY = 12  # keep last 12 snapshots (~3 hours at 15-min cadence)


def get_sbert():
    global _sbert
    if _sbert is None:
        from sentence_transformers import SentenceTransformer
        _sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert


def get_cluster_centroids():
    """
    Build SBERT centroid embeddings from our trained cluster top-terms.
    This is genuine model inference using our trained cluster knowledge.
    Cached after first call.
    """
    global _cluster_centroids, _cluster_labels, _cluster_meta
    if _cluster_centroids is not None:
        return _cluster_centroids, _cluster_labels, _cluster_meta

    cd = clusters_df()
    if cd.empty:
        return None, [], {}

    model = get_sbert()
    labels = cd["Cluster"].tolist()
    # encode each cluster's top terms as a representative centroid
    texts = cd["TopTerms"].fillna("").tolist()
    centroids = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # build metadata: peak velocity week per cluster from semantic_velocity.csv
    vd = velocity_df()
    meta: dict[str, dict] = {}
    for _, row in cd.iterrows():
        meta[row["Cluster"]] = {
            "size": int(row["Size"]),
            "top_terms": row["TopTerms"],
        }

    # attach cluster-level historical peak from overall velocity (proxy)
    if not vd.empty:
        peak_row = vd.loc[vd["semantic_velocity"].idxmax()]
        for lbl in labels:
            meta[lbl]["peak_week"] = str(peak_row["week"])
            meta[lbl]["peak_velocity"] = round(float(peak_row["semantic_velocity"]), 4)

    _cluster_centroids = centroids
    _cluster_labels = labels
    _cluster_meta = meta
    return _cluster_centroids, _cluster_labels, _cluster_meta


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def gdelt_df() -> pd.DataFrame:
    return _load(ROOT / "data" / "gdelt_processed.csv")


def velocity_df() -> pd.DataFrame:
    df = _load(ROOT / "reports" / "deep_learning" / "semantic_velocity.csv")
    if df.empty:
        return df
    df["week_start"] = df["week"].str.split("/").str[0]
    return df.sort_values("week_start").reset_index(drop=True)


def clusters_df() -> pd.DataFrame:
    return _load(ROOT / "reports" / "deep_learning" / "cluster_summary.csv")


def parse_themes(v) -> list:
    if isinstance(v, list):
        return v
    try:
        return ast.literal_eval(v)
    except Exception:
        return []


# ── Status ────────────────────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    gd = gdelt_df()
    vd = velocity_df()
    cd = clusters_df()
    max_vel  = float(vd["semantic_velocity"].max()) if not vd.empty else 0
    peak_week = str(vd.loc[vd["semantic_velocity"].idxmax(), "week"]) if not vd.empty else "N/A"

    # Data freshness
    now = datetime.now(timezone.utc)
    if _last_refresh_utc:
        age_seconds = int((now - _last_refresh_utc).total_seconds())
    else:
        # estimate from file mtime
        p = ROOT / "data" / "gdelt_processed.csv"
        age_seconds = int(now.timestamp() - p.stat().st_mtime) if p.exists() else -1

    return {
        "gdelt_records":      len(gd),
        "velocity_weeks":     len(vd),
        "clusters":           len(cd),
        "max_velocity":       round(max_vel, 4),
        "peak_rupture_week":  peak_week,
        "headlines_analyzed": 1_244_184,
        "years_covered":      19,
        "gdelt_timestamp":    str(gd["DATE"].max()) if not gd.empty and "DATE" in gd.columns else None,
        "data_age_seconds":   age_seconds,
        "refresh_in_progress": _refresh_in_progress,
        "next_refresh_in":    max(0, GDELT_REFRESH_INTERVAL - age_seconds) if age_seconds >= 0 else None,
    }


# ── Velocity ──────────────────────────────────────────────────────────────────
@app.get("/api/velocity")
def velocity(limit: int = 0):
    df = velocity_df()
    if df.empty:
        return []
    threshold = df["semantic_velocity"].mean() + df["semantic_velocity"].std()
    if limit > 0:
        df = df.tail(limit)
    return [
        {
            "week": r["week"],
            "week_start": r["week_start"],
            "v": round(float(r["semantic_velocity"]), 4),
            "is_rupture": bool(r["semantic_velocity"] >= threshold),
        }
        for _, r in df.iterrows()
    ]


@app.get("/api/velocity/top")
def top_ruptures(n: int = 5):
    df = velocity_df()
    if df.empty:
        return []
    top = df.nlargest(n, "semantic_velocity")
    return [
        {"week": r["week"], "week_start": r["week_start"], "v": round(float(r["semantic_velocity"]), 4)}
        for _, r in top.iterrows()
    ]


# ── Clusters ──────────────────────────────────────────────────────────────────
@app.get("/api/clusters")
def clusters():
    df = clusters_df()
    if df.empty:
        return []
    return df.to_dict("records")


# ── GDELT ─────────────────────────────────────────────────────────────────────
@app.get("/api/gdelt/themes")
def gdelt_themes(top_n: int = 20):
    df = gdelt_df()
    if df.empty:
        return []
    all_t: list[str] = []
    for v in df["theme_list"].dropna():
        all_t.extend(parse_themes(v))
    return [{"theme": t, "count": c} for t, c in Counter(all_t).most_common(top_n)]


@app.get("/api/gdelt/articles")
def gdelt_articles(limit: int = 30):
    df = gdelt_df()
    if df.empty:
        return []
    cols = [c for c in ["SOURCECOMMONNAME", "DOCUMENTIDENTIFIER", "tone_value", "theme_list", "DATE"] if c in df.columns]
    rows = df[cols].head(limit).to_dict("records")
    for r in rows:
        r["themes_parsed"] = parse_themes(r.get("theme_list", []))[:5]
    return rows


@app.get("/api/gdelt/keywords")
def hot_keywords():
    df = gdelt_df()
    if df.empty:
        return []
    all_t: list[str] = []
    for v in df["theme_list"].dropna():
        all_t.extend(parse_themes(v))
    roots = [t.lower().split("_")[0] for t, _ in Counter(all_t).most_common(60)]
    return list(dict.fromkeys(roots))[:16]


@app.post("/api/gdelt/refresh")
def refresh_gdelt(windows: int = 16):
    if _refresh_in_progress:
        raise HTTPException(409, "Refresh already in progress")
    try:
        count = _do_gdelt_refresh(windows=windows)
        return {"records": count, "windows": windows}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/gdelt/freshness")
def gdelt_freshness():
    now = datetime.now(timezone.utc)
    if _last_refresh_utc:
        age = int((now - _last_refresh_utc).total_seconds())
    else:
        p = ROOT / "data" / "gdelt_processed.csv"
        age = int(now.timestamp() - p.stat().st_mtime) if p.exists() else -1
    return {
        "age_seconds":       age,
        "age_human":         _age_human(age),
        "refresh_in_progress": _refresh_in_progress,
        "next_refresh_in":   max(0, GDELT_REFRESH_INTERVAL - age) if age >= 0 else None,
        "is_stale":          age > GDELT_REFRESH_INTERVAL,
    }


def _age_human(seconds: int) -> str:
    if seconds < 0:    return "unknown"
    if seconds < 60:   return f"{seconds}s ago"
    if seconds < 3600: return f"{seconds // 60}m ago"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m ago"


# ── Live Trend Detection ──────────────────────────────────────────────────────

def _compute_snapshot_trends():
    """
    Core dynamic trend detection using our trained model.

    Strategy (fast):
      1. Count unique GDELT theme codes + their frequencies
      2. Encode top-N unique themes with SBERT → weighted centroid
         (avoids encoding all 17K articles)
      3. Classify each theme into our 13 trained clusters (nearest centroid)
      4. Aggregate → cluster distribution for this snapshot
      5. Store centroid; compare to previous snapshot → live V_s
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    gd = gdelt_df()
    if gd.empty:
        return None

    gd = gd.copy()
    gd["_tp"] = gd["theme_list"].apply(parse_themes)

    # Theme frequency across all articles
    tc = Counter(t for tp in gd["_tp"] for t in tp)
    if not tc:
        return None

    # Encode top 300 unique themes (covers ~95% of frequency mass)
    top_themes = [t for t, _ in tc.most_common(300)]
    theme_counts = np.array([tc[t] for t in top_themes], dtype=float)
    theme_weights = theme_counts / theme_counts.sum()

    model = get_sbert()
    theme_texts = [t.replace("_", " ").lower() for t in top_themes]
    theme_embs = model.encode(theme_texts, batch_size=128, normalize_embeddings=True,
                               show_progress_bar=False)  # shape: (300, 384)

    # Weighted centroid of entire snapshot
    snapshot_centroid = (theme_embs * theme_weights[:, None]).sum(axis=0)
    norm = np.linalg.norm(snapshot_centroid)
    if norm > 0:
        snapshot_centroid /= norm

    # Classify each theme into our 13 trained clusters
    centroids, cluster_labels, _ = get_cluster_centroids()
    cluster_dist: dict[str, float] = {lbl: 0.0 for lbl in cluster_labels}

    if centroids is not None and len(cluster_labels):
        sims = cos_sim(theme_embs, centroids)          # (300, 13)
        assignments = np.argmax(sims, axis=1)          # (300,)
        for i, cluster_idx in enumerate(assignments):
            lbl = cluster_labels[cluster_idx]
            cluster_dist[lbl] = cluster_dist.get(lbl, 0.0) + float(theme_weights[i])

    # Live semantic velocity vs previous snapshot
    live_vs = 0.0
    prev_snapshot = _snapshot_history[-1] if _snapshot_history else None
    if prev_snapshot is not None:
        prev_c = prev_snapshot["centroid"]
        live_vs = float(1.0 - cos_sim(
            snapshot_centroid.reshape(1, -1),
            prev_c.reshape(1, -1)
        )[0, 0])

    # Trend direction: compare cluster shares to previous snapshot
    trend_delta: dict[str, float] = {}
    if prev_snapshot:
        prev_dist = prev_snapshot["cluster_dist"]
        for lbl in cluster_labels:
            trend_delta[lbl] = round(cluster_dist.get(lbl, 0) - prev_dist.get(lbl, 0), 4)

    return {
        "centroid": snapshot_centroid,
        "cluster_dist": cluster_dist,
        "trend_delta": trend_delta,
        "live_vs": round(live_vs, 4),
        "total_articles": len(gd),
        "unique_themes": len(tc),
        "timestamp": str(gd["DATE"].max()) if "DATE" in gd.columns else "unknown",
    }


@app.get("/api/trends")
def live_trends():
    """
    Dynamic trend detection endpoint.
    Returns current cluster distribution, live semantic velocity,
    and trend direction (rising/falling clusters) using our trained model.
    """
    result = _compute_snapshot_trends()
    if result is None:
        raise HTTPException(404, "No GDELT data. Refresh first.")

    # Store snapshot in rolling history
    global _snapshot_history
    _snapshot_history.append({
        "centroid": result["centroid"],
        "cluster_dist": result["cluster_dist"],
        "timestamp": result["timestamp"],
    })
    if len(_snapshot_history) > _MAX_HISTORY:
        _snapshot_history.pop(0)

    # Historical velocity baseline from trained model
    vd = velocity_df()
    hist_threshold = 0.0
    rupture_label = ""
    if not vd.empty:
        vm = float(vd["semantic_velocity"].mean())
        vs = float(vd["semantic_velocity"].std())
        hist_threshold = round(vm + vs, 4)
        if result["live_vs"] >= hist_threshold:
            rupture_label = "RUPTURE"
        elif result["live_vs"] >= vm:
            rupture_label = "ELEVATED"
        else:
            rupture_label = "STABLE"

    _, cluster_labels, cluster_meta = get_cluster_centroids()

    # Historical baseline share per cluster from training corpus
    total_trained = sum(cluster_meta[lbl]["size"] for lbl in cluster_labels if lbl in cluster_meta)
    hist_share = {
        lbl: (cluster_meta[lbl]["size"] / total_trained * 100) if total_trained > 0 else 0
        for lbl in cluster_labels
    }

    clusters_out = []
    for lbl in cluster_labels:
        live_share  = round(result["cluster_dist"].get(lbl, 0) * 100, 2)
        snap_delta  = round(result["trend_delta"].get(lbl, 0) * 100, 3)
        hist        = round(hist_share.get(lbl, 0), 2)
        vs_hist     = round(live_share - hist, 2)   # deviation from 19-yr norm
        ratio       = round(live_share / hist, 2) if hist > 0 else 1.0
        meta        = cluster_meta.get(lbl, {})

        # Prefer snapshot delta when available; fall back to vs-historical
        delta = snap_delta if any(abs(c) > 0 for c in result["trend_delta"].values()) else vs_hist

        clusters_out.append({
            "cluster":       lbl,
            "share":         live_share,
            "delta":         delta,
            "hist_share":    hist,
            "vs_hist":       vs_hist,
            "ratio":         ratio,
            "trending":      vs_hist > 1.0,     # >1pp above historical norm
            "declining":     vs_hist < -1.0,    # >1pp below historical norm
            "top_terms":     meta.get("top_terms", ""),
            "size":          meta.get("size", 0),
        })

    # Sort by current share descending
    clusters_out.sort(key=lambda x: x["share"], reverse=True)

    # Velocity history (last N snapshots)
    vel_history = []
    for i, snap in enumerate(_snapshot_history):
        vel_history.append({
            "snapshot": i + 1,
            "timestamp": snap["timestamp"],
        })

    return {
        "live_vs":          result["live_vs"],
        "rupture_label":    rupture_label,
        "hist_threshold":   hist_threshold,
        "total_articles":   result["total_articles"],
        "unique_themes":    result["unique_themes"],
        "snapshot_count":   len(_snapshot_history),
        "timestamp":        result["timestamp"],
        "clusters":         clusters_out,
    }


@app.get("/api/trends/history")
def trends_history():
    """Returns cluster distribution across all stored snapshots for evolution chart."""
    if not _snapshot_history:
        return []
    _, cluster_labels, _ = get_cluster_centroids()
    out = []
    for snap in _snapshot_history:
        entry: dict = {"timestamp": snap["timestamp"]}
        for lbl in cluster_labels:
            entry[lbl] = round(snap["cluster_dist"].get(lbl, 0) * 100, 2)
        out.append(entry)
    return out


# ── Spikes ────────────────────────────────────────────────────────────────────
@app.get("/api/spikes")
def spikes(limit: int = 20):
    p = ROOT / "reports" / "topic_modeling" / "12_spikes_anchors" / "spike_events.csv"
    df = _load(p)
    if df.empty:
        return []
    return df.head(limit).to_dict("records")


# ── Topic Watchdog ────────────────────────────────────────────────────────────
class WatchdogRequest(BaseModel):
    keyword: str


@app.post("/api/watchdog")
def watchdog(req: WatchdogRequest):
    from sklearn.metrics.pairwise import cosine_similarity

    gd = gdelt_df()
    if gd.empty:
        raise HTTPException(404, "No GDELT data. Refresh first.")

    kw_words = [w for w in req.keyword.strip().lower().split() if len(w) > 2]
    if not kw_words:
        kw_words = [req.keyword.strip().lower()]

    gd = gd.copy()
    gd["_tp"] = gd["theme_list"].apply(parse_themes)
    gd["_ts"] = gd["_tp"].apply(lambda t: " ".join(t).lower())
    src = gd.get("SOURCECOMMONNAME", pd.Series("", index=gd.index)).fillna("").str.lower()
    url = gd.get("DOCUMENTIDENTIFIER", pd.Series("", index=gd.index)).fillna("").str.lower()

    mask = pd.Series(False, index=gd.index)
    for w in kw_words:
        mask |= gd["_ts"].str.contains(w, na=False) | src.str.contains(w, na=False) | url.str.contains(w, na=False)

    matched = gd[mask].copy()
    if matched.empty:
        return {
            "verdict": "NO DATA", "z_score": 0, "article_count": 0,
            "avg_tone": 0, "avg_impact": 0, "articles": [], "themes": [],
            "time_series": [], "model_insight": None,
        }

    # ── SBERT: encode matched articles only ───────────────────────────────────
    model = get_sbert()
    matched = matched.reset_index(drop=True)
    matched["_tf"] = matched["_tp"].apply(lambda t: " ".join(t) if t else "general news")
    texts = matched["_tf"].tolist()

    mat_embs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    live_centroid = mat_embs.mean(axis=0, keepdims=True)

    # per-article uniqueness vs the matched-set centroid
    sims = cosine_similarity(mat_embs, live_centroid).flatten()
    matched["su"] = 1.0 - sims
    matched["ta"] = pd.to_numeric(matched.get("tone_value", 0), errors="coerce").fillna(0).abs()
    matched["si"] = matched["su"] * matched["ta"]

    # ── Model inference: classify into our 13 trained clusters ───────────────
    model_insight = None
    centroids, cluster_labels, cluster_meta = get_cluster_centroids()
    if centroids is not None and len(cluster_labels) > 0:
        # cosine similarity between live keyword centroid and each trained cluster centroid
        cluster_sims = cosine_similarity(live_centroid, centroids).flatten()
        best_idx = int(np.argmax(cluster_sims))
        matched_cluster = cluster_labels[best_idx]
        cluster_confidence = round(float(cluster_sims[best_idx]), 4)
        meta = cluster_meta.get(matched_cluster, {})

        # historical baseline z-score using our 19-year velocity data
        vd = velocity_df()
        hist_z = 0.0
        hist_context = "No historical data available"
        if not vd.empty:
            vel_mean = float(vd["semantic_velocity"].mean())
            vel_std  = float(vd["semantic_velocity"].std())
            # proxy: treat keyword article share of snapshot as a velocity signal
            snapshot_size = max(len(gd), 1)
            kw_share = len(matched) / snapshot_size
            # scale to comparable range (historical velocities are 0.1–0.65)
            scaled = kw_share * vel_mean * 20
            hist_z = round((scaled - vel_mean) / (vel_std + 1e-9), 2)
            rupture_threshold = round(vel_mean + vel_std, 4)
            hist_context = (
                f"19-year velocity baseline: μ={vel_mean:.4f}, σ={vel_std:.4f}. "
                f"Rupture threshold: {rupture_threshold}. "
                f"Historical peak: {meta.get('peak_week','—')} (V_s={meta.get('peak_velocity','—')})."
            )

        # all cluster similarity scores for the radar chart
        cluster_scores = [
            {"cluster": cluster_labels[i], "similarity": round(float(cluster_sims[i]), 4)}
            for i in np.argsort(cluster_sims)[::-1]
        ]

        model_insight = {
            "matched_cluster": matched_cluster,
            "cluster_confidence": cluster_confidence,
            "cluster_top_terms": meta.get("top_terms", ""),
            "cluster_size": meta.get("size", 0),
            "historical_z": hist_z,
            "historical_context": hist_context,
            "peak_week": meta.get("peak_week", "—"),
            "peak_velocity": meta.get("peak_velocity", 0),
            "cluster_scores": cluster_scores[:5],  # top 5 closest clusters
        }

    # ── z-score: article count vs per-theme distribution in snapshot ──────────
    tc = Counter(t for tp in gd["_tp"] for t in tp)
    if tc:
        arr = np.array(list(tc.values()), dtype=float)
        z = float((len(matched) - arr.mean()) / (arr.std() + 1e-9))
    else:
        z = 0.0
    verdict = "BREAKING" if z >= 2.5 else "TRENDING" if z >= 1.0 else "NORMAL"

    # ── Top articles by SBERT impact score ────────────────────────────────────
    articles = []
    for _, row in matched.nlargest(10, "si").iterrows():
        articles.append({
            "source": str(row.get("SOURCECOMMONNAME", "")),
            "url": str(row.get("DOCUMENTIDENTIFIER", "")),
            "tone": round(float(row.get("tone_value", 0) or 0), 2),
            "impact": round(float(row.get("si", 0)), 4),
            "uniqueness": round(float(row.get("su", 0)), 4),
        })

    # ── Dominant themes ───────────────────────────────────────────────────────
    all_t: list[str] = []
    for v in matched["_tp"]:
        all_t.extend(v)
    themes = [{"theme": t, "count": c} for t, c in Counter(all_t).most_common(12)]

    # ── Time series (15-min bins) ─────────────────────────────────────────────
    time_series: list[dict] = []
    if "DATE" in matched.columns:
        try:
            matched["_dt"] = pd.to_datetime(
                matched["DATE"].astype(str), format="%Y%m%d%H%M%S", errors="coerce"
            )
            ts = (
                matched.dropna(subset=["_dt"])
                .set_index("_dt")
                .resample("15min")
                .size()
                .reset_index(name="count")
            )
            time_series = [{"time": str(r["_dt"]), "count": int(r["count"])} for _, r in ts.iterrows()]
        except Exception:
            pass

    return {
        "verdict": verdict,
        "z_score": round(z, 2),
        "article_count": len(matched),
        "avg_tone": round(float(pd.to_numeric(matched.get("tone_value", pd.Series([0])), errors="coerce").mean()), 3),
        "avg_impact": round(float(matched["si"].mean()), 4),
        "articles": articles,
        "themes": themes,
        "time_series": time_series,
        "model_insight": model_insight,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
