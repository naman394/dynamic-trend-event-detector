"""
Dynamic Trend & Event Detector — Streamlit Interactive Dashboard
Run: streamlit run dashboard.py
"""

import ast
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
DL = REPORTS / "deep_learning"
TM = REPORTS / "topic_modeling" / "11_lda_sbert"
SPIKES = REPORTS / "topic_modeling" / "12_spikes_anchors"
SRC = ROOT / "src"

VELOCITY_CSV   = DL / "semantic_velocity.csv"
CLUSTER_CSV    = DL / "cluster_summary.csv"
TOPICS_CSV     = TM / "topics_over_time.csv"
SPIKE_CSV      = SPIKES / "spike_events.csv"
GDELT_CSV      = DATA / "gdelt_processed.csv"
IMPACT_CSV     = REPORTS / "event_impact_scores.csv"
RUPTURE_CSV    = REPORTS / "rupture_verification.csv"
HEADLINES_CSV  = DATA / "news_headlines.csv"
FULL_CSV       = DATA / "abcnews-date-text 2.csv"

# optional outputs from the new src scripts
NER_WEEKLY     = REPORTS / "ner_weekly_freq.csv"
NER_ENTITIES   = REPORTS / "ner_entities.csv"
LSTM_CSV       = REPORTS / "lstm_anomalies.csv"
GRANGER_HTML   = REPORTS / "granger_causality.html"
GRANGER_CSV    = REPORTS / "granger_edges.csv"
MULTI_CSV      = REPORTS / "multilingual_signals.csv"

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Semantic Radar — Dynamic Trend Detector",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_velocity():
    df = pd.read_csv(VELOCITY_CSV)
    df["week_start"] = df["week"].str.split("/").str[0]
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start").reset_index(drop=True)
    return df


@st.cache_data(ttl=300)
def load_clusters():
    return pd.read_csv(CLUSTER_CSV)


@st.cache_data(ttl=300)
def load_topics():
    df = pd.read_csv(TOPICS_CSV)
    df["bin_mid"] = pd.to_datetime(df["bin_mid"], errors="coerce")
    df = df.dropna(subset=["bin_mid"])
    df = df[df["topic_id"] >= 0]   # drop HDBSCAN noise label -1
    return df


@st.cache_data(ttl=300)
def load_spikes():
    df = pd.read_csv(SPIKE_CSV)
    df["spike_date"] = pd.to_datetime(df["spike_date"], errors="coerce")
    return df


@st.cache_data(ttl=60)   # short TTL so refresh button works
def load_gdelt():
    if not GDELT_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(GDELT_CSV)
    return df


@st.cache_data(ttl=60)
def load_impact():
    if not IMPACT_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(IMPACT_CSV).nlargest(50, "impact_score")
    return df


@st.cache_data
def load_headlines():
    path = FULL_CSV if FULL_CSV.exists() else HEADLINES_CSV
    df = pd.read_csv(path, dtype={"publish_date": str})
    df.columns = [c.strip() for c in df.columns]
    # normalise column names
    if "headline_text" not in df.columns and len(df.columns) >= 2:
        df.columns = ["publish_date", "headline_text"]
    df["publish_date"] = pd.to_datetime(df["publish_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["publish_date"])
    return df


@st.cache_data
def load_ruptures():
    if not RUPTURE_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(RUPTURE_CSV)


@st.cache_resource(show_spinner="Loading SBERT model…")
def load_sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def watchdog_search(keyword: str, gdelt_df: pd.DataFrame):
    """
    Filter GDELT snapshot for keyword, compute impact scores via SBERT,
    and return (filtered_df, z_score, verdict).

    Search strategy (any word in the keyword must match):
      1. GDELT theme codes  — e.g. "economy" matches ECON_*, EPU_ECONOMY_*
      2. Source name        — news outlet name
      3. Article URL        — company/topic names often appear in URLs
    """
    import ast as _ast
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    kw_full = keyword.strip().lower()
    # search each word independently so "tata motors" finds articles with "tata" OR "motors"
    kw_words = [w for w in kw_full.split() if len(w) > 2]
    if not kw_words:
        kw_words = [kw_full]

    # parse theme_list if still a string
    def _themes(v):
        if isinstance(v, list):
            return v
        try:
            return _ast.literal_eval(v)
        except Exception:
            return []

    gdelt_df = gdelt_df.copy()
    gdelt_df["_themes_parsed"] = gdelt_df["theme_list"].apply(_themes)
    gdelt_df["_theme_str"] = gdelt_df["_themes_parsed"].apply(lambda t: " ".join(t).lower())

    src_col = gdelt_df.get("SOURCECOMMONNAME", pd.Series("", index=gdelt_df.index)).fillna("").str.lower()
    url_col = gdelt_df.get("DOCUMENTIDENTIFIER", pd.Series("", index=gdelt_df.index)).fillna("").str.lower()

    # build mask: any word matches any field
    mask = pd.Series(False, index=gdelt_df.index)
    for w in kw_words:
        mask |= (
            gdelt_df["_theme_str"].str.contains(w, na=False)
            | src_col.str.contains(w, na=False)
            | url_col.str.contains(w, na=False)
        )
    matched = gdelt_df[mask].copy()

    if matched.empty:
        return pd.DataFrame(), 0.0, "NO DATA"

    # ── SBERT impact scoring ─────────────────────────────────────────────────
    model = load_sbert()

    matched["_theme_str_full"] = matched["_themes_parsed"].apply(
        lambda t: " ".join(t) if t else "general news"
    )
    all_theme_strings = gdelt_df["_theme_str"].replace("", "general news").tolist()
    matched_strings   = matched["_theme_str_full"].tolist()

    # encode all GDELT records for centroid, then encode matched
    all_embs     = model.encode(all_theme_strings,   batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    matched_embs = model.encode(matched_strings,      batch_size=64,  normalize_embeddings=True, show_progress_bar=False)

    global_centroid = all_embs.mean(axis=0, keepdims=True)
    sims = cosine_similarity(matched_embs, global_centroid).flatten()
    matched["semantic_uniqueness"] = 1.0 - sims
    matched["tone_abs"]    = matched["tone_value"].abs()
    matched["impact_score"] = matched["semantic_uniqueness"] * matched["tone_abs"]

    # ── z-score verdict ──────────────────────────────────────────────────────
    # Count articles per theme across full snapshot, compare keyword count
    from collections import Counter
    theme_counts = Counter(
        t for themes in gdelt_df["_themes_parsed"] for t in themes
    )
    counts = np.array(list(theme_counts.values()), dtype=float)
    mu, sigma = counts.mean(), counts.std()
    kw_count = float(len(matched))
    z = (kw_count - mu) / (sigma + 1e-9)

    if z >= 2.5:
        verdict = "BREAKING"
    elif z >= 1.5:
        verdict = "TRENDING"
    else:
        verdict = "NORMAL"

    return matched, z, verdict


def headlines_in_week(headlines_df, week_str):
    """Return headlines published in the given week string 'YYYY-MM-DD/YYYY-MM-DD'."""
    try:
        start_s, end_s = week_str.split("/")
        start = pd.to_datetime(start_s)
        end   = pd.to_datetime(end_s)
    except Exception:
        return pd.DataFrame()
    mask = (headlines_df["publish_date"] >= start) & (headlines_df["publish_date"] <= end)
    return headlines_df[mask].head(50)


def velocity_fig(vel_df, highlight_week=None):
    threshold = vel_df["semantic_velocity"].mean() + vel_df["semantic_velocity"].std()
    spikes = vel_df[vel_df["semantic_velocity"] >= threshold]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vel_df["week_start"], y=vel_df["semantic_velocity"],
        mode="lines", name="Semantic Velocity",
        line=dict(color="#4C9BE8", width=1.5),
        hovertemplate="<b>%{customdata}</b><br>Velocity: %{y:.4f}<extra></extra>",
        customdata=vel_df["week"],
    ))
    fig.add_trace(go.Scatter(
        x=spikes["week_start"], y=spikes["semantic_velocity"],
        mode="markers", name="Rupture Spikes",
        marker=dict(color="#FF4B4B", size=8, symbol="diamond"),
        hovertemplate="<b>%{customdata}</b><br>Velocity: %{y:.4f}<extra></extra>",
        customdata=spikes["week"],
    ))
    fig.add_hline(y=threshold, line_dash="dot", line_color="orange",
                  annotation_text=f"Rupture threshold ({threshold:.3f})")

    if highlight_week:
        row = vel_df[vel_df["week"] == highlight_week]
        if not row.empty:
            fig.add_vline(x=str(row["week_start"].iloc[0]), line_color="#FFD700",
                          line_width=2, annotation_text="Selected")

    fig.update_layout(
        height=420, template="plotly_dark",
        title="Semantic Velocity — Narrative Rupture Timeline",
        xaxis_title="Week", yaxis_title="Cosine Distance (V_s)",
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ── sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Semantic Radar")
    st.markdown("**Dynamic Trend & Event Detector**")
    st.caption("Navnit Naman · Kanhaiya Kumar\nNewton School of Technology")
    st.divider()

    vel_df = load_velocity()
    top_spikes = vel_df.nlargest(5, "semantic_velocity")[["week", "semantic_velocity"]]
    st.markdown("**Top 5 Rupture Weeks**")
    for _, r in top_spikes.iterrows():
        st.markdown(f"- `{r['week']}` — **{r['semantic_velocity']:.4f}**")

    st.divider()
    clusters_df = load_clusters()
    st.markdown("**13 Semantic Clusters**")
    for _, r in clusters_df.iterrows():
        st.markdown(f"- **{r['Cluster']}** ({r['Size']:,}) — {r['TopTerms'][:35]}…")


# ── tabs ───────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🌊 Velocity & Ruptures",
    "📊 Cluster Evolution",
    "🌐 GDELT Live Feed",
    "🔍 Keyword Search",
    "🏷️ NER Entities",
    "🤖 LSTM Anomaly",
    "🕸️ Causal Graph",
    "🌍 Multilingual",
    "🎯 Topic Watchdog",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Semantic Velocity & Rupture Explorer
# ══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Semantic Velocity — Narrative Rupture Timeline")
    st.markdown(
        "**V_s(t) = 1 − cosine_similarity(centroid\\_t, centroid\\_t−1)** "
        "— measures how far the aggregate meaning of news shifted week-over-week."
    )

    vel_df = load_velocity()
    headlines_df = load_headlines()
    ruptures_df  = load_ruptures()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Weeks", len(vel_df))
    col2.metric("Max Velocity", f"{vel_df['semantic_velocity'].max():.4f}")
    col3.metric(
        "Peak Rupture Week",
        vel_df.loc[vel_df['semantic_velocity'].idxmax(), 'week']
    )

    # week selector
    threshold = vel_df["semantic_velocity"].mean() + vel_df["semantic_velocity"].std()
    spike_weeks = vel_df[vel_df["semantic_velocity"] >= threshold]["week"].tolist()

    sel_week = st.selectbox(
        "Select a week to explore (spikes highlighted with ⚡):",
        options=["— none —"] + [
            f"⚡ {w}" if w in spike_weeks else w for w in vel_df["week"].tolist()
        ],
    )
    clean_week = sel_week.replace("⚡ ", "").strip() if sel_week != "— none —" else None

    st.plotly_chart(
        velocity_fig(vel_df, highlight_week=clean_week),
        use_container_width=True,
    )

    if clean_week:
        row = vel_df[vel_df["week"] == clean_week].iloc[0]
        v = row["semantic_velocity"]
        st.markdown(f"### Week `{clean_week}` — Velocity **{v:.4f}**")

        if v >= threshold:
            st.error(f"🔴 **RUPTURE DETECTED** — velocity {v:.4f} exceeds threshold {threshold:.4f}")
        else:
            st.success(f"🟢 Stable week — velocity {v:.4f} is within normal range")

        # check rupture_verification for context
        if not ruptures_df.empty and "week" in ruptures_df.columns:
            match = ruptures_df[ruptures_df["week"] == clean_week]
            if not match.empty:
                r = match.iloc[0]
                st.markdown("**Verified rupture context (top headlines that week):**")
                for h in str(r.get("sample_headlines", "")).split("|"):
                    if h.strip():
                        st.markdown(f"- {h.strip()}")

        # pull headlines from corpus
        week_headlines = headlines_in_week(headlines_df, clean_week)
        if not week_headlines.empty:
            st.markdown(f"**{len(week_headlines)} headlines from corpus that week:**")
            st.dataframe(
                week_headlines[["publish_date", "headline_text"]].rename(
                    columns={"publish_date": "Date", "headline_text": "Headline"}
                ),
                use_container_width=True, height=280,
            )
        else:
            st.info("No headlines found for this week in the loaded corpus.")

    st.divider()
    st.subheader("Full Velocity Table")
    st.dataframe(
        vel_df[["week", "semantic_velocity"]].rename(
            columns={"week": "Week", "semantic_velocity": "V_s"}
        ).sort_values("V_s", ascending=False),
        use_container_width=True, height=300,
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Cluster Evolution
# ══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Topic Cluster Evolution (2019–2021)")
    st.markdown("Select a topic cluster to see how its narrative proportion evolved over time.")

    topics_df   = load_topics()
    clusters_df = load_clusters()

    if topics_df.empty:
        st.warning("topics_over_time.csv not found. Run notebook 11 first.")
    else:
        # build topic menu from clusters + raw IDs
        unique_ids = sorted(topics_df["topic_id"].unique())
        cluster_labels = {r["Cluster"]: r["TopTerms"] for _, r in clusters_df.iterrows()}

        col_a, col_b = st.columns([1, 2])
        with col_a:
            mode = st.radio("View mode", ["Single topic", "Compare clusters", "Top-N topics"])
            if mode == "Single topic":
                sel_id = st.selectbox("Topic ID", unique_ids)
                n_smooth = st.slider("Smoothing window (bins)", 1, 5, 1)
            elif mode == "Compare clusters":
                sel_ids = st.multiselect("Topic IDs", unique_ids, default=unique_ids[:5])
            else:
                top_n = st.slider("Show top N topics by total volume", 3, 20, 8)

        with col_b:
            if mode == "Single topic":
                sub = topics_df[topics_df["topic_id"] == sel_id].sort_values("bin_mid")
                sub = sub.copy()
                sub["proportion_smooth"] = sub["proportion"].rolling(n_smooth, center=True, min_periods=1).mean()
                fig = px.line(
                    sub, x="bin_mid", y="proportion_smooth",
                    title=f"Topic {sel_id} — proportion over time",
                    labels={"bin_mid": "Date", "proportion_smooth": "Proportion"},
                    template="plotly_dark",
                )
                fig.update_traces(line_color="#4C9BE8")
                # mark spikes
                spikes_df = load_spikes()
                if not spikes_df.empty:
                    tsub = spikes_df[spikes_df["topic_id"] == sel_id]
                    for _, sr in tsub.iterrows():
                        fig.add_vline(
                            x=str(sr["spike_date"]), line_color="#FF4B4B",
                            line_dash="dash", line_width=1,
                            annotation_text=f"Z={sr['velocity_z']:.1f}",
                        )
                fig.update_layout(height=380, margin=dict(l=40, r=20, t=50, b=40))
                st.plotly_chart(fig, use_container_width=True)

                # spike info
                if not spikes_df.empty:
                    tsub = spikes_df[spikes_df["topic_id"] == sel_id]
                    if not tsub.empty:
                        st.markdown("**Detected spikes for this topic:**")
                        st.dataframe(
                            tsub[["spike_date", "velocity", "velocity_z", "keywords"]],
                            use_container_width=True,
                        )

            elif mode == "Compare clusters":
                sub = topics_df[topics_df["topic_id"].isin(sel_ids)]
                fig = px.line(
                    sub, x="bin_mid", y="proportion", color="topic_id",
                    title="Topic proportion comparison",
                    labels={"bin_mid": "Date", "proportion": "Proportion", "topic_id": "Topic"},
                    template="plotly_dark",
                )
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

            else:  # Top-N
                top_ids = (
                    topics_df.groupby("topic_id")["proportion"]
                    .sum().nlargest(top_n).index.tolist()
                )
                sub = topics_df[topics_df["topic_id"].isin(top_ids)]
                fig = px.line(
                    sub, x="bin_mid", y="proportion", color="topic_id",
                    title=f"Top {top_n} topics by cumulative proportion",
                    labels={"bin_mid": "Date", "proportion": "Proportion", "topic_id": "Topic"},
                    template="plotly_dark",
                )
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

        # K-Means cluster summary always shown below
        st.divider()
        st.subheader("K-Means Cluster Summary (K=13, full corpus 2003–2021)")
        st.dataframe(clusters_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — GDELT Live Feed
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("GDELT Live Intelligence Feed")
    st.markdown(
        "GDELT GKG 2.0 updates every **15 minutes** across 100+ languages. "
        "Click **Refresh** to fetch the latest snapshot and recompute impact scores."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Refresh GDELT Now", type="primary", use_container_width=True):
            with st.spinner("Fetching latest GDELT snapshot…"):
                result = subprocess.run(
                    [sys.executable, str(SRC / "gdelt_processor.py")],
                    capture_output=True, text=True, cwd=str(ROOT),
                )
            if result.returncode == 0:
                st.success("GDELT refreshed successfully!")
                st.cache_data.clear()
            else:
                st.error(f"GDELT fetch failed:\n{result.stderr[-400:]}")

        windows = st.selectbox("Time window", ["15 min (1)", "1 hour (4)", "4 hours (16)", "24 hours (96)"])
        win_map  = {"15 min (1)": 1, "1 hour (4)": 4, "4 hours (16)": 16, "24 hours (96)": 96}
        if st.button("🔄 Refresh with wider window", use_container_width=True):
            n_win = win_map[windows]
            with st.spinner(f"Fetching {n_win} GDELT window(s)…"):
                result = subprocess.run(
                    [sys.executable, "-c",
                     f"import sys; sys.path.insert(0,'.'); "
                     f"from src.gdelt_fetcher import fetch_last_n_gkg; "
                     f"df = fetch_last_n_gkg(n={n_win}, save_to='data/gdelt_processed.csv'); "
                     f"print(f'Fetched {{len(df):,}} records')"],
                    capture_output=True, text=True, cwd=str(ROOT),
                )
            if result.returncode == 0:
                st.success(result.stdout.strip() or "Done.")
                st.cache_data.clear()
            else:
                st.error(result.stderr[-300:])

    with col2:
        gdelt_df = load_gdelt()
        if gdelt_df.empty:
            st.warning("No GDELT data. Click Refresh to fetch.")
        else:
            st.metric("Records loaded", len(gdelt_df))
            if "DATE" in gdelt_df.columns:
                st.caption(f"Latest timestamp: `{gdelt_df['DATE'].max()}`")

            # theme frequency chart
            all_themes = []
            for tl in gdelt_df["theme_list"].dropna():
                try:
                    themes = ast.literal_eval(tl) if isinstance(tl, str) else tl
                    all_themes.extend(themes)
                except Exception:
                    pass

            if all_themes:
                theme_counts = (
                    pd.Series(all_themes)
                    .value_counts()
                    .head(20)
                    .reset_index()
                )
                theme_counts.columns = ["Theme", "Count"]
                fig = px.bar(
                    theme_counts, x="Count", y="Theme", orientation="h",
                    title="Top 20 GDELT Themes (current snapshot)",
                    template="plotly_dark", color="Count",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(height=500, margin=dict(l=20, r=20, t=50, b=30), yaxis={"autorange": "reversed"})
                st.plotly_chart(fig, use_container_width=True)

    # impact scores
    st.divider()
    st.subheader("Top Impact Events (S_I = Semantic Uniqueness × |Tone|)")
    impact_df = load_impact()
    if impact_df.empty:
        st.info("Run notebook 09 to generate event_impact_scores.csv.")
    else:
        cols_show = [c for c in ["SOURCECOMMONNAME", "impact_score", "semantic_uniqueness", "tone_abs", "DOCUMENTIDENTIFIER"] if c in impact_df.columns]
        st.dataframe(impact_df[cols_show].head(20), use_container_width=True)

        fig2 = px.histogram(
            impact_df, x="impact_score", nbins=50,
            title="Impact Score Distribution",
            template="plotly_dark", color_discrete_sequence=["#FF4B4B"],
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — Keyword Search
# ══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Keyword Search — Cluster Membership & Velocity History")

    query = st.text_input("Enter a keyword or phrase:", placeholder="e.g. bushfire, coronavirus, trump")

    clusters_df = load_clusters()
    topics_df   = load_topics()
    headlines_df = load_headlines()
    vel_df      = load_velocity()

    if query.strip():
        q = query.strip().lower()

        # 1. Match against K-Means cluster terms
        matched_clusters = clusters_df[
            clusters_df["TopTerms"].str.lower().str.contains(q, na=False)
        ]

        st.markdown("---")
        st.subheader(f"Results for: `{q}`")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Matching K-Means Clusters:**")
            if matched_clusters.empty:
                st.info("No cluster directly contains this term. Showing headlines below.")
            else:
                for _, r in matched_clusters.iterrows():
                    st.success(f"**{r['Cluster']}** ({r['Size']:,} docs)\n\n_{r['TopTerms']}_")

        with col_r:
            # 2. Match against topics_over_time keywords via spike_events
            spikes_df = load_spikes()
            if not spikes_df.empty and "keywords" in spikes_df.columns:
                matched_spikes = spikes_df[
                    spikes_df["keywords"].str.lower().str.contains(q, na=False)
                ]
                st.markdown(f"**Spikes containing `{q}` (Phase 4):**")
                if matched_spikes.empty:
                    st.info("No spikes matched this keyword.")
                else:
                    st.dataframe(
                        matched_spikes[["spike_date", "velocity_z", "keywords"]].head(10),
                        use_container_width=True,
                    )

        # 3. Topic proportion chart for matched topic IDs
        if not topics_df.empty and not spikes_df.empty:
            matched_topic_ids = (
                spikes_df[spikes_df["keywords"].str.lower().str.contains(q, na=False)]["topic_id"]
                .unique()[:5]
            )
            if len(matched_topic_ids) > 0:
                sub = topics_df[topics_df["topic_id"].isin(matched_topic_ids)]
                fig = px.line(
                    sub, x="bin_mid", y="proportion", color="topic_id",
                    title=f"Proportion over time for topics matching '{q}'",
                    template="plotly_dark",
                    labels={"bin_mid": "Date", "proportion": "Proportion", "topic_id": "Topic"},
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

        # 4. Matching headlines from corpus
        st.markdown(f"**Headlines containing `{q}` (from corpus):**")
        matching_headlines = headlines_df[
            headlines_df["headline_text"].str.lower().str.contains(q, na=False)
        ].copy()

        if matching_headlines.empty:
            st.info("No headlines matched in the loaded corpus.")
        else:
            st.metric("Total matching headlines", len(matching_headlines))

            # frequency over time
            matching_headlines["year_month"] = matching_headlines["publish_date"].dt.to_period("M").astype(str)
            freq_over_time = matching_headlines.groupby("year_month").size().reset_index(name="count")
            fig2 = px.bar(
                freq_over_time, x="year_month", y="count",
                title=f"Frequency of '{q}' in headlines over time",
                template="plotly_dark",
                labels={"year_month": "Month", "count": "Count"},
                color="count", color_continuous_scale="Reds",
            )
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(
                matching_headlines[["publish_date", "headline_text"]]
                .rename(columns={"publish_date": "Date", "headline_text": "Headline"})
                .head(100),
                use_container_width=True, height=260,
            )


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — NER Entity Explorer
# ══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Named Entity Recognition — Who & Where Drives Each Rupture?")
    st.markdown(
        "Run `python src/ner_entity_tracker.py` to generate entity frequency data. "
        "This surfaces which **people**, **places**, and **organisations** appear at each narrative spike."
    )

    if not NER_WEEKLY.exists():
        st.warning("NER output not found. Run: `python src/ner_entity_tracker.py`")
        st.code("python src/ner_entity_tracker.py", language="bash")
    else:
        ner_df = pd.read_csv(NER_WEEKLY, parse_dates=["week"])

        entity_types = ner_df["entity_type"].unique().tolist() if "entity_type" in ner_df.columns else []
        col1, col2 = st.columns([1, 2])

        with col1:
            etype = st.selectbox("Entity type", entity_types or ["PERSON", "GPE", "ORG"])
            top_n = st.slider("Top N entities", 5, 30, 10)

        with col2:
            sub = ner_df[ner_df["entity_type"] == etype] if "entity_type" in ner_df.columns else ner_df
            top_entities = sub.groupby("entity")["count"].sum().nlargest(top_n).index.tolist()
            sel_entity = st.selectbox("Select entity to trace", top_entities)

        entity_sub = sub[sub["entity"] == sel_entity].sort_values("week")
        fig = px.line(
            entity_sub, x="week", y="count",
            title=f"Weekly frequency of '{sel_entity}' ({etype})",
            template="plotly_dark",
        )
        # overlay velocity spikes
        vel_df = load_velocity()
        threshold = vel_df["semantic_velocity"].mean() + vel_df["semantic_velocity"].std()
        for _, r in vel_df[vel_df["semantic_velocity"] >= threshold].iterrows():
            fig.add_vline(x=str(r["week_start"]), line_color="#FF4B4B",
                          line_dash="dot", line_width=1)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Red dotted lines = semantic velocity rupture weeks")

        if NER_ENTITIES.exists():
            st.divider()
            st.subheader("All Entity Data")
            full_ner = pd.read_csv(NER_ENTITIES)
            st.dataframe(full_ner.head(500), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — LSTM Anomaly Detection
# ══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("LSTM Autoencoder — Deep Anomaly Detection")
    st.markdown(
        "Trains on pre-2020 topic velocity patterns. High **reconstruction error** = "
        "the model cannot explain the current pattern from past behaviour = anomaly."
    )

    if not LSTM_CSV.exists():
        st.warning("LSTM output not found. Run: `python src/lstm_anomaly_detector.py`")
        st.code("python src/lstm_anomaly_detector.py", language="bash")
    else:
        lstm_df = pd.read_csv(LSTM_CSV, parse_dates=["date"])

        threshold_lstm = lstm_df["reconstruction_error"].mean() + 2 * lstm_df["reconstruction_error"].std()
        anomalies = lstm_df[lstm_df["reconstruction_error"] >= threshold_lstm]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total time bins", len(lstm_df))
        col2.metric("Anomalies detected", len(anomalies))
        col3.metric("Threshold", f"{threshold_lstm:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lstm_df["date"], y=lstm_df["reconstruction_error"],
            mode="lines", name="Reconstruction Error",
            line=dict(color="#4C9BE8", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=anomalies["date"], y=anomalies["reconstruction_error"],
            mode="markers", name="Anomaly",
            marker=dict(color="#FF4B4B", size=9, symbol="x"),
        ))
        fig.add_hline(y=threshold_lstm, line_dash="dot", line_color="orange",
                      annotation_text="Anomaly threshold (μ + 2σ)")
        fig.update_layout(
            height=400, template="plotly_dark",
            title="LSTM Reconstruction Error — Topic Velocity Time Series",
            xaxis_title="Date", yaxis_title="MSE Reconstruction Error",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detected Anomalies")
        st.dataframe(anomalies.sort_values("reconstruction_error", ascending=False),
                     use_container_width=True)

        # compare with Z-score
        if SPIKE_CSV.exists():
            st.divider()
            st.subheader("LSTM vs Z-Score Comparison")
            spike_df = load_spikes()
            zscore_dates = set(spike_df["spike_date"].dt.strftime("%Y-%m").unique())
            lstm_dates   = set(anomalies["date"].dt.strftime("%Y-%m").unique())
            only_lstm  = lstm_dates - zscore_dates
            only_z     = zscore_dates - lstm_dates
            both       = lstm_dates & zscore_dates

            c1, c2, c3 = st.columns(3)
            c1.metric("Detected by both", len(both))
            c2.metric("LSTM only (missed by Z-score)", len(only_lstm))
            c3.metric("Z-score only (missed by LSTM)", len(only_z))
            st.caption("Month-level comparison (LSTM runs on topic bins, Z-score on individual topic series)")


# ══════════════════════════════════════════════════════════════════════════
# TAB 7 — Granger Causality Graph
# ══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("Granger Causality — Topic Influence Network")
    st.markdown(
        "Shows which topics **statistically predict** future movements in other topics. "
        "A directed edge A → B means: past values of A help predict B beyond B's own history."
    )

    if not GRANGER_HTML.exists():
        st.warning("Causality graph not found. Run: `python src/granger_causality.py`")
        st.code("python src/granger_causality.py", language="bash")
    else:
        with open(GRANGER_HTML, "r") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=650, scrolling=True)

        if GRANGER_CSV.exists():
            st.divider()
            st.subheader("Significant Causal Edges (p < 0.05)")
            edges_df = pd.read_csv(GRANGER_CSV)
            fig = px.scatter(
                edges_df, x="f_stat", y="p_value",
                hover_data=["cause", "effect", "lag"],
                color="lag", size="f_stat",
                title="Granger Edges — F-statistic vs p-value",
                template="plotly_dark",
                labels={"f_stat": "F-Statistic", "p_value": "p-value"},
            )
            fig.add_hline(y=0.05, line_dash="dot", line_color="red",
                          annotation_text="p=0.05 significance threshold")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                edges_df.sort_values("f_stat", ascending=False),
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════
# TAB 8 — Multilingual SBERT
# ══════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("Multilingual Semantic Expansion")
    st.markdown(
        "Uses `paraphrase-multilingual-mpnet-base-v2` to project GDELT themes and "
        "ABC News headlines into the **same language-agnostic 768-dim space**. "
        "Detects cross-lingual semantic overlap and early signals from non-English sources."
    )

    if not MULTI_CSV.exists():
        st.warning("Multilingual output not found. Run: `python src/multilingual_sbert.py`")
        st.code("python src/multilingual_sbert.py", language="bash")
    else:
        multi_df = pd.read_csv(MULTI_CSV)

        if "source_lang" in multi_df.columns:
            lang_counts = multi_df["source_lang"].value_counts().reset_index()
            lang_counts.columns = ["Language", "Count"]
            fig = px.pie(lang_counts, values="Count", names="Language",
                         title="GDELT Sources by Language",
                         template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        if "cross_sim" in multi_df.columns:
            fig2 = px.histogram(
                multi_df, x="cross_sim", nbins=40,
                title="Cross-lingual Semantic Similarity (GDELT themes vs ABC News)",
                template="plotly_dark", color_discrete_sequence=["#9B59B6"],
                labels={"cross_sim": "Cosine Similarity"},
            )
            st.plotly_chart(fig2, use_container_width=True)

        if "similarity_rank" in multi_df.columns:
            st.subheader("Top GDELT Events Semantically Closest to ABC News Corpus")
            top_matches = multi_df.nlargest(20, "cross_sim")
            show_cols = [c for c in ["SOURCECOMMONNAME", "source_lang", "cross_sim", "theme_string", "DOCUMENTIDENTIFIER"] if c in top_matches.columns]
            st.dataframe(top_matches[show_cols], use_container_width=True)

        st.divider()
        st.subheader("Raw Multilingual Signals")
        st.dataframe(multi_df.head(200), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 9 — Topic Watchdog
# ══════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.header("🎯 Topic Watchdog — Real-Time Keyword Intelligence")
    st.markdown(
        "Type any keyword, brand, or topic. The system scans the live GDELT feed, "
        "scores every matching article with **SBERT impact scoring**, and tells you "
        "whether coverage is **Normal**, **Trending**, or **Breaking**."
    )

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        wdog_keyword = st.text_input(
            "Enter keyword to monitor:",
            placeholder="e.g.  Tata Motors · Modi · AI regulation · Ukraine",
            label_visibility="collapsed",
        )
    with col_btn:
        wdog_go = st.button("Watch 🔍", type="primary", use_container_width=True)

    # refresh GDELT inline
    with st.expander("⚙️ Fetch fresh GDELT data before watching"):
        w_col1, w_col2 = st.columns([2, 1])
        with w_col1:
            wdog_window = st.selectbox(
                "Time window",
                ["15 min (1)", "1 hour (4)", "4 hours (16)", "24 hours (96)"],
                key="wdog_window",
            )
        with w_col2:
            if st.button("🔄 Refresh GDELT", use_container_width=True, key="wdog_refresh"):
                n = {"15 min (1)": 1, "1 hour (4)": 4, "4 hours (16)": 16, "24 hours (96)": 96}[wdog_window]
                with st.spinner(f"Fetching {n} GDELT snapshot(s)…"):
                    res = subprocess.run(
                        [sys.executable, "-c",
                         f"import sys; sys.path.insert(0,'.'); "
                         f"from src.gdelt_fetcher import fetch_last_n_gkg; "
                         f"df = fetch_last_n_gkg(n={n}, save_to='data/gdelt_processed.csv'); "
                         f"print(f'Fetched {{len(df):,}} records')"],
                        capture_output=True, text=True, cwd=str(ROOT),
                    )
                if res.returncode == 0:
                    st.success(res.stdout.strip() or "GDELT refreshed!")
                    st.cache_data.clear()
                else:
                    st.error(res.stderr[-300:])

    # ── "What's hot right now" hint ───────────────────────────────────────────
    gdelt_hint = load_gdelt()
    if not gdelt_hint.empty:
        import ast as _ast_hint
        from collections import Counter as _C
        _all_t = []
        for v in gdelt_hint["theme_list"].dropna():
            try:
                _all_t.extend(_ast_hint.literal_eval(v) if isinstance(v, str) else v)
            except Exception:
                pass
        _top = [t.lower().split("_")[0] for t, _ in _C(_all_t).most_common(40)]
        _unique_hints = list(dict.fromkeys(_top))[:12]  # dedupe, keep order
        st.caption(
            f"💡 **{len(gdelt_hint):,} articles** in current snapshot · "
            f"Try: " + "  ·  ".join(f"`{h}`" for h in _unique_hints)
        )

    st.divider()

    if wdog_keyword.strip() and wdog_go:
        gdelt_df = load_gdelt()
        if gdelt_df.empty:
            st.warning("No GDELT data loaded. Use the panel above to fetch fresh data first.")
        else:
            with st.spinner(f"Scanning GDELT for **{wdog_keyword}** and scoring with SBERT…"):
                matched_df, z_score, verdict = watchdog_search(wdog_keyword, gdelt_df)

            # ── Verdict banner ────────────────────────────────────────────────
            if verdict == "BREAKING":
                st.error(f"🔴 **BREAKING** — `{wdog_keyword}` is spiking at **{z_score:.1f}σ** above normal coverage")
            elif verdict == "TRENDING":
                st.warning(f"🟡 **TRENDING** — `{wdog_keyword}` is elevated at **{z_score:.1f}σ** above normal")
            elif verdict == "NORMAL":
                st.success(f"🟢 **NORMAL** — `{wdog_keyword}` is at routine coverage levels (z = {z_score:.1f})")
            else:
                st.info(f"No articles found mentioning **{wdog_keyword}** in the current GDELT snapshot.")

            if not matched_df.empty:
                # ── Top metrics ───────────────────────────────────────────────
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Articles Found", len(matched_df))
                m2.metric("Avg Tone", f"{matched_df['tone_value'].mean():.2f}")
                m3.metric("Avg Impact Score", f"{matched_df['impact_score'].mean():.3f}")
                m4.metric(
                    "Tone",
                    "😟 Negative" if matched_df["tone_value"].mean() < -1
                    else ("😊 Positive" if matched_df["tone_value"].mean() > 1 else "😐 Neutral")
                )

                st.divider()

                col_left, col_right = st.columns(2)

                # ── Article volume over time ───────────────────────────────────
                with col_left:
                    st.subheader("Coverage volume over time")
                    if "DATE" in matched_df.columns:
                        try:
                            matched_df["_dt"] = pd.to_datetime(
                                matched_df["DATE"].astype(str), format="%Y%m%d%H%M%S", errors="coerce"
                            )
                            time_series = (
                                matched_df.dropna(subset=["_dt"])
                                .set_index("_dt")
                                .resample("15min")
                                .size()
                                .reset_index(name="count")
                            )
                            if not time_series.empty:
                                fig_time = px.bar(
                                    time_series, x="_dt", y="count",
                                    labels={"_dt": "Time", "count": "Articles"},
                                    title=f'Articles per 15-min window — "{wdog_keyword}"',
                                    template="plotly_dark",
                                    color="count",
                                    color_continuous_scale="Reds",
                                )
                                fig_time.update_layout(height=320, showlegend=False,
                                                       margin=dict(l=20, r=20, t=50, b=30))
                                st.plotly_chart(fig_time, use_container_width=True)
                        except Exception:
                            st.info("Could not parse timestamps for time-series chart.")
                    else:
                        st.info("DATE column not available for time chart.")

                # ── Tone distribution ─────────────────────────────────────────
                with col_right:
                    st.subheader("Tone distribution")
                    fig_tone = px.histogram(
                        matched_df, x="tone_value", nbins=30,
                        labels={"tone_value": "Tone Score"},
                        title=f'Sentiment spread — "{wdog_keyword}"',
                        template="plotly_dark",
                        color_discrete_sequence=["#9B59B6"],
                    )
                    fig_tone.add_vline(x=0, line_dash="dot", line_color="white",
                                       annotation_text="Neutral")
                    fig_tone.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=30))
                    st.plotly_chart(fig_tone, use_container_width=True)

                st.divider()

                # ── Top impactful articles ────────────────────────────────────
                st.subheader(f"Top articles by SBERT Impact Score")
                top_articles = matched_df.nlargest(10, "impact_score")
                for i, (_, row) in enumerate(top_articles.iterrows(), 1):
                    url  = str(row.get("DOCUMENTIDENTIFIER", "")).strip()
                    src  = str(row.get("SOURCECOMMONNAME", "Unknown source")).strip()
                    tone = row.get("tone_value", 0.0)
                    si   = row.get("impact_score", 0.0)
                    tone_icon = "😟" if tone < -1 else ("😊" if tone > 1 else "😐")
                    with st.container():
                        c1, c2, c3 = st.columns([5, 1, 1])
                        with c1:
                            if url.startswith("http"):
                                st.markdown(f"**{i}.** [{src}]({url})")
                            else:
                                st.markdown(f"**{i}.** {src}")
                        with c2:
                            st.markdown(f"Tone: {tone_icon} `{tone:.2f}`")
                        with c3:
                            st.markdown(f"S_I: `{si:.3f}`")

                st.divider()

                # ── Top themes in matched articles ─────────────────────────────
                st.subheader("Dominant themes in matched articles")
                import ast as _ast2
                from collections import Counter as _Counter

                all_t = []
                for v in matched_df["theme_list"]:
                    try:
                        themes = _ast2.literal_eval(v) if isinstance(v, str) else v
                        all_t.extend(themes)
                    except Exception:
                        pass

                if all_t:
                    theme_freq = (
                        pd.DataFrame(_Counter(all_t).most_common(15), columns=["Theme", "Count"])
                    )
                    fig_themes = px.bar(
                        theme_freq, x="Count", y="Theme", orientation="h",
                        title="Top GDELT themes in matched articles",
                        template="plotly_dark",
                        color="Count", color_continuous_scale="Blues",
                    )
                    fig_themes.update_layout(
                        height=420, margin=dict(l=20, r=20, t=50, b=30),
                        yaxis={"autorange": "reversed"},
                    )
                    st.plotly_chart(fig_themes, use_container_width=True)

    elif not wdog_keyword.strip():
        st.info("Enter a keyword above and click **Watch** to start monitoring.")
