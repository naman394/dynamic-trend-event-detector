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
                    [sys.executable, str(SRC / "gdelt_fetcher.py"), f"--windows={n_win}"],
                    capture_output=True, text=True, cwd=str(ROOT),
                )
            st.info(result.stdout[-300:] or "Done.")

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
