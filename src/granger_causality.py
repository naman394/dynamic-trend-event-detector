"""
Granger Causality Network — Topic Influence Graph.

Tests whether past values of topic A statistically predict topic B (beyond B's own history).
Significant pairs (p < 0.05) become directed edges in a network.

Output:
  reports/granger_edges.csv          — significant causal edges with F-stat, p-value, lag
  reports/granger_causality.html     — interactive Plotly network graph

Usage:
    python src/granger_causality.py
    python src/granger_causality.py --top-topics 15 --max-lag 4 --p-threshold 0.05

Requirements:
    pip install statsmodels networkx
"""

import argparse
import sys
import warnings
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
TM      = REPORTS / "topic_modeling" / "11_lda_sbert"

TOPICS_CSV   = TM / "topics_over_time.csv"
OUT_EDGES    = REPORTS / "granger_edges.csv"
OUT_HTML     = REPORTS / "granger_causality.html"


# ── data loading ─────────────────────────────────────────────────────────────
def load_pivot(top_n=20):
    """Load topics_over_time.csv and return a (time × topic) pivot DataFrame."""
    if not TOPICS_CSV.exists():
        print(f"ERROR: {TOPICS_CSV} not found. Run notebook 11 first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(TOPICS_CSV)
    df["bin_mid"] = pd.to_datetime(df["bin_mid"], errors="coerce")
    df = df.dropna(subset=["bin_mid"])
    df = df[df["topic_id"] >= 0]

    # top N topics by total proportion
    top_ids = (
        df.groupby("topic_id")["proportion"].sum()
        .nlargest(top_n).index.tolist()
    )
    df = df[df["topic_id"].isin(top_ids)]

    pivot = (
        df.pivot_table(index="bin_mid", columns="topic_id",
                       values="proportion", aggfunc="mean")
        .sort_index()
        .fillna(method="ffill")
        .fillna(0.0)
    )
    print(f"Pivot shape: {pivot.shape}  ({pivot.shape[0]} time bins × {pivot.shape[1]} topics)")
    return pivot


# ── stationarity check + differencing ────────────────────────────────────────
def make_stationary(series, alpha=0.05):
    """
    Return differenced series if ADF test fails stationarity.
    One round of differencing is usually sufficient for topic proportions.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        pval = adfuller(series.dropna(), autolag="AIC")[1]
        if pval > alpha:
            return series.diff().dropna()
    except Exception:
        pass
    return series


# ── Granger test ─────────────────────────────────────────────────────────────
def run_granger_tests(pivot, max_lag=4, p_threshold=0.05):
    """
    Run Granger causality tests for all ordered (cause, effect) pairs.
    Returns DataFrame of significant edges.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        print("ERROR: statsmodels not installed.  pip install statsmodels", file=sys.stderr)
        sys.exit(1)

    topic_ids = pivot.columns.tolist()
    n         = len(topic_ids)
    total     = n * (n - 1)
    edges     = []

    print(f"Testing {total} directed pairs (max_lag={max_lag})…")

    for idx, (cause, effect) in enumerate(permutations(topic_ids, 2)):
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{total}", flush=True)

        cause_s  = make_stationary(pivot[cause])
        effect_s = make_stationary(pivot[effect])

        # align on shared index
        combined = pd.DataFrame({"effect": effect_s, "cause": cause_s}).dropna()
        if len(combined) < max_lag * 4:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = grangercausalitytests(
                    combined[["effect", "cause"]],
                    maxlag=max_lag,
                    verbose=False,
                )
        except Exception:
            continue

        # find best lag (lowest p-value for F-test)
        for lag in range(1, max_lag + 1):
            f_stat = results[lag][0]["ssr_ftest"][0]
            p_val  = results[lag][0]["ssr_ftest"][1]
            if p_val < p_threshold:
                edges.append({
                    "cause":    cause,
                    "effect":   effect,
                    "lag":      lag,
                    "f_stat":   round(f_stat, 4),
                    "p_value":  round(p_val, 6),
                })

    edges_df = pd.DataFrame(edges)
    if not edges_df.empty:
        # keep only best lag per (cause, effect) pair
        edges_df = (
            edges_df.sort_values("p_value")
            .groupby(["cause", "effect"], as_index=False)
            .first()
        )
    print(f"Significant edges found: {len(edges_df)}")
    return edges_df


# ── topic labels ─────────────────────────────────────────────────────────────
def build_label_map(pivot):
    """Try to load spike_events keywords for topic labels; fallback to Topic-{id}."""
    spike_csv = REPORTS / "topic_modeling" / "12_spikes_anchors" / "spike_events.csv"
    label_map = {tid: f"Topic-{tid}" for tid in pivot.columns}

    if spike_csv.exists():
        sp = pd.read_csv(spike_csv)
        for tid in pivot.columns:
            sub = sp[sp["topic_id"] == tid]
            if not sub.empty:
                kws = sub.iloc[sub["velocity_z"].values.argmax()]["keywords"]
                label_map[tid] = f"{tid}: {str(kws)[:25]}"
    return label_map


# ── Plotly network graph ──────────────────────────────────────────────────────
def build_plotly_graph(edges_df, pivot, label_map):
    try:
        import networkx as nx
        import plotly.graph_objects as go
    except ImportError:
        print("ERROR: networkx/plotly not installed.  pip install networkx plotly", file=sys.stderr)
        sys.exit(1)

    G = nx.DiGraph()
    topic_ids = pivot.columns.tolist()
    G.add_nodes_from(topic_ids)

    for _, row in edges_df.iterrows():
        G.add_edge(
            row["cause"], row["effect"],
            weight=row["f_stat"],
            lag=row["lag"],
            p_value=row["p_value"],
        )

    # layout
    pos = nx.spring_layout(G, seed=42, k=2.5)

    # node sizes proportional to out-degree (how many things they cause)
    out_degrees = dict(G.out_degree())
    node_sizes  = [10 + out_degrees.get(n, 0) * 6 for n in G.nodes()]

    # edges
    edge_x, edge_y, edge_hover = [], [], []
    annotations = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_hover.append(
            f"{label_map.get(u, u)} → {label_map.get(v, v)}<br>"
            f"Lag: {data['lag']}  F={data['weight']:.2f}  p={data['p_value']:.4f}"
        )
        # arrowhead annotation
        annotations.append(dict(
            ax=x0, ay=y0, axref="x", ayref="y",
            x=x1,  y=y1,  xref="x",  yref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2,
            arrowwidth=max(0.5, data["weight"] / 10),
            arrowcolor="#FF6B6B",
        ))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#555"),
        hoverinfo="none",
    )

    # nodes
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = [label_map.get(n, str(n)) for n in G.nodes()]
    node_hover = [
        f"<b>{label_map.get(n, n)}</b><br>"
        f"Out-degree (causes): {out_degrees.get(n, 0)}<br>"
        f"In-degree (caused by): {G.in_degree(n)}"
        for n in G.nodes()
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=[out_degrees.get(n, 0) for n in G.nodes()],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Out-degree<br>(causal influence)"),
            line=dict(width=1, color="white"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Granger Causality Network — Topic Influence Graph<br>"
                     "<sub>Edge A→B: past values of A predict B (p<0.05). "
                     "Node size = causal influence (out-degree).</sub>",
                font=dict(size=16),
            ),
            showlegend=False,
            hovermode="closest",
            annotations=annotations,
            template="plotly_dark",
            height=700,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=80, b=20),
        ),
    )

    return fig, G


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Granger causality topic influence graph")
    parser.add_argument("--top-topics",   type=int,   default=20,   help="top N topics to include")
    parser.add_argument("--max-lag",      type=int,   default=4,    help="max lag for Granger test")
    parser.add_argument("--p-threshold",  type=float, default=0.05, help="significance threshold")
    args = parser.parse_args()

    REPORTS.mkdir(exist_ok=True)

    pivot     = load_pivot(top_n=args.top_topics)
    label_map = build_label_map(pivot)

    edges_df = run_granger_tests(pivot, max_lag=args.max_lag, p_threshold=args.p_threshold)

    if edges_df.empty:
        print("No significant causal relationships found at the given threshold.")
        print("Try --max-lag 6 or --p-threshold 0.10")
        return

    # add readable labels
    edges_df["cause_label"]  = edges_df["cause"].map(label_map)
    edges_df["effect_label"] = edges_df["effect"].map(label_map)

    edges_df.to_csv(OUT_EDGES, index=False)
    print(f"Edges saved: {OUT_EDGES}")

    # ── report ──
    print("\n── Top 15 Causal Edges (by F-statistic) ──────────────────────")
    top = edges_df.nlargest(15, "f_stat")
    for _, r in top.iterrows():
        print(
            f"  {label_map.get(r['cause'], r['cause']):<30} → "
            f"{label_map.get(r['effect'], r['effect']):<30}  "
            f"lag={r['lag']}  F={r['f_stat']:.2f}  p={r['p_value']:.4f}"
        )

    # ── most causally influential topics ──
    top_causes = edges_df.groupby("cause")["f_stat"].sum().nlargest(5)
    print("\n── Most Causally Influential Topics ──────────────────────────")
    for tid, fsum in top_causes.items():
        print(f"  {label_map.get(tid, tid):<35}  total-F={fsum:.2f}")

    # ── Plotly graph ──
    fig, G = build_plotly_graph(edges_df, pivot, label_map)
    fig.write_html(str(OUT_HTML))
    print(f"\nGraph saved: {OUT_HTML}")

    # quick NetworkX summary
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} significant edges")

    import networkx as nx
    if G.number_of_nodes() > 0:
        strongly = list(nx.strongly_connected_components(G))
        largest  = max(strongly, key=len)
        print(f"Largest strongly-connected component: {len(largest)} topics")

    print("\nDone.")


if __name__ == "__main__":
    main()
