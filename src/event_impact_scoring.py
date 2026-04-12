"""
Event Impact Scoring — SBERT-based
=============================================
Purpose in the pipeline
-----------------------
This module sits AFTER the Deep Learning model (SBERT K-Means) and uses its
output to focus GDELT verification and scoring on periods that matter most.

Full pipeline position:
    Baseline TF-IDF
        → Advanced ML LDA
            → Deep Learning SBERT K-Means  ← finds clusters + rupture weeks
                → GDELT Processor
                    → GDELT Analysis
                        → THIS MODULE  ← verifies + scores GDELT events
                            → Visualise Results

What this module does
---------------------
1. GDELT Verification:
   Load the semantic velocity results from the DL model. Identify the top
   "Narrative Rupture" weeks (high V_s spike). Pull ABC News headlines from
   those windows to understand what caused the shift.

2. Event Impact Scoring (SBERT-based):
   For every GDELT record, compute:
       S_I = Semantic_Uniqueness × Tonal_Intensity
   where Semantic_Uniqueness is now the SBERT cosine distance from the
   GDELT snapshot's global centroid (richer than TF-IDF Phase 1).

   High S_I = unique theme + high emotional intensity = priority signal.

Why SBERT over TF-IDF for scoring?
-----------------------------------
- TF-IDF: "TAX_DISEASE" and "HEALTH_PANDEMIC" are two different tokens
- SBERT: they land in the same region of embedding space → properly scored
  as related (not independent) events.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import ast
import os

sns.set_theme(style="whitegrid")
OUTPUT_DIR = 'reports'
DL_DIR     = 'reports/deep_learning'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Load DL model rupture output ─────────────────────────────────────────
def load_rupture_windows(top_n: int = 3) -> list[dict]:
    """
    Read semantic velocity CSV from DL model.
    Return the top-N rupture weeks sorted by velocity (descending).
    """
    vel_path = f'{DL_DIR}/semantic_velocity.csv'
    if not os.path.exists(vel_path):
        print(f"[WARN] {vel_path} not found. Run deep_learning.py first.")
        return []
    vel_df = pd.read_csv(vel_path).sort_values('semantic_velocity', ascending=False)
    ruptures = vel_df.head(top_n).to_dict('records')
    print(f"\n── Top {top_n} Narrative Rupture Windows (from DL model) ──")
    for r in ruptures:
        print(f"  {r['week']}  →  V_s = {r['semantic_velocity']:.4f}")
    return ruptures


# ── 2. GDELT cross-reference at rupture windows ──────────────────────────────
def verify_gdelt_at_ruptures(headlines_df: pd.DataFrame,
                              ruptures: list[dict]) -> pd.DataFrame:
    """
    For each rupture window:
      - pull ABC News headlines from ±3 days of that week
      - show the dominant topic shift
    Returns a summary DataFrame.
    """
    records = []
    print("\n── GDELT Verification: ABC Headlines Around Rupture Weeks ──")
    for r in ruptures:
        # Parse week bounds from string like "2003-05-26/2003-06-01"
        try:
            parts  = r['week'].split('/')
            w_start = pd.to_datetime(parts[0]) - pd.Timedelta(days=3)
            w_end   = pd.to_datetime(parts[1]) + pd.Timedelta(days=3)
        except Exception:
            continue

        window = headlines_df[
            (headlines_df['publish_date'] >= w_start) &
            (headlines_df['publish_date'] <= w_end)
        ]
        if window.empty:
            continue

        # Get 5 representative headlines
        sample = window['headline_text'].sample(min(5, len(window)),
                                                random_state=42).tolist()
        records.append({
            'rupture_week':     r['week'],
            'velocity':         round(r['semantic_velocity'], 4),
            'headline_count':   len(window),
            'sample_headlines': ' | '.join(sample),
        })
        print(f"\n  Week: {r['week']}  (V_s={r['semantic_velocity']:.4f})")
        print(f"  Headlines in window: {len(window)}")
        for h in sample:
            print(f"    • {h}")

    return pd.DataFrame(records)


# ── 3. GDELT Event Impact Scoring (SBERT) ────────────────────────────────────
def score_gdelt_events_sbert(gdelt_df: pd.DataFrame,
                              model: SentenceTransformer) -> pd.DataFrame:
    """
    Compute S_I = Semantic_Uniqueness × |Tone| for every GDELT record.
    Semantic_Uniqueness = SBERT cosine distance from global snapshot centroid.
    """
    print("\n── Computing SBERT-based Impact Scores for GDELT records ──")

    gdelt_df = gdelt_df.copy()
    gdelt_df['theme_string'] = gdelt_df['theme_list'].apply(
        lambda l: ' '.join(l) if isinstance(l, list) else ''
    )
    gdelt_df = gdelt_df[gdelt_df['theme_string'].str.strip() != ''].copy()
    print(f"  GDELT records with themes: {len(gdelt_df):,}")

    # SBERT encode all theme strings
    print("  Encoding GDELT themes with SBERT …")
    embeddings = model.encode(
        gdelt_df['theme_string'].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Global snapshot centroid
    global_centroid = embeddings.mean(axis=0, keepdims=True)

    # Semantic uniqueness: distance from centroid
    sims = cosine_similarity(embeddings, global_centroid).flatten()
    gdelt_df['semantic_uniqueness'] = 1.0 - sims

    # Tonal intensity
    gdelt_df['tone_abs']     = gdelt_df['tone_value'].abs()
    gdelt_df['impact_score'] = gdelt_df['semantic_uniqueness'] * gdelt_df['tone_abs']

    print(f"  Impact score range: {gdelt_df['impact_score'].min():.4f} – "
          f"{gdelt_df['impact_score'].max():.4f}")
    return gdelt_df


# ── 4. Main pipeline ─────────────────────────────────────────────────────────
def run():
    # ── Load GDELT data ──────────────────────────────────────────────────────
    gdelt_path = 'data/gdelt_processed.csv'
    if not os.path.exists(gdelt_path):
        print(f"[SKIP] {gdelt_path} not found. Run gdelt_processor.py first.")
        return

    gdelt_df = pd.read_csv(gdelt_path)
    gdelt_df['theme_list'] = gdelt_df['theme_list'].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else []
    )

    # ── Load headlines for rupture verification ──────────────────────────────
    hl_df = pd.read_csv('data/news_headlines.csv')
    hl_df['publish_date'] = pd.to_datetime(hl_df['publish_date'], format='%Y%m%d')

    # ── Step 1: Identify rupture weeks from DL model ─────────────────────────
    ruptures = load_rupture_windows(top_n=3)

    # ── Step 2: Verify GDELT themes at rupture weeks ─────────────────────────
    rupture_summary = verify_gdelt_at_ruptures(hl_df, ruptures)
    if not rupture_summary.empty:
        rupture_summary.to_csv(f'{OUTPUT_DIR}/rupture_verification.csv', index=False)
        print(f"\n  Rupture summary saved → {OUTPUT_DIR}/rupture_verification.csv")

    # ── Step 3: SBERT impact scoring of all GDELT events ─────────────────────
    print("\nLoading SBERT model …")
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    gdelt_scored = score_gdelt_events_sbert(gdelt_df, sbert)

    # Top 5 most impactful events
    top5 = gdelt_scored.sort_values('impact_score', ascending=False).head(5)
    print("\n── Top 5 GDELT Events by Impact Score (SBERT) ──")
    for _, row in top5.iterrows():
        print(f"  Source  : {row.get('SOURCECOMMONNAME', 'N/A')}")
        print(f"  Tone    : {row['tone_value']:.2f}   Uniqueness: {row['semantic_uniqueness']:.4f}")
        print(f"  S_I     : {row['impact_score']:.4f}")
        print(f"  Themes  : {str(row['theme_list'])[:90]}…")
        print()

    # ── Save ─────────────────────────────────────────────────────────────────
    gdelt_scored.to_csv(f'{OUTPUT_DIR}/event_impact_scores.csv', index=False)

    # ── Visualise ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Event Impact Scoring — SBERT-based\n'
                 '(After Deep Learning SBERT K-Means + GDELT Verification)',
                 fontsize=12, fontweight='bold')

    sns.histplot(gdelt_scored['impact_score'], bins=35, kde=True,
                 color='#E74C3C', ax=axes[0])
    axes[0].set_title('Distribution of S_I Scores', fontweight='bold')
    axes[0].set_xlabel('S_I = SBERT Uniqueness × |Tone|')

    top_src = (gdelt_scored.groupby('SOURCECOMMONNAME')['impact_score']
               .mean().sort_values(ascending=False).head(12))
    sns.barplot(x=top_src.values, y=top_src.index,
                palette='viridis', ax=axes[1])
    axes[1].set_title('Average Impact Score by Source', fontweight='bold')
    axes[1].set_xlabel('Mean S_I')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/event_impact_analysis.png', dpi=150)
    plt.close()
    print(f"Chart saved → {OUTPUT_DIR}/event_impact_analysis.png")
    print("✅ Event Impact Scoring complete.")


if __name__ == "__main__":
    run()
