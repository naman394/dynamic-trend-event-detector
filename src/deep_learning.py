"""
Deep Learning Model: K-Means Clustering on SBERT Embeddings
=============================================================
Architecture (from project specification):
  1. SBERT (all-MiniLM-L6-v2) → dense 384-dim semantic embeddings
  2. UMAP (3-component) → dimensionality reduction for visualisation
  3. K-Means → discover Topic Clusters (Topic A, Topic B, ...)
  4. Temporal analysis → Growth Velocity per cluster over time
  5. Semantic Velocity → cosine distance between weekly embedding centroids
  6. GDELT Verification → cross-reference discovered topics with GDELT themes

Why K-Means on SBERT over plain LDA?
  - SBERT embeddings are contextual (transformer-based); LDA uses raw counts
  - K-Means in embedding space groups headlines by *semantic meaning*, not word overlap
  - UMAP 3D projection enables visual inspection of cluster separation
  - Semantic Velocity captures continuous narrative shift (not discrete topic changes)
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
import ast

sns.set_theme(style="whitegrid")
OUTPUT_DIR = 'reports/deep_learning'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Load & Pre-process Data ───────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv('data/abcnews-date-text 2.csv')
df['publish_date'] = pd.to_datetime(df['publish_date'], format='%Y%m%d')
df['headline_text'] = df['headline_text'].astype(str)

# Stratified sample by year so all 19 years (2003-2021) are equally represented.
# 50,000 total → ~2,631 per year — balances SBERT encoding time vs. coverage.
SAMPLE_SIZE = 50_000
df['_yr'] = df['publish_date'].dt.year
n_years  = df['_yr'].nunique()
per_year = SAMPLE_SIZE // n_years
df = pd.concat(
    [grp.sample(min(len(grp), per_year), random_state=42)
     for _, grp in df.groupby('_yr')]
).drop(columns='_yr').reset_index(drop=True)
print(f"Stratified sample: {len(df):,} headlines across {n_years} years "
      f"(≈{per_year:,}/year) for SBERT encoding")


# ── 2. SBERT Embeddings ──────────────────────────────────────────────────────
print("\nLoading SBERT model (all-MiniLM-L6-v2) …")
sbert = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding headlines → 384-dim embeddings …")
embeddings = sbert.encode(
    df['headline_text'].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,    # unit vectors → cosine sim = dot product
)
embeddings = np.array(embeddings)
print(f"Embedding matrix: {embeddings.shape}")


# ── 3. UMAP Dimensionality Reduction ────────────────────────────────────────
print("\nRunning UMAP (3 components) for visualisation …")
reducer_3d = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1,
                        metric='cosine', random_state=42)
umap_3d = reducer_3d.fit_transform(embeddings)     # shape (N, 3)

reducer_2d = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        metric='cosine', random_state=42)
umap_2d = reducer_2d.fit_transform(embeddings)     # shape (N, 2)

df['umap_x'] = umap_3d[:, 0]
df['umap_y'] = umap_3d[:, 1]
df['umap_z'] = umap_3d[:, 2]
df['umap2_x'] = umap_2d[:, 0]
df['umap2_y'] = umap_2d[:, 1]
print("UMAP projections complete.")


# ── 4. K-Means: Optimal K via Elbow + Silhouette ────────────────────────────
print("\nSearching for optimal number of clusters (K=2…10) …")
K_RANGE = range(2, 11)
inertias, silhouettes = [], []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    inertias.append(km.inertia_)
    sil = silhouette_score(embeddings, labels, sample_size=5000, random_state=42)
    silhouettes.append(sil)
    print(f"  K={k:2d}  Inertia={km.inertia_:,.0f}  Silhouette={sil:.4f}")

# Elbow + Silhouette plot
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(list(K_RANGE), inertias, 'o-', color='#3498DB', linewidth=2)
axes[0].set_title('Elbow Method — Inertia vs K', fontweight='bold')
axes[0].set_xlabel('Number of Clusters K')
axes[0].set_ylabel('Within-Cluster Sum of Squares')
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].plot(list(K_RANGE), silhouettes, 's-', color='#E74C3C', linewidth=2)
axes[1].set_title('Silhouette Score vs K\n(higher = better separated)', fontweight='bold')
axes[1].set_xlabel('Number of Clusters K')
axes[1].set_ylabel('Silhouette Score')
axes[1].grid(True, linestyle='--', alpha=0.5)

best_k = int(np.argmax(silhouettes)) + 2    # K_RANGE starts at 2
axes[1].axvline(best_k, color='#27AE60', linestyle='--',
                label=f'Best K={best_k}')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/optimal_k_selection.png', dpi=150)
plt.close()
print(f"\nBest K by silhouette: {best_k}")


# ── 5. Final K-Means with Best K ─────────────────────────────────────────────
print(f"\nFitting final K-Means (K={best_k}) …")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(embeddings)
df['cluster'] = cluster_labels

sil_final = silhouette_score(embeddings, cluster_labels,
                             sample_size=min(5000, len(df)), random_state=42)
ch_score  = calinski_harabasz_score(embeddings, cluster_labels)
print(f"Silhouette Score       : {sil_final:.4f}  (higher = better, max 1.0)")
print(f"Calinski-Harabasz Score: {ch_score:.2f}    (higher = better)")


# ── 6. Cluster Top Terms (TF-IDF) ────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

df['clean_text'] = df['headline_text'].apply(clean)
tfidf_v = TfidfVectorizer(stop_words='english', max_features=3000,
                           sublinear_tf=True, ngram_range=(1, 2))
tfidf_m = tfidf_v.fit_transform(df['clean_text'])
feat = tfidf_v.get_feature_names_out()

cluster_terms = {}
for c in range(best_k):
    mask = (df['cluster'] == c).to_numpy()   # numpy bool required for scipy sparse indexing
    cluster_vec = np.asarray(tfidf_m[mask].mean(axis=0)).flatten()
    top_idx = cluster_vec.argsort()[::-1][:8]
    cluster_terms[c] = [feat[i] for i in top_idx]
    print(f"  Cluster {c}: {', '.join(cluster_terms[c])}")


# ── 7. 3D Scatter Plot (SBERT+UMAP) ─────────────────────────────────────────
print("\nGenerating 3D scatter plot …")
PALETTE = sns.color_palette('tab10', best_k)

fig = plt.figure(figsize=(13, 9))
ax  = fig.add_subplot(111, projection='3d')

for c in range(best_k):
    mask = df['cluster'] == c
    ax.scatter(df.loc[mask, 'umap_x'],
               df.loc[mask, 'umap_y'],
               df.loc[mask, 'umap_z'],
               c=[PALETTE[c]], s=6, alpha=0.55, label=f'Topic {chr(65+c)}')

ax.set_title('SBERT K-Means: 3D UMAP Projection of Topic Clusters\n'
             f'(K={best_k}, Silhouette={sil_final:.3f})',
             fontsize=12, fontweight='bold')
ax.set_xlabel('UMAP-1')
ax.set_ylabel('UMAP-2')
ax.set_zlabel('UMAP-3')
ax.legend(loc='upper left', markerscale=3, fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/sbert_kmeans_3d_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {OUTPUT_DIR}/sbert_kmeans_3d_scatter.png")


# ── 8. 2D Scatter (cleaner for publication) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))
for c in range(best_k):
    mask = df['cluster'] == c
    ax.scatter(df.loc[mask, 'umap2_x'], df.loc[mask, 'umap2_y'],
               c=[PALETTE[c]], s=8, alpha=0.5,
               label=f'Topic {chr(65+c)}: {cluster_terms[c][0]}')
ax.set_title('SBERT K-Means: 2D UMAP Topic Cluster Map', fontsize=13, fontweight='bold')
ax.set_xlabel('UMAP Dimension 1')
ax.set_ylabel('UMAP Dimension 2')
ax.legend(markerscale=3, fontsize=9, title='Cluster (top term)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/sbert_kmeans_2d_scatter.png', dpi=150)
plt.close()


# ── 9. Growth Velocity: cluster size over time ──────────────────────────────
print("\nCalculating Growth Velocity …")
df['year_month'] = df['publish_date'].dt.to_period('M')
growth = (df.groupby(['year_month', 'cluster'])
           .size()
           .unstack(fill_value=0))

fig, ax = plt.subplots(figsize=(14, 5))
for c in range(best_k):
    if c in growth.columns:
        growth[c].plot(ax=ax, linewidth=1.8, marker='o', markersize=2,
                       color=PALETTE[c], label=f'Topic {chr(65+c)}')
ax.set_title('Growth Velocity: Monthly Topic Cluster Size Over Time',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Headlines')
ax.set_xlabel('Month')
ax.legend(title='Cluster', bbox_to_anchor=(1.01, 1))
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/cluster_growth_velocity.png', dpi=150)
plt.close()


# ── 10. Semantic Velocity: weekly embedding centroid shift ───────────────────
print("\nCalculating Semantic Velocity (weekly centroid shift) …")
df['week'] = df['publish_date'].dt.to_period('W')
weeks = sorted(df['week'].unique())

velocities = []
prev_centroid = None
for week in weeks:
    week_mask = df['week'] == week
    week_embeds = embeddings[week_mask.values]
    if len(week_embeds) == 0:
        continue
    centroid = week_embeds.mean(axis=0)
    if prev_centroid is not None:
        sim = cosine_similarity([prev_centroid], [centroid])[0][0]
        velocity = float(1.0 - sim)
        velocities.append({'week': str(week), 'semantic_velocity': velocity})
    prev_centroid = centroid

vel_df = pd.DataFrame(velocities)

if not vel_df.empty:
    max_vel = vel_df['semantic_velocity'].max()
    rupture_week = vel_df.loc[vel_df['semantic_velocity'].idxmax(), 'week']
    print(f"  Max Semantic Velocity: {max_vel:.4f} (Week: {rupture_week})")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(range(len(vel_df)), vel_df['semantic_velocity'],
                    alpha=0.25, color='#9B59B6')
    ax.plot(range(len(vel_df)), vel_df['semantic_velocity'],
            color='#8E44AD', linewidth=1.5, marker='o', markersize=3)
    ax.axhline(vel_df['semantic_velocity'].mean(), color='#E74C3C',
               linestyle='--', label=f"Mean velocity={vel_df['semantic_velocity'].mean():.3f}")
    top_idx = vel_df['semantic_velocity'].idxmax()
    ax.axvline(top_idx, color='#E67E22', linestyle='--',
               label=f"Narrative rupture: {rupture_week}")
    ax.set_xticks(range(0, len(vel_df), max(1, len(vel_df)//12)))
    ax.set_xticklabels(vel_df['week'].iloc[::max(1, len(vel_df)//12)],
                       rotation=45, ha='right', fontsize=8)
    ax.set_title('Semantic Velocity (SBERT Weekly Centroid Shift)\n'
                 'Spikes indicate Narrative Ruptures / Major Event Transitions',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Cosine Distance from Previous Week')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/semantic_velocity.png', dpi=150)
    plt.close()
    vel_df.to_csv(f'{OUTPUT_DIR}/semantic_velocity.csv', index=False)
    print(f"  Saved semantic velocity data → {OUTPUT_DIR}/semantic_velocity.csv")


# ── 11. GDELT Verification ───────────────────────────────────────────────────
gdelt_path = 'data/gdelt_processed.csv'
if os.path.exists(gdelt_path):
    print("\nPerforming GDELT cross-verification …")
    gdelt_df = pd.read_csv(gdelt_path)
    gdelt_df['theme_list'] = gdelt_df['theme_list'].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else []
    )

    from collections import Counter
    all_gdelt_themes = Counter(
        t for row in gdelt_df['theme_list'] for t in row
    )
    top_gdelt = set(t for t, _ in all_gdelt_themes.most_common(30))

    print("\n  Cluster ↔ GDELT Theme Cross-Reference:")
    for c, terms in cluster_terms.items():
        matches = [t for t in terms
                   if any(t.lower() in gt.lower() or gt.lower() in t.lower()
                          for gt in top_gdelt)]
        print(f"  Topic {chr(65+c)}: {', '.join(terms[:5])}")
        print(f"    GDELT matches : {matches if matches else '(indirect — see top GDELT themes)'}")
else:
    print("\n  [SKIP] GDELT data not found — run gdelt_processor.py first.")


# ── 12. Cluster Summary CSV ──────────────────────────────────────────────────
summary = pd.DataFrame([{
    'Cluster': c,
    'Label': f'Topic {chr(65+c)}',
    'Size': int((df['cluster'] == c).sum()),
    'TopTerms': ', '.join(cluster_terms[c]),
} for c in range(best_k)])
summary.to_csv(f'{OUTPUT_DIR}/cluster_summary.csv', index=False)

# Verification Report
with open(f'{OUTPUT_DIR}/verification_report.txt', 'w') as f:
    f.write("MODEL INTEGRITY REPORT — DEEP LEARNING (SBERT + K-Means)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Model              : SBERT (all-MiniLM-L6-v2)\n")
    f.write(f"Embeddings dim     : 384\n")
    f.write(f"Reduction          : UMAP (3D + 2D)\n")
    f.write(f"Clustering         : K-Means (K={best_k})\n")
    f.write(f"Documents          : {len(df):,}\n")
    f.write(f"Silhouette Score   : {sil_final:.4f}\n")
    f.write(f"Calinski-Harabasz  : {ch_score:.2f}\n")
    if not vel_df.empty:
        f.write(f"Max Sem Velocity   : {max_vel:.4f} (Week {rupture_week})\n")
    f.write("=" * 60 + "\n\nClusters:\n")
    for _, row in summary.iterrows():
        f.write(f"  Topic {row['Label'][-1]} ({row['Size']:,} docs): {row['TopTerms']}\n")

print(f"\nAll outputs saved to {OUTPUT_DIR}/")
print(f"✅ Deep Learning pipeline complete.")
print(f"\nKey results:")
print(f"  Optimal clusters    : K={best_k}")
print(f"  Silhouette Score    : {sil_final:.4f}")
print(f"  Calinski-Harabasz   : {ch_score:.2f}")
if not vel_df.empty:
    print(f"  Max Semantic Velocity: {max_vel:.4f} (Week {rupture_week})")
