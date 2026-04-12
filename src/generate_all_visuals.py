"""
Generate all publication-quality visuals for all implemented models:
  - Baseline TF-IDF
  - Advanced ML LDA
  - Deep Learning SBERT K-Means (3D scatter, semantic velocity, growth velocity)
  - GDELT Impact Scoring

Reads pre-computed results from reports/ rather than re-fitting models,
making this a fast regeneration script.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs('reports', exist_ok=True)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\b\w{1,2}\b', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# ── 1. Baseline TF-IDF ───────────────────────────────────────────────────────
def generate_baseline_viz() -> None:
    print("Generating Baseline visuals …")
    df = pd.read_csv('data/news_headlines.csv')
    df['clean_text'] = df['headline_text'].astype(str).apply(clean_text)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000,
                                 sublinear_tf=True, ngram_range=(1, 2), min_df=5)
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    feature_names = vectorizer.get_feature_names_out()
    col_sums = np.asarray(tfidf_matrix.sum(axis=0)).flatten()

    top_df = (pd.DataFrame({'Term': feature_names, 'TF-IDF Weight': col_sums})
              .sort_values('TF-IDF Weight', ascending=False).head(15))

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(data=top_df, x='TF-IDF Weight', y='Term', palette='Blues_d', ax=ax)
    ax.set_title('Baseline: Top 15 Discriminative Terms (TF-IDF)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cumulative TF-IDF Score', fontsize=11)
    plt.tight_layout()
    plt.savefig('reports/baseline_tfidf.png', dpi=150)
    plt.close()
    print("  Saved reports/baseline_tfidf.png")


# ── 2. Advanced ML LDA ───────────────────────────────────────────────────────
def generate_advanced_viz() -> None:
    print("Generating Advanced ML (LDA) visuals …")

    # Use pre-computed topic CSV if available, otherwise re-fit
    topic_csv = 'reports/advanced_ml/verified_topics.csv'
    if os.path.exists(topic_csv):
        topics = pd.read_csv(topic_csv)
        print("  Using cached LDA topics from reports/advanced_ml/verified_topics.csv")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('off')
        table_data = [[row['Topic'], row['Keywords'][:60]] for _, row in topics.iterrows()]
        ax.table(cellText=table_data, colLabels=['Topic', 'Top Keywords'],
                 loc='center', cellLoc='left')
        ax.set_title('Advanced ML: LDA Topic Keywords', fontsize=13, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('reports/advanced_lda.png', dpi=150, bbox_inches='tight')
        plt.close()
    else:
        df = pd.read_csv('data/news_headlines.csv')
        df['clean_text'] = df['headline_text'].astype(str).apply(clean_text)
        tf_v = CountVectorizer(max_df=0.90, min_df=5, stop_words='english', max_features=5000)
        tf = tf_v.fit_transform(df['clean_text'])
        vocab = tf_v.get_feature_names_out()
        lda = LatentDirichletAllocation(n_components=5, max_iter=25,
                                        learning_method='online', random_state=42)
        lda.fit(tf)
        fig, axes = plt.subplots(1, 5, figsize=(20, 8))
        for idx, (topic, ax) in enumerate(zip(lda.components_, axes)):
            top_idx = topic.argsort()[:-11:-1]
            words = [vocab[i] for i in top_idx]
            sns.barplot(x=topic[top_idx], y=words, ax=ax,
                        palette=sns.color_palette('viridis', 10))
            ax.set_title(f'Topic {idx+1}', fontweight='bold')
            ax.set_xlabel('Weight')
            ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig('reports/advanced_lda.png', dpi=150, bbox_inches='tight')
        plt.close()
    print("  Saved reports/advanced_lda.png")


# ── 3. Deep Learning — SBERT K-Means ────────────────────────────────────────
def generate_deep_learning_viz() -> None:
    print("Generating Deep Learning visuals (from cached results) …")
    dl_dir = 'reports/deep_learning'

    # 3a. Semantic Velocity (from cached CSV)
    vel_csv = f'{dl_dir}/semantic_velocity.csv'
    if os.path.exists(vel_csv):
        vel_df = pd.read_csv(vel_csv)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(range(len(vel_df)), vel_df['semantic_velocity'],
                        alpha=0.22, color='#9B59B6')
        ax.plot(range(len(vel_df)), vel_df['semantic_velocity'],
                color='#8E44AD', linewidth=1.5, marker='o', markersize=3)
        max_idx = vel_df['semantic_velocity'].idxmax()
        max_val = vel_df.loc[max_idx, 'semantic_velocity']
        max_wk  = vel_df.loc[max_idx, 'week']
        ax.axvline(max_idx, color='#E67E22', linewidth=2, linestyle='--',
                   label=f'Narrative Rupture: {max_wk} (V_s={max_val:.3f})')
        step = max(1, len(vel_df)//12)
        ax.set_xticks(range(0, len(vel_df), step))
        ax.set_xticklabels(vel_df['week'].iloc[::step], rotation=45, ha='right', fontsize=7)
        ax.set_title('Semantic Velocity (SBERT Weekly Centroid Shift)',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('$V_s$')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(f'{dl_dir}/semantic_velocity_summary.png', dpi=150)
        plt.close()
        print(f"  Saved {dl_dir}/semantic_velocity_summary.png")

    # 3b. Cluster summary table
    cluster_csv = f'{dl_dir}/cluster_summary.csv'
    if os.path.exists(cluster_csv):
        clusters = pd.read_csv(cluster_csv)
        colors = sns.color_palette('tab10', len(clusters))

        fig, ax = plt.subplots(figsize=(11, 4))
        bars = ax.barh(clusters['Cluster'], clusters['Size'],
                       color=colors, edgecolor='white', linewidth=1.5)
        for bar, (_, row) in zip(bars, clusters.iterrows()):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    row['TopTerms'][:45], va='center', fontsize=8, color='#2C3E50')
        ax.set_title('SBERT K-Means — Topic Cluster Sizes & Top Terms',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Headlines')
        plt.tight_layout()
        plt.savefig(f'{dl_dir}/cluster_sizes_summary.png', dpi=150)
        plt.close()
        print(f"  Saved {dl_dir}/cluster_sizes_summary.png")


# ── 4. Event Impact (SBERT) ──────────────────────────────────────────────────
def generate_impact_viz() -> None:
    print("Generating Event Impact visuals …")
    impact_csv = 'reports/event_impact_scores.csv'
    if not os.path.exists(impact_csv):
        print("  [SKIP] event_impact_scores.csv not found — run notebook 09 first.")
        return

    df = pd.read_csv(impact_csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Event Impact Scoring — SBERT-based', fontsize=13, fontweight='bold')

    sns.histplot(df['impact_score'], bins=35, kde=True,
                 color='#E74C3C', ax=axes[0])
    axes[0].set_title('S_I Distribution')
    axes[0].set_xlabel('S_I = SBERT_Uniqueness × |Tone|')

    top_src = (df.groupby('SOURCECOMMONNAME')['impact_score']
               .mean().sort_values(ascending=False).head(12))
    sns.barplot(x=top_src.values, y=top_src.index, palette='viridis', ax=axes[1])
    axes[1].set_title('Avg Impact Score by Source')
    axes[1].set_xlabel('Mean S_I')

    plt.tight_layout()
    plt.savefig('reports/event_impact_analysis.png', dpi=150)
    plt.close()
    print("  Saved reports/event_impact_analysis.png")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_baseline_viz()
    generate_advanced_viz()
    generate_deep_learning_viz()
    generate_impact_viz()
    print("\n✅ All visuals regenerated successfully.")
