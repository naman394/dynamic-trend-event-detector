"""
Baseline Model: TF-IDF based Trend Detection
=============================================
Approach: Term Frequency–Inverse Document Frequency (TF-IDF)
Why TF-IDF?
  - Normalises raw term counts by document frequency, penalising ubiquitous words
  - Fast, interpretable, no training required — ideal for establishing a benchmark
  - Allows temporal burst analysis by tracking per-day term weights over time
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import os

sns.set_theme(style="whitegrid")

# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = 'reports/baseline'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Load & Pre-process Data ───────────────────────────────────────────────
df = pd.read_csv('data/news_headlines.csv')
df['publish_date'] = pd.to_datetime(df['publish_date'], format='%Y%m%d')

# Clean: lowercase, remove punctuation/digits, strip extra whitespace
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)          # keep only letters
    text = re.sub(r'\b\w{1,2}\b', ' ', text)        # remove 1-2 char tokens
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['headline_text'].apply(clean_text)
print(f"Dataset: {len(df):,} headlines  |  date range: "
      f"{df['publish_date'].min().date()} → {df['publish_date'].max().date()}")


# ── 2. Fit TF-IDF Vectoriser ─────────────────────────────────────────────────
# sublinear_tf=True applies log(1+tf) dampening so one very long document
# with repeated terms doesn't dominate the whole corpus.
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=1000,
    sublinear_tf=True,      # log-scale term frequency
    ngram_range=(1, 2),     # capture single words AND 2-word phrases
    min_df=5,               # ignore terms appearing in fewer than 5 docs
)
tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
feature_names = vectorizer.get_feature_names_out()

print(f"Vocabulary size: {len(feature_names):,} terms")


# ── 3. Verification 1 — Global Keyword Ranking ───────────────────────────────
col_sums = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
ranking = (pd.DataFrame({'term': feature_names, 'tfidf_sum': col_sums})
           .sort_values('tfidf_sum', ascending=False)
           .reset_index(drop=True))

print("\nTop 20 Global Terms (TF-IDF ranked):")
print(ranking.head(20).to_string(index=False))

# Save full ranking
ranking.to_csv(f'{OUTPUT_DIR}/global_term_ranking.csv', index=False)


# ── 4. Verification 2 — Temporal Burst Analysis ──────────────────────────────
top_keywords = ranking.head(5)['term'].tolist()

df_daily = df.copy()
for kw in top_keywords:
    # Use word-boundary regex for accurate matching
    pattern = r'(?<![a-z])' + re.escape(kw) + r'(?![a-z])'
    df_daily[kw] = df_daily['clean_text'].str.contains(pattern, regex=True).astype(int)

burst_stats = df_daily.groupby('publish_date')[top_keywords].sum()

fig, ax = plt.subplots(figsize=(14, 5))
burst_stats.plot(ax=ax, linewidth=1.5, marker='o', markersize=3)
ax.set_title('Baseline — Keyword Burst Detection (Daily Frequencies)', fontsize=14, fontweight='bold')
ax.set_ylabel('Daily Occurrence Count')
ax.set_xlabel('Date')
ax.legend(title='Term', bbox_to_anchor=(1.01, 1), loc='upper left')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/baseline_burst_verification.png', dpi=150)
plt.close()
print(f"\nSaved burst chart → {OUTPUT_DIR}/baseline_burst_verification.png")


# ── 5. Verification 3 — Per-Day Top Term Table ──────────────────────────────
daily_top_terms = []
unique_days = sorted(df['publish_date'].unique())[:15]   # first 15 days

for day in unique_days:
    day_mask = df['publish_date'] == day
    day_tfidf = vectorizer.transform(df.loc[day_mask, 'clean_text'])
    day_sums = np.asarray(day_tfidf.sum(axis=0)).flatten()
    best_idx = int(day_sums.argmax())
    daily_top_terms.append({
        'date': day.strftime('%Y-%m-%d'),
        'headlines': int(day_mask.sum()),
        'top_term': feature_names[best_idx],
        'tfidf_intensity': round(float(day_sums[best_idx]), 4),
    })

results_df = pd.DataFrame(daily_top_terms)
print("\nPer-Day Top Term (first 15 days):")
print(results_df.to_string(index=False))
results_df.to_csv(f'{OUTPUT_DIR}/baseline_verification_results.csv', index=False)


# ── 6. Publication-quality Visual — Top 15 Terms ────────────────────────────
top15 = ranking.head(15)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top15['term'][::-1], top15['tfidf_sum'][::-1],
               color=sns.color_palette('Blues_d', 15))
ax.set_xlabel('Cumulative TF-IDF Score', fontsize=12)
ax.set_title('Baseline: Top 15 Discriminative Terms (TF-IDF)', fontsize=14, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.5)
for i, bar in enumerate(bars):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{bar.get_width():.2f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/baseline_top15_terms.png', dpi=150)
plt.close()
print(f"Saved top-15 chart → {OUTPUT_DIR}/baseline_top15_terms.png")

print("\n✅ Baseline pipeline complete.")
