import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create reports dir
if not os.path.exists('reports/baseline'):
    os.makedirs('reports/baseline')

# Load data
df = pd.read_csv('data/news_headlines.csv')
df['publish_date'] = pd.to_datetime(df['publish_date'], format='%Y%m%d')
df['headline_text'] = df['headline_text'].str.lower()

# Baseline: TF-IDF for most significant words
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = vectorizer.fit_transform(df['headline_text'])
feature_names = vectorizer.get_feature_names_out()

# --- VERIFICATION STEP 1: Global Keyword Ranking ---
sums = tfidf_matrix.sum(axis=0)
data = [(term, sums[0, col]) for col, term in enumerate(feature_names)]
ranking = pd.DataFrame(data, columns=['term', 'tfidf_sum']).sort_values('tfidf_sum', ascending=False)

print("Baseline: Top 20 Global Terms (Results):")
print(ranking.head(20))

# --- VERIFICATION STEP 2: Temporal Burst Analysis (Novelty Verification) ---
# We track how often the top keywords appear over time
top_keywords = ranking.head(5)['term'].tolist()
df_daily = df.copy()
for kw in top_keywords:
    df_daily[kw] = df_daily['headline_text'].str.contains(rf'\b{kw}\b', regex=True).astype(int)

burst_stats = df_daily.groupby('publish_date')[top_keywords].sum()

plt.figure(figsize=(12, 6))
sns.lineplot(data=burst_stats)
plt.title('Baseline Verification: Keyword Burst Detection (Daily Frequencies)', fontsize=14)
plt.ylabel('Daily Frequency')
plt.xlabel('Date')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('reports/baseline/baseline_burst_verification.png', dpi=300)
print("\nGenerated reports/baseline/baseline_burst_verification.png for verification.")

# --- VERIFICATION STEP 3: Topic Concentration Metric ---
# We verify if TF-IDF effectively separates days into unique 'events'
daily_top_terms = []
unique_days = df['publish_date'].unique()[:10] # Track first 10 days for clarity

for day in unique_days:
    day_df = df[df['publish_date'] == day]
    day_tfidf = vectorizer.transform(day_df['headline_text'])
    day_sums = day_tfidf.sum(axis=0)
    best_idx = day_sums.argmax()
    best_term = feature_names[best_idx]
    daily_top_terms.append({'date': day, 'top_term': best_term, 'intensity': day_sums[0, best_idx]})

results_df = pd.DataFrame(daily_top_terms)
print("\nBaseline verification: Daily Event Identification Results:")
print(results_df)
results_df.to_csv('reports/baseline/baseline_verification_results.csv', index=False)
