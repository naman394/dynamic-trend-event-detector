import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load data
df = pd.read_csv('data/news_headlines.csv')

# Preprocessing: simple lower and noise removal
df['headline_text'] = df['headline_text'].str.lower()

# Baseline: TF-IDF for most significant words
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = vectorizer.fit_transform(df['headline_text'])

feature_names = vectorizer.get_feature_names_out()
dense = tfidf_matrix.todense()
denselist = dense.tolist()

# Get top 10 words overall
sums = tfidf_matrix.sum(axis=0)
data = []
for col, term in enumerate(feature_names):
    data.append((term, sums[0, col]))

ranking = pd.DataFrame(data, columns=['term', 'rank'])
words = ranking.sort_values('rank', ascending=False)

print("Top 10 Terms in Baseline (Overall):")
print(words.head(10))

# Topic extraction per day (Demo for one specific day)
sample_date = df['publish_date'].iloc[0]
day_df = df[df['publish_date'] == sample_date]
print(f"\nTop Terms for date {sample_date}:")
day_tfidf = vectorizer.transform(day_df['headline_text'])
day_sums = day_tfidf.sum(axis=0)
day_data = []
for col, term in enumerate(feature_names):
    day_data.append((term, day_sums[0, col]))
day_ranking = pd.DataFrame(day_data, columns=['term', 'rank'])
print(day_ranking.sort_values('rank', ascending=False).head(5))
