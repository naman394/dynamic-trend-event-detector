import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
data_path = 'data/news_headlines.csv'
df = pd.read_csv(data_path)

# Convert publish_date to datetime
df['publish_date'] = pd.to_datetime(df['publish_date'], format='%Y%m%d')

# EDA: Headlines over time
print(f"Total Headlines: {len(df)}")
print(f"Date Range: {df['publish_date'].min()} to {df['publish_date'].max()}")

# Group by day
daily_counts = df.groupby('publish_date').size()

plt.figure(figsize=(12, 6))
daily_counts.plot(kind='line')
plt.title('Headlines Frequency Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Headlines')
plt.grid(True)
plt.savefig('reports/eda_temporal_dist.png')
print("Saved temporal distribution plot to reports/eda_temporal_dist.png")

# Text stats
df['headline_len'] = df['headline_text'].str.len()
plt.figure(figsize=(10, 5))
sns.histplot(df['headline_len'], bins=20, kde=True)
plt.title('Headline Length Distribution')
plt.savefig('reports/eda_text_stats.png')
print("Saved text stats plot to reports/eda_text_stats.png")

# Display first few rows
print(df.head())
