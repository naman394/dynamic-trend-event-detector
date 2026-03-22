import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast

# Load processed GDELT
df = pd.read_csv('data/gdelt_processed.csv')

# theme_list is saved as string representation of list, convert back
df['theme_list'] = df['theme_list'].apply(ast.literal_eval)

# Flatten themes
all_themes = [theme for sublist in df['theme_list'] for theme in sublist]
theme_counts = Counter(all_themes)

# Top 15 GDELT Themes
top_themes = theme_counts.most_common(15)
print("Top 15 GDELT Themes:")
for theme, count in top_themes:
    print(f"{theme}: {count}")

# Visualize
theme_df = pd.DataFrame(top_themes, columns=['Theme', 'Count'])
plt.figure(figsize=(12, 8))
sns.barplot(x='Count', y='Theme', data=theme_df, palette='viridis')
plt.title('Top Themes in GDELT 15-Min Snapshot')
plt.savefig('reports/gdelt_top_themes.png')
print("Saved GDELT visualization to reports/gdelt_top_themes.png")

# Sentiment Analysis per Theme (for top themes)
top_theme_names = [t[0] for t in top_themes]
theme_tone = []

for theme in top_theme_names:
    avg_tone = df[df['theme_list'].apply(lambda l: theme in l)]['tone_value'].mean()
    theme_tone.append({'Theme': theme, 'AvgTone': avg_tone})

tone_df = pd.DataFrame(theme_tone)
print("\nAverage Tone (Sentiment) for Top GDELT Themes:")
print(tone_df)
