import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_velocity():
    if os.path.exists('reports/semantic_velocity.csv'):
        df = pd.read_csv('reports/semantic_velocity.csv')
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='week', y='velocity', marker='o', color='#636EFA')
        plt.xticks(rotation=45, ha='right')
        plt.title('Semantic Velocity Over Time (Narrative Shift Magnitude)', fontsize=14)
        plt.xlabel('Week Interval', fontsize=12)
        plt.ylabel('Velocity (Cosine Distance)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('reports/semantic_velocity_plot.png', dpi=300)
        print("Generated reports/semantic_velocity_plot.png")

def visualize_impact():
    if os.path.exists('reports/event_impact_scores.csv'):
        df = pd.read_csv('reports/event_impact_scores.csv')
        
        # Plot top 15 most impactful sources
        plt.figure(figsize=(12, 6))
        top_sources = df.groupby('SOURCECOMMONNAME')['impact_score'].mean().sort_values(ascending=False).head(15)
        sns.barplot(x=top_sources.values, y=top_sources.index, palette='viridis')
        plt.title('Average Event Impact Score by Source', fontsize=14)
        plt.xlabel('Average Impact Score ($S_I$)', fontsize=12)
        plt.ylabel('Source', fontsize=12)
        plt.tight_layout()
        plt.savefig('reports/impact_by_source.png', dpi=300)
        print("Generated reports/impact_by_source.png")

        # Distribution of impact scores
        plt.figure(figsize=(10, 6))
        sns.histplot(df['impact_score'], bins=30, kde=True, color='#EF553B')
        plt.title('Distribution of Event Impact Scores ($S_I$)', fontsize=14)
        plt.xlabel('Impact Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        plt.savefig('reports/impact_distribution.png', dpi=300)
        print("Generated reports/impact_distribution.png")

if __name__ == "__main__":
    if not os.path.exists('reports'):
        os.makedirs('reports')
    visualize_velocity()
    visualize_impact()
