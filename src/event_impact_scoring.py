import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

def calculate_impact_score():
    print("--- 🚀 Extra Mile: Event Impact Scoring System ---")
    
    # Load processed GDELT data (the real-time stream)
    data_path = 'data/gdelt_processed.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run gdelt_processor.py first.")
        return

    df = pd.read_csv(data_path)
    
    # We'll simulate a 'stream' by grouping by GKGRECORDID or just using indices 
    # since this snapshot is from a single 15-min window.
    # To demonstrate 'Impact Scoring', we'll calculate it per record or per theme cluster.
    
    print("Loading Sentence-BERT for Semantic Impact calculation...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Objective: Identify 'Impactful' records
    # Impact = Semantic Uniqueness (Embedding distance from mean) * Intensity (|Tone|)
    
    # 1. Generate Embeddings for Themes (represented as strings)
    df['theme_string'] = df['theme_list'].apply(lambda l: " ".join(eval(l)) if isinstance(l, str) else "")
    embeddings = model.encode(df['theme_string'].tolist())
    
    # 2. Calculate the 'Global Narrative Centroid' for this snapshot
    global_centroid = np.mean(embeddings, axis=0)
    
    # 3. Calculate 'Semantic Magnitude' (Distance from centroid)
    # Higher distance means the record is an outlier/unique narrative
    similarities = cosine_similarity(embeddings, [global_centroid]).flatten()
    df['semantic_magnitude'] = 1 - similarities
    
    # 4. Calculate Final Impact Score
    # We use abs(tone) because highly positive OR highly negative events are both impactful.
    # High impact = Unique narrative + High emotional intensity.
    df['impact_score'] = df['semantic_magnitude'] * df['tone_value'].abs()
    
    # 5. Get Top 5 Impactful Events
    top_impact = df.sort_values('impact_score', ascending=False).head(5)
    
    print("\n--- Top 5 Impactful Events (Extra Mile Analysis) ---")
    for idx, row in top_impact.iterrows():
        print(f"ID: {row['GKGRECORDID']}")
        print(f"Source: {row['SOURCECOMMONNAME']}")
        print(f"Tone Intensity: {abs(row['tone_value']):.2f}")
        print(f"Semantic Uniqueness: {row['semantic_magnitude']:.4f}")
        print(f"FINAL IMPACT SCORE: {row['impact_score']:.4f}")
        print(f"Themes: {row['theme_list'][:100]}...") # Truncated print
        print("-" * 30)

    # Save to reports
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    df.to_csv('reports/event_impact_scores.csv', index=False)
    print("\nFull Impact Scores saved to reports/event_impact_scores.csv")

if __name__ == "__main__":
    calculate_impact_score()
