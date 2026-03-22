import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load data
df = pd.read_csv('data/news_headlines.csv')
df['publish_date'] = pd.to_datetime(df['publish_date'], format='%Y%m%d')

# Bucket by time (e.g., weekly)
df['week'] = df['publish_date'].dt.to_period('W')
weeks = sorted(df['week'].unique())

print("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

velocities = []
prev_centroid = None

print("Calculating Semantic Velocity using SBERT Embeddings (High-Precision Shift Detection)...")

for i in range(len(weeks)):
    week_headlines = df[df['week'] == weeks[i]]['headline_text'].tolist()
    
    if len(week_headlines) == 0: continue
    
    # Generate embeddings for current week
    week_embeddings = model.encode(week_headlines)
    
    # Calculate Semantic Centroid
    current_centroid = np.mean(week_embeddings, axis=0)
    
    if prev_centroid is not None:
        # Measure shift between consecutive weeks
        similarity = cosine_similarity([prev_centroid], [current_centroid])[0][0]
        velocity = 1 - similarity
        
        # Identification of Narrative Ruptures (High velocity)
        velocities.append({'week': str(weeks[i]), 'velocity': velocity})
        print(f"Week {weeks[i-1]} -> {weeks[i]}: Semantic Velocity = {velocity:.4f}")
    
    prev_centroid = current_centroid

# Save results
if not os.path.exists('reports'):
    os.makedirs('reports')

velocity_df = pd.DataFrame(velocities)
if not velocity_df.empty:
    velocity_df.to_csv('reports/semantic_velocity_sbert.csv', index=False)
    print("\nResults saved to reports/semantic_velocity_sbert.csv")
    
    # Alert for 'Narrative Rupture'
    max_vel = velocity_df['velocity'].max()
    rupture_week = velocity_df[velocity_df['velocity'] == max_vel]['week'].values[0]
    print(f"\n🚨 POTENTIAL NARRATIVE RUPTURE DETECTED: {rupture_week} (Velocity: {max_vel:.4f})")
else:
    print("\nNot enough data points to calculate velocity.")
