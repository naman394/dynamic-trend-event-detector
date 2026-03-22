import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora

def run_advanced_pipeline():
    # Setup paths
    OUTPUT_DIR = 'reports/advanced_ml'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load data
    df = pd.read_csv('data/news_headlines.csv')
    df['headline_text'] = df['headline_text'].astype(str)
    texts = [doc.split() for doc in df['headline_text']]

    # Create Dictionary/Corpus for Gensim Coherence
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Vectorize for LDA (Scikit-Learn)
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf_matrix = tf_vectorizer.fit_transform(df['headline_text'])
    tf_feature_names = tf_vectorizer.get_feature_names_out()

    # Model training
    num_topics = 5
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=15, 
                                    learning_method='online', random_state=42)
    lda.fit(tf_matrix)

    # --- VERIFICATION 1: Model Fitting Metrics ---
    perplexity = lda.perplexity(tf_matrix)
    log_likelihood = lda.score(tf_matrix)
    print(f"Confidence Metric: Model Perplexity = {perplexity:.2f}")
    print(f"Confidence Metric: Log-Likelihood = {log_likelihood:.2f}")

    # --- VERIFICATION 2: Topic Coherence (Cv) using Gensim ---
    # Convert Sklearn topics to list for Gensim
    topics_top_words = []
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[:-11:-1]
        top_features = [tf_feature_names[i] for i in top_features_ind]
        topics_top_words.append(top_features)

    # Calculate Coherence Score (Cv)
    print("Calculating Topic Coherence (this may take a minute)...")
    coherence_model = CoherenceModel(topics=topics_top_words, texts=texts, 
                                     dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(f"Verification Metric: Topic Coherence (Cv Score) = {coherence_score:.4f}")

    # --- VERIFICATION 3: Topic Discrimination (Overlap Proof) ---
    from sklearn.metrics.pairwise import cosine_similarity
    topic_sim = cosine_similarity(lda.components_)
    sns.heatmap(topic_sim, annot=True, cmap='RdYlGn', xticklabels=range(1, 6), yticklabels=range(1, 6))
    plt.title('Verification: Topic Discrimination Matrix (Lower Overlap = Better)')
    plt.savefig(f'{OUTPUT_DIR}/topic_separation_heatmap.png', dpi=300)
    plt.close()

    # --- VERIFICATION 4: Confidence Distribution ---
    doc_topic_dist = lda.transform(tf_matrix)
    max_probs = doc_topic_dist.max(axis=1)
    plt.figure(figsize=(10, 5))
    sns.histplot(max_probs, bins=30, kde=True, color='green')
    plt.title(f'Verification: Model Confidence Distribution (Avg: {max_probs.mean():.2f})')
    plt.xlabel('Topic Probability Prediction')
    plt.savefig(f'{OUTPUT_DIR}/model_confidence_distribution.png', dpi=300)
    plt.close()

    # Save final report logs
    with open(f'{OUTPUT_DIR}/verification_report.txt', 'w') as f:
        f.write(f"MODEL INTEGRITY REPORT - ADVANCED ML (LDA)\n")
        f.write(f"="*40 + "\n")
        f.write(f"1. Model Fit (Perplexity): {perplexity:.2f}\n")
        f.write(f"2. Log-Likelihood: {log_likelihood:.2f}\n")
        f.write(f"3. Topic Coherence (Cv): {coherence_score:.4f}\n")
        f.write(f"4. Average Topic Confidence: {max_probs.mean():.4f}\n")
        f.write(f"="*40 + "\n")

    # Final result output
    topics_list = []
    for idx, words in enumerate(topics_top_words):
        topics_list.append({"Topic": idx+1, "Keywords": ", ".join(words)})
    pd.DataFrame(topics_list).to_csv(f'{OUTPUT_DIR}/verified_topics.csv', index=False)
    print(f"Integrity Report and verified results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    run_advanced_pipeline()
