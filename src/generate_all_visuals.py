import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
try:
    from bertopic import BERTopic
except ImportError:
    BERTopic = None

# Custom theme for high quality visuals
sns.set_theme(style="whitegrid", palette="muted")
COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

def generate_baseline_viz():
    print("Generating Baseline Visuals...")
    df = pd.read_csv('data/news_headlines.csv')
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df['headline_text'].astype(str))
    
    feature_names = vectorizer.get_feature_names_out()
    sums = tfidf_matrix.sum(axis=0)
    data = [(term, sums[0, col]) for col, term in enumerate(feature_names)]
    
    top_df = pd.DataFrame(data, columns=['Term', 'TF-IDF Weight']).sort_values('TF-IDF Weight', ascending=False).head(15)
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=top_df, x='TF-IDF Weight', y='Term', palette="Blues_d")
    plt.title('Baseline: Top 15 Discriminative Terms (TF-IDF Weighting)', fontsize=15, fontweight='bold')
    plt.xlabel('Cumulative TF-IDF Score', fontsize=12)
    plt.ylabel('Term', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/baseline_tfidf.png', dpi=300)
    print("Saved reports/baseline_tfidf.png")

def generate_advanced_viz():
    print("Generating Advanced ML (LDA) Visuals...")
    df = pd.read_csv('data/news_headlines.csv')
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = tf_vectorizer.fit_transform(df['headline_text'].astype(str))
    tf_feature_names = tf_vectorizer.get_feature_names_out()

    num_topics = 5
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', random_state=42)
    lda.fit(tf)

    fig, axes = plt.subplots(1, 5, figsize=(20, 8), sharey=True)
    fig.suptitle('Advanced ML: LDA Topic Feature Distribution', fontsize=20, fontweight='bold', y=1.05)

    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[:-11:-1]
        top_features = [tf_feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        sns.barplot(x=weights, y=top_features, ax=ax, palette=sns.color_palette("viridis", 10))
        ax.set_title(f'Topic {topic_idx + 1}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Weight', fontsize=10)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('reports/advanced_lda.png', dpi=300, bbox_inches='tight')
    print("Saved reports/advanced_lda.png")

def generate_deep_learning_viz():
    print("Generating Deep Learning (BERTopic) Visuals...")
    if BERTopic is None:
        print("BERTopic not installed, skipping DL viz.")
        return
        
    df = pd.read_csv('data/news_headlines.csv')
    docs = df['headline_text'].astype(str).tolist()
    
    # We use a lighter model for speed in this turn if needed, but the original script does full fit.
    # To ensure the visual matches the 'deep_learning.py' output:
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)
    
    # Topic Frequency Bar Chart
    info = topic_model.get_topic_info()
    # Filter out -1 (Outliers) for the top chart
    info = info[info['Topic'] != -1].head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=info, x='Count', y='Name', palette="magma")
    plt.title('Deep Learning: Top 10 Narrative Clusters (BERTopic Sizes)', fontsize=15, fontweight='bold')
    plt.xlabel('Number of Headlines', fontsize=12)
    plt.ylabel('Thematic Cluster Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/deep_learning_clusters.png', dpi=300)
    print("Saved reports/deep_learning_clusters.png")

if __name__ == "__main__":
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    generate_baseline_viz()
    generate_advanced_viz()
    generate_deep_learning_viz()
    print("\n--- All visual representations generated successfully ---")
