import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Load data
df = pd.read_csv('data/news_headlines.csv')

# Vectorize for LDA (LDA typically uses CountVectorizer)
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(df['headline_text'])
tf_feature_names = tf_vectorizer.get_feature_names_out()

# LDA model
num_topics = 5
lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', random_state=42)
lda.fit(tf)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
print("LDA Probabilistic Topics:")
display_topics(lda, tf_feature_names, no_top_words)
