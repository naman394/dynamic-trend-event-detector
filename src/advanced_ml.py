"""
Advanced ML Model: Latent Dirichlet Allocation (LDA) Topic Modelling
=====================================================================
Approach: Probabilistic topic modelling with LDA (Blei et al., 2003)

Coherence metric: Gensim C_V (preferred over UMass when available)
  - C_V uses Normalised PMI (NPMI) + indirect confirmation measure
  - Ranges 0 → 1 (higher = more coherent, more human-interpretable)
  - Outperforms UMass in correlation with human judgements
    (Röder et al., 2015 "Exploring the Space of Topic Coherence Measures")

Fallback: UMass coherence (manual implementation, no external dependency)
  - Used when gensim is unavailable (e.g. Python 3.14 + no prebuilt wheel)
  - UMass(t) = Σ log[(D(w_i,w_j)+ε) / D(w_j)] — co-document frequency based
"""

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os

sns.set_theme(style="whitegrid")
OUTPUT_DIR = 'reports/advanced_ml'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Coherence helpers ────────────────────────────────────────────────────────

def _gensim_cv_coherence(topic_words, texts, dictionary):
    """
    Gensim C_V coherence (NPMI-based).
    Returns float in [0, 1]. Higher = more coherent.
    """
    from gensim.models.coherencemodel import CoherenceModel
    cm = CoherenceModel(topics=topic_words, texts=texts,
                        dictionary=dictionary, coherence='c_v')
    return cm.get_coherence()


def _umass_coherence(topic_words, doc_word_matrix, vocab):
    """
    Manual UMass coherence fallback — no external dependency.
    Returns float ≤ 0. Higher (less negative) = more coherent.
    """
    word_idx = {w: i for i, w in enumerate(vocab)}
    binary   = (doc_word_matrix > 0).astype('float32')
    doc_freq = np.asarray(binary.sum(axis=0)).flatten()
    scores   = []
    for words in topic_words:
        indices = [word_idx[w] for w in words if w in word_idx]
        topic_score, pairs = 0.0, 0
        for m in range(1, len(indices)):
            for l in range(m):
                wi, wj = indices[m], indices[l]
                co = float(binary[:, wi].multiply(binary[:, wj]).sum())
                dj = float(doc_freq[wj])
                if dj > 0:
                    topic_score += np.log((co + 1.0) / dj)
                    pairs += 1
        scores.append(topic_score / pairs if pairs > 0 else 0.0)
    return float(np.mean(scores))


def compute_coherence(topic_words, texts_tokenised, doc_word_matrix, vocab):
    """
    Try Gensim C_V first; fall back to UMass if gensim is unavailable.
    Returns (score, metric_name, metric_range).
    """
    try:
        from gensim import corpora
        dictionary = corpora.Dictionary(texts_tokenised)
        score = _gensim_cv_coherence(topic_words, texts_tokenised, dictionary)
        return score, 'C_V (Gensim)', '[0 → 1, higher = better]'
    except ImportError:
        score = _umass_coherence(topic_words, doc_word_matrix, vocab)
        return score, 'UMass (fallback)', '[−∞ → 0, higher = better]'


# ── Text cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\b\w{1,2}\b', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# ── 1. Load & pre-process ────────────────────────────────────────────────────
df = pd.read_csv('data/news_headlines.csv')
df['headline_text'] = df['headline_text'].astype(str)
df['clean_text']    = df['headline_text'].apply(clean_text)
print(f"Loaded {len(df):,} headlines")

# Tokenised texts for Gensim coherence model
texts_tokenised = [doc.split() for doc in df['clean_text']]


# ── 2. Vectorise (raw counts for LDA) ───────────────────────────────────────
tf_vectorizer = CountVectorizer(
    max_df=0.90, min_df=5,
    stop_words='english',
    max_features=5000,
)
tf_matrix = tf_vectorizer.fit_transform(df['clean_text'])
vocab     = tf_vectorizer.get_feature_names_out().tolist()
print(f"Vocabulary: {len(vocab):,} terms  |  Matrix: {tf_matrix.shape}")


# ── 3. Train LDA ─────────────────────────────────────────────────────────────
NUM_TOPICS = 5
lda = LatentDirichletAllocation(
    n_components=NUM_TOPICS,
    max_iter=25,
    learning_method='online',
    learning_offset=50.0,
    random_state=42,
)
lda.fit(tf_matrix)


# ── 4. Verification 1 — Model fit metrics ────────────────────────────────────
perplexity     = lda.perplexity(tf_matrix)
log_likelihood = lda.score(tf_matrix)
print(f"\nPerplexity     : {perplexity:.2f}  (lower = better)")
print(f"Log-Likelihood : {log_likelihood:.2f}  (higher = better)")


# ── 5. Extract top words ─────────────────────────────────────────────────────
topics_top_words: list[list[str]] = []
for topic in lda.components_:
    top_idx = topic.argsort()[:-11:-1]
    topics_top_words.append([vocab[i] for i in top_idx])


# ── 6. Verification 2 — Coherence (Gensim C_V preferred) ────────────────────
print("\nComputing topic coherence …")
coherence_score, metric_name, metric_range = compute_coherence(
    topics_top_words, texts_tokenised, tf_matrix, vocab
)
print(f"Coherence metric : {metric_name}")
print(f"Score            : {coherence_score:.4f}  {metric_range}")


# ── 7. Verification 3 — Topic discrimination heatmap ────────────────────────
topic_sim = cosine_similarity(lda.components_)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(topic_sim, annot=True, fmt='.2f', cmap='RdYlGn_r',
            xticklabels=[f'T{i+1}' for i in range(NUM_TOPICS)],
            yticklabels=[f'T{i+1}' for i in range(NUM_TOPICS)],
            linewidths=0.5, ax=ax)
ax.set_title('LDA — Topic Discrimination Matrix\n(low off-diagonal = distinct topics)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/topic_separation_heatmap.png', dpi=150)
plt.close()


# ── 8. Verification 4 — Confidence distribution ─────────────────────────────
doc_topic_dist = lda.transform(tf_matrix)
max_probs      = doc_topic_dist.max(axis=1)

fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(max_probs, bins=40, kde=True, color='#2ecc71', ax=ax)
ax.axvline(max_probs.mean(), color='red', linestyle='--',
           label=f'Mean = {max_probs.mean():.2f}')
ax.set_title('LDA — Model Confidence Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Max Topic Probability per Document')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/model_confidence_distribution.png', dpi=150)
plt.close()


# ── 9. Per-topic word bar charts ─────────────────────────────────────────────
fig, axes = plt.subplots(1, NUM_TOPICS, figsize=(20, 7))
fig.suptitle('LDA — Top-10 Words per Topic', fontsize=16, fontweight='bold')

for idx, (words, ax) in enumerate(zip(topics_top_words, axes)):
    weights = lda.components_[idx][[vocab.index(w) for w in words]]
    ax.barh(words[::-1], weights[::-1],
            color=sns.color_palette('viridis', len(words)))
    ax.set_title(f'Topic {idx+1}', fontweight='bold')
    ax.set_xlabel('Word Weight')
    ax.tick_params(axis='y', labelsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/lda_topic_words.png', dpi=150, bbox_inches='tight')
plt.close()


# ── 10. Document distribution ────────────────────────────────────────────────
dominant     = doc_topic_dist.argmax(axis=1)
topic_counts = pd.Series(dominant).value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8, 5))
topic_counts.plot(kind='bar', ax=ax,
                  color=sns.color_palette('tab10', NUM_TOPICS),
                  edgecolor='white')
ax.set_title('LDA — Document Distribution Across Topics',
             fontsize=12, fontweight='bold')
ax.set_xticklabels([f'Topic {i+1}' for i in topic_counts.index], rotation=30)
ax.set_ylabel('Number of Documents')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/topic_document_distribution.png', dpi=150)
plt.close()


# ── 11. Save reports ─────────────────────────────────────────────────────────
topics_list = [{'Topic': i+1, 'Keywords': ', '.join(w)}
               for i, w in enumerate(topics_top_words)]
pd.DataFrame(topics_list).to_csv(f'{OUTPUT_DIR}/verified_topics.csv', index=False)

with open(f'{OUTPUT_DIR}/verification_report.txt', 'w') as f:
    f.write("MODEL INTEGRITY REPORT — ADVANCED ML (LDA)\n")
    f.write("=" * 55 + "\n")
    f.write(f"Vocabulary size        : {len(vocab):,}\n")
    f.write(f"Topics                 : {NUM_TOPICS}\n")
    f.write(f"Iterations             : 25\n")
    f.write(f"Perplexity             : {perplexity:.2f}\n")
    f.write(f"Log-Likelihood         : {log_likelihood:.2f}\n")
    f.write(f"Coherence ({metric_name:<12}) : {coherence_score:.4f}  {metric_range}\n")
    f.write(f"Avg Max-Topic Prob     : {max_probs.mean():.4f}\n")
    f.write("=" * 55 + "\n\nTopics:\n")
    for row in topics_list:
        f.write(f"  Topic {row['Topic']}: {row['Keywords']}\n")

print(f"\nReports saved to {OUTPUT_DIR}/")
print("✅ Advanced ML pipeline complete.")
