# Presentation: Project 8 - Dynamic Trend & Event Detector

## Slide 1: Title & Introduction
- Project: Dynamic Trend & Event Detector
- Focus: Social Media Analytics & Journalism
- Objective: Track evolving narratives over time.

## Slide 2: The Challenge
- Information Overload: Millions of documents daily.
- Distinguishing "fleeting memes" vs "significant events".
- Need for temporal sensitivity.

## Slide 3: Tech Stack
- NLP: BERTopic, LDA.
- Infrastructure: Python, Scikit-learn, UMAP.
- Data: GDELT / ABC News Headlines.

## Slide 4: Baseline Approach (Level 5)
- Frequency-based (TF-IDF).
- Pros: Fast, interpretable.
- Cons: Ignores semantics/context.

## Slide 5: Advanced ML & DL (Level 7-9)
- Probabilistic: LDA (Latent Dirichlet Allocation).
- Embeddings: BERTopic (SBERT + HDBSCAN).
- Results: Dense clusters representing real-world news themes.

## Slide 6: Hybrid Logic & "Semantic Velocity"
- Logic: Distance between topic centroids over time.
- Metric: $1 - \text{cosine\_similarity}(C_t, C_{t+1})$.
- High Velocity = Emerging Event.

## Slide 7: Results
- Visualizing Topic Over Time (ToT).
- Correlation with news timestamps.

## Slide 8: Future Work
- Integration with real-time GDELT streams.
- Misinformation detection scoring.

## Slide 9: Conclusion
- Hybrid models provide the best balance of context and speed.
- Semantic velocity is a robust proxy for "breaking news".
