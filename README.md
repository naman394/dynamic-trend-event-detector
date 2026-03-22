# Dynamic Trend & Event Detector 🚀

A multi-stage NLP pipeline for monitoring global information streams, detecting emerging narratives, and quantifying societal shifts using **Sentence-BERT** and **GDELT**.

## 👥 Authors
- **Navnit Naman** (Enrollment: 230085) - B.Tech CSE (AI-ML)
- **Kanhaiya Kumar** (Enrollment: 230062) - B.Tech CSE (AI-ML)
- **Institution**: Newton School of Technology, Rishihood University

---

## 🏗️ System Architecture
The system bridges the gap between traditional frequency-based statistics and modern Transformer-based semantic trajectories:

1.  **Baseline (TF-IDF)**: Statistical importance weighting for key term extraction.
2.  **Advanced ML (LDA)**: Probabilistic latent variable modeling for thematic grouping.
3.  **Deep Learning (BERTopic)**: Context-aware clustering using SBERT, UMAP, and HDBSCAN.
4.  **Hybrid/Edge (Semantic Velocity)**: Tracking narrative evolution by measuring document centroid displacement in SBERT space.
5.  **Extra Mile (Event Impact Scoring)**: Quantifying breakthroughs by cross-referencing semantic uniqueness with GDELT emotional intensity (Tone).

---

## 📊 Key Innovations
### 1. Semantic Velocity ($V_s$)
Instead of just counting keywords, we calculate the velocity of information:
$$V_s = 1 - \text{cosine\_similarity}(\text{Centroid}_{T_n}, \text{Centroid}_{T_{n+1}})$$
High peaks indicate **Narrative Ruptures** (breaking events).

### 2. Event Impact Scoring ($S_I$)
Combines thematic uniqueness with sentiment magnitude to filter out noise:
$$S_I = \text{SemanticDistance} \times |\text{GDELT Tone}|$$

---

## 📂 Project Structure
```text
├── src/
│   ├── baseline.py          # TF-IDF keyword extraction
│   ├── advanced_ml.py       # LDA probabilistic modeling
│   ├── deep_learning.py     # BERTopic implementation
│   ├── hybrid_temporal.py   # SBERT-based Semantic Velocity
│   ├── gdelt_processor.py   # GDELT GKG V2 ingestion
│   ├── gdelt_analysis.py    # GDELT Theme-Tone analysis
│   └── event_impact_scoring.py # Extra Mile implementation
├── reports/
│   ├── report.tex           # Detailed academic LaTeX report
│   └── implementation_report.md # Summary of findings
├── run_all.py               # Orchestration pipeline
└── requirements.txt         # Project dependencies
```

## 🚀 Getting Started

### 1. Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Running the Pipeline
```bash
python run_all.py
```

## ⚖️ License
This project is part of the Advanced Agentic Coding series. Created for educational and research purposes.
