# Dynamic Trend & Event Detector

**Project:** Automated detection of narrative trends and event ruptures in news corpora using statistical and probabilistic NLP models.

**Authors:** Navnit Naman (230085) & Kanhaiya Kumar (230062) — Newton School of Technology, Rishihood University

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Current Phase Scope](#2-current-phase-scope)
3. [Why Each Model Was Chosen](#3-why-each-model-was-chosen)
4. [Architecture](#4-architecture)
5. [Corrections Made](#5-corrections-made)
6. [Modules Removed](#6-modules-removed)
7. [Repository Structure](#7-repository-structure)
8. [How to Run](#8-how-to-run)
9. [Results Summary](#9-results-summary)
10. [Roadmap](#10-roadmap)

---

## 1. Project Overview

This project analyses large-scale news headline datasets to:
- Identify the most discriminative terms and their temporal burst patterns (Baseline)
- Discover latent thematic groups and track topic evolution (Advanced ML)
- Cross-reference with global real-time news via the GDELT Global Knowledge Graph
- Score events by semantic uniqueness and tonal intensity (Impact Scoring)

### Dataset
| Property | Value |
|---|---|
| Source | ABC News archive |
| Size | ~20,000 headlines |
| Period | 2003–2012 |
| Format | `publish_date` (YYYYMMDD) + `headline_text` |

---

## 2. Project Scope

All modules are implemented and fully operational:

| # | Notebook | Module | Description |
|---|---|---|---|
| 01 | `01_eda.ipynb` | EDA | Data inspection, temporal and text distributions |
| 02 | `02_baseline_tfidf.ipynb` | Baseline (TF-IDF) | Global term ranking, burst detection, per-day top term |
| 03 | `03_advanced_ml_lda.ipynb` | Advanced ML (LDA) | 5-topic model, **Gensim C_V coherence**, discrimination heatmap |
| 04 | `04_gdelt_processor.ipynb` | GDELT Processor | Parse raw 15-min GKG snapshot → clean CSV |
| 05 | `05_gdelt_analysis.ipynb` | GDELT Analysis | Theme frequency, per-theme sentiment |
| 06 | `06_event_impact_scoring.ipynb` | Impact Scoring (TF-IDF baseline) | TF-IDF cosine distance × \|tone\| — lightweight reference |
| 07 | `07_visualize_results.ipynb` | Results Dashboard | Consolidated output from all models |
| 08 | `08_deep_learning_kmeans.ipynb` | **Deep Learning (SBERT + K-Means)** | K-Means on SBERT embeddings, UMAP 3D/2D, semantic velocity, growth velocity |
| 09 | `09_gdelt_verification_impact.ipynb` | **GDELT Verification & SBERT Impact** | Rupture-triggered GDELT verification, SBERT S_I scoring |

---

## 3. Why Each Model Was Chosen

### 3.1 Baseline — TF-IDF

**Problem addressed:** Establish an interpretable, dependency-free benchmark for keyword importance.

**Why TF-IDF:**

$$\text{TF-IDF}(t, d) = \log(1 + f_{t,d}) \times \left(\log\frac{N}{1 + df_t} + 1\right)$$

- No training required — immediate, unsupervised operation
- `IDF` naturally suppresses stop-word noise without a hand-coded list
- `log-TF` prevents a single high-frequency document from dominating
- Bigram support (`ngram_range=(1,2)`) captures phrases like "prime minister"
- Temporal burst analysis reveals which keywords spike on specific dates

**Limitations:** Treats each word as independent; cannot group semantically related terms. Addressed by LDA.

### 3.3 Deep Learning — K-Means on SBERT Embeddings

**Problem addressed:** LDA uses bag-of-words; semantically equivalent phrases like *"troops deployed"* and *"soldiers sent"* are invisible to it. SBERT solves this.

**Architecture (from project specification):**

```
Headlines → SBERT (384-dim) → UMAP (3D/2D) → K-Means Clustering
                                                    │
                              Growth Velocity ◄─────┤
                              Semantic Velocity ◄────┤
                              GDELT Verification ◄───┘
```

**Semantic Velocity:**
$$V_s(t) = 1 - \cos\text{-sim}\bigl(\bar{e}_{t-1},\, \bar{e}_t\bigr)$$

A spike in $V_s$ signals a **Narrative Rupture** — the corpus shifted topics suddenly due to a real-world event.

**Results:** K=5 optimal clusters discovered:
| Cluster | Label | Top Terms |
|---|---|---|
| Topic A | Crime/Courts | police, man, court, murder, charged |
| Topic B | Sports | cup, win, world, title, final |
| Topic C | Government/Policy | council, govt, plan, nsw, drought |
| Topic D | War/Security | iraq, war, baghdad, troops, saddam |
| Topic E | Health/Disasters | sars, crash, dead, hospital, missing |

**Max Semantic Velocity:** 0.1308 detected in Week 2003-05-26/06-01 (SARS outbreak + Iraq war transition period).

### 3.4 Advanced ML — LDA (Latent Dirichlet Allocation)

**Problem addressed:** Discover latent thematic groups; each document is a mixture of topics.

**Why LDA:**

$$p(w \mid \alpha, \beta) = \int p(\theta \mid \alpha) \prod_{n=1}^{N} \sum_{z_n} p(z_n \mid \theta)\, p(w_n \mid z_n, \beta)\, d\theta$$

- Probabilistic model: accounts for multi-topic documents (realistic for news)
- Raw count input (unlike TF-IDF): matches the Dirichlet-Multinomial generative assumption
- Allows tracking which topics rise/fall over time

**Corrections applied in this phase:**
- Preprocessing now strips punctuation, digits, 1–2 char tokens before vectorisation
- Increased `max_iter` from 15 → 25 for better convergence
- Added `learning_offset=50.0` to stabilise early learning
- **UMass coherence** implemented from scratch (gensim incompatible with Python 3.14)
- Added per-topic bar charts, document distribution chart, and discrimination heatmap

**Coherence metric — UMass (no external library):**

$$C_{\text{UMass}}(t) = \sum_{i=2}^{N} \sum_{j=1}^{i-1} \log \frac{D(w_i, w_j) + 1}{D(w_j)}$$

### 3.5 GDELT Integration

**Why GDELT:** Provides real-time geopolitical context (100+ languages, 15-min updates) with a structured theme taxonomy and quantitative tone score — dimensions not available in the raw headline corpus.

### 3.6 Event Impact Scoring (Phase 1)

**Formula:**

$$S_I = \underbrace{(1 - \cos\text{-sim}(\vec{v}_d, \vec{c}_{\text{global}}))}_{\text{Semantic Uniqueness (TF-IDF)}} \times \underbrace{|\text{tone}_d|}_{\text{Tonal Intensity}}$$

A record scores high only if it is **both** semantically unusual **and** tonally charged. This filters out:
- Common background topics (low uniqueness, ignored)
- Neutral unique records (low tone, ignored)

---

## 4. Architecture

```
Raw Data (news_headlines.csv)
        │
        ▼
    EDA (01_eda.ipynb)
        │
        ├──► Baseline TF-IDF (02_baseline_tfidf.ipynb)
        │         └── Global ranking, burst detection, per-day top term
        │
        ├──► Advanced ML LDA (03_advanced_ml_lda.ipynb)
        │         └── 5 topics, UMass coherence, heatmap, confidence
        │
        └──► Deep Learning: SBERT K-Means (08_deep_learning_kmeans.ipynb)
                  ├── SBERT (384-dim) → UMAP (3D+2D) → K-Means (K=5)
                  ├── Growth Velocity (monthly cluster size)
                  ├── Semantic Velocity V_s (weekly centroid shift)
                  └── GDELT Verification

GDELT GKG Snapshot (*.gkg.csv)
        │
        ▼
    GDELT Processor (04_gdelt_processor.ipynb)
        │
        ▼
    GDELT Analysis (05_gdelt_analysis.ipynb)
        │
        ▼
    Event Impact Scoring (06_event_impact_scoring.ipynb)
        │     S_I = TF-IDF Uniqueness × |Tone|
        ▼
    Results Dashboard (07_visualize_results.ipynb)
```

---

## 5. Corrections Made

### Baseline (`src/baseline.py`)

| Issue | Fix |
|---|---|
| No text preprocessing | Added: lowercase, remove punctuation/digits, strip 1–2 char tokens |
| Simple TF (no dampening) | Added `sublinear_tf=True` |
| Unigrams only | Added `ngram_range=(1, 2)` for bigram capture |
| Unsafe `argmax` on scipy matrix | Fixed: `np.asarray(day_sums).flatten().argmax()` |
| No word-boundary regex | Fixed: `(?<![a-z])` prefix/suffix guards |
| Missing vocabulary richness metric | Added `min_df=5` for noise reduction |

### Advanced ML (`src/advanced_ml.py`)

| Issue | Fix |
|---|---|
| `max_iter=15` (insufficient) | Increased to `max_iter=25` |
| Gensim coherence (incompatible with Py 3.14) | Replaced with manual UMass implementation |
| Raw whitespace-split texts for coherence | Coherence now uses the same cleaned vocabulary as the vectoriser |
| Missing per-topic visualisation | Added horizontal bar charts for each topic |
| Missing document-to-topic distribution | Added `topic_document_distribution.png` |
| No `learning_offset` | Added `learning_offset=50.0` |

### Event Impact Scoring (`src/event_impact_scoring.py`)

| Issue | Fix |
|---|---|
| SBERT dependency (unavailable) | Replaced with TF-IDF cosine distance |
| `eval()` on untrusted strings | Replaced with `ast.literal_eval` + safe fallback |

---

## 6. Modules Removed

| File | Reason |
|---|---|
| `src/hybrid_temporal.py` | Duplicate TF-IDF + SBERT hybrid — superseded by dedicated `baseline.py` + `deep_learning.py` |
| BERTopic integration | Replaced by the more controlled SBERT + K-Means approach in `deep_learning.py` |
| `reports/hybrid_impact/` | Output directory of removed hybrid module |

---

## 7. Repository Structure

```
dynamic-trend-event-detector/
├── data/
│   ├── news_headlines.csv          # ABC News headlines (2003–2012)
│   ├── 20260322044500.gkg.csv      # Raw GDELT GKG snapshot
│   └── gdelt_processed.csv         # Processed GDELT output
├── notebooks/                      # All analysis as executed Jupyter notebooks
│   ├── 00_run_all.ipynb            # Master orchestrator
│   ├── 01_eda.ipynb
│   ├── 02_baseline_tfidf.ipynb
│   ├── 03_advanced_ml_lda.ipynb
│   ├── 04_gdelt_processor.ipynb
│   ├── 05_gdelt_analysis.ipynb
│   ├── 06_event_impact_scoring.ipynb  # TF-IDF baseline scorer (see also 09)
│   ├── 07_visualize_results.ipynb
│   ├── 08_deep_learning_kmeans.ipynb  # SBERT K-Means + UMAP + semantic velocity
│   └── 09_gdelt_verification_impact.ipynb  # GDELT rupture verification + SBERT impact
├── src/                            # Python source (mirrors notebooks)
│   ├── eda.py
│   ├── baseline.py
│   ├── advanced_ml.py
│   ├── deep_learning.py            # SBERT K-Means (08)
│   ├── gdelt_processor.py
│   ├── gdelt_analysis.py
│   ├── event_impact_scoring.py     # SBERT-based (09)
│   ├── visualize_results.py
│   └── generate_all_visuals.py
├── reports/
│   ├── pipeline_overview.png       # Full pipeline diagram
│   ├── baseline/                   # Baseline charts + CSV
│   ├── advanced_ml/                # LDA charts, report, CSV
│   ├── eda/                        # EDA charts
│   └── event_impact_scores.csv     # Final impact scores
├── presentation/
├── run_all.py                      # CLI orchestrator (runs all src/*.py)
├── requirements.txt
└── setup.sh
```

---

## 8. How to Run

### One-time setup

```bash
cd dynamic-trend-event-detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=dtdetector_venv --display-name="Python (dtdetector)"
```

### Option A — Run notebooks (recommended)

```bash
# Execute all notebooks in sequence
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=dtdetector_venv \
  notebooks/00_run_all.ipynb
```

Or open and run individual notebooks in JupyterLab:
```bash
jupyter lab
```

### Option B — Run Python scripts directly

```bash
source venv/bin/activate
python run_all.py
```

---

## 9. Results Summary

Trained on **1,244,184 headlines** spanning 2003–2021 (full ABC News archive).

| Model | Key Metric | Value |
|---|---|---|
| TF-IDF | Vocabulary (bigrams) | 5,000 n-grams, min_df=50 |
| TF-IDF | Top global term | `police` (score 11,374) |
| LDA | Topics | **10** (19 years of data) |
| LDA | Vocabulary | 11,956 terms |
| LDA | Perplexity | 5,868.89 |
| LDA | Log-Likelihood | −50,235,963.57 |
| LDA | **Gensim C_V coherence** | **0.3575** [0→1, higher=better] |
| LDA | Gensim UMass coherence | −5.52 (−∞→0, higher=better) |
| LDA | Avg max topic prob | 0.4173 |
| **SBERT K-Means** | Training set | 49,989 (stratified 2,631/year) |
| **SBERT K-Means** | **Optimal K** | **3 clusters** |
| **SBERT K-Means** | **Silhouette Score** | **0.0213** |
| **SBERT K-Means** | **Calinski-Harabasz** | **917.21** |
| **SBERT K-Means** | **Max Semantic Velocity** | **0.6563** (Week 2006-09-11) |
| GDELT | Records processed | 564 |
| Impact Scoring (nb 06 baseline) | Method | TF-IDF cosine × \|tone\| |
| Impact Scoring (nb 09 SBERT) | Method | **SBERT cosine × \|tone\|** — rupture-triggered |

### Discovered Topic Clusters (SBERT K-Means, K=3)

| Cluster | Size | Theme | Top Terms |
|---|---|---|---|
| Topic A | 20,454 | Government / Society | new, council, govt, says, nsw, water, coronavirus, plan |
| Topic B | 15,768 | Sports / Culture | interview, win, australia, cup, australian, day, world |
| Topic C | 13,767 | Crime / Justice | police, man, court, crash, charged, murder, death, killed |

> With 19 years of diverse news (2003–2021), SBERT semantics converge on 3 broad megatopics.
> K=3 was the Silhouette optimum — `interview` in Topic B reflects ABC's *"interview"*-titled  
> broadcast clips, which are a structural feature of the dataset.

### Narrative Rupture Discovery

Max semantic velocity **V_s = 0.6563** at **Week 2006-09-11/2006-09-17** — aligns with:
- Howard Government's **media ownership law** changes introduced to Parliament (Sept 2006)
- North Korea nuclear test preparations (international shift)
- Major domestic sporting + judicial stories cross-cutting that week

All generated charts, CSVs, and reports are in `reports/`.

---

## 10. Roadmap

### Implemented ✅

| Module | Technology |
|---|---|
| TF-IDF Baseline | sklearn `TfidfVectorizer`, bigrams, sublinear TF — 1.24M headlines |
| LDA Advanced ML | sklearn LDA + **Gensim C_V coherence** (0.3575) — 10 topics, 1.24M docs |
| Deep Learning | SBERT `all-MiniLM-L6-v2` + UMAP + K-Means (K=3) — 50K stratified |
| Semantic Velocity | Weekly centroid shift (max rupture V_s=0.6563, Week 2006-09-11) |
| Growth Velocity | Monthly cluster size tracking |
| GDELT Processing | GKG theme + tone parsing |
| GDELT Verification | Rupture-triggered cross-reference with ABC News |
| SBERT Impact Scoring | S_I = SBERT_uniqueness × \|GDELT tone\| |

### Future Upgrades

| Priority | Module | Technology |
|---|---|---|
| Medium | BERTopic upgrade | Replace K-Means with HDBSCAN (auto cluster count) |
| Low | Real-time GDELT stream | Live GDELT API polling every 15 minutes |
| Low | Multi-corpus comparison | Extend to BBC / Reuters datasets |
