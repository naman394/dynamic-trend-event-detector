# Dynamic Trend & Event Detector

**Project:** Automated detection of narrative trends and event ruptures in news corpora using statistical, probabilistic, and deep learning NLP models.

**Authors:** Navnit Naman (230085) & Kanhaiya Kumar (230062) — Newton School of Technology, Rishihood University

**Step-by-step pipeline & file-by-file explanation (for viva / presentation):**  
- Notebook: [`notebooks/10_project_walkthrough.ipynb`](notebooks/10_project_walkthrough.ipynb) (interactive, **why** at each step)  
- Markdown: [`docs/PROJECT_WALKTHROUGH.md`](docs/PROJECT_WALKTHROUGH.md)

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Project Scope](#2-project-scope)
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
- Identify the most discriminative terms and their temporal burst patterns (Baseline TF-IDF)
- Discover latent thematic groups using probabilistic topic modelling (Advanced ML — LDA)
- Detect semantic topic clusters and narrative ruptures using deep learning (SBERT + K-Means)
- Cross-reference with global real-time news via the GDELT Global Knowledge Graph (live API feed)
- Score events by semantic uniqueness and tonal intensity (SBERT Impact Scoring)

### Dataset
| Property | Value |
|---|---|
| Source | ABC News archive (full corpus) |
| Size | **1,244,184 headlines** |
| Period | **2003–2021** (19 years) |
| Format | `publish_date` (YYYYMMDD) + `headline_text` |

### GDELT Live Feed
| Property | Value |
|---|---|
| Source | GDELT 2.0 Global Knowledge Graph |
| Update frequency | **Every 15 minutes** (live API) |
| Coverage | 100+ languages, 200+ countries |
| Fields used | Themes (V2THEMES), Tone (V2TONE), Source, URL |

---

## 2. Project Scope

All modules are implemented and fully operational:

| # | Notebook | Module | Description |
|---|---|---|---|
| 01 | `01_eda.ipynb` | EDA | Temporal distribution, text length stats across 1.24M headlines |
| 02 | `02_baseline_tfidf.ipynb` | Baseline (TF-IDF) | 5K-vocabulary bigram ranking, burst detection, per-day top term |
| 03 | `03_advanced_ml_lda.ipynb` | Advanced ML (LDA) | 10-topic model, **Gensim C_V coherence** (0.3575), discrimination heatmap |
| 04 | `04_gdelt_processor.ipynb` | GDELT Processor (Live) | **Live fetch from GDELT API** → parse GKG → clean CSV |
| 05 | `05_gdelt_analysis.ipynb` | GDELT Analysis | Theme frequency, per-theme sentiment from live snapshot |
| 06 | `06_event_impact_scoring.ipynb` | Impact Scoring (TF-IDF baseline) | TF-IDF cosine distance × \|tone\| — lightweight reference |
| 07 | `07_visualize_results.ipynb` | Results Dashboard | Consolidated output from all models |
| 08 | `08_deep_learning_kmeans.ipynb` | **Deep Learning (SBERT + K-Means)** | K-Means on SBERT sentence embeddings, UMAP 3D/2D, **K=13** topic clusters, semantic velocity, growth velocity |
| 09 | `09_gdelt_verification_impact.ipynb` | **GDELT Verification & SBERT Impact** | Rupture-triggered GDELT verification, SBERT S_I scoring |
| 10 | `10_project_walkthrough.ipynb` | **Project walkthrough** | Step-by-step guide + **why** each stage exists (no training) |
| 11 | `11_topic_modeling_lda_bertopic.ipynb` | **LDA vs SBERT** | 2019–2021: Gensim C_V sweep + **SBERT** (`all-MiniLM-L6-v2`) + UMAP + HDBSCAN, TF-IDF topic words, **`topics_over_time.csv`**, theme-keyword tables (spikes → notebook **12**) |
| 12 | `12_phase4_trend_and_events.ipynb` | **Spikes + anchors** | After notebook 11: builds `spike_events.csv` (per-topic z-score on growth-velocity > 2.5) and optional `anchor_ground_truth_detection.csv` |

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

**Parameters (tuned for 1.24M corpus):** `max_features=5000`, `min_df=50`, `sublinear_tf=True`

**Limitations:** Treats each word as independent; cannot group semantically related terms. Addressed by LDA.

### 3.2 Advanced ML — LDA (Latent Dirichlet Allocation)

**Problem addressed:** Discover latent thematic groups; each document is a mixture of topics.

**Why LDA:**

$$p(w \mid \alpha, \beta) = \int p(\theta \mid \alpha) \prod_{n=1}^{N} \sum_{z_n} p(z_n \mid \theta)\, p(w_n \mid z_n, \beta)\, d\theta$$

- Probabilistic model: accounts for multi-topic documents (realistic for news)
- Raw count input (unlike TF-IDF): matches the Dirichlet-Multinomial generative assumption
- Allows tracking which topics rise/fall over time

**Parameters (tuned for 1.24M corpus):** `n_components=10`, `max_features=15000`, `min_df=50`, `batch_size=4096`, `max_iter=20`

**Coherence evaluation:**
- **Gensim C_V coherence: 0.3575** [0→1, higher=better] — primary metric (correlates with human judgement)
- Gensim UMass coherence: −5.52 [−∞→0, higher=better] — secondary metric
- Fallback: manual UMass implementation if Gensim is unavailable (Python 3.14 incompatibility)

**Coherence metric — UMass:**

$$C_{\text{UMass}}(t) = \sum_{i=2}^{N} \sum_{j=1}^{i-1} \log \frac{D(w_i, w_j) + 1}{D(w_j)}$$

### 3.3 Deep Learning — K-Means on SBERT Sentence Embeddings

**Problem addressed:** LDA uses bag-of-words; semantically equivalent phrases like *"troops deployed"* and *"soldiers sent"* are invisible to it. SBERT solves this using **sentence embeddings**.

**How SBERT works:**
1. Full headline (sentence) → Tokenize into subword tokens
2. Pass through 6 Transformer layers with self-attention (every token attends to every other token)
3. Mean pooling → single 384-dim vector representing the entire sentence's meaning

> SBERT embeds **whole sentences**, not individual words. K-Means clusters these sentence vectors.
> TF-IDF is then used only to **label** each cluster with human-readable top terms.

**Architecture:**

```
Headlines → SBERT (384-dim sentence embeddings) → UMAP (3D/2D) → K-Means (K=13)
                                                                      │
                                    TF-IDF labeling (top terms) ◄─────┤
                                    Growth Velocity ◄─────────────────┤
                                    Semantic Velocity ◄───────────────┤
                                    GDELT Verification ◄──────────────┘
```

**Sampling strategy:** Stratified sample of **49,989 headlines** (≈2,631/year across all 19 years) to ensure equal coverage of every news era from 2003–2021.

**K selection:** Silhouette score search over K=2→15. K=13 was the optimum — with 19 years of diverse data, the corpus naturally contains 13 distinct semantic regions.

**Semantic Velocity:**
$$V_s(t) = 1 - \cos\text{-sim}\bigl(\bar{e}_{t-1},\, \bar{e}_t\bigr)$$

A spike in $V_s$ signals a **Narrative Rupture** — the corpus shifted topics suddenly due to a real-world event.

### 3.4 GDELT Live Integration

**Why GDELT:** Provides real-time geopolitical context (100+ languages, 15-min updates) with a structured theme taxonomy and quantitative tone score — dimensions not available in the raw headline corpus.

**Live feed:** `src/gdelt_fetcher.py` queries `http://data.gdeltproject.org/gdeltv2/lastupdate.txt` and downloads the newest GKG snapshot every time the pipeline runs. No stale static files.

| Mode | Command | Coverage |
|---|---|---|
| Latest snapshot | `./refresh_gdelt.sh` | Last 15 minutes |
| Last 1 hour | `./refresh_gdelt.sh 4` | 4 × 15-min windows |
| Last 4 hours | `./refresh_gdelt.sh 16` | 16 windows (~8K records) |
| Last 24 hours | `./refresh_gdelt.sh 96` | 96 windows (~50K records) |

### 3.5 Event Impact Scoring

**SBERT-based formula (notebook 09 — production):**

$$S_I = \underbrace{(1 - \cos\text{-sim}(\vec{v}_d^{\text{SBERT}}, \vec{c}_{\text{global}}^{\text{SBERT}}))}_{\text{Semantic Uniqueness}} \times \underbrace{|\text{tone}_d|}_{\text{Tonal Intensity}}$$

**TF-IDF baseline formula (notebook 06 — reference):**

$$S_I = \underbrace{(1 - \cos\text{-sim}(\vec{v}_d^{\text{TF-IDF}}, \vec{c}_{\text{global}}^{\text{TF-IDF}}))}_{\text{Semantic Uniqueness}} \times \underbrace{|\text{tone}_d|}_{\text{Tonal Intensity}}$$

| Property | Notebook 06 (TF-IDF) | Notebook 09 (SBERT) |
|---|---|---|
| Semantic method | TF-IDF cosine distance | **SBERT embedding distance** |
| Synonym handling | No | **Yes** (semantically aware) |
| DL model link | Independent | **Uses rupture weeks from DL model** |
| Recommended for | Quick baseline reference | **Production scoring** |

---

## 4. Architecture

```
Full ABC News Corpus (1,244,184 headlines, 2003–2021)
        │
        ▼
    EDA (01_eda.ipynb)
        │
        ├──► Baseline TF-IDF (02_baseline_tfidf.ipynb)
        │         └── 5K bigrams, burst detection, per-day top term
        │
        ├──► Advanced ML LDA (03_advanced_ml_lda.ipynb)
        │         └── 10 topics, Gensim C_V=0.3575, heatmap, confidence
        │
        └──► Deep Learning: SBERT K-Means (08_deep_learning_kmeans.ipynb)
                  ├── SBERT (384-dim sentences) → UMAP (3D+2D) → K-Means (K=13)
                  ├── Growth Velocity (monthly cluster size)
                  ├── Semantic Velocity V_s (weekly centroid shift)
                  └── 13 topic clusters across 19 years
                           │
GDELT Live API (updated every 15 min)
        │
        ▼
    GDELT Fetcher (src/gdelt_fetcher.py) ← downloads newest GKG
        │
        ▼
    GDELT Processor (04_gdelt_processor.ipynb)
        │
        ▼
    GDELT Analysis (05_gdelt_analysis.ipynb)
        │
        ├──► TF-IDF Impact Scorer (06_event_impact_scoring.ipynb) ← baseline
        │
        └──► SBERT Impact Scorer + Rupture Verification (09_gdelt_verification_impact.ipynb)
                  │     S_I = SBERT_Uniqueness × |Tone|
                  │     Uses semantic velocity ruptures from DL model
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
| `min_df=5` too permissive for 1.24M | Increased to `min_df=50` |
| `max_features=1000` too small | Increased to `max_features=5000` |

### Advanced ML (`src/advanced_ml.py`)

| Issue | Fix |
|---|---|
| `max_iter=15` (insufficient) | Increased to `max_iter=20` |
| Gensim coherence (incompatible with Py 3.14) | Gensim installed in Python 3.11 venv; fallback manual UMass if unavailable |
| Raw whitespace-split texts for coherence | Coherence now uses the same cleaned vocabulary as the vectoriser |
| Missing per-topic visualisation | Added horizontal bar charts for each topic |
| Missing document-to-topic distribution | Added `topic_document_distribution.png` |
| No `learning_offset` | Added `learning_offset=50.0` |
| Only 5 topics for 19-year corpus | Increased to `n_components=10` |
| No `batch_size` tuning | Added `batch_size=4096` for faster online LDA on 1.24M docs |
| `max_features=5000` | Increased to `max_features=15000` for richer vocabulary |

### Deep Learning (`src/deep_learning.py`)

| Issue | Fix |
|---|---|
| Only 2003 data (20K sample) | Switched to full 1.24M corpus with stratified 50K sample |
| Random sampling missed years | Stratified: 2,631 headlines/year across all 19 years |
| K search only 2–10 (found false K=3) | Extended to K=2–15; true optimum K=13 |
| `n_neighbors=15` too low for 50K | Increased to `n_neighbors=30` |
| `sample_size=2000` for Silhouette | Increased to `sample_size=5000` |

### GDELT Pipeline

| Issue | Fix |
|---|---|
| Static .gkg.csv file (stale within 15 min) | New `gdelt_fetcher.py` — live API download every run |
| Event Impact Scoring used TF-IDF only | Notebook 09: SBERT-based scoring + rupture verification |

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
│   ├── abcnews-date-text 2.csv    # Full ABC News corpus (1.24M headlines, 2003–2021)
│   ├── news_headlines.csv          # Legacy 20K sample (2003 only)
│   ├── 20260322044500.gkg.csv      # Static GDELT snapshot (fallback)
│   ├── gdelt_processed.csv         # Most recent processed GDELT output (live)
│   └── gdelt_cache/                # Auto-cached downloaded GDELT ZIPs
├── notebooks/                      # All analysis as executed Jupyter notebooks
│   ├── 00_run_all.ipynb            # Master orchestrator
│   ├── 01_eda.ipynb                # EDA on full 1.24M dataset
│   ├── 02_baseline_tfidf.ipynb     # TF-IDF (5K bigrams, min_df=50)
│   ├── 03_advanced_ml_lda.ipynb    # LDA (10 topics, Gensim C_V=0.3575) [kernel: dtdetector311]
│   ├── 04_gdelt_processor.ipynb    # GDELT live fetch + parse
│   ├── 05_gdelt_analysis.ipynb     # Theme frequency + sentiment
│   ├── 06_event_impact_scoring.ipynb  # TF-IDF baseline scorer (see also 09)
│   ├── 07_visualize_results.ipynb  # Consolidated results dashboard
│   ├── 08_deep_learning_kmeans.ipynb  # SBERT + UMAP + K-Means (K=13)
│   ├── 09_gdelt_verification_impact.ipynb  # GDELT rupture verification + SBERT impact
│   └── 10_project_walkthrough.ipynb          # Step-by-step + rationale (viva / onboarding)
├── src/                            # Python source (mirrors notebooks)
│   ├── eda.py
│   ├── baseline.py
│   ├── advanced_ml.py
│   ├── deep_learning.py            # SBERT K-Means (mirrors notebook 08)
│   ├── gdelt_fetcher.py            # Live GDELT API fetcher (new)
│   ├── gdelt_processor.py          # GKG parser (calls fetcher)
│   ├── gdelt_analysis.py
│   ├── event_impact_scoring.py     # SBERT-based impact (mirrors notebook 09)
│   ├── visualize_results.py
│   └── generate_all_visuals.py     # Regenerate all charts from cached results
├── reports/
│   ├── pipeline_overview.png       # Full pipeline diagram
│   ├── baseline/                   # Baseline charts + CSV
│   ├── advanced_ml/                # LDA charts, report, CSV
│   ├── deep_learning/              # SBERT scatter plots, velocity charts, cluster CSV
│   ├── eda/                        # EDA charts
│   ├── event_impact_scores.csv     # SBERT-scored impact results
│   └── rupture_verification.csv    # Narrative rupture headline evidence
├── refresh_gdelt.sh                # One-command live GDELT refresh script
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

**For Gensim coherence** (requires Python ≤ 3.12):
```bash
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt gensim>=4.3
python -m ipykernel install --user --name=dtdetector311 --display-name="Python 3.11 (dtdetector)"
```

### Option A — Run notebooks (recommended)

```bash
jupyter lab
# Then run notebooks 01 → 02 → 03 → 08 → 04 → 05 → 09 → 06 → 07
```

### Option B — Run Python scripts directly

```bash
source venv/bin/activate
python run_all.py
```

### Option C — Refresh only GDELT (live data)

```bash
./refresh_gdelt.sh           # latest 15-min snapshot
./refresh_gdelt.sh 4         # last 1 hour
./refresh_gdelt.sh 16        # last 4 hours
./refresh_gdelt.sh 96        # last 24 hours
```

### Option D — LDA vs BERTopic on 2019–2021 clean headlines

Use the **Python 3.11** environment (`venv311`) so Gensim, BERTopic, and HDBSCAN install cleanly.

**Recommended:** run **[`notebooks/11_topic_modeling_lda_bertopic.ipynb`](notebooks/11_topic_modeling_lda_bertopic.ipynb)** — the full pipeline lives in the notebook (no script calls). Set `MAX_DOCS = None` in the first code cell for the full ~92k corpus, or an integer (e.g. `5000`) for a faster run.

Optional CLI equivalents (same outputs): `python src/prepare_df_clean.py` then `python src/topic_modeling_lda_bertopic.py` (add `--max-docs N` for a smoke test).

Outputs land under `reports/topic_modeling/` (coherence sweep plot, CSVs, Plotly HTML) and `models/sbert_topic_bundle.joblib`.

**Spike events:** after `topics_over_time.csv` exists, run  
`python src/spike_events_from_topics_over_time.py`  
to write `reports/topic_modeling/spike_events.csv` (per-topic **z-score on growth-velocity** > 2.5 + keywords). Run **notebook 12** after notebook 11 (or this CLI alone if `topics_over_time.csv` already exists).

---

## 9. Results Summary

Trained on **1,244,184 headlines** spanning **2003–2021** (full ABC News archive).

### Model Metrics

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
| **SBERT K-Means** | Training set | 49,989 headlines (stratified 2,631/year) |
| **SBERT K-Means** | **Optimal K** | **13 clusters** (search range K=2–15) |
| **SBERT K-Means** | **Silhouette Score** | **0.0226** |
| **SBERT K-Means** | **Calinski-Harabasz** | **454.89** |
| **SBERT K-Means** | **Max Semantic Velocity** | **0.6563** (Week 2006-09-11) |
| GDELT | Data source | **Live API** (updated every 15 min) |
| Impact Scoring (nb 06) | Method | TF-IDF cosine × \|tone\| (baseline) |
| Impact Scoring (nb 09) | Method | **SBERT cosine × \|tone\|** (production, rupture-triggered) |

### Discovered Topic Clusters (SBERT K-Means, K=13)

| Cluster | Size | Theme | Top Terms |
|---|---|---|---|
| Topic A | 2,403 | Water / Climate | water, rain, drought, flood, weather, cyclone, storm |
| Topic B | 4,718 | Rural / Agriculture | farmers, cattle, rural, indigenous, wa, farm, search |
| Topic C | 4,488 | Politics / Elections | council, election, govt, government, mp, minister, labor |
| Topic D | 1,399 | **COVID / Pandemics** | coronavirus, covid, vaccine, cases, flu, outbreak, quarantine |
| Topic E | 3,879 | International / War | iraq, china, trump, korea, iran, war, attack |
| Topic F | 1,354 | **Bushfires** | bushfire, blaze, fires, firefighters, bushfires, arson |
| Topic G | 3,072 | Fatal Incidents | crash, death, dies, killed, dead, man, car |
| Topic H | 4,991 | Economy / Business | market, price, industry, business, budget, pay, deal |
| Topic I | 4,660 | Geography / Cities | australia, sydney, nsw, melbourne, queensland, adelaide |
| Topic J | 4,911 | ABC Interviews | interview, speaks, abc, news, drum, talks, country |
| Topic K | 3,450 | Health / Education | health, hospital, school, care, doctors, cancer, funding |
| Topic L | 5,248 | Sports | win, cup, wins, world, final, england, league, open |
| Topic M | 5,416 | Crime / Courts | police, man, court, charged, murder, charges, accused |

> K=13 was selected via Silhouette score search (K=2–15). With 19 years of diverse Australian
> news (2003–2021), the corpus naturally separates into 13 distinct semantic regions — including
> era-specific clusters like COVID (2020–2021) and Bushfires (2019–2020 Black Summer).

### Narrative Rupture Discovery

Max semantic velocity **V_s = 0.6563** at **Week 2006-09-11/2006-09-17** — aligns with:
- Howard Government's **media ownership law** changes introduced to Parliament (Sept 2006)
- North Korea nuclear test preparations (international shift)
- Major domestic sporting + judicial stories cross-cutting that week

### GDELT Live Feed

GDELT data is fetched live from the GDELT 2.0 API every time the pipeline runs.
Use `./refresh_gdelt.sh` to pull the latest 15-minute snapshot and re-run all
GDELT-dependent notebooks (05, 09) with fresh data.

All generated charts, CSVs, and reports are in `reports/`.

---

## 10. Roadmap

### Implemented

| Module | Technology |
|---|---|
| TF-IDF Baseline | sklearn `TfidfVectorizer`, bigrams, sublinear TF — 1.24M headlines |
| LDA Advanced ML | sklearn LDA + **Gensim C_V coherence** (0.3575) — 10 topics, 1.24M docs |
| Deep Learning | SBERT `all-MiniLM-L6-v2` + UMAP + K-Means (**K=13**) — 50K stratified |
| Semantic Velocity | Weekly centroid shift (max rupture V_s=0.6563, Week 2006-09-11) |
| Growth Velocity | Monthly cluster size tracking across 19 years |
| GDELT Live Feed | `gdelt_fetcher.py` — downloads from API every 15 minutes |
| GDELT Verification | Rupture-triggered cross-reference with ABC News headlines |
| SBERT Impact Scoring | S_I = SBERT_uniqueness × \|GDELT tone\| |
| Refresh Script | `refresh_gdelt.sh` — one-command live data refresh |

### Future Upgrades

| Priority | Module | Technology |
|---|---|---|
| Medium | BERTopic upgrade | Replace K-Means with HDBSCAN (auto cluster count) |
| Low | Multi-corpus comparison | Extend to BBC / Reuters datasets |
