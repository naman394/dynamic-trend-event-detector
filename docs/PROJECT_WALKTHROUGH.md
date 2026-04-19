# Project Walkthrough — Dynamic Trend & Event Detector

This document explains **the order in which work proceeds**, **what each script and notebook does**, and **how the pieces connect**. Use it when presenting or defending the project.

**Jupyter version (step-by-step + “why” for each stage):** [`notebooks/10_project_walkthrough.ipynb`](../notebooks/10_project_walkthrough.ipynb)

---

## 1. End-to-end pipeline (recommended order)

Run from the project root with the virtual environment activated.

| Step | What | Notebook (same logic as `src/`) | Main outputs |
|------|------|--------------------------------|--------------|
| 0 | Optional: orchestrate everything | `00_run_all.ipynb` | Runs downstream steps in order |
| 1 | Explore the corpus | `01_eda.ipynb` | `reports/eda_*.png` |
| 2 | Baseline: TF-IDF trends | `02_baseline_tfidf.ipynb` | `reports/baseline/` |
| 3 | Advanced: LDA topics | `03_advanced_ml_lda.ipynb` | `reports/advanced_ml/` (use **Python 3.11** kernel for Gensim) |
| 4 | Deep learning: SBERT + clusters + velocity | `08_deep_learning_kmeans.ipynb` | `reports/deep_learning/` |
| 5 | GDELT: fetch + parse | `04_gdelt_processor.ipynb` | `data/gdelt_processed.csv` |
| 6 | GDELT: themes & tone | `05_gdelt_analysis.ipynb` | `reports/gdelt_*.png` |
| 7 | Verification + SBERT impact | `09_gdelt_verification_impact.ipynb` | `reports/event_impact_scores.csv`, rupture tables |
| 8 | Optional: TF-IDF impact baseline | `06_event_impact_scoring.ipynb` | Baseline comparison |
| 9 | Dashboard | `07_visualize_results.ipynb` | Consolidated figures |

**Why GDELT comes after deep learning:** rupture weeks from `semantic_velocity.csv` are used to focus verification and interpretation. The live GDELT layer can be refreshed anytime (`./refresh_gdelt.sh`).

**CLI alternative:** `python run_all.py` runs the `src/*.py` scripts in the same logical order (see `run_all.py`).

---

## 2. Data you need

| File | Role |
|------|------|
| `data/abcnews-date-text 2.csv` | Full ABC News headlines (~1.24M rows, 2003–2021). Used by EDA, baseline, LDA, SBERT. |
| `data/gdelt_processed.csv` | Produced by the GDELT processor (live fetch or local `.gkg.csv` fallback). |
| `reports/deep_learning/semantic_velocity.csv` | Produced by `08` / `deep_learning.py`; required before full GDELT verification + impact in `09`. |

---

## 3. File-by-file: Python scripts (`src/`)

### `run_all.py`
- **Purpose:** Single entry point that runs each stage as a subprocess in order.
- **What it does:** Loops over `SCRIPTS` (eda → baseline → advanced_ml → deep_learning → gdelt_processor → gdelt_analysis → event_impact_scoring → visualize_results), prints stdout, creates `reports/` if missing.
- **When to use:** Quick full pipeline without Jupyter.

### `eda.py`
- **Purpose:** Exploratory data analysis on the headline CSV.
- **What it does:** Loads `data/abcnews-date-text 2.csv`, parses dates, plots daily headline counts (`reports/eda_temporal_dist.png`), plots headline length distribution (`reports/eda_text_stats.png`), prints sample rows.

### `baseline.py`
- **Purpose:** **Baseline ML** — frequency-style signal via TF-IDF (not raw counts only; IDF down-weights common words).
- **What it does:**
  - `clean_text`: lowercase, strip non-letters, drop 1–2 character tokens.
  - `TfidfVectorizer`: up to 5k features, bigrams, `min_df=50`, `sublinear_tf=True`.
  - Global term ranking → CSV.
  - Burst analysis: for top global terms, marks presence per day and plots activity.
- **Outputs:** `reports/baseline/` (CSVs, burst chart, etc.).

### `advanced_ml.py`
- **Purpose:** **Advanced ML** — probabilistic topic modelling (LDA).
- **What it does:**
  - Same style of text cleaning as baseline.
  - `CountVectorizer` on raw counts (LDA expects counts, not TF-IDF).
  - Trains `LatentDirichletAllocation` with 10 topics, online learning, large `batch_size` for the big corpus.
  - **Coherence:** tries Gensim **C_V** first; if Gensim is missing, falls back to manual **UMass**.
  - Saves topic words, heatmaps, confidence-style plots, verification text report.
- **Note:** Gensim often needs **Python ≤ 3.12**; notebook `03` uses a 3.11 kernel in your setup.

### `deep_learning.py`
- **Purpose:** **DL track** — sentence embeddings + clustering + temporal metrics.
- **What it does:**
  - Loads full corpus, **stratified sample ~50k** (equal per calendar year) for SBERT cost.
  - **SBERT** (`all-MiniLM-L6-v2`): one 384-d embedding per headline (sentence-level).
  - **UMAP:** 3D and 2D projections for plots.
  - **K-Means:** searches K in a range, picks best by silhouette (script may use K=2…10; notebook may extend to K=15 — keep them in sync if you change behaviour).
  - **Cluster labels:** TF-IDF top terms *per cluster* (human-readable names; clustering is still SBERT-based).
  - **Growth velocity:** cluster sizes over time windows.
  - **Semantic velocity:** weekly cosine shift between embedding centroids → `semantic_velocity.csv` (rupture detection).
  - Optional **GDELT string overlap** for sanity check.
- **Outputs:** `reports/deep_learning/` (plots, CSVs, verification report).

### `gdelt_fetcher.py`
- **Purpose:** Download **live** GDELT GKG data (updates every ~15 minutes).
- **What it does:** Reads `lastupdate.txt` or `masterfilelist.txt` from GDELT’s HTTP site, downloads `.gkg.csv.zip`, parses TSV, extracts theme list and tone, can merge multiple time windows.

### `gdelt_processor.py`
- **Purpose:** Turn GDELT into `data/gdelt_processed.csv`.
- **What it does:** Prefers **live fetch** via `gdelt_fetcher`; if that fails, parses a local `*.gkg.csv`; last resort: existing processed CSV.

### `gdelt_analysis.py`
- **Purpose:** Summarise the processed GDELT table.
- **What it does:** Counts themes, plots top themes, computes average tone per top theme, saves bar charts under `reports/`.

### `event_impact_scoring.py`
- **Purpose:** **Extra mile — event impact** using SBERT + GDELT.
- **What it does:**
  - Reads **rupture windows** from `reports/deep_learning/semantic_velocity.csv`.
  - Encodes GDELT theme strings with SBERT; scores **semantic uniqueness** vs global centroid.
  - **S_I = uniqueness × |tone|** (tone from GDELT).
  - Can summarise headlines around rupture weeks for narrative verification.
- **Outputs:** `reports/event_impact_scores.csv`, plots.

### `visualize_results.py`
- **Purpose:** Legacy helper plots for semantic velocity / impact **if** those CSV paths exist (some paths expect older filenames).
- **What it does:** `visualize_velocity()`, `visualize_impact()` — quick charts in `reports/`.

### `generate_all_visuals.py`
- **Purpose:** Regenerate figures from cached results **without** full retraining (faster than notebooks).
- **What it does:** Rebuilds selected plots for baseline, LDA, deep learning (from `reports/`), impact. **Note:** may still point at `news_headlines.csv` for a fast baseline demo — main pipeline uses the full `abcnews` file in baseline/notebooks.

### `setup.sh` / `generate_pptx.py`
- **Purpose:** Environment setup and optional PowerPoint generation for presentations (separate from the modelling pipeline).

---

## 4. Notebooks (mirror + document the code)

| Notebook | Mirrors | Extra value |
|----------|---------|-------------|
| `00_run_all.ipynb` | Orchestration | Cells to run pipeline in order |
| `01_eda.ipynb` | `eda.py` | Markdown + charts for reports |
| `02_baseline_tfidf.ipynb` | `baseline.py` | Methodology diagrams, tables |
| `03_advanced_ml_lda.ipynb` | `advanced_ml.py` | LDA + Gensim diagrams, coherence explanation |
| `04_gdelt_processor.ipynb` | `gdelt_processor` + `gdelt_fetcher` | Live API explanation |
| `05_gdelt_analysis.ipynb` | `gdelt_analysis.py` | Theme/tone visuals |
| `06_event_impact_scoring.ipynb` | TF-IDF baseline scorer | Compare with `09` |
| `07_visualize_results.ipynb` | Dashboard | Pulls multiple `reports/` outputs |
| `08_deep_learning_kmeans.ipynb` | `deep_learning.py` | Full SBERT/UMAP/K workflow, often **newer** than script (e.g. K range) |
| `09_gdelt_verification_impact.ipynb` | `event_impact_scoring` + verification story | SBERT impact + rupture narrative |

**Rule of thumb:** Notebooks are the **presentation and experiment** layer; `src/` is what `run_all.py` executes. Keep behaviour aligned when you change parameters.

---

## 5. Supporting scripts

### `refresh_gdelt.sh`
- Clears `data/gdelt_cache/` (optional), fetches latest snapshot(s), re-runs notebooks 05 and 09 (or as you configured).
- Use when you want **fresh global news** without re-running SBERT/LDA.

### `requirements.txt`
- Lists `pandas`, `scikit-learn`, `sentence-transformers`, `umap-learn`, `requests`, `gensim` (note version/Python constraints), Jupyter tools.

---

## 6. How to explain “hybrid / edge” without old `hybrid_temporal.py`

The removed **hybrid** module is conceptually replaced by a **pipeline**:

1. **Lexical baseline (TF-IDF)** — fast, interpretable bursts.  
2. **Probabilistic topics (LDA)** — document–topic mixtures.  
3. **Neural clustering + time (SBERT + velocities)** — meaning + when the narrative shifts.  
4. **External layer (GDELT)** — global themes/tone and live refresh.  
5. **Impact scoring** — SBERT on GDELT themes × tone.

That combination **is** the hybrid design — implemented as separate, testable stages rather than one monolithic file.

---

## 7. Quick checklist before a demo

1. [ ] Full headline CSV present (`abcnews-date-text 2.csv`).  
2. [ ] `08` run successfully → `semantic_velocity.csv` exists.  
3. [ ] GDELT: run `04` or `./refresh_gdelt.sh` so `gdelt_processed.csv` is current.  
4. [ ] `09` run for impact scores and verification narrative.  
5. [ ] Open `07` for the dashboard.

---

*Generated for project explanation and viva preparation. Align version numbers in `deep_learning.py` vs notebook if you change K-range or sample size.*
