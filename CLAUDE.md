# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Environment Setup

Two Python environments are used because Gensim requires Python ≤ 3.12:

```bash
# Primary environment (most modules)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=dtdetector_venv --display-name="Python (dtdetector)"

# Python 3.11 environment (LDA coherence via Gensim, notebook 03 and 11)
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=dtdetector311 --display-name="Python 3.11 (dtdetector)"
```

After setup, use `./setup.sh` as a shorthand for the primary environment.

---

## Running the Pipeline

**Full pipeline (Python scripts):**
```bash
source venv/bin/activate
python run_all.py
```

**Notebooks (recommended — executed in order):**
```bash
jupyter lab
# Run: 01 → 02 → 03 → 08 → 04 → 05 → 09 → 06 → 07
```

**Individual scripts:**
```bash
python src/eda.py
python src/baseline.py
python src/advanced_ml.py           # use venv311 for Gensim coherence
python src/deep_learning.py         # SBERT + K-Means, outputs rupture weeks
python src/gdelt_processor.py       # calls gdelt_fetcher.py for live data
python src/gdelt_analysis.py
python src/event_impact_scoring.py  # SBERT S_I scoring
python src/visualize_results.py
```

**Interactive dashboard:**
```bash
streamlit run dashboard.py
```

**Refresh GDELT live data:**
```bash
./refresh_gdelt.sh           # latest 15-min snapshot
./refresh_gdelt.sh 4         # last 1 hour (4 windows)
./refresh_gdelt.sh 16        # last 4 hours
./refresh_gdelt.sh 96        # last 24 hours
```

**LDA vs SBERT topic modeling (Phase 4, notebooks 11–13):**
```bash
source venv311/bin/activate
python src/prepare_df_clean.py
python src/topic_modeling_lda_bertopic.py          # full corpus
python src/topic_modeling_lda_bertopic.py --max-docs 5000  # smoke test
python src/spike_events_from_topics_over_time.py   # requires topics_over_time.csv
```

**GDELT BigQuery spike validation (optional, requires GCP credentials):**
```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
gcloud auth application-default login
python src/gdelt_bigquery_spike_validation.py --max-spikes 10 --window-days 30 --max-plots 5
```

---

## Architecture

### Data Flow

```
data/abcnews-date-text 2.csv  (1,244,184 headlines, 2003–2021)
        │
        ├─► src/eda.py                     → reports/eda/
        ├─► src/baseline.py                → reports/baseline/       (TF-IDF 5K bigrams)
        ├─► src/advanced_ml.py             → reports/advanced_ml/    (LDA 10 topics)
        └─► src/deep_learning.py           → reports/deep_learning/  (SBERT K-Means K=13)
                                                     │
                                             rupture_weeks.pkl
                                                     │
GDELT Live API (every 15 min)                        │
        │                                            │
        ▼                                            ▼
src/gdelt_fetcher.py ──► data/gdelt_processed.csv
        │
        ├─► src/gdelt_analysis.py          → reports/gdelt_*.png
        └─► src/event_impact_scoring.py    ← uses rupture_weeks.pkl
                │  S_I = SBERT_uniqueness × |tone|
                └─► reports/event_impact_scores.csv

Phase 4 (notebook 11–13 / venv311):
  src/prepare_df_clean.py ──► models/df_clean.pkl
  src/topic_modeling_lda_bertopic.py ──► reports/topic_modeling/11_lda_sbert/topics_over_time.csv
  src/spike_events_from_topics_over_time.py ──► reports/topic_modeling/12_spikes_anchors/spike_events.csv
  src/gdelt_bigquery_spike_validation.py ──► reports/topic_modeling/13_gdelt_bigquery/
```

### Module Responsibilities

| File | Role |
|---|---|
| `src/gdelt_fetcher.py` | Downloads live GKG ZIPs from GDELT API; parses TSV into clean DataFrame; caches to `data/gdelt_cache/` |
| `src/gdelt_processor.py` | Calls `gdelt_fetcher.py` and applies additional cleaning |
| `src/deep_learning.py` | Stratified sample (49,989 headlines), SBERT embeddings (384-dim), UMAP, K-Means (K=13), semantic velocity, outputs `rupture_weeks.pkl` |
| `src/event_impact_scoring.py` | SBERT impact score S_I = uniqueness × tone; reads `rupture_weeks.pkl` from deep learning |
| `src/topic_modeling_lda_bertopic.py` | LDA coherence sweep (Gensim C_V) + SBERT/HDBSCAN on 2019–2021 subset; outputs `topics_over_time.csv` and `models/sbert_topic_bundle.joblib` |
| `src/spike_events_from_topics_over_time.py` | Z-score (> 2.5) on per-topic growth velocity from `topics_over_time.csv` → `spike_events.csv` |
| `dashboard.py` | Streamlit dashboard reading all CSV outputs; 5-min cache (`@st.cache_data(ttl=300)`) |

### Key Design Decisions

- **Two venvs by design:** `venv` (default) uses the system Python; `venv311` (Python 3.11) is required for Gensim ≥ 4.3 which is incompatible with Python 3.14. Notebooks 03 and 11 must use the `dtdetector311` kernel.
- **GDELT is always live:** `gdelt_fetcher.py` hits the API on every run. Pass `--local` to `run_all.py` or use the static fallback `data/20260322044500.gkg.csv` only when offline.
- **rupture_weeks.pkl dependency:** `event_impact_scoring.py` (nb 09) reads rupture weeks produced by `deep_learning.py` (nb 08). Run the deep learning step first or the impact scorer falls back to all weeks.
- **Phase 4 is standalone:** Notebooks 11–13 and their `src/` counterparts form a self-contained sub-pipeline on the 2019–2021 clean headlines. They do not depend on notebooks 01–09.
- **Spike detection uses z-score, not absolute threshold:** each topic's velocity series is normalised independently so a spike in a small topic is comparable to one in a large topic.

### Output Directories

```
reports/
├── baseline/           ← TF-IDF charts and top-term CSV
├── advanced_ml/        ← LDA topic charts, coherence, heatmap
├── deep_learning/      ← SBERT scatter, velocity CSVs, cluster summary
├── eda/                ← Temporal and text-length charts
├── topic_modeling/
│   ├── 11_lda_sbert/   ← LDA sweep, SBERT HTML, topics_over_time.csv
│   ├── 12_spikes_anchors/ ← spike_events.csv, anchor_ground_truth*.csv
│   └── 13_gdelt_bigquery/ ← gdelt_validation.csv/.html, GDELT_ANCHOR_EVENT_VALIDATION.md
├── event_impact_scores.csv
└── rupture_verification.csv
```

---

## Dataset

- **Primary corpus:** `data/abcnews-date-text 2.csv` — 1,244,184 headlines, 2003–2021, columns: `publish_date` (YYYYMMDD int), `headline_text`
- **Legacy sample:** `data/news_headlines.csv` — 20K headlines, 2003 only (used in older notebooks)
- **GDELT processed:** `data/gdelt_processed.csv` — most recent live fetch output
- **Phase 4 clean corpus:** `models/df_clean.pkl` — 2019–2021 headlines after `prepare_df_clean.py`

---

## Advanced Modules (src/)

Four additional modules added after the core pipeline:

| Module | Run command | Output |
|---|---|---|
| `ner_entity_tracker.py` | `python src/ner_entity_tracker.py` | `reports/ner_weekly_freq.csv`, `reports/ner_entities.csv` (requires `en_core_web_sm`: `python -m spacy download en_core_web_sm`) |
| `lstm_anomaly_detector.py` | `python src/lstm_anomaly_detector.py` | `reports/lstm_anomalies.csv` (requires PyTorch) |
| `granger_causality.py` | `python src/granger_causality.py` | `reports/granger_causality.html`, `reports/granger_edges.csv` |
| `multilingual_sbert.py` | `python src/multilingual_sbert.py` | `reports/multilingual_signals.csv` |

These outputs are consumed by the dashboard's optional sidebar tabs.
