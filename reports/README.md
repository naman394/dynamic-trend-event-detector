# Generated outputs (by pipeline stage)

Paths are relative to the repository root. Re-run the listed notebook or script to regenerate.

| Stage | Location | Produced by |
| --- | --- | --- |
| EDA | `eda/` | `01_eda.ipynb` |
| Baseline TF-IDF | `baseline/` | `02_baseline_tfidf.ipynb` |
| LDA (full corpus) | `advanced_ml/` | `03_advanced_ml_lda.ipynb` |
| GDELT live snapshot | `gdelt_*.png`, `data/gdelt_processed.csv` | `04`–`05` notebooks |
| Deep learning (SBERT) | `deep_learning/` | `08_deep_learning_kmeans.ipynb` |
| Impact + ruptures | `event_impact_scores.csv`, `rupture_verification.csv` | `09_gdelt_verification_impact.ipynb` |
| **Topic modeling 2019–2021** | `topic_modeling/11_lda_sbert/` | `11_topic_modeling_lda_bertopic.ipynb` or `src/topic_modeling_lda_bertopic.py` |
| **Spikes + anchor tables** | `topic_modeling/12_spikes_anchors/` | `12_phase4_trend_and_events.ipynb` or `src/spike_events_from_topics_over_time.py`, `src/anchor_ground_truth_report.py` |
| **GDELT BigQuery validation** | `topic_modeling/13_gdelt_bigquery/` | `13_gdelt_spike_bigquery_validation.ipynb` or `src/gdelt_bigquery_spike_validation.py` |

Legacy one-off files (BERTopic HTML, old Phase-4 CSVs) were removed; the current stack uses **LDA + SBERT/HDBSCAN** in notebook 11 only.
