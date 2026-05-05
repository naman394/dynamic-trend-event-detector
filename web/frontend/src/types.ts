export interface VelocityPoint {
  week: string
  week_start: string
  v: number
  is_rupture: boolean
}

export interface Cluster {
  Cluster: string
  Size: number
  TopTerms: string
}

export interface GdeltTheme {
  theme: string
  count: number
}

export interface GdeltArticle {
  SOURCECOMMONNAME: string
  DOCUMENTIDENTIFIER: string
  tone_value: number
  theme_list: string
  DATE?: string
  themes_parsed?: string[]
}

export interface WatchdogArticle {
  source: string
  url: string
  tone: number
  impact: number
  uniqueness: number
}

export interface ModelInsight {
  matched_cluster: string
  cluster_confidence: number
  cluster_top_terms: string
  cluster_size: number
  historical_z: number
  historical_context: string
  peak_week: string
  peak_velocity: number
  cluster_scores: { cluster: string; similarity: number }[]
}

export interface WatchdogResult {
  verdict: 'BREAKING' | 'TRENDING' | 'NORMAL' | 'NO DATA'
  z_score: number
  article_count: number
  avg_tone: number
  avg_impact: number
  articles: WatchdogArticle[]
  themes: GdeltTheme[]
  time_series: { time: string; count: number }[]
  model_insight: ModelInsight | null
}

export interface StatusData {
  gdelt_records: number
  velocity_weeks: number
  clusters: number
  max_velocity: number
  peak_rupture_week: string
  headlines_analyzed: number
  years_covered: number
  gdelt_timestamp: string | null
  data_age_seconds: number
  refresh_in_progress: boolean
  next_refresh_in: number | null
}

export interface TrendCluster {
  cluster: string
  share: number
  delta: number
  hist_share: number
  vs_hist: number
  ratio: number
  trending: boolean
  declining: boolean
  top_terms: string
  size: number
}

export interface TrendSnapshot {
  live_vs: number
  rupture_label: 'RUPTURE' | 'ELEVATED' | 'STABLE' | ''
  hist_threshold: number
  total_articles: number
  unique_themes: number
  snapshot_count: number
  timestamp: string
  clusters: TrendCluster[]
}

export interface SpikeEvent {
  topic_id: number
  spike_date: string
  velocity: number
  velocity_z: number
  keywords: string
}

export interface LiveAlertTheme {
  theme: string
  count: number
}

export interface LiveAlertEvent {
  source: string
  url: string
  tone: number
  themes: string[]
}

export interface LiveAlertCluster {
  cluster: string
  delta: number
}

export interface LiveAlert {
  id: string
  detected_at: string
  gdelt_timestamp: string
  live_vs: number
  z_score: number
  rupture_label: 'RUPTURE' | 'ELEVATED'
  top_themes: LiveAlertTheme[]
  top_events: LiveAlertEvent[]
  rising_clusters: LiveAlertCluster[]
  snapshot_count: number
}

export interface HybridClusterRow {
  cluster: string
  size: number
  silhouette_seeded: number
  silhouette_random: number
  top_terms: string
  low_confidence_pct?: number
}

export interface HybridImpactRow {
  source?: string
  url?: string
  tone?: number
  sbert_uniqueness?: number
  lda_rarity?: number
  si_hybrid?: number
  [key: string]: unknown
}

export interface AblationRow {
  Model: string
  Precision: number
  Recall: number
  F1: number
  Mean_Lead_Days: number
  TP?: number
  FP?: number
  FN?: number
}

export interface ValidationRow {
  rank: number
  week: string
  velocity: number
  z_score: number
  matched_event: string
}

export interface ValidationProof {
  rows: ValidationRow[]
  precision: Record<string, number>
  stats: {
    mean: number
    std: number
    threshold_1s: number
    threshold_2s: number
    total_weeks: number
  }
  diagnostic: string
}
