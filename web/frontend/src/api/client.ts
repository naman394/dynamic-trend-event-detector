import axios from 'axios'
import type { StatusData, VelocityPoint, Cluster, GdeltTheme, GdeltArticle, WatchdogResult, SpikeEvent, TrendSnapshot, LiveAlert, HybridClusterRow, HybridImpactRow, AblationRow, ValidationProof } from '../types'

const api = axios.create({ baseURL: '/api', timeout: 90000 })

export const getStatus      = () => api.get<StatusData>('/status').then(r => r.data)
export const getVelocity    = (limit = 0) => api.get<VelocityPoint[]>(`/velocity?limit=${limit}`).then(r => r.data)
export const getTopRuptures = (n = 5) => api.get<VelocityPoint[]>(`/velocity/top?n=${n}`).then(r => r.data)
export const getClusters    = () => api.get<Cluster[]>('/clusters').then(r => r.data)
export const getThemes      = (n = 20) => api.get<GdeltTheme[]>(`/gdelt/themes?top_n=${n}`).then(r => r.data)
export const getArticles    = (n = 30) => api.get<GdeltArticle[]>(`/gdelt/articles?limit=${n}`).then(r => r.data)
export const getKeywords    = () => api.get<string[]>('/gdelt/keywords').then(r => r.data)
export const getSpikes      = () => api.get<SpikeEvent[]>('/spikes').then(r => r.data)
export const refreshGdelt   = (windows = 16) => api.post<{ records: number }>(`/gdelt/refresh?windows=${windows}`).then(r => r.data)
export const runWatchdog    = (keyword: string) => api.post<WatchdogResult>('/watchdog', { keyword }).then(r => r.data)
export const getLiveTrends  = () => api.get<TrendSnapshot>('/trends').then(r => r.data)
export const getAlerts      = (limit = 20) => api.get<LiveAlert[]>(`/alerts?limit=${limit}`).then(r => r.data)
export const getLatestAlert = () => api.get<LiveAlert | null>('/alerts/latest').then(r => r.data)

export const getHybridSummary    = () => api.get<HybridClusterRow[]>('/hybrid/summary').then(r => r.data)
export const getHybridImpact     = (limit = 20) => api.get<HybridImpactRow[]>(`/hybrid/impact?limit=${limit}`).then(r => r.data)
export const getHybridAlpha      = () => api.get<{ alpha: number | null; text: string }>('/hybrid/alpha').then(r => r.data)

export const getAblationTable      = () => api.get<AblationRow[]>('/ablation/table').then(r => r.data)
export const getAblationDiagnostic = () => api.get<{ text: string }>('/ablation/diagnostic').then(r => r.data)

export const getValidationProof    = () => api.get<ValidationProof>('/validation/proof').then(r => r.data)
