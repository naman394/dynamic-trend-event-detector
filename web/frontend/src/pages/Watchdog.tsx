import { useState, useEffect, useRef } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis } from 'recharts'
import { Search, Radio, ExternalLink, RefreshCw, TrendingUp, AlertTriangle, CheckCircle, Brain, Layers, History } from 'lucide-react'
import { runWatchdog, getKeywords, refreshGdelt } from '../api/client'
import type { WatchdogResult, ModelInsight } from '../types'

const VERDICT = {
  BREAKING: { label:'BREAKING', textColor:'text-red-700',   bg:'bg-red-50',   border:'border-red-200',   icon:AlertTriangle,  iconColor:'text-red-600',   pill:'bg-red-600' },
  TRENDING:  { label:'TRENDING', textColor:'text-amber-700', bg:'bg-amber-50', border:'border-amber-200', icon:TrendingUp,     iconColor:'text-amber-600', pill:'bg-amber-500' },
  NORMAL:    { label:'NORMAL',   textColor:'text-green-700', bg:'bg-green-50', border:'border-green-200', icon:CheckCircle,   iconColor:'text-green-600', pill:'bg-green-600' },
  'NO DATA': { label:'NO DATA', textColor:'text-slate-500', bg:'bg-blue-50', border:'border-blue-200',  icon:Radio,         iconColor:'text-slate-500',pill:'bg-gray-400' },
}

function VerdictBanner({ result, keyword }: { result: WatchdogResult; keyword: string }) {
  const cfg = VERDICT[result.verdict]
  const Icon = cfg.icon
  return (
    <div className={`${cfg.bg} border ${cfg.border} rounded-2xl p-6 animate-slide-up`}>
      <div className="flex items-center gap-4 flex-wrap">
        <div className={`w-12 h-12 rounded-2xl bg-white border ${cfg.border} flex items-center justify-center shadow-sm`}>
          <Icon size={22} className={cfg.iconColor} />
        </div>
        <div>
          <div className="flex items-center gap-2 mb-0.5">
            <span className={`text-2xl font-extrabold tracking-tight ${cfg.textColor}`}>{cfg.label}</span>
            <span className={`text-[11px] font-mono text-white ${cfg.pill} px-2 py-0.5 rounded-full`}>
              z = {result.z_score.toFixed(2)}σ
            </span>
          </div>
          <div className="text-sm text-slate-500">
            <span className="font-semibold text-slate-900">"{keyword}"</span>
            {result.verdict === 'BREAKING' && ' is spiking — unusually high coverage detected'}
            {result.verdict === 'TRENDING' && ' is elevated above the normal baseline'}
            {result.verdict === 'NORMAL'   && ' is within normal coverage range'}
            {result.verdict === 'NO DATA'  && ' — no matching articles in current snapshot'}
          </div>
        </div>
        <div className="ml-auto flex gap-8">
          {[
            { label:'Articles',  value: result.article_count.toLocaleString(), color:'text-slate-900' },
            { label:'Avg Tone',  value: result.avg_tone.toFixed(2), color: result.avg_tone < -1 ? 'text-red-600' : result.avg_tone > 1 ? 'text-green-600' : 'text-slate-500' },
            { label:'Avg S_I',   value: result.avg_impact.toFixed(4), color:'text-blue-600' },
          ].map(m => (
            <div key={m.label} className="text-right">
              <div className={`text-lg font-bold font-mono ${m.color}`}>{m.value}</div>
              <div className="text-[11px] text-slate-500 uppercase tracking-wide">{m.label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

const CLUSTER_EMOJI: Record<string, string> = {
  'Topic A': '🌊', 'Topic B': '🌾', 'Topic C': '🏛️', 'Topic D': '🦠',
  'Topic E': '🌍', 'Topic F': '🔥', 'Topic G': '🚨', 'Topic H': '💹',
  'Topic I': '🗺️', 'Topic J': '📻', 'Topic K': '🏥', 'Topic L': '⚽', 'Topic M': '⚖️',
}
const CLUSTER_DESC: Record<string, string> = {
  'Topic A': 'Weather, climate events, floods, drought',
  'Topic B': 'Rural, farming, indigenous communities',
  'Topic C': 'Politics, elections, government policy',
  'Topic D': 'COVID-19, vaccines, pandemic response',
  'Topic E': 'International affairs, war, geopolitics',
  'Topic F': 'Bushfires, disasters, emergency events',
  'Topic G': 'Accidents, crashes, crime, missing persons',
  'Topic H': 'Economy, markets, business, budget',
  'Topic I': 'Geography, cities, local news',
  'Topic J': 'Media, interviews, commentary',
  'Topic K': 'Health, hospitals, education, social services',
  'Topic L': 'Sports — cricket, AFL, rugby, tennis',
  'Topic M': 'Crime, courts, charges, murder trials',
}

function ModelInsightPanel({ insight }: { insight: ModelInsight }) {
  const conf = Math.round(insight.cluster_confidence * 100)
  const emoji = CLUSTER_EMOJI[insight.matched_cluster] ?? '📰'
  const desc  = CLUSTER_DESC[insight.matched_cluster] ?? insight.cluster_top_terms

  return (
    <div className="bg-white border border-blue-100 rounded-2xl shadow-sm overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-4 flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-white/20 flex items-center justify-center">
          <Brain size={18} className="text-white" />
        </div>
        <div>
          <div className="font-bold text-white text-sm">Model Insight</div>
          <div className="text-blue-100 text-xs">19-year trained semantic classifier · SBERT K-Means K=13</div>
        </div>
      </div>

      <div className="p-6 grid grid-cols-3 gap-6">
        {/* Cluster match */}
        <div className="col-span-1">
          <div className="text-[11px] font-bold text-slate-500 uppercase tracking-widest mb-3 flex items-center gap-1.5">
            <Layers size={11} /> Narrative Cluster
          </div>
          <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 text-center">
            <div className="text-3xl mb-2">{emoji}</div>
            <div className="font-extrabold text-slate-900 text-sm">{insight.matched_cluster}</div>
            <div className="text-[11px] text-slate-500 mt-1 leading-snug">{desc}</div>
            <div className="mt-3 flex items-center justify-center gap-1.5">
              <div className="h-1.5 flex-1 bg-blue-100 rounded-full overflow-hidden">
                <div className="h-full bg-blue-600 rounded-full" style={{ width: `${conf}%` }} />
              </div>
              <span className="text-[11px] font-mono font-bold text-blue-600">{conf}%</span>
            </div>
            <div className="text-[10px] text-slate-500 mt-1">semantic match</div>
          </div>
          <div className="mt-3 text-[10px] text-slate-500 font-mono bg-blue-50 border border-blue-100 rounded-lg px-3 py-2 leading-relaxed">
            Trained on {insight.cluster_size.toLocaleString()} headlines
          </div>
        </div>

        {/* Top 5 cluster similarities */}
        <div className="col-span-1">
          <div className="text-[11px] font-bold text-slate-500 uppercase tracking-widest mb-3">
            Nearest Clusters
          </div>
          <div className="space-y-2">
            {insight.cluster_scores.map((s, i) => (
              <div key={s.cluster} className="flex items-center gap-2">
                <span className="text-sm w-5 shrink-0">{CLUSTER_EMOJI[s.cluster] ?? '📰'}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-0.5">
                    <span className={`text-[11px] font-medium truncate ${i === 0 ? 'text-blue-600' : 'text-slate-600'}`}>{s.cluster}</span>
                    <span className="text-[10px] font-mono text-slate-500 shrink-0 ml-1">{(s.similarity * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1 bg-blue-50 rounded-full overflow-hidden">
                    <div className={`h-full rounded-full ${i === 0 ? 'bg-blue-600' : 'bg-blue-200'}`}
                      style={{ width: `${s.similarity * 100}%` }} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Historical baseline */}
        <div className="col-span-1">
          <div className="text-[11px] font-bold text-slate-500 uppercase tracking-widest mb-3 flex items-center gap-1.5">
            <History size={11} /> Historical Baseline
          </div>
          <div className="space-y-3">
            <div className="bg-slate-50 border border-blue-100 rounded-xl p-3">
              <div className="text-[10px] text-slate-500 mb-0.5 uppercase tracking-wide">All-time peak week</div>
              <div className="font-mono text-xs font-bold text-slate-900">{insight.peak_week}</div>
              <div className="text-[11px] text-red-600 font-semibold mt-0.5">V_s = {insight.peak_velocity}</div>
            </div>
            <div className={`rounded-xl p-3 border ${insight.historical_z >= 1.5 ? 'bg-red-50 border-red-200' : insight.historical_z >= 0.5 ? 'bg-amber-50 border-amber-200' : 'bg-green-50 border-green-200'}`}>
              <div className="text-[10px] text-slate-500 mb-0.5 uppercase tracking-wide">Historical signal</div>
              <div className={`text-lg font-extrabold font-mono ${insight.historical_z >= 1.5 ? 'text-red-600' : insight.historical_z >= 0.5 ? 'text-amber-600' : 'text-green-600'}`}>
                {insight.historical_z > 0 ? '+' : ''}{insight.historical_z}σ
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">vs 19-year baseline</div>
            </div>
            <div className="text-[10px] text-slate-500 leading-relaxed font-mono bg-slate-50 border border-blue-100 rounded-lg px-3 py-2">
              {insight.historical_context.split('.')[0]}.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Watchdog() {
  const [keyword, setKeyword] = useState('')
  const [result, setResult] = useState<WatchdogResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [hotKeywords, setHotKeywords] = useState<string[]>([])
  const [windows, setWindows] = useState(16)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => { getKeywords().then(setHotKeywords).catch(() => {}); inputRef.current?.focus() }, [])

  async function search(kw?: string) {
    const q = (kw ?? keyword).trim()
    if (!q) return
    if (kw) setKeyword(kw)
    setLoading(true); setResult(null)
    try { setResult(await runWatchdog(q)) }
    catch (e: any) { alert(e?.response?.data?.detail ?? 'Search failed') }
    finally { setLoading(false) }
  }

  async function doRefresh() {
    setRefreshing(true)
    try { const r = await refreshGdelt(windows); alert(`Fetched ${r.records.toLocaleString()} records`); getKeywords().then(setHotKeywords) }
    catch { alert('Refresh failed') }
    finally { setRefreshing(false) }
  }

  return (
    <div className="p-8 max-w-5xl">
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-7 h-7 rounded-lg bg-blue-600 flex items-center justify-center">
            <Radio size={14} className="text-white" />
          </div>
          <span className="section-label">Topic Watchdog</span>
        </div>
        <h1 className="page-title mb-2">Real-Time Keyword Intelligence</h1>
        <p className="page-sub">
          Type any keyword. SBERT scores every matching GDELT article and flags if coverage is{' '}
          <span className="text-red-600 font-semibold">Breaking</span>,{' '}
          <span className="text-amber-600 font-semibold">Trending</span>, or{' '}
          <span className="text-green-600 font-semibold">Normal</span>.
        </p>
      </div>

      {/* Search box */}
      <div className="bg-white border border-blue-200 rounded-2xl p-6 shadow-sm mb-5">
        <div className="flex gap-3 mb-5">
          <div className="relative flex-1">
            <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              ref={inputRef}
              value={keyword}
              onChange={e => setKeyword(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && search()}
              placeholder="Enter a keyword, brand, or topic…"
              className="input-field pl-10"
            />
          </div>
          <button onClick={() => search()} disabled={loading} className="btn-primary min-w-[110px] flex items-center justify-center gap-2">
            {loading ? <><RefreshCw size={14} className="animate-spin" /> Scanning…</> : <><Search size={14} /> Watch</>}
          </button>
        </div>
        {hotKeywords.length > 0 && (
          <div>
            <p className="text-[11px] text-slate-500 font-semibold uppercase tracking-widest mb-2">Hot right now</p>
            <div className="flex flex-wrap gap-2">
              {hotKeywords.map(kw => (
                <button key={kw} onClick={() => search(kw)}
                  className="px-3 py-1 rounded-lg text-xs bg-blue-50 text-blue-600 border border-blue-200 hover:bg-blue-600 hover:text-white transition-all duration-150 capitalize font-medium">
                  {kw}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Refresh panel */}
      <details className="mb-6 bg-white border border-blue-100 rounded-xl p-4 shadow-sm">
        <summary className="text-sm text-slate-500 cursor-pointer select-none hover:text-blue-600 transition-colors font-medium">
          ⚙️ Fetch fresh GDELT data
        </summary>
        <div className="flex items-center gap-3 mt-4">
          <select value={windows} onChange={e => setWindows(Number(e.target.value))} className="input-field flex-1 text-sm py-2">
            <option value={1}>15 min (1 snapshot)</option>
            <option value={4}>1 hour (4 snapshots)</option>
            <option value={16}>4 hours (16 snapshots)</option>
            <option value={96}>24 hours (96 snapshots)</option>
          </select>
          <button onClick={doRefresh} disabled={refreshing} className="btn-ghost text-sm whitespace-nowrap flex items-center gap-2">
            <RefreshCw size={13} className={refreshing ? 'animate-spin' : ''} />
            {refreshing ? 'Fetching…' : 'Refresh'}
          </button>
        </div>
      </details>

      {/* Loading */}
      {loading && (
        <div className="bg-white border border-blue-100 rounded-2xl p-10 flex flex-col items-center gap-4 shadow-sm">
          <div className="w-12 h-12 rounded-full border-3 border-blue-100 border-t-accent-blue animate-spin" style={{ borderWidth: 3 }} />
          <div className="text-slate-500 text-sm">Scanning GDELT feed with SBERT…</div>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <div className="space-y-5 animate-slide-up">
          <VerdictBanner result={result} keyword={keyword} />

          {result.model_insight && (
            <ModelInsightPanel insight={result.model_insight} />
          )}

          {result.verdict !== 'NO DATA' && result.article_count > 0 && (
            <>
              <div className="grid grid-cols-2 gap-5">
                {result.time_series.length > 0 && (
                  <div className="bg-white border border-blue-100 rounded-2xl p-5 shadow-sm">
                    <div className="text-sm font-bold text-slate-900 mb-4">Coverage volume (15-min bins)</div>
                    <ResponsiveContainer width="100%" height={160}>
                      <AreaChart data={result.time_series}>
                        <defs>
                          <linearGradient id="tsG" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%"  stopColor="#2563eb" stopOpacity={0.15} />
                            <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <XAxis dataKey="time" hide />
                        <YAxis hide />
                        <Tooltip content={({ active, payload }) =>
                          active && payload?.length
                            ? <div className="bg-white border border-blue-100 shadow rounded-lg px-2 py-1 text-xs font-mono text-blue-600">{payload[0].value} articles</div>
                            : null
                        } />
                        <Area type="monotone" dataKey="count" stroke="#2563eb" strokeWidth={2} fill="url(#tsG)" dot={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                )}
                {result.themes.length > 0 && (
                  <div className="bg-white border border-blue-100 rounded-2xl p-5 shadow-sm">
                    <div className="text-sm font-bold text-slate-900 mb-4">Dominant themes</div>
                    <ResponsiveContainer width="100%" height={160}>
                      <BarChart data={result.themes.slice(0,8)} layout="vertical">
                        <XAxis type="number" hide />
                        <YAxis type="category" dataKey="theme" width={130} tick={{ fontSize:10, fill:'#64748b' }} />
                        <Tooltip content={({ active, payload }) =>
                          active && payload?.length
                            ? <div className="bg-white border border-blue-100 shadow rounded-lg px-2 py-1 text-xs text-blue-600 font-mono">{payload[0].payload.theme}: {payload[0].value}</div>
                            : null
                        } />
                        <Bar dataKey="count" radius={[0,4,4,0]}>
                          {result.themes.slice(0,8).map((_,i) => <Cell key={i} fill={`rgba(37,99,235,${0.9-i*0.08})`} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>

              <div className="bg-white border border-blue-100 rounded-2xl p-6 shadow-sm">
                <div className="text-sm font-bold text-slate-900 mb-4">Top Articles by SBERT Impact Score (S_I)</div>
                <div className="space-y-2">
                  {result.articles.map((a, i) => (
                    <div key={i} className="flex items-center gap-4 p-3 rounded-xl bg-blue-50/50 hover:bg-blue-50 transition-colors border border-blue-100/0 hover:border-blue-200">
                      <div className="w-6 h-6 rounded-full bg-blue-600 text-white text-[11px] font-bold flex items-center justify-center shrink-0">{i+1}</div>
                      <div className="flex-1 min-w-0">
                        {a.url.startsWith('http')
                          ? <a href={a.url} target="_blank" rel="noreferrer" className="text-sm text-slate-900 hover:text-blue-600 flex items-center gap-1.5 truncate font-medium">
                              {a.source || a.url} <ExternalLink size={11} className="shrink-0 text-slate-500" />
                            </a>
                          : <span className="text-sm text-slate-900 font-medium truncate">{a.source || '—'}</span>
                        }
                      </div>
                      <div className="flex items-center gap-4 shrink-0 text-xs font-mono">
                        <span className={a.tone < -1 ? 'text-red-600' : a.tone > 1 ? 'text-green-600' : 'text-slate-500'}>
                          {a.tone > 0 ? '+' : ''}{a.tone.toFixed(2)}
                        </span>
                        <span className="text-blue-600 font-bold">S_I {a.impact.toFixed(3)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {result.verdict === 'NO DATA' && (
            <div className="bg-white border border-blue-100 rounded-2xl p-10 text-center shadow-sm">
              <Radio size={32} className="mx-auto mb-3 text-blue-200" />
              <div className="font-semibold text-slate-900">No articles found for "{keyword}"</div>
              <div className="text-sm text-slate-500 mt-1">Try a broader keyword or refresh GDELT data above.</div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
