import { useEffect, useState, useCallback } from 'react'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
         ResponsiveContainer, Tooltip, BarChart, Bar, Cell, XAxis, YAxis } from 'recharts'
import { Activity, TrendingUp, TrendingDown, Zap, RefreshCw, Brain, AlertTriangle } from 'lucide-react'
import { getLiveTrends } from '../api/client'
import type { TrendSnapshot } from '../types'

const CLUSTER_EMOJI: Record<string, string> = {
  'Topic A':'🌊','Topic B':'🌾','Topic C':'🏛️','Topic D':'🦠',
  'Topic E':'🌍','Topic F':'🔥','Topic G':'🚨','Topic H':'💹',
  'Topic I':'🗺️','Topic J':'📻','Topic K':'🏥','Topic L':'⚽','Topic M':'⚖️',
}
const CLUSTER_COLORS = [
  '#2563eb','#7c3aed','#16a34a','#dc2626','#d97706',
  '#0891b2','#db2777','#ea580c','#059669','#4f46e5',
  '#ca8a04','#0284c7','#be185d',
]

const RUPTURE_CFG = {
  RUPTURE:  { bg:'bg-red-50',    border:'border-red-200',    text:'text-red-700',    pill:'bg-red-600',    label:'NARRATIVE RUPTURE', icon: AlertTriangle },
  ELEVATED: { bg:'bg-amber-50',  border:'border-amber-200',  text:'text-amber-700',  pill:'bg-amber-500',  label:'ELEVATED VELOCITY', icon: Zap },
  STABLE:   { bg:'bg-green-50',  border:'border-green-200',  text:'text-green-700',  pill:'bg-green-600',  label:'STABLE',            icon: Activity },
  '':       { bg:'bg-blue-50',   border:'border-blue-100',   text:'text-blue-700',   pill:'bg-blue-500',   label:'COMPUTING…',        icon: Activity },
}

function VelocityGauge({ vs, threshold }: { vs: number; threshold: number }) {
  const pct = Math.min((vs / Math.max(threshold * 1.5, 0.01)) * 100, 100)
  const color = vs >= threshold ? '#dc2626' : vs >= threshold * 0.6 ? '#d97706' : '#16a34a'
  return (
    <div className="flex flex-col items-center">
      <div className="relative w-36 h-20 overflow-hidden">
        <svg viewBox="0 0 120 60" className="w-full h-full">
          <path d="M10,60 A50,50 0 0,1 110,60" fill="none" stroke="#dbeafe" strokeWidth="10" strokeLinecap="round"/>
          <path d="M10,60 A50,50 0 0,1 110,60" fill="none" stroke={color} strokeWidth="10"
            strokeLinecap="round" strokeDasharray={`${pct * 1.57} 157`}/>
          <text x="60" y="55" textAnchor="middle" fontSize="13" fontWeight="800" fill={color} fontFamily="monospace">
            {vs.toFixed(4)}
          </text>
        </svg>
      </div>
      <div className="text-[10px] text-slate-500 font-mono mt-0.5">V_s(live)</div>
      <div className="text-[10px] text-slate-500 mt-0.5">threshold: <span className="text-red-500 font-mono">{threshold.toFixed(4)}</span></div>
    </div>
  )
}

export default function Trends() {
  const [data, setData]       = useState<TrendSnapshot | null>(null)
  const [loading, setLoading] = useState(true)
  const [lastFetch, setLastFetch] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(false)

  const fetch = useCallback(async () => {
    setLoading(true)
    try {
      const d = await getLiveTrends()
      setData(d)
      setLastFetch(new Date().toLocaleTimeString())
    } finally { setLoading(false) }
  }, [])

  useEffect(() => { fetch() }, [fetch])

  useEffect(() => {
    if (!autoRefresh) return
    const id = setInterval(fetch, 60_000)
    return () => clearInterval(id)
  }, [autoRefresh, fetch])

  const ruptureCfg = RUPTURE_CFG[data?.rupture_label ?? '']
  const RuptureIcon = ruptureCfg.icon

  const rising   = data?.clusters.filter(c => c.trending)  ?? []
  const falling  = data?.clusters.filter(c => c.declining) ?? []
  const radarData = (data?.clusters ?? []).map(c => ({
    cluster: c.cluster.replace('Topic ', ''),
    value: Math.round(c.share * 10) / 10,
  }))

  return (
    <div className="p-8 max-w-7xl">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center">
            <Brain size={14} className="text-white" />
          </div>
          <span className="section-label">Dynamic Trend Detection</span>
        </div>
        <div className="flex items-start justify-between gap-6">
          <div>
            <h1 className="page-title mb-2">Live Narrative Radar</h1>
            <p className="page-sub">
              Our SBERT K-Means model classifies live GDELT news into 13 trained narrative clusters in real-time.
              Semantic velocity V_s measures how fast the global narrative is shifting — same formula used on 19 years of training data.
            </p>
          </div>
          <div className="flex items-center gap-3 shrink-0 mt-1">
            {lastFetch && <span className="text-xs text-slate-500 font-mono bg-white border border-blue-100 px-3 py-1.5 rounded-lg">{lastFetch}</span>}
            <button onClick={() => setAutoRefresh(a => !a)}
              className={`text-xs px-3 py-1.5 rounded-lg border font-medium transition-all ${autoRefresh ? 'bg-green-600 text-white border-green-600' : 'bg-white text-slate-500 border-blue-100 hover:border-blue-300'}`}>
              {autoRefresh ? '⏱ Auto ON' : '⏱ Auto OFF'}
            </button>
            <button onClick={fetch} disabled={loading}
              className="btn-primary flex items-center gap-2 text-sm">
              <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
              {loading ? 'Detecting…' : 'Detect Now'}
            </button>
          </div>
        </div>
      </div>

      {loading && !data ? (
        <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-16 flex flex-col items-center gap-4">
          <div className="w-12 h-12 rounded-full border-2 border-blue-100 border-t-blue-600 animate-spin" />
          <div className="text-slate-500 text-sm text-center">
            Running SBERT inference on live GDELT snapshot…<br/>
            <span className="text-xs text-slate-400">Classifying news into 13 trained narrative universes</span>
          </div>
        </div>
      ) : data && (
        <>
          {/* Velocity banner */}
          <div className={`${ruptureCfg.bg} border ${ruptureCfg.border} rounded-2xl p-5 mb-6 flex items-center gap-6 flex-wrap`}>
            <div className={`w-11 h-11 rounded-2xl bg-white border ${ruptureCfg.border} flex items-center justify-center shadow-sm`}>
              <RuptureIcon size={20} className={ruptureCfg.text} />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-0.5">
                <span className={`text-xl font-extrabold ${ruptureCfg.text}`}>{ruptureCfg.label}</span>
                <span className={`text-[11px] font-mono text-white ${ruptureCfg.pill} px-2 py-0.5 rounded-full`}>
                  V_s = {data.live_vs.toFixed(4)}
                </span>
              </div>
              <div className="text-sm text-slate-500">
                Live semantic velocity vs trained rupture threshold{' '}
                <span className="font-mono text-red-500">{data.hist_threshold.toFixed(4)}</span>
                {' · '}Snapshot: {data.total_articles.toLocaleString()} articles · {data.unique_themes} unique themes classified
              </div>
            </div>
            <VelocityGauge vs={data.live_vs} threshold={data.hist_threshold} />
          </div>

          {/* Stats row */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            {[
              { label: 'Live articles',    value: data.total_articles.toLocaleString(), color: 'text-slate-900' },
              { label: 'Unique themes',    value: data.unique_themes,                   color: 'text-blue-600'  },
              { label: 'Rising clusters',  value: rising.length,                        color: 'text-green-600' },
              { label: 'Declining clusters', value: falling.length,                     color: 'text-red-600'   },
            ].map(s => (
              <div key={s.label} className="bg-white rounded-2xl border border-blue-100 shadow-sm p-4">
                <div className={`text-2xl font-bold ${s.color}`}>{s.value}</div>
                <div className="text-xs text-slate-500 mt-0.5 font-medium">{s.label}</div>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-3 gap-6 mb-6">
            {/* Radar chart */}
            <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-6">
              <h2 className="font-bold text-slate-900 mb-1 text-sm">Narrative Radar</h2>
              <p className="text-[11px] text-slate-500 mb-4">Share of live news per trained cluster</p>
              <ResponsiveContainer width="100%" height={250}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#dbeafe" />
                  <PolarAngleAxis dataKey="cluster" tick={{ fontSize: 10, fill: '#64748b' }} />
                  <PolarRadiusAxis tick={{ fontSize: 9 }} axisLine={false} />
                  <Radar dataKey="value" stroke="#2563eb" fill="#2563eb" fillOpacity={0.18} strokeWidth={2} />
                  <Tooltip formatter={(v: number) => [`${v}%`, 'Share']} />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Bar chart - cluster shares */}
            <div className="col-span-2 bg-white rounded-2xl border border-blue-100 shadow-sm p-6">
              <h2 className="font-bold text-slate-900 mb-1 text-sm">Cluster Distribution — Current Snapshot</h2>
              <p className="text-[11px] text-slate-500 mb-4">% of live GDELT coverage classified into each trained narrative universe</p>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={data.clusters} layout="vertical" margin={{ left: 8, right: 40 }}>
                  <XAxis type="number" tick={{ fontSize: 10, fill: '#94a3b8' }} tickLine={false} unit="%" />
                  <YAxis type="category" dataKey="cluster" width={70} tick={{ fontSize: 10, fill: '#64748b' }} tickLine={false} />
                  <Tooltip formatter={(v: number) => [`${v}%`, 'Share']} />
                  <Bar dataKey="share" radius={[0, 6, 6, 0]}>
                    {data.clusters.map((c, i) => (
                      <Cell key={c.cluster}
                        fill={c.trending ? '#16a34a' : c.declining ? '#dc2626' : CLUSTER_COLORS[i % 13]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Rising / Falling */}
          <div className="grid grid-cols-2 gap-6 mb-6">
            <div className="bg-white rounded-2xl border border-green-100 shadow-sm p-5">
              <div className="flex items-center gap-2 mb-1">
                <TrendingUp size={16} className="text-green-600" />
                <h3 className="font-bold text-slate-900 text-sm">Rising Narratives</h3>
                <span className="ml-auto text-xs text-green-600 font-mono bg-green-50 border border-green-200 px-2 py-0.5 rounded-full">
                  above 19-yr norm
                </span>
              </div>
              <p className="text-[10px] text-slate-400 mb-4">Clusters with more live coverage than their historical baseline</p>
              {rising.map((c) => (
                <div key={c.cluster} className="flex items-center gap-3 py-2.5 border-b border-green-50 last:border-0">
                  <span className="text-xl">{CLUSTER_EMOJI[c.cluster] ?? '📰'}</span>
                  <div className="flex-1 min-w-0">
                    <div className="font-semibold text-slate-900 text-sm">{c.cluster}</div>
                    <div className="text-[11px] text-slate-500 truncate">{c.top_terms}</div>
                    <div className="text-[10px] text-slate-400 mt-0.5">
                      historical norm: <span className="font-mono">{c.hist_share.toFixed(1)}%</span>
                    </div>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="font-mono text-sm font-bold text-green-600">+{c.vs_hist.toFixed(1)}pp</div>
                    <div className="text-[11px] font-mono text-green-700 bg-green-50 border border-green-200 px-1.5 py-0.5 rounded mt-0.5">
                      {c.ratio.toFixed(1)}× norm
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="bg-white rounded-2xl border border-red-100 shadow-sm p-5">
              <div className="flex items-center gap-2 mb-1">
                <TrendingDown size={16} className="text-red-600" />
                <h3 className="font-bold text-slate-900 text-sm">Declining Narratives</h3>
                <span className="ml-auto text-xs text-red-600 font-mono bg-red-50 border border-red-200 px-2 py-0.5 rounded-full">
                  below 19-yr norm
                </span>
              </div>
              <p className="text-[10px] text-slate-400 mb-4">Clusters with less live coverage than their historical baseline</p>
              {falling.map((c) => (
                <div key={c.cluster} className="flex items-center gap-3 py-2.5 border-b border-red-50 last:border-0">
                  <span className="text-xl">{CLUSTER_EMOJI[c.cluster] ?? '📰'}</span>
                  <div className="flex-1 min-w-0">
                    <div className="font-semibold text-slate-900 text-sm">{c.cluster}</div>
                    <div className="text-[11px] text-slate-500 truncate">{c.top_terms}</div>
                    <div className="text-[10px] text-slate-400 mt-0.5">
                      historical norm: <span className="font-mono">{c.hist_share.toFixed(1)}%</span>
                    </div>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="font-mono text-sm font-bold text-red-600">{c.vs_hist.toFixed(1)}pp</div>
                    <div className="text-[11px] font-mono text-red-700 bg-red-50 border border-red-200 px-1.5 py-0.5 rounded mt-0.5">
                      {c.ratio.toFixed(1)}× norm
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Full cluster grid */}
          <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-6">
            <h2 className="font-bold text-slate-900 mb-1 text-sm">All 13 Narrative Universes — Ranked by Current Coverage</h2>
            <p className="text-[11px] text-slate-500 mb-5">Live GDELT articles classified using our SBERT K-Means model trained on 1.24M ABC News headlines (2003–2021)</p>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {data.clusters.map((c, i) => {
                const color = c.trending ? '#16a34a' : c.declining ? '#dc2626' : CLUSTER_COLORS[
                  ['Topic A','Topic B','Topic C','Topic D','Topic E','Topic F','Topic G',
                   'Topic H','Topic I','Topic J','Topic K','Topic L','Topic M'].indexOf(c.cluster) % 13
                ]
                return (
                  <div key={c.cluster} className="border border-blue-100 rounded-xl p-3.5"
                    style={{ borderLeftColor: color, borderLeftWidth: 3 }}>
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-lg">{CLUSTER_EMOJI[c.cluster] ?? '📰'}</span>
                      <div className="flex-1 min-w-0">
                        <div className="font-bold text-xs text-slate-900">{c.cluster}</div>
                      </div>
                      <div className="flex items-center gap-1">
                        {c.trending && <TrendingUp size={12} className="text-green-600" />}
                        {c.declining && <TrendingDown size={12} className="text-red-600" />}
                        <span className="font-mono text-xs font-bold text-slate-900">{c.share.toFixed(1)}%</span>
                      </div>
                    </div>
                    <div className="h-1.5 bg-blue-50 rounded-full overflow-hidden mb-2">
                      <div className="h-full rounded-full transition-all duration-700" style={{ width: `${Math.min(c.share * 4, 100)}%`, background: color }} />
                    </div>
                    <div className="text-[10px] text-slate-500 truncate">{c.top_terms}</div>
                    <div className={`text-[10px] font-mono font-semibold mt-1 ${c.vs_hist > 0 ? 'text-green-600' : 'text-red-500'}`}>
                      {c.vs_hist > 0 ? '+' : ''}{c.vs_hist.toFixed(1)}pp vs 19-yr norm · {c.ratio.toFixed(1)}×
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
