import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { Rss, RefreshCw, ExternalLink, Globe } from 'lucide-react'
import { getThemes, getArticles, refreshGdelt } from '../api/client'
import type { GdeltTheme, GdeltArticle } from '../types'

function ToneBadge({ tone }: { tone: number }) {
  if (tone < -2) return <span className="text-[10px] font-mono font-semibold text-red-600 bg-red-50 border border-red-200 px-2 py-0.5 rounded-full">{tone.toFixed(1)}</span>
  if (tone >  2) return <span className="text-[10px] font-mono font-semibold text-green-600 bg-green-50 border border-green-200 px-2 py-0.5 rounded-full">+{tone.toFixed(1)}</span>
  return <span className="text-[10px] font-mono text-slate-500 bg-blue-50 border border-blue-200 px-2 py-0.5 rounded-full">{tone.toFixed(1)}</span>
}

export default function LiveFeed() {
  const [themes, setThemes] = useState<GdeltTheme[]>([])
  const [articles, setArticles] = useState<GdeltArticle[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [windows, setWindows] = useState(16)
  const [lastUpdated, setLastUpdated] = useState<string | null>(null)

  async function load() {
    setLoading(true)
    try {
      const [t, a] = await Promise.all([getThemes(25), getArticles(40)])
      setThemes(t); setArticles(a); setLastUpdated(new Date().toLocaleTimeString())
    } finally { setLoading(false) }
  }

  async function doRefresh() {
    setRefreshing(true)
    try { const r = await refreshGdelt(windows); await load(); alert(`Fetched ${r.records.toLocaleString()} records`) }
    catch { alert('Refresh failed') }
    finally { setRefreshing(false) }
  }

  useEffect(() => { load() }, [])

  return (
    <div className="p-8 max-w-6xl">
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-7 h-7 rounded-lg bg-green-600 flex items-center justify-center">
            <Rss size={14} className="text-white" />
          </div>
          <span className="section-label">GDELT Live Intelligence</span>
        </div>
        <div className="flex items-start justify-between gap-6">
          <div>
            <h1 className="page-title mb-2">Global News Feed</h1>
            <p className="page-sub">GDELT 2.0 Global Knowledge Graph — 100+ languages, 200+ countries, updated every 15 minutes.</p>
          </div>
          <div className="flex items-center gap-3 shrink-0 mt-1">
            {lastUpdated && <span className="text-xs text-slate-500 font-mono bg-white border border-blue-100 px-3 py-1.5 rounded-lg">{lastUpdated}</span>}
            <select value={windows} onChange={e => setWindows(Number(e.target.value))} className="input-field text-xs py-2 w-32">
              <option value={1}>15 min</option>
              <option value={4}>1 hour</option>
              <option value={16}>4 hours</option>
              <option value={96}>24 hours</option>
            </select>
            <button onClick={doRefresh} disabled={refreshing} className="btn-primary flex items-center gap-2 text-sm">
              <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
              {refreshing ? 'Fetching…' : 'Refresh'}
            </button>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="bg-white rounded-2xl border border-blue-100 shadow-sm flex items-center justify-center h-64">
          <div className="flex flex-col items-center gap-3 text-slate-500">
            <div className="w-8 h-8 rounded-full border-2 border-blue-100 border-t-accent-blue animate-spin" />
            <span className="text-sm">Loading GDELT data…</span>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="grid grid-cols-3 gap-4">
            {[
              { label:'Records in snapshot', value: articles.length > 0 ? '17K+' : '—',  color:'text-green-600' },
              { label:'Unique themes',        value: themes.length,                         color:'text-slate-900' },
              { label:'Top theme count',      value: themes[0]?.count.toLocaleString()??'—', color:'text-blue-600' },
            ].map(s => (
              <div key={s.label} className="bg-white rounded-2xl border border-blue-100 shadow-sm p-5">
                <div className={`text-2xl font-bold ${s.color}`}>{s.value}</div>
                <div className="text-xs text-slate-500 mt-1 font-medium">{s.label}</div>
              </div>
            ))}
          </div>

          <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-6">
            <h2 className="font-bold text-slate-900 mb-5">Top 25 GDELT Themes — Current Snapshot</h2>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={themes} layout="vertical" margin={{ left:8, right:16 }}>
                <XAxis type="number" hide />
                <YAxis type="category" dataKey="theme" width={210} tick={{ fontSize:10, fill:'#64748b' }} tickLine={false} />
                <Tooltip content={({ active, payload }) =>
                  active && payload?.length
                    ? <div className="bg-white border border-blue-100 shadow-lg rounded-xl px-3 py-2 text-xs">
                        <div className="font-mono text-slate-500 mb-1">{payload[0].payload.theme}</div>
                        <div className="font-bold text-blue-600">{payload[0].value} articles</div>
                      </div>
                    : null
                } />
                <Bar dataKey="count" radius={[0,6,6,0]}>
                  {themes.map((_,i) => <Cell key={i} fill={`rgba(37,99,235,${1-(i/themes.length)*0.65})`} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-6">
            <div className="flex items-center justify-between mb-5">
              <h2 className="font-bold text-slate-900">Recent Articles</h2>
              <span className="text-xs text-slate-500 font-mono bg-blue-50 border border-blue-100 px-3 py-1 rounded-full">{articles.length} shown</span>
            </div>
            <div className="space-y-1.5">
              {articles.map((a, i) => (
                <div key={i} className="flex items-center gap-4 p-3 rounded-xl hover:bg-blue-50 transition-colors group border border-transparent hover:border-blue-100">
                  <Globe size={13} className="text-blue-300 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-slate-900 truncate">{a.SOURCECOMMONNAME||'Unknown source'}</span>
                      {a.DOCUMENTIDENTIFIER?.startsWith('http') && (
                        <a href={a.DOCUMENTIDENTIFIER} target="_blank" rel="noreferrer" className="opacity-0 group-hover:opacity-100 transition-opacity">
                          <ExternalLink size={11} className="text-blue-600" />
                        </a>
                      )}
                    </div>
                    <div className="flex gap-1.5 mt-1 flex-wrap">
                      {(a.themes_parsed??[]).slice(0,4).map(t => (
                        <span key={t} className="text-[10px] font-mono text-slate-500 bg-blue-50 border border-blue-100 px-1.5 py-0.5 rounded">
                          {t.split('_').slice(0,2).join('_')}
                        </span>
                      ))}
                    </div>
                  </div>
                  <ToneBadge tone={a.tone_value??0} />
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
