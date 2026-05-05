import { useEffect, useState, useRef } from 'react'
import { ComposedChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, CartesianGrid } from 'recharts'
import { Zap, ChevronRight, Radio, TrendingUp, ExternalLink } from 'lucide-react'
import { getVelocity, getTopRuptures, getAlerts } from '../api/client'
import type { VelocityPoint, LiveAlert } from '../types'

const ALERT_CFG = {
  RUPTURE:  { bg: 'bg-red-50',   border: 'border-red-200',   pill: 'bg-red-600',   text: 'text-red-700',   icon: Zap,        label: 'RUPTURE'  },
  ELEVATED: { bg: 'bg-amber-50', border: 'border-amber-200', pill: 'bg-amber-500', text: 'text-amber-700', icon: TrendingUp, label: 'ELEVATED' },
}

function LiveAlertsFeed({ alerts, lastPoll }: { alerts: LiveAlert[]; lastPoll: Date | null }) {
  if (alerts.length === 0) {
    return (
      <div className="bg-white border border-blue-100 rounded-2xl p-8 flex flex-col items-center gap-3 shadow-sm mb-8">
        <Radio size={28} className="text-blue-200" />
        <div className="text-sm font-semibold text-slate-700">No alerts detected yet</div>
        <div className="text-xs text-slate-500 text-center max-w-sm">
          Alerts fire automatically every 15 minutes when live GDELT semantic velocity exceeds the
          19-year historical baseline by ≥1σ (ELEVATED) or ≥2σ (RUPTURE).
        </div>
        {lastPoll && (
          <div className="text-[11px] text-slate-400 font-mono">
            Last checked: {lastPoll.toLocaleTimeString()}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-4 mb-8">
      {alerts.map(alert => {
        const cfg = ALERT_CFG[alert.rupture_label]
        const Icon = cfg.icon
        const detectedAt = new Date(alert.detected_at)
        return (
          <div key={alert.id} className={`${cfg.bg} border ${cfg.border} rounded-2xl p-5 shadow-sm`}>
            {/* Header row */}
            <div className="flex items-start justify-between gap-4 mb-4">
              <div className="flex items-center gap-3">
                <div className={`w-9 h-9 rounded-xl ${cfg.pill} flex items-center justify-center shrink-0`}>
                  <Icon size={16} className="text-white" />
                </div>
                <div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className={`text-base font-extrabold tracking-tight ${cfg.text}`}>{cfg.label}</span>
                    <span className="text-[11px] font-mono text-white bg-slate-700 px-2 py-0.5 rounded-full">
                      V_s = {alert.live_vs.toFixed(4)}
                    </span>
                    <span className={`text-[11px] font-mono text-white ${cfg.pill} px-2 py-0.5 rounded-full`}>
                      z = {alert.z_score > 0 ? '+' : ''}{alert.z_score}σ
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-500 mt-0.5 font-mono">
                    {detectedAt.toLocaleString()} · GDELT: {alert.gdelt_timestamp}
                  </div>
                </div>
              </div>
              <div className="text-[10px] text-slate-400 font-mono shrink-0">
                snapshot #{alert.snapshot_count}
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              {/* Top themes */}
              <div>
                <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">
                  Dominant Themes
                </div>
                <div className="space-y-1">
                  {alert.top_themes.slice(0, 5).map((t, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <div className="h-1.5 rounded-full bg-slate-200 flex-1 overflow-hidden">
                        <div
                          className={`h-full rounded-full ${alert.rupture_label === 'RUPTURE' ? 'bg-red-500' : 'bg-amber-500'}`}
                          style={{ width: `${Math.min(100, (t.count / (alert.top_themes[0]?.count || 1)) * 100)}%` }}
                        />
                      </div>
                      <span className="text-[10px] font-mono text-slate-600 truncate w-36">
                        {t.theme.replace(/_/g, ' ').toLowerCase()}
                      </span>
                      <span className="text-[10px] text-slate-400 shrink-0">{t.count}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Rising clusters */}
              <div>
                <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">
                  Surging Clusters
                </div>
                {alert.rising_clusters.length === 0 ? (
                  <div className="text-[11px] text-slate-400">No cluster shift detected</div>
                ) : (
                  <div className="space-y-1.5">
                    {alert.rising_clusters.map((c, i) => (
                      <div key={i} className="flex items-center justify-between bg-white/60 rounded-lg px-2.5 py-1.5 border border-white">
                        <span className="text-[11px] font-semibold text-slate-700">{c.cluster}</span>
                        <span className={`text-[11px] font-mono font-bold ${c.delta > 0 ? 'text-green-600' : 'text-red-600'}`}>
                          +{c.delta.toFixed(1)}pp
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Top events */}
              <div>
                <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">
                  High-Impact Sources
                </div>
                <div className="space-y-1.5">
                  {alert.top_events.slice(0, 4).map((ev, i) => (
                    <div key={i} className="flex items-center gap-2 bg-white/60 rounded-lg px-2.5 py-1.5 border border-white">
                      <span className={`text-[10px] font-mono shrink-0 ${ev.tone < -1 ? 'text-red-600' : ev.tone > 1 ? 'text-green-600' : 'text-slate-500'}`}>
                        {ev.tone > 0 ? '+' : ''}{ev.tone.toFixed(1)}
                      </span>
                      {ev.url.startsWith('http') ? (
                        <a href={ev.url} target="_blank" rel="noreferrer"
                          className="text-[11px] text-blue-600 hover:underline truncate flex items-center gap-1">
                          {ev.source || ev.url}
                          <ExternalLink size={9} className="shrink-0" />
                        </a>
                      ) : (
                        <span className="text-[11px] text-slate-700 truncate">{ev.source || '—'}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

const CustomDot = (props: any) => {
  const { cx, cy, payload } = props
  if (!payload.is_rupture) return null
  return <circle cx={cx} cy={cy} r={4} fill="#dc2626" stroke="#fff" strokeWidth={1.5} />
}

const RTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload as VelocityPoint
  return (
    <div className="bg-white border border-blue-100 shadow-lg rounded-xl px-4 py-3 text-xs">
      <div className="font-mono text-slate-500 mb-2">{d.week}</div>
      <div className={`text-base font-bold ${d.is_rupture ? 'text-red-600' : 'text-blue-600'}`}>V_s = {d.v.toFixed(4)}</div>
      {d.is_rupture && <div className="mt-1.5 flex items-center gap-1.5 text-red-600 font-medium"><Zap size={11} /> Narrative Rupture</div>}
    </div>
  )
}

export default function Ruptures() {
  const [velocity, setVelocity] = useState<VelocityPoint[]>([])
  const [topRuptures, setTopRuptures] = useState<VelocityPoint[]>([])
  const [selected, setSelected] = useState<VelocityPoint | null>(null)
  const [loading, setLoading] = useState(true)
  const [alerts, setAlerts] = useState<LiveAlert[]>([])
  const [lastPoll, setLastPoll] = useState<Date | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    Promise.all([getVelocity(), getTopRuptures(10)])
      .then(([v, r]) => { setVelocity(v); setTopRuptures(r) })
      .finally(() => setLoading(false))
  }, [])

  // Poll live alerts every 30 seconds
  useEffect(() => {
    function fetchAlerts() {
      getAlerts(10)
        .then(data => { setAlerts(data); setLastPoll(new Date()) })
        .catch(() => {})
    }
    fetchAlerts()
    pollRef.current = setInterval(fetchAlerts, 30_000)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  const mean = velocity.length ? velocity.reduce((s,p) => s+p.v,0)/velocity.length : 0
  const std  = velocity.length ? Math.sqrt(velocity.reduce((s,p) => s+(p.v-mean)**2,0)/velocity.length) : 0
  const threshold = mean + std
  const ruptures = velocity.filter(v => v.is_rupture)
  const maxV = velocity.length ? Math.max(...velocity.map(v => v.v)) : 1

  return (
    <div className="p-8 max-w-6xl">
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-7 h-7 rounded-lg bg-red-600 flex items-center justify-center">
            <Zap size={14} className="text-white" />
          </div>
          <span className="section-label">Narrative Ruptures</span>
        </div>
        <h1 className="page-title mb-2">Semantic Velocity Timeline</h1>
        <p className="page-sub">Week-over-week narrative shift across 19 years of news. Red markers = ruptures where the news world changed meaning suddenly.</p>
      </div>

      {/* ── Live Alerts Feed ─────────────────────────────────────────── */}
      <div className="mb-2">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-7 h-7 rounded-lg bg-red-600 flex items-center justify-center">
            <Radio size={14} className="text-white" />
          </div>
          <span className="section-label">Live Alerts</span>
          {lastPoll && (
            <span className="ml-auto text-[11px] text-slate-400 font-mono">
              auto-refreshes every 30s · last: {lastPoll.toLocaleTimeString()}
            </span>
          )}
        </div>
        <p className="page-sub mb-4">
          Automatically fires when live GDELT velocity exceeds the 19-year baseline by ≥1σ.
          Each alert includes the dominant themes, surging topic clusters, and highest-tone sources.
        </p>
        <LiveAlertsFeed alerts={alerts} lastPoll={lastPoll} />
      </div>

      {/* ── Historical velocity stats ──────────────────────────────── */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        {[
          { label:'Weeks Analyzed',   value: velocity.length,                            color:'text-slate-900' },
          { label:'Ruptures Detected', value: ruptures.length,                            color:'text-red-600' },
          { label:'Max Velocity',      value: maxV.toFixed(4),                            color:'text-red-600' },
          { label:'Rupture Rate',      value: `${((ruptures.length/(velocity.length||1))*100).toFixed(1)}%`, color:'text-amber-600' },
        ].map(s => (
          <div key={s.label} className="bg-white rounded-2xl border border-blue-100 shadow-sm p-5">
            <div className={`text-2xl font-bold ${s.color}`}>{s.value}</div>
            <div className="text-xs text-slate-500 mt-1 font-medium">{s.label}</div>
          </div>
        ))}
      </div>

      <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-6 mb-8">
        <div className="flex items-center justify-between mb-5">
          <div>
            <h2 className="font-bold text-slate-900">V_s(t) = 1 − cosine_similarity(centroid_t, centroid_t−1)</h2>
            <p className="text-xs text-slate-500 mt-0.5">Click a spike to inspect · <span className="text-red-600 font-medium">Red dots</span> = rupture threshold exceeded</p>
          </div>
        </div>
        {loading ? (
          <div className="h-64 flex items-center justify-center text-slate-500">
            <div className="w-8 h-8 rounded-full border-2 border-blue-100 border-t-accent-blue animate-spin" />
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={velocity} onClick={e => e?.activePayload && setSelected(e.activePayload[0].payload)}>
              <defs>
                <linearGradient id="rG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#2563eb" stopOpacity={0.12} />
                  <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#dbeafe" vertical={false} />
              <XAxis dataKey="week_start" tick={{ fontSize:10, fill:'#94a3b8' }} tickLine={false} interval={51} />
              <YAxis tick={{ fontSize:10, fill:'#94a3b8' }} tickLine={false} axisLine={false} />
              <Tooltip content={<RTooltip />} />
              <ReferenceLine y={threshold} stroke="#dc2626" strokeDasharray="5 5" strokeOpacity={0.6}
                label={{ value:`Rupture ≥ ${threshold.toFixed(3)}`, position:'right', fontSize:10, fill:'#dc2626' }} />
              <Area type="monotone" dataKey="v" stroke="#2563eb" strokeWidth={2}
                fill="url(#rG)" dot={<CustomDot />}
                activeDot={{ r:6, fill:'#dc2626', stroke:'#fff', strokeWidth:2 }} />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2">
          {selected ? (
            <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-6 animate-slide-up">
              <div className="flex items-center gap-3 mb-4">
                {selected.is_rupture
                  ? <span className="badge-breaking"><Zap size={12} /> Rupture Week</span>
                  : <span className="badge-normal">Stable Week</span>
                }
                <span className="font-mono text-sm text-slate-500">{selected.week}</span>
              </div>
              <div className={`text-4xl font-extrabold mb-1 ${selected.is_rupture ? 'text-red-600' : 'text-blue-600'}`}>
                {selected.v.toFixed(4)}
              </div>
              <div className="text-sm text-slate-500 mb-4">Semantic velocity (V_s)</div>
              <div className="p-4 rounded-xl bg-blue-50 text-xs text-slate-500 font-mono leading-relaxed border border-blue-100">
                {selected.is_rupture
                  ? `⚡ Rupture detected. V_s = ${selected.v.toFixed(4)} exceeds threshold ${threshold.toFixed(4)}.\nThe news corpus shifted significantly — a major real-world event likely caused this.`
                  : `✓ Stable week. V_s = ${selected.v.toFixed(4)} is within normal range (threshold: ${threshold.toFixed(4)}).`
                }
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-10 flex items-center justify-center min-h-[160px]">
              <div className="text-center text-slate-500">
                <Zap size={28} className="mx-auto mb-3 text-blue-200" />
                <div className="text-sm">Click any point on the chart to inspect that week</div>
              </div>
            </div>
          )}
        </div>

        <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold text-slate-900 text-sm">Top 10 Ruptures</h3>
            <Zap size={14} className="text-red-600" />
          </div>
          <div className="space-y-2">
            {topRuptures.map((r, i) => (
              <button key={r.week} onClick={() => setSelected(r)}
                className={`w-full flex items-center gap-3 p-2.5 rounded-xl text-left transition-all ${
                  selected?.week === r.week ? 'bg-red-50 border border-red-200' : 'hover:bg-blue-50 border border-transparent'
                }`}>
                <div className="w-5 h-5 rounded-full bg-red-100 text-red-600 text-[10px] font-bold flex items-center justify-center shrink-0">{i+1}</div>
                <div className="flex-1 min-w-0">
                  <div className="text-[11px] font-mono text-slate-900 truncate">{r.week}</div>
                  <div className="text-[10px] text-red-600 font-semibold">{r.v.toFixed(4)}</div>
                </div>
                <ChevronRight size={12} className="text-slate-500 shrink-0" />
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
