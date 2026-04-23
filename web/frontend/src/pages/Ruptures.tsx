import { useEffect, useState } from 'react'
import { ComposedChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, CartesianGrid } from 'recharts'
import { Zap, ChevronRight } from 'lucide-react'
import { getVelocity, getTopRuptures } from '../api/client'
import type { VelocityPoint } from '../types'

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

  useEffect(() => {
    Promise.all([getVelocity(), getTopRuptures(10)])
      .then(([v, r]) => { setVelocity(v); setTopRuptures(r) })
      .finally(() => setLoading(false))
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
