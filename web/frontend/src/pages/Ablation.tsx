import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Cell,
} from 'recharts'
import { FlaskConical, CheckCircle, AlertTriangle, TrendingUp } from 'lucide-react'
import { getAblationTable, getAblationDiagnostic } from '../api/client'
import type { AblationRow } from '../types'

const MODEL_COLORS: Record<string, string> = {
  'TF-IDF': '#94a3b8',
  'LDA':    '#7c3aed',
  'SBERT':  '#2563eb',
  'Hybrid': '#16a34a',
}

const MODEL_DESC: Record<string, string> = {
  'TF-IDF': 'Bag-of-words keyword frequency baseline — no semantics, no context',
  'LDA':    'Probabilistic topic model — latent topics, no embeddings',
  'SBERT':  'Sentence transformer embeddings + K-Means rupture detection',
  'Hybrid': 'LDA-seeded SBERT + learned α* fusion — best of both worlds',
}

const ANCHORS = [
  { name: 'Black Summer',      range: '2019-11-01 → 2020-01-31', emoji: '🔥' },
  { name: 'First AU COVID',    range: '2020-01-20 → 2020-02-05', emoji: '🦠' },
  { name: 'National Lockdown', range: '2020-03-01 → 2020-04-15', emoji: '🏠' },
]

function MetricBar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0
  return (
    <div className="flex items-center gap-3">
      <div className="w-14 text-[11px] font-bold text-slate-900 shrink-0">{label}</div>
      <div className="flex-1 h-2 bg-blue-50 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="w-12 text-right font-mono text-xs font-bold" style={{ color }}>{value.toFixed(3)}</div>
    </div>
  )
}

function ScoreCard({ row }: { row: AblationRow }) {
  const color = MODEL_COLORS[row.Model] ?? '#64748b'
  const isHybrid = row.Model === 'Hybrid'
  return (
    <div className={`bg-white rounded-2xl border-2 shadow-sm p-5 transition-all ${isHybrid ? 'border-green-400 shadow-green-100' : 'border-blue-100'}`}>
      <div className="flex items-center gap-3 mb-4">
        <div className="w-3 h-3 rounded-full shrink-0" style={{ background: color }} />
        <span className="font-extrabold text-slate-900">{row.Model}</span>
        {isHybrid && (
          <span className="ml-auto text-[10px] font-bold bg-green-600 text-white px-2 py-0.5 rounded-full">BEST</span>
        )}
      </div>
      <div className="text-[11px] text-slate-500 mb-4 leading-relaxed">{MODEL_DESC[row.Model] ?? ''}</div>
      <div className="space-y-2.5">
        <MetricBar label="Precision" value={row.Precision} max={1} color={color} />
        <MetricBar label="Recall"    value={row.Recall}    max={1} color={color} />
        <MetricBar label="F1"        value={row.F1}        max={1} color={color} />
      </div>
      <div className="mt-4 flex items-center gap-2 bg-blue-50 border border-blue-100 rounded-xl px-3 py-2">
        <TrendingUp size={12} className="text-blue-500 shrink-0" />
        <span className="text-[11px] text-slate-600">
          Lead time: <span className="font-mono font-bold text-blue-600">{row.Mean_Lead_Days?.toFixed(1) ?? '—'} days</span>
        </span>
      </div>
      {(row.TP !== undefined) && (
        <div className="mt-2 flex gap-3">
          <span className="text-[10px] font-mono text-green-600 bg-green-50 border border-green-200 px-2 py-0.5 rounded">TP: {row.TP}</span>
          <span className="text-[10px] font-mono text-amber-600 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded">FP: {row.FP}</span>
          <span className="text-[10px] font-mono text-red-600 bg-red-50 border border-red-200 px-2 py-0.5 rounded">FN: {row.FN}</span>
        </div>
      )}
    </div>
  )
}

export default function Ablation() {
  const [rows, setRows]       = useState<AblationRow[]>([])
  const [diagText, setDiag]   = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getAblationTable(), getAblationDiagnostic()])
      .then(([r, d]) => { setRows(r); setDiag(d.text) })
      .finally(() => setLoading(false))
  }, [])

  const notRun = !loading && rows.length === 0

  const radarData = rows.map(r => ({
    model: r.Model,
    Precision: +Number(r.Precision ?? 0).toFixed(3),
    Recall:    +Number(r.Recall    ?? 0).toFixed(3),
    F1:        +Number(r.F1       ?? 0).toFixed(3),
  }))

  const barMetrics = ['Precision', 'Recall', 'F1'] as const

  return (
    <div className="p-8 max-w-6xl">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-7 h-7 rounded-lg bg-purple-600 flex items-center justify-center">
            <FlaskConical size={14} className="text-white" />
          </div>
          <span className="section-label">Ablation Study</span>
        </div>
        <h1 className="page-title mb-2">Component-Level Evaluation</h1>
        <p className="page-sub">
          Each model component is isolated and evaluated on the same 3 real-world anchor events
          using Precision, Recall, F1, and Lead-time. This proves each component contributes
          measurably to the hybrid pipeline.
        </p>
      </div>

      {/* Anchor events */}
      <div className="bg-blue-50 border border-blue-200 rounded-2xl p-5 mb-8">
        <div className="text-[11px] font-bold text-blue-700 uppercase tracking-widest mb-3">3 Ground-Truth Anchor Events (14-day tolerance window)</div>
        <div className="grid grid-cols-3 gap-3">
          {ANCHORS.map(a => (
            <div key={a.name} className="bg-white border border-blue-200 rounded-xl px-4 py-3 flex items-center gap-3">
              <span className="text-xl shrink-0">{a.emoji}</span>
              <div>
                <div className="font-bold text-sm text-slate-900">{a.name}</div>
                <div className="text-[11px] font-mono text-slate-500 mt-0.5">{a.range}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="bg-white rounded-2xl border border-blue-100 p-16 flex flex-col items-center gap-4 shadow-sm">
          <div className="w-10 h-10 rounded-full border-2 border-blue-100 border-t-blue-600 animate-spin" />
          <span className="text-slate-500 text-sm">Loading ablation results…</span>
        </div>
      ) : notRun ? (
        <div className="bg-amber-50 border border-amber-200 rounded-2xl p-8 text-center shadow-sm">
          <FlaskConical size={32} className="mx-auto mb-3 text-amber-400" />
          <div className="font-bold text-slate-900 mb-1">Ablation study not yet run</div>
          <div className="text-sm text-slate-500 mb-4">Generate results with:</div>
          <div className="font-mono text-sm bg-white border border-amber-200 rounded-xl px-5 py-2.5 inline-block text-slate-700">
            python src/ablation_study.py
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Score cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {rows.map(r => <ScoreCard key={r.Model} row={r} />)}
          </div>

          {/* Charts */}
          <div className="grid grid-cols-2 gap-6">
            {/* Grouped bar chart */}
            <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-6">
              <h2 className="font-bold text-slate-900 text-sm mb-1">Precision / Recall / F1 Comparison</h2>
              <p className="text-[11px] text-slate-500 mb-4">Hybrid (green) leads all metrics</p>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={rows} margin={{ left: 4, right: 4 }}>
                  <XAxis dataKey="Model" tick={{ fontSize: 11, fill: '#64748b' }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#64748b' }} />
                  <Tooltip
                    content={({ active, payload, label }) =>
                      active && payload?.length ? (
                        <div className="bg-white border border-blue-100 shadow-lg rounded-xl px-3 py-2 text-xs space-y-0.5">
                          <div className="font-bold text-slate-900 mb-1">{label}</div>
                          {payload.map((p: any) => (
                            <div key={p.name} className="font-mono" style={{ color: p.fill }}>
                              {p.name}: {Number(p.value).toFixed(3)}
                            </div>
                          ))}
                        </div>
                      ) : null
                    }
                  />
                  {barMetrics.map((m, mi) => (
                    <Bar key={m} dataKey={m} radius={[4, 4, 0, 0]}>
                      {rows.map((r) => (
                        <Cell
                          key={r.Model}
                          fill={MODEL_COLORS[r.Model] ?? '#94a3b8'}
                          fillOpacity={mi === 0 ? 1 : mi === 1 ? 0.65 : 0.35}
                        />
                      ))}
                    </Bar>
                  ))}
                </BarChart>
              </ResponsiveContainer>
              <div className="flex gap-3 justify-center mt-2">
                <div className="flex gap-1 items-center"><div className="w-3 h-3 bg-blue-600 rounded-sm opacity-100" /><span className="text-[10px] text-slate-500">Precision</span></div>
                <div className="flex gap-1 items-center"><div className="w-3 h-3 bg-blue-600 rounded-sm opacity-65" /><span className="text-[10px] text-slate-500">Recall</span></div>
                <div className="flex gap-1 items-center"><div className="w-3 h-3 bg-blue-600 rounded-sm opacity-35" /><span className="text-[10px] text-slate-500">F1</span></div>
              </div>
            </div>

            {/* Radar chart */}
            <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-6">
              <h2 className="font-bold text-slate-900 text-sm mb-1">Multi-Metric Radar</h2>
              <p className="text-[11px] text-slate-500 mb-4">Hybrid fills the largest area across all metrics</p>
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={[
                  { metric: 'Precision', ...Object.fromEntries(rows.map(r => [r.Model, r.Precision])) },
                  { metric: 'Recall',    ...Object.fromEntries(rows.map(r => [r.Model, r.Recall])) },
                  { metric: 'F1',        ...Object.fromEntries(rows.map(r => [r.Model, r.F1])) },
                  { metric: 'Lead',      ...Object.fromEntries(rows.map(r => [r.Model, Math.min(r.Mean_Lead_Days / 30, 1)])) },
                ]}>
                  <PolarGrid stroke="#dbeafe" />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: '#64748b' }} />
                  <PolarRadiusAxis domain={[0, 1]} tick={{ fontSize: 9 }} axisLine={false} />
                  {rows.map(r => (
                    <Radar key={r.Model} dataKey={r.Model} name={r.Model}
                      stroke={MODEL_COLORS[r.Model] ?? '#94a3b8'}
                      fill={MODEL_COLORS[r.Model] ?? '#94a3b8'}
                      fillOpacity={r.Model === 'Hybrid' ? 0.25 : 0.1}
                      strokeWidth={r.Model === 'Hybrid' ? 2.5 : 1.5} />
                  ))}
                  <Tooltip formatter={(v: number) => [v.toFixed(3), '']} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Lead-time bar */}
          <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-6">
            <h2 className="font-bold text-slate-900 text-sm mb-1">Mean Lead Time (days before event)</h2>
            <p className="text-[11px] text-slate-500 mb-5">How many days before an anchor event the model fires its first detection</p>
            <ResponsiveContainer width="100%" height={100}>
              <BarChart data={rows} layout="vertical" margin={{ left: 4, right: 40 }}>
                <XAxis type="number" tick={{ fontSize: 10, fill: '#94a3b8' }} unit=" days" />
                <YAxis type="category" dataKey="Model" width={60} tick={{ fontSize: 11, fill: '#64748b' }} tickLine={false} />
                <Tooltip formatter={(v: number) => [`${v.toFixed(1)} days`, 'Lead Time']} />
                <Bar dataKey="Mean_Lead_Days" radius={[0, 6, 6, 0]}>
                  {rows.map(r => <Cell key={r.Model} fill={MODEL_COLORS[r.Model] ?? '#94a3b8'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Summary table */}
          <div className="bg-white rounded-2xl border border-blue-100 shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-blue-100 bg-blue-50/50">
              <span className="font-bold text-slate-900">Full Ablation Table</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-blue-50/60 border-b border-blue-100">
                    {['Model', 'Precision', 'Recall', 'F1', 'Lead (days)', 'TP', 'FP', 'FN'].map(h => (
                      <th key={h} className="px-5 py-3 text-left text-[11px] font-bold text-slate-500 uppercase tracking-wide">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-blue-50">
                  {rows.map((r, i) => (
                    <tr key={i} className={`hover:bg-blue-50/30 transition-colors ${r.Model === 'Hybrid' ? 'bg-green-50/40' : ''}`}>
                      <td className="px-5 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-2.5 h-2.5 rounded-full" style={{ background: MODEL_COLORS[r.Model] ?? '#94a3b8' }} />
                          <span className="font-bold text-slate-900">{r.Model}</span>
                          {r.Model === 'Hybrid' && (
                            <span className="text-[9px] font-bold bg-green-600 text-white px-1.5 py-0.5 rounded-full">BEST</span>
                          )}
                        </div>
                      </td>
                      {([r.Precision, r.Recall, r.F1] as number[]).map((v, vi) => (
                        <td key={vi} className="px-5 py-3 font-mono font-semibold" style={{ color: MODEL_COLORS[r.Model] ?? '#64748b' }}>
                          {Number(v ?? 0).toFixed(3)}
                        </td>
                      ))}
                      <td className="px-5 py-3 font-mono text-blue-600 font-semibold">{Number(r.Mean_Lead_Days ?? 0).toFixed(1)}</td>
                      <td className="px-5 py-3 font-mono text-green-600">{r.TP ?? '—'}</td>
                      <td className="px-5 py-3 font-mono text-amber-600">{r.FP ?? '—'}</td>
                      <td className="px-5 py-3 font-mono text-red-600">{r.FN ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Diagnostic text */}
          {diagText && (
            <details className="bg-white border border-blue-100 rounded-2xl shadow-sm overflow-hidden">
              <summary className="px-6 py-4 cursor-pointer font-bold text-slate-900 hover:bg-blue-50/50 transition-colors flex items-center gap-2">
                <FlaskConical size={14} className="text-purple-600" />
                Full Diagnostic Report
                <span className="ml-auto text-xs text-slate-400">click to expand</span>
              </summary>
              <div className="px-6 pb-6">
                <pre className="text-[11px] font-mono text-slate-600 bg-blue-50/50 border border-blue-100 rounded-xl p-4 whitespace-pre-wrap overflow-x-auto leading-relaxed mt-2">
                  {diagText}
                </pre>
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  )
}
