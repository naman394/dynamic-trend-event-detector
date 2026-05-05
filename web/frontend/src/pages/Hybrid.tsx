import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts'
import { GitMerge, Zap, Target, TrendingUp, ChevronRight } from 'lucide-react'
import { getHybridSummary, getHybridImpact, getHybridAlpha } from '../api/client'
import type { HybridClusterRow, HybridImpactRow } from '../types'

const COLORS = [
  '#2563eb','#7c3aed','#16a34a','#dc2626','#d97706',
  '#0891b2','#db2777','#ea580c','#059669','#4f46e5',
  '#ca8a04','#0284c7','#be185d',
]

function AlphaBox({ text }: { text: string }) {
  const alphaMatch = text.match(/alpha[^0-9]*([0-9.]+)/i)
  const mrrMatch   = text.match(/mrr[^0-9]*([0-9.]+)/i)
  const alpha = alphaMatch?.[1] ?? null
  const mrr   = mrrMatch?.[1]   ?? null

  return (
    <div className="bg-gradient-to-br from-green-50 to-emerald-50 border border-green-200 rounded-2xl p-6 shadow-sm">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-7 h-7 rounded-lg bg-green-600 flex items-center justify-center">
          <Target size={14} className="text-white" />
        </div>
        <span className="text-xs font-mono text-green-700 uppercase tracking-widest font-semibold">Learned Fusion Weight</span>
      </div>
      <div className="grid grid-cols-2 gap-4 mb-5">
        <div className="bg-white border border-green-200 rounded-xl p-4 text-center">
          <div className="text-3xl font-extrabold text-green-600 font-mono">{alpha ? `α* = ${alpha}` : '—'}</div>
          <div className="text-xs text-slate-500 mt-1">Optimal fusion weight</div>
          <div className="text-[10px] text-slate-400 mt-0.5 font-mono">scipy.optimize · MRR objective</div>
        </div>
        <div className="bg-white border border-green-200 rounded-xl p-4 text-center">
          <div className="text-3xl font-extrabold text-emerald-600 font-mono">{mrr ? `${mrr}` : '—'}</div>
          <div className="text-xs text-slate-500 mt-1">Best MRR achieved</div>
          <div className="text-[10px] text-slate-400 mt-0.5 font-mono">on 3 anchor events</div>
        </div>
      </div>
      <div className="bg-white border border-green-100 rounded-xl p-4">
        <div className="text-[11px] font-bold text-slate-500 uppercase tracking-widest mb-2">Hybrid Formula</div>
        <div className="font-mono text-sm text-green-700 bg-green-50 border border-green-200 rounded-lg px-4 py-2.5">
          S_I<sup>hybrid</sup> = (α* · SBERT_uniq + (1−α*) · LDA_rarity) × |τ|
        </div>
        <div className="text-[11px] text-slate-500 mt-3 leading-relaxed">
          α* is learned via MRR maximisation on Black Summer, First AU COVID, and National Lockdown anchor events.
          A high α* favours SBERT uniqueness; a low α* favours LDA rarity.
        </div>
      </div>
      {text && (
        <details className="mt-4">
          <summary className="text-xs text-slate-500 cursor-pointer hover:text-green-600 font-medium">Full optimisation report ↓</summary>
          <pre className="mt-2 text-[11px] font-mono text-slate-600 bg-white border border-green-100 rounded-lg p-3 whitespace-pre-wrap overflow-x-auto leading-relaxed">
            {text}
          </pre>
        </details>
      )}
    </div>
  )
}

function SilhouetteChart({ rows }: { rows: HybridClusterRow[] }) {
  const data = rows.map(r => ({
    name: r.cluster?.replace('Topic ', '') ?? '?',
    seeded: +Number(r.silhouette_seeded ?? 0).toFixed(4),
    random: +Number(r.silhouette_random ?? 0).toFixed(4),
  }))

  return (
    <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-6">
      <div className="mb-4">
        <h2 className="font-bold text-slate-900 text-sm">LDA-Seeded vs Random Init — Silhouette Scores</h2>
        <p className="text-[11px] text-slate-500 mt-0.5">
          LDA seeds provide better K-Means starting centroids than random initialisation.
          Higher silhouette = tighter, better-separated clusters.
        </p>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ left: 4, right: 4 }}>
          <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748b' }} />
          <YAxis tick={{ fontSize: 10, fill: '#64748b' }} domain={['auto', 'auto']} />
          <Tooltip
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <div className="bg-white border border-blue-100 shadow-lg rounded-xl px-3 py-2 text-xs">
                  <div className="font-mono text-slate-500 mb-1">Topic {label}</div>
                  <div className="text-blue-600 font-bold">Seeded: {payload[0]?.value}</div>
                  <div className="text-slate-500">Random: {payload[1]?.value}</div>
                </div>
              ) : null
            }
          />
          <Bar dataKey="seeded" name="LDA-Seeded" fill="#2563eb" radius={[4, 4, 0, 0]} />
          <Bar dataKey="random" name="Random Init" fill="#cbd5e1" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <div className="flex items-center gap-4 mt-3 justify-center">
        <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm bg-blue-600" /><span className="text-[11px] text-slate-500">LDA-Seeded</span></div>
        <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm bg-slate-300" /><span className="text-[11px] text-slate-500">Random Init</span></div>
      </div>
    </div>
  )
}

function ImpactTable({ rows }: { rows: HybridImpactRow[] }) {
  if (!rows.length) return (
    <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-10 text-center">
      <Zap size={28} className="mx-auto mb-3 text-blue-200" />
      <div className="font-semibold text-slate-900">No hybrid impact scores yet</div>
      <div className="text-sm text-slate-500 mt-1 font-mono">python src/hybrid_pipeline.py</div>
    </div>
  )

  const maxSi = Math.max(...rows.map(r => r.si_hybrid ?? 0))

  return (
    <div className="bg-white rounded-2xl border border-blue-100 shadow-sm overflow-hidden">
      <div className="px-6 py-4 border-b border-blue-100 flex items-center justify-between bg-blue-50/50">
        <span className="font-bold text-slate-900">Top {rows.length} Events by Hybrid Impact Score</span>
        <span className="text-xs font-mono text-slate-500 bg-white border border-blue-100 px-3 py-1 rounded-full">S_I<sup>hybrid</sup></span>
      </div>
      <div className="divide-y divide-blue-50">
        {rows.map((r, i) => {
          const si    = r.si_hybrid ?? 0
          const pct   = maxSi > 0 ? (si / maxSi) * 100 : 0
          const tone  = r.tone ?? 0
          const url   = String(r.url ?? r.DOCUMENTIDENTIFIER ?? '')
          const src   = String(r.source ?? r.SOURCECOMMONNAME ?? `Event ${i + 1}`)
          return (
            <div key={i} className="px-6 py-4 hover:bg-blue-50/40 transition-colors flex items-start gap-4">
              <div className="w-7 h-7 rounded-full bg-blue-100 text-blue-700 text-[11px] font-bold flex items-center justify-center shrink-0 mt-0.5">{i + 1}</div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  {url.startsWith('http')
                    ? <a href={url} target="_blank" rel="noreferrer"
                        className="text-sm font-semibold text-slate-900 hover:text-blue-600 truncate max-w-xs">{src}</a>
                    : <span className="text-sm font-semibold text-slate-900 truncate">{src}</span>
                  }
                  <span className={`text-[11px] font-mono shrink-0 ${tone < -1 ? 'text-red-600' : tone > 1 ? 'text-green-600' : 'text-slate-500'}`}>
                    tone {tone > 0 ? '+' : ''}{Number(tone).toFixed(2)}
                  </span>
                </div>
                <div className="flex items-center gap-3 mt-2">
                  <div className="flex-1 h-1.5 bg-blue-50 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-600 rounded-full transition-all duration-500" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="text-xs font-mono font-bold text-blue-600 shrink-0">S_I {Number(si).toFixed(4)}</span>
                </div>
                <div className="flex gap-4 mt-1.5">
                  <span className="text-[10px] text-slate-400 font-mono">
                    SBERT_uniq: <span className="text-slate-600">{Number(r.sbert_uniqueness ?? 0).toFixed(4)}</span>
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono">
                    LDA_rarity: <span className="text-slate-600">{Number(r.lda_rarity ?? 0).toFixed(4)}</span>
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function Hybrid() {
  const [summary, setSummary] = useState<HybridClusterRow[]>([])
  const [impact,  setImpact]  = useState<HybridImpactRow[]>([])
  const [alphaText, setAlphaText] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getHybridSummary(), getHybridImpact(15), getHybridAlpha()])
      .then(([s, imp, a]) => { setSummary(s); setImpact(imp); setAlphaText(a.text) })
      .finally(() => setLoading(false))
  }, [])

  const notRun = !loading && summary.length === 0 && impact.length === 0

  const radarData = summary.map(r => ({
    cluster: r.cluster?.replace('Topic ', '') ?? '?',
    seeded:  +Number(r.silhouette_seeded ?? 0).toFixed(4),
    random:  +Number(r.silhouette_random ?? 0).toFixed(4),
  }))

  return (
    <div className="p-8 max-w-6xl">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-7 h-7 rounded-lg bg-green-600 flex items-center justify-center">
            <GitMerge size={14} className="text-white" />
          </div>
          <span className="section-label">Hybrid Innovation</span>
        </div>
        <h1 className="page-title mb-2">LDA × SBERT Synergistic Pipeline</h1>
        <p className="page-sub">
          Three-stage hybrid: LDA topic keywords seed K-Means centroids (forward pass) →
          SBERT cluster confidence down-weights low-confidence seeds (feedback loop) →
          learned α* fuses SBERT uniqueness + LDA rarity into a single impact score.
        </p>
      </div>

      {/* Pipeline stages */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        {[
          { step:'①', title:'LDA → K-Means Seeds', desc:'LDA top keywords are SBERT-encoded into (10, 384) seed centroids for K-Means init=seeds, n_init=1', color:'blue', icon: TrendingUp },
          { step:'②', title:'SBERT Confidence Feedback', desc:'Bottom 25% low-confidence cluster assignments flag weak LDA seeds. SBERT corrects LDA geometry.', color:'purple', icon: Zap },
          { step:'③', title:'Learned α* Fusion', desc:'α* is optimised via MRR on 3 anchor events. Hybrid S_I fuses SBERT uniqueness and LDA rarity.', color:'green', icon: Target },
        ].map(s => {
          const Icon = s.icon
          const border: Record<string,string> = { blue:'border-blue-200', purple:'border-purple-200', green:'border-green-200' }
          const bg: Record<string,string> = { blue:'bg-blue-50', purple:'bg-purple-50', green:'bg-green-50' }
          const ic: Record<string,string> = { blue:'bg-blue-600', purple:'bg-purple-600', green:'bg-green-600' }
          const tc: Record<string,string> = { blue:'text-blue-700', purple:'text-purple-700', green:'text-green-700' }
          return (
            <div key={s.step} className={`${bg[s.color]} border ${border[s.color]} rounded-2xl p-5 shadow-sm`}>
              <div className="flex items-center gap-3 mb-3">
                <div className={`w-8 h-8 rounded-xl ${ic[s.color]} flex items-center justify-center`}>
                  <Icon size={15} className="text-white" />
                </div>
                <span className={`text-lg font-extrabold ${tc[s.color]}`}>{s.step}</span>
              </div>
              <div className={`font-bold text-sm text-slate-900 mb-1`}>{s.title}</div>
              <div className="text-[11px] text-slate-500 leading-relaxed">{s.desc}</div>
            </div>
          )
        })}
      </div>

      {loading ? (
        <div className="bg-white rounded-2xl border border-blue-100 p-16 flex flex-col items-center gap-4 shadow-sm">
          <div className="w-10 h-10 rounded-full border-2 border-blue-100 border-t-blue-600 animate-spin" />
          <span className="text-slate-500 text-sm">Loading hybrid results…</span>
        </div>
      ) : notRun ? (
        <div className="bg-amber-50 border border-amber-200 rounded-2xl p-8 text-center shadow-sm">
          <GitMerge size={32} className="mx-auto mb-3 text-amber-400" />
          <div className="font-bold text-slate-900 mb-1">Hybrid pipeline not yet run</div>
          <div className="text-sm text-slate-500 mb-4">Generate results with:</div>
          <div className="font-mono text-sm bg-white border border-amber-200 rounded-xl px-5 py-2.5 inline-block text-slate-700">
            python src/hybrid_pipeline.py
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Alpha optimisation */}
          <AlphaBox text={alphaText} />

          {/* Cluster stats */}
          {summary.length > 0 && (
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-5">
                <div className="text-2xl font-bold text-blue-600">{summary.length}</div>
                <div className="text-xs text-slate-500 mt-0.5 font-medium">Clusters evaluated</div>
              </div>
              <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-5">
                <div className="text-2xl font-bold text-green-600">
                  {summary.filter(r =>
                    Number(r.silhouette_seeded) > Number(r.silhouette_random)
                  ).length} / {summary.length}
                </div>
                <div className="text-xs text-slate-500 mt-0.5 font-medium">Clusters improved by seeding</div>
              </div>
              <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-5">
                <div className="text-2xl font-bold text-purple-600">
                  {summary.length > 0
                    ? (summary.reduce((s, r) => s + Number(r.low_confidence_pct ?? 0), 0) / summary.length).toFixed(1) + '%'
                    : '—'}
                </div>
                <div className="text-xs text-slate-500 mt-0.5 font-medium">Avg low-confidence assignments</div>
              </div>
            </div>
          )}

          {/* Silhouette chart + radar side by side */}
          {summary.length > 0 && (
            <div className="grid grid-cols-2 gap-6">
              <SilhouetteChart rows={summary} />
              <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-6">
                <h2 className="font-bold text-slate-900 text-sm mb-1">Silhouette Radar — Seeded vs Random</h2>
                <p className="text-[11px] text-slate-500 mb-4">Radial comparison across all clusters</p>
                <ResponsiveContainer width="100%" height={250}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#dbeafe" />
                    <PolarAngleAxis dataKey="cluster" tick={{ fontSize: 10, fill: '#64748b' }} />
                    <PolarRadiusAxis tick={{ fontSize: 9 }} axisLine={false} />
                    <Radar dataKey="seeded" name="Seeded" stroke="#2563eb" fill="#2563eb" fillOpacity={0.2} strokeWidth={2} />
                    <Radar dataKey="random" name="Random" stroke="#cbd5e1" fill="#cbd5e1" fillOpacity={0.15} strokeWidth={1.5} />
                    <Tooltip formatter={(v: number) => [v.toFixed(4), '']} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Impact table */}
          <ImpactTable rows={impact} />

          {/* Cluster detail table */}
          {summary.length > 0 && (
            <div className="bg-white rounded-2xl border border-blue-100 shadow-sm overflow-hidden">
              <div className="px-6 py-4 border-b border-blue-100 bg-blue-50/50">
                <span className="font-bold text-slate-900">Cluster-Level Hybrid Metrics</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-blue-50/60 border-b border-blue-100">
                      {['Cluster','Size','Silhouette (Seeded)','Silhouette (Random)','Δ Improvement','Low Conf %','Top Terms'].map(h => (
                        <th key={h} className="px-4 py-3 text-left text-[11px] font-bold text-slate-500 uppercase tracking-wide whitespace-nowrap">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-blue-50">
                    {summary.map((r, i) => {
                      const delta = Number(r.silhouette_seeded ?? 0) - Number(r.silhouette_random ?? 0)
                      return (
                        <tr key={i} className="hover:bg-blue-50/30 transition-colors">
                          <td className="px-4 py-3 font-semibold text-slate-900">
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 rounded-full shrink-0" style={{ background: COLORS[i % 13] }} />
                              {r.cluster}
                            </div>
                          </td>
                          <td className="px-4 py-3 font-mono text-slate-700">{Number(r.size ?? 0).toLocaleString()}</td>
                          <td className="px-4 py-3 font-mono text-blue-600 font-semibold">{Number(r.silhouette_seeded ?? 0).toFixed(4)}</td>
                          <td className="px-4 py-3 font-mono text-slate-500">{Number(r.silhouette_random ?? 0).toFixed(4)}</td>
                          <td className={`px-4 py-3 font-mono font-bold ${delta > 0 ? 'text-green-600' : 'text-red-500'}`}>
                            {delta > 0 ? '+' : ''}{delta.toFixed(4)}
                          </td>
                          <td className="px-4 py-3 font-mono text-slate-500">{Number(r.low_confidence_pct ?? 0).toFixed(1)}%</td>
                          <td className="px-4 py-3 text-[11px] text-slate-500 max-w-xs truncate">{r.top_terms ?? '—'}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
