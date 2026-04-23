import { useEffect, useState } from 'react'
import { BarChart2, Info } from 'lucide-react'
import { getSpikes } from '../api/client'
import type { SpikeEvent } from '../types'

export default function Impact() {
  const [spikes, setSpikes] = useState<SpikeEvent[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => { getSpikes().then(setSpikes).finally(() => setLoading(false)) }, [])

  return (
    <div className="p-8 max-w-4xl">
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-7 h-7 rounded-lg bg-amber-500 flex items-center justify-center">
            <BarChart2 size={14} className="text-white" />
          </div>
          <span className="section-label">Impact Scoring</span>
        </div>
        <h1 className="page-title mb-2">Spike Events & Impact Analysis</h1>
        <p className="page-sub">Per-topic z-score spike detection (z ≥ 2.5σ) from Phase 4 analysis. Each spike is a statistically abnormal surge in topic growth velocity.</p>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-2xl p-5 mb-8">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-600/10 flex items-center justify-center shrink-0 mt-0.5">
            <Info size={15} className="text-blue-600" />
          </div>
          <div>
            <div className="text-sm font-bold text-slate-900 mb-2">SBERT Impact Formula</div>
            <div className="font-mono text-sm text-blue-600 bg-white border border-blue-200 rounded-xl px-4 py-2.5 inline-block">
              S_I = (1 − cosine_sim(v_SBERT, centroid_SBERT)) × |tone|
            </div>
            <div className="text-xs text-slate-500 mt-2 leading-relaxed">
              High S_I = semantically unique article with strong emotional tone = high real-world impact signal.
            </div>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-10 flex items-center justify-center">
          <div className="w-8 h-8 rounded-full border-2 border-blue-100 border-t-accent-blue animate-spin" />
        </div>
      ) : spikes.length === 0 ? (
        <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-10 text-center">
          <BarChart2 size={32} className="mx-auto mb-3 text-blue-200" />
          <div className="font-bold text-slate-900">No spike events found</div>
          <div className="text-sm text-slate-500 mt-1">Run notebook 12 to generate spike_events.csv</div>
          <div className="font-mono text-xs mt-3 bg-blue-50 border border-blue-200 px-4 py-2 rounded-lg inline-block text-slate-500">
            python src/spike_events_from_topics_over_time.py
          </div>
        </div>
      ) : (
        <div className="bg-white rounded-2xl border border-blue-100 shadow-sm overflow-hidden">
          <div className="px-6 py-4 border-b border-blue-100 flex items-center justify-between bg-blue-50/50">
            <span className="font-bold text-slate-900">{spikes.length} spike events detected</span>
            <span className="text-xs text-slate-500 font-mono bg-white border border-blue-100 px-3 py-1 rounded-full">z ≥ 2.5σ threshold</span>
          </div>
          <div className="divide-y divide-blue-50">
            {spikes.map((s, i) => (
              <div key={i} className="px-6 py-4 hover:bg-blue-50/50 transition-colors flex items-start gap-4">
                <div className="w-7 h-7 rounded-full bg-amber-100 text-amber-700 text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">{i+1}</div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-3 mb-1.5">
                    <span className="font-mono text-sm font-semibold text-slate-900">{s.spike_date}</span>
                    <span className="text-xs text-blue-600 bg-blue-50 border border-blue-200 px-2 py-0.5 rounded-full font-semibold">Topic {s.topic_id}</span>
                  </div>
                  <div className="text-xs text-slate-500 mb-2">
                    z-score: <span className="text-red-600 font-mono font-bold">{Number(s.velocity_z).toFixed(2)}σ</span>
                    {' · '}velocity: <span className="text-slate-900 font-mono font-semibold">{Number(s.velocity).toFixed(4)}</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {String(s.keywords??'').split(',').slice(0,8).map(k => (
                      <span key={k} className="text-[11px] font-mono text-slate-500 bg-blue-50 border border-blue-100 px-2 py-0.5 rounded">{k.trim()}</span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
