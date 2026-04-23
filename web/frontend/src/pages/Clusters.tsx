import { useEffect, useState } from 'react'
import { Layers, Users } from 'lucide-react'
import { getClusters } from '../api/client'
import type { Cluster } from '../types'

const COLORS = ['#2563eb','#7c3aed','#16a34a','#dc2626','#d97706','#0891b2','#db2777','#ea580c','#059669','#4f46e5','#ca8a04','#0284c7','#be185d']
const EMOJI:  Record<string,string> = { 'Topic A':'🌊','Topic B':'🌾','Topic C':'🏛️','Topic D':'🦠','Topic E':'🌍','Topic F':'🔥','Topic G':'🚨','Topic H':'💹','Topic I':'🗺️','Topic J':'📻','Topic K':'🏥','Topic L':'⚽','Topic M':'⚖️' }
const DESC:   Record<string,string> = {
  'Topic A':'Weather, climate events, floods, drought and water systems',
  'Topic B':'Rural Australia, farming, cattle, indigenous communities',
  'Topic C':'Australian politics, elections, government policy',
  'Topic D':'COVID-19, vaccines, pandemic response (2020–2021)',
  'Topic E':'International affairs, war, geopolitics, Trump era',
  'Topic F':'Australian bushfires, Black Summer 2019–2020',
  'Topic G':'Fatal accidents, crashes, homicides, missing persons',
  'Topic H':'Economy, markets, business, budget, industry',
  'Topic I':'Australian geography, cities, states, local news',
  'Topic J':'ABC News interviews, media appearances, commentary',
  'Topic K':'Health system, hospitals, education, social services',
  'Topic L':'Sports — cricket, AFL, rugby, tennis, Olympics',
  'Topic M':'Crime, police, courts, charges, murder trials',
}

export default function Clusters() {
  const [clusters, setClusters] = useState<Cluster[]>([])
  const [selected, setSelected] = useState<Cluster | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => { getClusters().then(c => { setClusters(c); setSelected(c[0]??null) }).finally(() => setLoading(false)) }, [])

  const total = clusters.reduce((s,c) => s+(c.Size??0), 0)

  return (
    <div className="p-8 max-w-6xl">
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-7 h-7 rounded-lg bg-purple-600 flex items-center justify-center">
            <Layers size={14} className="text-white" />
          </div>
          <span className="section-label">Semantic Clusters</span>
        </div>
        <h1 className="page-title mb-2">13 Narrative Universes</h1>
        <p className="page-sub">SBERT K-Means (K=13) on 49,989 stratified headlines. Each cluster is a semantic region of the news landscape — discovered without any labels.</p>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-8">
        {[
          { label:'Semantic clusters', value:'13', color:'text-purple-600' },
          { label:'Headlines clustered', value: total.toLocaleString(), color:'text-slate-900' },
          { label:'Silhouette score', value:'0.0226', color:'text-blue-600' },
        ].map(s => (
          <div key={s.label} className="bg-white rounded-2xl border border-blue-100 shadow-sm p-5">
            <div className={`text-2xl font-bold ${s.color}`}>{s.value}</div>
            <div className="text-xs text-slate-500 mt-1 font-medium">{s.label}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2 grid grid-cols-2 gap-3 content-start">
          {loading
            ? [...Array(13)].map((_,i) => <div key={i} className="bg-white rounded-2xl h-32 animate-pulse border border-blue-100" />)
            : clusters.map((c, i) => {
                const color = COLORS[i%13]
                const pct = total > 0 ? (c.Size/total)*100 : 0
                const isSel = selected?.Cluster === c.Cluster
                return (
                  <button key={c.Cluster} onClick={() => setSelected(c)}
                    className={`bg-white rounded-2xl p-4 text-left border-2 transition-all duration-200 hover:shadow-md ${isSel ? 'shadow-md' : 'border-blue-100 hover:border-blue-300'}`}
                    style={isSel ? { borderColor: color, boxShadow:`0 0 0 3px ${color}15` } : {}}>
                    <div className="flex items-center gap-2.5 mb-3">
                      <span className="text-xl">{EMOJI[c.Cluster]??'📰'}</span>
                      <div>
                        <div className="font-bold text-sm text-slate-900">{c.Cluster}</div>
                        <div className="text-[10px] font-mono text-slate-500">{c.Size?.toLocaleString()} docs · {pct.toFixed(1)}%</div>
                      </div>
                    </div>
                    <div className="h-1.5 rounded-full bg-blue-100 overflow-hidden mb-2.5">
                      <div className="h-full rounded-full" style={{ width:`${Math.min(pct*4,100)}%`, background:color }} />
                    </div>
                    <div className="text-[11px] text-slate-500 line-clamp-1">{c.TopTerms}</div>
                  </button>
                )
              })
          }
        </div>

        <div className="sticky top-8">
          {selected && (() => {
            const idx = clusters.findIndex(c => c.Cluster === selected.Cluster)
            const color = COLORS[idx%13]
            const pct = total > 0 ? (selected.Size/total)*100 : 0
            return (
              <div className="bg-white rounded-2xl border-2 shadow-md p-6 animate-fade-in" style={{ borderColor: color }}>
                <div className="text-3xl mb-3">{EMOJI[selected.Cluster]??'📰'}</div>
                <div className="font-extrabold text-xl text-slate-900 mb-0.5">{selected.Cluster}</div>
                <div className="text-xs font-mono text-slate-500 mb-4">Rank #{idx+1} of 13 clusters</div>

                <div className="flex items-center gap-2 mb-4 p-3 rounded-xl bg-blue-50 border border-blue-100">
                  <Users size={13} className="text-blue-600" />
                  <span className="text-sm font-bold text-slate-900">{selected.Size?.toLocaleString()}</span>
                  <span className="text-xs text-slate-500">headlines ({pct.toFixed(1)}%)</span>
                </div>

                <div className="h-2 rounded-full bg-blue-100 overflow-hidden mb-5">
                  <div className="h-full rounded-full transition-all duration-500" style={{ width:`${Math.min(pct*5,100)}%`, background:color }} />
                </div>

                <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wide mb-1.5">Description</div>
                <p className="text-sm text-slate-900 leading-relaxed mb-5">{DESC[selected.Cluster]??selected.TopTerms}</p>

                <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wide mb-2">Top Terms</div>
                <div className="flex flex-wrap gap-1.5">
                  {selected.TopTerms?.split(',').slice(0,10).map(t => (
                    <span key={t} className="px-2 py-0.5 rounded-md text-[11px] font-mono border font-medium"
                      style={{ borderColor:`${color}50`, color, background:`${color}12` }}>
                      {t.trim()}
                    </span>
                  ))}
                </div>
              </div>
            )
          })()}
        </div>
      </div>
    </div>
  )
}
