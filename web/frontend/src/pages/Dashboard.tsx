import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { Activity, Zap, Database, Layers, TrendingUp, ArrowRight, Clock } from 'lucide-react'
import { getStatus, getVelocity, getClusters, getTopRuptures } from '../api/client'
import type { StatusData, VelocityPoint, Cluster } from '../types'

const CLUSTER_COLORS = [
  '#2563eb','#7c3aed','#16a34a','#dc2626','#d97706',
  '#0891b2','#db2777','#ea580c','#059669','#4f46e5',
  '#ca8a04','#0284c7','#be185d',
]

function StatCard({ icon: Icon, label, value, sub, color }: {
  icon: React.ElementType; label: string; value: string | number; sub?: string; color: string
}) {
  const icons: Record<string,string> = { blue:'text-blue-600', red:'text-red-600', green:'text-green-600', purple:'text-purple-600' }
  const bg: Record<string,string> = { blue:'bg-blue-50', red:'bg-red-50', green:'bg-green-50', purple:'bg-purple-50' }
  const border: Record<string,string> = { blue:'border-blue-200', red:'border-red-200', green:'border-green-200', purple:'border-purple-200' }
  return (
    <div className={`bg-white rounded-2xl p-5 border ${border[color]} shadow-sm animate-slide-up`}>
      <div className={`w-9 h-9 rounded-xl ${bg[color]} flex items-center justify-center mb-4`}>
        <Icon size={17} className={icons[color]} />
      </div>
      <div className="text-2xl font-extrabold text-slate-900">{value}</div>
      <div className="text-sm text-slate-500 mt-0.5 font-medium">{label}</div>
      {sub && <div className="text-[11px] text-slate-500/70 mt-1 font-mono">{sub}</div>}
    </div>
  )
}

const VTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-blue-100 shadow-lg rounded-xl px-3 py-2 text-xs">
      <div className="text-slate-500 font-mono mb-1">{label}</div>
      <div className={`font-bold ${payload[0].value > 0.4 ? 'text-red-600' : 'text-blue-600'}`}>
        V_s = {Number(payload[0].value).toFixed(4)}
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [status, setStatus] = useState<StatusData | null>(null)
  const [velocity, setVelocity] = useState<VelocityPoint[]>([])
  const [clusters, setClusters] = useState<Cluster[]>([])
  const [topRuptures, setTopRuptures] = useState<VelocityPoint[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getStatus(), getVelocity(), getClusters(), getTopRuptures(5)])
      .then(([s, v, c, r]) => { setStatus(s); setVelocity(v); setClusters(c); setTopRuptures(r) })
      .finally(() => setLoading(false))
  }, [])

  const mean = velocity.length ? velocity.reduce((s,p) => s+p.v, 0) / velocity.length : 0
  const std  = velocity.length ? Math.sqrt(velocity.reduce((s,p) => s+(p.v-mean)**2, 0) / velocity.length) : 0
  const threshold = mean + std

  return (
    <div className="p-8 max-w-7xl">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <div className="relative w-2.5 h-2.5">
            <div className="pulse-dot w-2.5 h-2.5 rounded-full bg-green-600 text-green-600" />
          </div>
          <span className="text-xs font-semibold text-green-600 tracking-widest uppercase">System Active</span>
        </div>
        <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight mb-2">
          Narrative Intelligence <span className="text-blue-600">Dashboard</span>
        </h1>
        <p className="text-slate-500 max-w-2xl text-[15px]">
          Real-time detection of narrative trends and event ruptures across 19 years of news —
          powered by SBERT, K-Means, and GDELT live intelligence.
        </p>
      </div>

      {/* Stats */}
      {loading ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {[...Array(4)].map((_,i) => <div key={i} className="bg-white rounded-2xl h-32 animate-pulse border border-blue-100" />)}
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard icon={Database}  label="Headlines Analyzed"   value="1.24M"     sub="ABC News 2003–2021"       color="blue"   />
          <StatCard icon={Zap}       label="Max Rupture Velocity" value={status?.max_velocity?.toFixed(4) ?? '—'} sub={status?.peak_rupture_week} color="red" />
          <StatCard icon={Layers}    label="Semantic Clusters"    value={status?.clusters ?? 13} sub="SBERT K-Means K=13"  color="purple" />
          <StatCard icon={Activity}  label="GDELT Live Records"   value={status?.gdelt_records?.toLocaleString() ?? '—'} sub="Updated every 15 min" color="green" />
        </div>
      )}

      {/* Velocity + Top ruptures */}
      <div className="grid grid-cols-3 gap-6 mb-8">
        <div className="col-span-2 bg-white rounded-2xl border border-blue-100 shadow-sm p-6">
          <div className="flex items-center justify-between mb-5">
            <div>
              <h2 className="font-bold text-slate-900">Semantic Velocity — 2003–2021</h2>
              <p className="text-xs text-slate-500 mt-0.5">Narrative shift magnitude (V_s) week-over-week</p>
            </div>
            <Link to="/ruptures" className="text-xs text-blue-600 hover:underline flex items-center gap-1 font-medium">
              Explore <ArrowRight size={12} />
            </Link>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={velocity} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="vGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#2563eb" stopOpacity={0.18} />
                  <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="week_start" hide />
              <YAxis hide domain={[0,'auto']} />
              <Tooltip content={<VTooltip />} />
              <ReferenceLine y={threshold} stroke="#dc2626" strokeDasharray="4 4" strokeOpacity={0.5} />
              <Area type="monotone" dataKey="v" stroke="#2563eb" strokeWidth={2}
                fill="url(#vGrad)" dot={false} activeDot={{ r:4, fill:'#2563eb', stroke:'#fff', strokeWidth:2 }} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-bold text-slate-900 text-sm">Top Ruptures</h2>
            <div className="w-7 h-7 rounded-lg bg-red-50 flex items-center justify-center">
              <Zap size={13} className="text-red-600" />
            </div>
          </div>
          <div className="space-y-3">
            {topRuptures.map((r, i) => (
              <div key={r.week} className="flex items-start gap-3">
                <div className="w-5 h-5 rounded-full bg-red-100 text-red-600 text-[10px] font-bold flex items-center justify-center shrink-0 mt-0.5">{i+1}</div>
                <div className="flex-1 min-w-0">
                  <div className="font-mono text-[11px] text-slate-900 truncate">{r.week}</div>
                  <div className="text-xs text-red-600 font-semibold">V_s = {r.v.toFixed(4)}</div>
                </div>
              </div>
            ))}
          </div>
          <Link to="/ruptures" className="mt-4 block text-center text-xs text-blue-600 hover:underline font-medium">
            View all ruptures →
          </Link>
        </div>
      </div>

      {/* Clusters */}
      <div className="bg-white rounded-2xl border border-blue-100 shadow-sm p-6 mb-8">
        <div className="flex items-center justify-between mb-5">
          <div>
            <h2 className="font-bold text-slate-900">13 Semantic Clusters</h2>
            <p className="text-xs text-slate-500 mt-0.5">SBERT K-Means on 49,989 stratified headlines (2003–2021)</p>
          </div>
          <Link to="/clusters" className="text-xs text-blue-600 hover:underline flex items-center gap-1 font-medium">
            Explore <ArrowRight size={12} />
          </Link>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
          {clusters.map((c, i) => (
            <Link key={c.Cluster} to="/clusters"
              className="border border-blue-100 rounded-xl p-3.5 hover:border-blue-300 hover:shadow-sm transition-all duration-200 bg-blue-50/40 hover:bg-blue-50">
              <div className="flex items-center gap-2 mb-1.5">
                <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: CLUSTER_COLORS[i%13] }} />
                <span className="text-xs font-semibold text-slate-900 truncate">{c.Cluster}</span>
              </div>
              <div className="text-[11px] text-slate-500 line-clamp-2 leading-relaxed">{c.TopTerms}</div>
              <div className="mt-2 text-[10px] font-mono text-slate-500/70">{c.Size?.toLocaleString()} docs</div>
            </Link>
          ))}
        </div>
      </div>

      {/* CTAs */}
      <div className="grid grid-cols-2 gap-4">
        <Link to="/watchdog" className="bg-white border border-blue-100 rounded-2xl p-6 flex items-center gap-4 hover:border-blue-300 hover:shadow-md transition-all duration-200 group">
          <div className="w-11 h-11 rounded-xl bg-blue-600 flex items-center justify-center shadow-sm group-hover:bg-blue-700 transition-colors">
            <TrendingUp size={18} className="text-white" />
          </div>
          <div>
            <div className="font-bold text-slate-900">Topic Watchdog</div>
            <div className="text-xs text-slate-500 mt-0.5">Monitor any keyword in real-time</div>
          </div>
          <ArrowRight size={16} className="text-slate-500 ml-auto group-hover:text-blue-600 transition-colors" />
        </Link>
        <Link to="/feed" className="bg-white border border-blue-100 rounded-2xl p-6 flex items-center gap-4 hover:border-green-200 hover:shadow-md transition-all duration-200 group">
          <div className="w-11 h-11 rounded-xl bg-green-600 flex items-center justify-center shadow-sm group-hover:bg-green-700 transition-colors">
            <Clock size={18} className="text-white" />
          </div>
          <div>
            <div className="font-bold text-slate-900">Live GDELT Feed</div>
            <div className="text-xs text-slate-500 mt-0.5">Global news intelligence, 15-min updates</div>
          </div>
          <ArrowRight size={16} className="text-slate-500 ml-auto group-hover:text-green-600 transition-colors" />
        </Link>
      </div>
    </div>
  )
}
