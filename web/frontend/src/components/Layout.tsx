import { NavLink, Outlet } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { Activity, Layers, Zap, Rss, Github, Brain, RefreshCw, FlaskConical, Network, ShieldCheck } from 'lucide-react'

const nav = [
  { to: '/',             icon: Activity,      label: 'Dashboard',     group: 'main' },
  { to: '/trends',       icon: Brain,         label: 'Trend Radar',   group: 'main', badge: 'LIVE' },
  { to: '/ruptures',     icon: Zap,           label: 'Ruptures',      group: 'main' },
  { to: '/clusters',     icon: Layers,        label: 'Clusters',      group: 'main' },
  { to: '/feed',         icon: Rss,           label: 'Live Feed',     group: 'main' },
  { to: '/ablation',     icon: FlaskConical,  label: 'Ablation Study',group: 'research' },
  { to: '/validation',   icon: ShieldCheck,   label: 'Validation',    group: 'research', badge: 'P@5=100%' },
  { to: '/architecture', icon: Network,       label: 'Architecture',  group: 'research' },
]

function useFreshness() {
  const [age, setAge]         = useState<string | null>(null)
  const [stale, setStale]     = useState(false)
  const [spinning, setSpinning] = useState(false)

  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch('/api/gdelt/freshness')
        const d = await r.json()
        setAge(d.age_human)
        setStale(d.is_stale)
        setSpinning(d.refresh_in_progress)
      } catch {}
    }
    poll()
    const id = setInterval(poll, 30_000)
    return () => clearInterval(id)
  }, [])

  return { age, stale, spinning }
}

export default function Layout() {
  const { age, stale, spinning } = useFreshness()

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="w-64 shrink-0 fixed top-0 left-0 h-screen z-30 bg-white border-r border-blue-100 shadow-sm flex flex-col">
        {/* Logo */}
        <div className="px-6 py-5 border-b border-blue-100">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center shadow-md">
              <svg viewBox="0 0 24 24" fill="none" className="w-5 h-5 text-white" stroke="currentColor" strokeWidth={2}>
                <circle cx="12" cy="12" r="2" fill="currentColor" />
                <circle cx="12" cy="12" r="5" strokeDasharray="2 2" />
                <circle cx="12" cy="12" r="9" strokeDasharray="3 3" opacity={0.6} />
              </svg>
            </div>
            <div>
              <div className="font-bold text-slate-900 tracking-tight text-sm">Semantic Radar</div>
              <div className="flex items-center gap-1.5 mt-0.5">
                <div className="w-1.5 h-1.5 rounded-full bg-green-600 animate-pulse" />
                <span className="text-[10px] text-slate-500 font-mono tracking-wide">LIVE · v1.0</span>
              </div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 overflow-y-auto space-y-0.5">
          <p className="px-3 pt-1 pb-2 text-[10px] font-semibold text-slate-500 uppercase tracking-widest">Intelligence</p>
          {nav.filter(n => n.group === 'main').map(({ to, icon: Icon, label, badge }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-500 hover:text-slate-900 hover:bg-blue-50'
                }`
              }
            >
              <Icon size={15} />
              <span className="flex-1">{label}</span>
              {badge && (
                <span className="text-[9px] font-bold bg-green-500 text-white px-1.5 py-0.5 rounded-full tracking-wide">
                  {badge}
                </span>
              )}
            </NavLink>
          ))}
          <p className="px-3 pt-4 pb-2 text-[10px] font-semibold text-slate-500 uppercase tracking-widest">Research</p>
          {nav.filter(n => n.group === 'research').map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-500 hover:text-slate-900 hover:bg-indigo-50'
                }`
              }
            >
              <Icon size={15} />
              <span className="flex-1">{label}</span>
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-4 py-4 border-t border-blue-100 bg-blue-50/50">
          {/* GDELT freshness */}
          <div className={`flex items-center gap-2 mb-3 px-3 py-2 rounded-lg border text-[11px] font-mono ${stale ? 'bg-amber-50 border-amber-200 text-amber-700' : 'bg-green-50 border-green-200 text-green-700'}`}>
            <RefreshCw size={11} className={spinning ? 'animate-spin' : ''} />
            <span className="flex-1">{spinning ? 'Fetching GDELT…' : age ? `GDELT: ${age}` : 'GDELT: checking…'}</span>
            {stale && !spinning && <span className="text-amber-600 font-bold">STALE</span>}
            {!stale && !spinning && <span className="text-green-600 font-bold">LIVE</span>}
          </div>
          <div className="text-[11px] text-slate-500 leading-relaxed">
            <div className="font-semibold text-slate-900/70 mb-0.5">Newton School of Technology</div>
            Navnit Naman · Kanhaiya Kumar
          </div>
          <a
            href="https://github.com/naman394/dynamic-trend-event-detector"
            target="_blank" rel="noreferrer"
            className="mt-2.5 flex items-center gap-1.5 text-xs text-slate-500 hover:text-blue-600 transition-colors"
          >
            <Github size={12} /> View on GitHub
          </a>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 ml-64 min-h-screen grid-bg">
        <Outlet />
      </main>
    </div>
  )
}
