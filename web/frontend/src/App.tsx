import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Watchdog from './pages/Watchdog'
import Ruptures from './pages/Ruptures'
import Clusters from './pages/Clusters'
import LiveFeed from './pages/LiveFeed'
import Trends from './pages/Trends'
import Ablation from './pages/Ablation'
import Architecture from './pages/Architecture'
import Validation from './pages/Validation'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="trends" element={<Trends />} />
          <Route path="watchdog" element={<Watchdog />} />
          <Route path="ruptures" element={<Ruptures />} />
          <Route path="clusters" element={<Clusters />} />
          <Route path="feed" element={<LiveFeed />} />
          <Route path="ablation" element={<Ablation />} />
          <Route path="architecture" element={<Architecture />} />
          <Route path="validation" element={<Validation />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
