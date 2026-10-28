import { useEffect, useState } from 'react'
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import AboutPage from './pages/AboutPage'
import TrainingPage from './pages/TrainingPage'
import InferencePage from './pages/InferencePage'

export default function App() {
  const [hasModel, setHasModel] = useState<boolean | null>(null)

  useEffect(() => {
    fetch('/api/models/active')
      .then(r => r.json())
      .then(d => setHasModel(d.model !== null))
      .catch(() => setHasModel(false))
  }, [])

  return (
    <BrowserRouter>
      <nav className="nav">
        <NavLink to="/">About</NavLink>
        <NavLink to="/training">Train & Evaluate</NavLink>
        <NavLink to="/inference">Inference</NavLink>
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<AboutPage />} />
          <Route path="/training" element={<TrainingPage hasModel={hasModel} />} />
          <Route path="/inference" element={<InferencePage hasModel={hasModel} />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}
