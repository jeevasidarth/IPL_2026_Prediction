import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [metadata, setMetadata] = useState({ teams: [], squads: {}, venues: [] })
  const [selection, setSelection] = useState({ 
    team1: 'CSK', 
    team2: 'RCB', 
    team1_xi: [],
    team2_xi: [],
    venue: 'M Chinnaswamy Stadium, Bengaluru',
    afternoon: 0,
    hum_i1: 70,
    dew_i1: 22,
    hum_i2: 85,
    dew_i2: 30
  })
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState('connecting')

  // Fetch Metadata on Load
  useEffect(() => {
    fetch('http://localhost:8000/metadata')
      .then(res => {
        if (!res.ok) throw new Error("Backend Error")
        return res.json()
      })
      .then(data => {
        setMetadata(data)
        setApiStatus('connected')
      })
      .catch(err => {
        console.error("Backend not reached:", err)
        setApiStatus('error')
      })
  }, [])

  const togglePlayer = (teamKey, playerName) => {
    const field = teamKey === 'team1' ? 'team1_xi' : 'team2_xi';
    const current = selection[field];
    if (current.includes(playerName)) {
      setSelection({...selection, [field]: current.filter(p => p !== playerName)});
    } else if (current.length < 12) {
      setSelection({...selection, [field]: [...current, playerName]});
    }
  }

  const handlePredict = async () => {
    setLoading(true)
    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(selection)
      })
      const data = await res.json()
      setPrediction(data)
    } catch (err) {
      alert("Error: Ensure backend server is running (python scripts/server.py)")
    } finally {
      setLoading(false)
    }
  }

  const PerformanceList = ({ players, type }) => (
    <div className="performer-list">
      {players.map((p, i) => (
        <div key={i} className="performer-row">
          <span className="player-name">{p.name || p[0]}</span>
          <span className="player-stat neon-text">
            {type === 'bat' ? `${Math.round(p.runs || p[1])} runs` : `${(p.econ || p[1]).toFixed(1)} econ`}
          </span>
        </div>
      ))}
    </div>
  )

  return (
    <div className="app-container">
      {/* Background Overlay */}
      <div className="bg-overlay"></div>
      
      <header className="header">
        <h1 className="neon-text">IPL 2026 <span className="highlight">Hierarchical Engine</span></h1>
        <p className="subtitle">Bottom-Up Player Performance Analytics</p>
        <div className={`status-badge ${apiStatus}`}>
          {apiStatus === 'connected' ? '● System Online' : apiStatus === 'connecting' ? '○ Connecting...' : '!! System Offline'}
        </div>
      </header>

      <main className="dashboard">
        {/* Squad Selection Section */}
        <section className="squad-builder-grid">
          {['team1', 'team2'].map(tKey => {
            const teamName = selection[tKey];
            const squad = metadata.squads[teamName] || { Batters: [], Bowlers: [] };
            const allPlayers = Array.from(new Set([...squad.Batters, ...squad.Bowlers])).sort();
            const selectedCount = selection[`${tKey}_xi`].length;

            return (
              <div key={tKey} className="glass-card squad-box">
                <h3 className="neon-text">{teamName} Squad Builder ({selectedCount}/12)</h3>
                <p className="hint">Select 11 + 1 Impact Player</p>
                <div className="player-pool">
                  {allPlayers.map(p => (
                    <button 
                      key={p} 
                      className={`player-chip ${selection[`${tKey}_xi`].includes(p) ? 'active' : ''}`}
                      onClick={() => togglePlayer(tKey, p)}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
        </section>

        {/* Global Settings Section */}
        <section className="glass-card selector-panel">
          <div className="input-group">
            <label>Team 1 (Bat First Selection)</label>
            <select value={selection.team1} onChange={e => setSelection({...selection, team1: e.target.value})}>
              {metadata.teams.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>

          <div className="vs-badge neon-text">VS</div>

          <div className="input-group">
            <label>Team 2</label>
            <select value={selection.team2} onChange={e => setSelection({...selection, team2: e.target.value})}>
              {metadata.teams.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>

          <div className="input-group venue-group">
            <label>Match Venue</label>
            <select value={selection.venue} onChange={e => setSelection({...selection, venue: e.target.value})}>
              {metadata.venues.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>

          <div className="input-group env-group">
            <label>I1 Humidity ({selection.hum_i1}%)</label>
            <input 
              type="range" min="30" max="95" 
              value={selection.hum_i1} 
              onChange={e => setSelection({...selection, hum_i1: parseInt(e.target.value)})} 
            />
          </div>

          <div className="input-group env-group">
            <label>I1 Dew ({selection.dew_i1})</label>
            <input 
              type="range" min="10" max="40" 
              value={selection.dew_i1} 
              onChange={e => setSelection({...selection, dew_i1: parseInt(e.target.value)})} 
            />
          </div>

          <div className="input-group env-group i2-highlight">
            <label>I2 Humidity ({selection.hum_i2}%)</label>
            <input 
              type="range" min="30" max="95" 
              value={selection.hum_i2} 
              onChange={e => setSelection({...selection, hum_i2: parseInt(e.target.value)})} 
            />
          </div>

          <div className="input-group env-group i2-highlight">
            <label>I2 Dew ({selection.dew_i2})</label>
            <input 
              type="range" min="10" max="40" 
              value={selection.dew_i2} 
              onChange={e => setSelection({...selection, dew_i2: parseInt(e.target.value)})} 
            />
          </div>

          <button 
            className={`predict-btn ${loading ? 'loading' : ''} ${apiStatus !== 'connected' || selection.team1_xi.length < 11 || selection.team2_xi.length < 11 ? 'disabled' : ''}`} 
            onClick={handlePredict}
            disabled={apiStatus !== 'connected' || loading || selection.team1_xi.length < 11 || selection.team2_xi.length < 11}
          >
            {loading ? 'Simulating...' : 
             apiStatus !== 'connected' ? 'READYING SYSTEM...' :
             (selection.team1_xi.length < 11 || selection.team2_xi.length < 11) ? 'SELECT PLAYING XI' : 
             'EXECUTE PREDICTION'}
          </button>
        </section>

        {/* Results Section */}
        {prediction && prediction.status === 'success' && (
          <section className="results-container">
            {[prediction.scenario_a, prediction.scenario_b].map((scenario, idx) => (

              <div key={idx} className="glass-card result-card">
                <div className="card-header">
                  <h3>If {scenario.bat_first} Bats First</h3>
                  <div className="win-prob-pill neon-border">
                    Win Prob: {scenario.win_prob}%
                  </div>
                </div>

                <div className="score-summary">
                  <span className="label">PROJECTED TOTAL</span>
                  <span className="big-score neon-text">{scenario.target}</span>
                </div>

                <div className="analysis-grid">
                  <div className="analysis-col">
                    <h4>Top Batter Projections</h4>
                    <PerformanceList players={scenario.notable_performers} type="bat" />
                  </div>
                  <div className="analysis-col border-left">
                    <h4>Bowling Economics</h4>
                    <PerformanceList players={scenario.bowlers} type="bowl" />
                  </div>
                </div>
              </div>
            ))}
          </section>
        )}
      </main>

      <footer className="footer">
        <p>Verified against 2025 hold-out set | MAE: 16.3 runs</p>
      </footer>
    </div>
  )
}

export default App
