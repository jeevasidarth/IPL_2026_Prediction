import { useState, useEffect } from 'react'
import { ComposedChart, Line, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import './App.css'

function App() {
  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const [metadata, setMetadata] = useState({ teams: [], squads: {}, venues: [] })
  const [selection, setSelection] = useState({ 
    team1: 'CSK', 
    team2: 'RCB', 
    team1_xi: [],
    team2_xi: [],
    venue: 'M Chinnaswamy Stadium, Bengaluru',
    afternoon: 0,
    hum_i1: null,
    dew_i1: null,
    hum_i2: null,
    dew_i2: null
  })
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState('connecting')
  const [view, setView] = useState('match') // 'match', 'tournament', or 'versus'
  const [tournamentResults, setTournamentResults] = useState(null)
  const [compSelection, setCompSelection] = useState({ p1: '', p2: '', venue: 'Wankhede Stadium, Mumbai' })
  const [compResult, setCompResult] = useState(null)
  const [compLoading, setCompLoading] = useState(false)

  // Derived list of all players
  const allPlayers = Object.values(metadata.squads).reduce((acc, squad) => {
    return [...acc, ...squad.Batters, ...squad.Bowlers];
  }, []).sort();
  const uniquePlayers = Array.from(new Set(allPlayers));

  const handleCompare = async () => {
    if (!compSelection.p1 || !compSelection.p2) return;
    setCompLoading(true)
    try {
      const res = await fetch(`${API_BASE}/compare-players`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          player1: compSelection.p1,
          player2: compSelection.p2,
          venue: compSelection.venue
        })
      })
      const data = await res.json()
      if (data.status === 'success') setCompResult(data.comparison)
    } catch (err) {
      console.error(err)
    } finally {
      setCompLoading(false)
    }
  }

  // Fetch Metadata on Load
  useEffect(() => {
    fetch(`${API_BASE}/metadata`)
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

    // Fetch Tournament Results
    fetch(`${API_BASE}/tournament-results`)
      .then(res => res.json())
      .then(data => {
        if (data.status === 'success') {
          setTournamentResults(data.results)
        }
      })
      .catch(err => console.error("Tournament Results Fetch Error:", err))
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
      const res = await fetch(`${API_BASE}/predict`, {
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

  const handleSmartPick = async (tKey) => {
    const teamName = selection[tKey];
    try {
      const res = await fetch(`${API_BASE}/smart-pick`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          team1: teamName, 
          venue: selection.venue,
          team2: '', // Dummy
          team1_xi: [], team2_xi: [] // Dummy
        })
      });
      const data = await res.json();
      if (data.status === 'success') {
        setSelection(prev => ({ ...prev, [`${tKey}_xi`]: data.xi.slice(0, 12) }));
      }
    } catch (err) {
      console.error("Smart Pick UI Error:", err);
    }
  };

  const [inspectingPlayer, setInspectingPlayer] = useState(null)

  const RosterProjections = ({ projections, title }) => (
    <div className="roster-container">
      <h4 className="roster-title">{title}</h4>
      <div className="roster-grid">
        {Object.entries(projections).map(([name, stats], i) => (
          <div 
            key={i} 
            className="player-stat-strip glass-card"
            onClick={() => setInspectingPlayer({ name, ...stats })}
          >
            <span className="p-name">{name}</span>
            <span className="p-val neon-text">
              {stats.role === 'batter' ? `${Math.round(stats.runs)} runs` : `${stats.wkts.toFixed(1)} wkts`}
            </span>
          </div>
        ))}
      </div>
    </div>
  )

  const PlayerPhaseGraph = ({ role, phases }) => {
    if (!phases || phases.length === 0) return <div className="insight-note">Detailed phase graph not available yet.</div>;
    return (
      <div className="player-graph-container glass-card">
        <h4 className="graph-title">Phase-Wise Performance Graph</h4>
        <div className="pg-grid">
          {phases.map((p, i) => (
            <div key={i} className="pg-bar-group">
              <div className="pg-labels">
                <span className="p-phase-name">{p.phase}</span>
                {role === 'batter' ? 
                  <span className="pg-sr">{p.sr} SR <span className="pg-sub">({p.runs} runs)</span></span> : 
                  <span className="pg-econ">{p.econ} Econ <span className="pg-sub">({p.wkts} wkts)</span></span>
                }
              </div>
              <div className="pg-track">
                {role === 'batter' ? (
                  <div className="pg-fill pg-fill-bat" style={{ width: `${Math.min((p.sr/250)*100, 100)}%` }}></div>
                ) : (
                  <div className="pg-fill pg-fill-bowl" style={{ width: `${Math.min((p.econ/15)*100, 100)}%` }}></div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const PlayerDetailModal = ({ player, onClose }) => {
    if (!player) return null;
    return (
      <div className="modal-overlay" onClick={onClose}>
        <div className="modal-content glass-card" onClick={e => e.stopPropagation()}>
          <button className="close-btn" onClick={onClose}>×</button>
          <h2 className="neon-text">{player.name}</h2>
          <p className="role-tag">{player.role.toUpperCase()}</p>
          
          <div className="detail-grid">
            {player.role === 'batter' ? (
              <>
                <div className="detail-item">
                  <span className="d-label">PROJECTED RUNS</span>
                  <span className="d-val">{Math.round(player.runs)}</span>
                </div>
                <div className="detail-item">
                  <span className="d-label">EXPECTED BALLS</span>
                  <span className="d-val">{Math.round(player.balls)}</span>
                </div>
                <div className="detail-item">
                  <span className="d-label">PREDICTED SR</span>
                  <span className="d-val">{player.balls > 0 ? ((player.runs/player.balls)*100).toFixed(1) : '0.0'}</span>
                </div>
              </>
            ) : (
              <>
                <div className="detail-item">
                  <span className="d-label">EST. ECONOMY</span>
                  <span className="d-val">{player.econ.toFixed(1)}</span>
                </div>
                <div className="detail-item">
                  <span className="d-label">EXPECTED WICKETS</span>
                  <span className="d-val">{player.wkts.toFixed(1)}</span>
                </div>
                <div className="detail-item">
                  <span className="d-label">OVERS TO BE BOWLED</span>
                  <span className="d-val">{player.overs.toFixed(1)}</span>
                </div>
              </>
            )}
          </div>
          
          <PlayerPhaseGraph role={player.role} phases={player.phases} />
          
          <p className="insight-note">Predictions factor in Phase Intensity (Powerplay/Middle/Death) and current Stadium conditions.</p>
        </div>
      </div>
    )
  }

  const IntensityCurve = ({ phases }) => (
    <div className="intensity-container">
      <h4 className="intensity-title">Match Intensity Curve (SR & Econ)</h4>
      <div className="intensity-grid">
        {phases.map((p, i) => (
          <div key={i} className="intensity-bar-group">
            <div className="intensity-labels">
              <span>{p.phase}</span>
              <span className="sr-val">{p.sr} SR</span>
            </div>
            <div className="intensity-track">
              <div 
                className="intensity-fill sr-fill" 
                style={{ width: `${Math.min((p.sr/250)*100, 100)}%` }}
              ></div>
            </div>
            <div className="intensity-track econ-track">
              <div 
                className="intensity-fill econ-fill" 
                style={{ width: `${Math.min((p.econ/15)*100, 100)}%` }}
              ></div>
              <span className="econ-val">{p.econ} Econ</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )

const SearchableSelect = ({ value, onChange, options, placeholder }) => {
  const [search, setSearch] = useState('');
  const [open, setOpen] = useState(false);
  
  const filtered = options
    .filter(o => !search || o.toLowerCase().includes(search.toLowerCase()))
    .slice(0, 15);

  return (
    <div className="searchable-select-container">
      <input 
        type="text" 
        placeholder={placeholder} 
        value={open ? search : (value || '')} 
        onChange={e => {setSearch(e.target.value); setOpen(true)}}
        onFocus={() => {setOpen(true); setSearch('')}}
        onBlur={() => setTimeout(() => setOpen(false), 300)}
        className="search-input"
      />
      {open && (
        <div className="search-dropdown glass-card">
          {filtered.map(o => (
            <div 
              key={o} 
              className={`search-option ${value === o ? 'selected' : ''}`} 
              /* Use onMouseDown instead of onClick to beat the onBlur event */
              onMouseDown={(e) => {
                e.preventDefault(); // Prevent focus loss from input
                onChange(o);
                setOpen(false);
                setSearch(o);
              }}
            >
              {o}
              {value === o && <span className="check-mark"> ✓</span>}
            </div>
          ))}
          {filtered.length === 0 && <div className="search-option disabled">No matches found</div>}
        </div>
      )}
    </div>
  );
}

const MatchWorm = ({ scenarioA, scenarioB }) => {
  if (!scenarioA.worm_data || !scenarioB.worm_data) return null;
  
  const lineData = scenarioA.worm_data.map((val, i) => ({
    over: i,
    [scenarioA.bat_first]: val,
    [scenarioB.bat_first]: scenarioB.worm_data[i]
  }));

  const wkt1Data = (scenarioA.worm_wickets || []).map(w => ({
    over: w.over,
    [scenarioA.bat_first]: w.runs
  }));

  const wkt2Data = (scenarioB.worm_wickets || []).map(w => ({
    over: w.over,
    [scenarioB.bat_first]: w.runs
  }));

  return (
    <div className="worm-chart-container glass-card">
      <h3 className="neon-text">CUMULATIVE RUN PROGRESS & WICKETS (WORM)</h3>
      <ResponsiveContainer width="100%" height={400} aspect={2}>
        <ComposedChart data={lineData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
            <XAxis dataKey="over" stroke="rgba(255,255,255,0.5)" label={{ value: 'OVERS', position: 'insideBottom', offset: -5, fill: 'white', fontSize: 10 }} />
            <YAxis stroke="rgba(255,255,255,0.5)" label={{ value: 'RUNS', angle: -90, position: 'insideLeft', fill: 'white', fontSize: 10 }} />
            <Tooltip 
              contentStyle={{ background: 'rgba(15, 23, 42, 0.9)', border: '1px solid #00f2ff', borderRadius: '10px' }}
              itemStyle={{ color: '#00f2ff' }}
            />
            <Legend verticalAlign="top" height={36}/>
            <Line type="monotone" dataKey={scenarioA.bat_first} stroke="#00f2ff" strokeWidth={3} dot={false} activeDot={{ r: 8 }} animationDuration={2000} />
            <Line type="monotone" dataKey={scenarioB.bat_first} stroke="#ff0055" strokeWidth={3} dot={false} activeDot={{ r: 8 }} animationDuration={2000} />
            
            <Scatter data={wkt1Data} fill="#fff" stroke="#ff0055" strokeWidth={2} name={`${scenarioA.bat_first} WKT`} />
            <Scatter data={wkt2Data} fill="#fff" stroke="#00f2ff" strokeWidth={2} name={`${scenarioB.bat_first} WKT`} />
          </ComposedChart>
        </ResponsiveContainer>
      <p className="insight-note">Circles indicate projected wickets based on phase-level pressure and bowler efficiency.</p>
    </div>
  );
}

  const StandingsTable = ({ standings }) => (
    <div className="standings-container glass-card">
      <h3 className="neon-text">IPL 2026 Points Table</h3>
      <table className="standings-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Team</th>
            <th>P</th>
            <th>W</th>
            <th>L</th>
            <th>NRR</th>
            <th>Pts</th>
          </tr>
        </thead>
        <tbody>
          {standings.map((s, i) => (
            <tr key={i} className={i < 4 ? 'playoff-zone' : ''}>
              <td>{s.rank}</td>
              <td className="team-name-cell">{s.team}</td>
              <td>{s.played}</td>
              <td>{s.wins}</td>
              <td>{s.losses}</td>
              <td className={s.nrr >= 0 ? 'pos-nrr' : 'neg-nrr'}>{s.nrr}</td>
              <td className="pts-cell">{s.points}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )

  const PlayoffBracket = ({ playoffs, champion }) => (
    <div className="playoffs-container">
      <div className="champion-reveal glass-card neon-border">
        <span className="trophy-icon">🏆</span>
        <h2>IPL 2026 CHAMPIONS</h2>
        <h1 className="champion-name neon-text">{champion}</h1>
      </div>

      <div className="bracket-grid">
        <div className="bracket-card glass-card">
          <h4>Qualifier 1</h4>
          <div className="match-row">
            <span>{playoffs.qualifier_1.team1} vs {playoffs.qualifier_1.team2}</span>
            <span className="winner-label">Winner: {playoffs.qualifier_1.winner}</span>
          </div>
        </div>
        <div className="bracket-card glass-card">
          <h4>Eliminator</h4>
          <div className="match-row">
            <span>{playoffs.eliminator.team1} vs {playoffs.eliminator.team2}</span>
            <span className="winner-label">Winner: {playoffs.eliminator.winner}</span>
          </div>
        </div>
        <div className="bracket-card glass-card">
          <h4>Qualifier 2</h4>
          <div className="match-row">
            <span>{playoffs.qualifier_2.team1} vs {playoffs.qualifier_2.team2}</span>
            <span className="winner-label">Winner: {playoffs.qualifier_2.winner}</span>
          </div>
        </div>
        <div className="bracket-card glass-card final-card highlight-border">
          <h4>The Grand Final</h4>
          <div className="match-row">
            <span>{playoffs.final.team1} vs {playoffs.final.team2}</span>
            <span className="winner-label champion-glow">Champion: {playoffs.final.winner}</span>
          </div>
        </div>
      </div>
    </div>
  )

  return (
    <div className="app-container">
      {/* Background Overlay */}
      <div className="bg-overlay"></div>
      
      <header className="header">
        <h1 className="neon-text">IPL 2026 <span className="highlight">Hierarchical Engine</span></h1>
        <p className="subtitle">Bottom-Up Player Performance Analytics</p>
        
        <div className="view-toggle">
          <button 
            className={`toggle-btn ${view === 'match' ? 'active' : ''}`}
            onClick={() => setView('match')}
          >
            Match Predictor
          </button>
          <button 
            className={`toggle-btn ${view === 'tournament' ? 'active' : ''}`}
            onClick={() => setView('tournament')}
          >
            Tournament View
          </button>
          <button 
            className={`toggle-btn ${view === 'versus' ? 'active' : ''}`}
            onClick={() => setView('versus')}
          >
            PvP Duel
          </button>
        </div>

        <div className={`status-badge ${apiStatus}`}>
          {apiStatus === 'connected' ? '● System Online' : apiStatus === 'connecting' ? '○ Connecting...' : '!! System Offline'}
        </div>
      </header>

      <main className="dashboard">
        <PlayerDetailModal player={inspectingPlayer} onClose={() => setInspectingPlayer(null)} />

        {view === 'match' ? (
          <>
            {/* Squad Selection Section */}
            <section className="squad-builder-grid">
              {['team1', 'team2'].map(tKey => {
                const teamName = selection[tKey];
                const squad = metadata.squads[teamName] || { Batters: [], Bowlers: [] };
                const allPlayers = Array.from(new Set([...squad.Batters, ...squad.Bowlers])).sort();
                const selectedCount = selection[`${tKey}_xi`].length;

                return (
                  <div key={tKey} className="glass-card squad-box">
                    <div className="squad-header">
                      <h3 className="neon-text">{teamName} Squad Builder ({selectedCount}/12)</h3>
                      <button 
                        className="smart-pick-btn" 
                        onClick={() => handleSmartPick(tKey)}
                        title="AI Auto-Select Optimal XI"
                      >
                        ✨ AI AUTO-PICK
                      </button>
                    </div>
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
                <label>I1 Humidity {selection.hum_i1 ? `(${selection.hum_i1}%)` : '(Auto)'}</label>
                <input 
                  type="number" min="30" max="95" placeholder="Auto"
                  value={selection.hum_i1 || ''} 
                  onChange={e => setSelection({...selection, hum_i1: e.target.value ? parseInt(e.target.value) : null})} 
                  className="env-input"
                />
              </div>

              <div className="input-group env-group">
                <label>I1 Dew {selection.dew_i1 ? `(${selection.dew_i1})` : '(Auto)'}</label>
                <input 
                  type="number" min="10" max="40" placeholder="Auto"
                  value={selection.dew_i1 || ''} 
                  onChange={e => setSelection({...selection, dew_i1: e.target.value ? parseInt(e.target.value) : null})} 
                  className="env-input"
                />
              </div>

              <div className="input-group env-group i2-highlight">
                <label>I2 Humidity {selection.hum_i2 ? `(${selection.hum_i2}%)` : '(Auto)'}</label>
                <input 
                  type="number" min="30" max="95" placeholder="Auto"
                  value={selection.hum_i2 || ''} 
                  onChange={e => setSelection({...selection, hum_i2: e.target.value ? parseInt(e.target.value) : null})} 
                  className="env-input"
                />
              </div>

              <div className="input-group env-group i2-highlight">
                <label>I2 Dew {selection.dew_i2 ? `(${selection.dew_i2})` : '(Auto)'}</label>
                <input 
                  type="number" min="10" max="40" placeholder="Auto"
                  value={selection.dew_i2 || ''} 
                  onChange={e => setSelection({...selection, dew_i2: e.target.value ? parseInt(e.target.value) : null})} 
                  className="env-input"
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
                        <RosterProjections projections={scenario.projections} title="Squad Projections" />
                      </div>
                      <div className="analysis-col border-left">
                        <IntensityCurve phases={scenario.phase_stats || []} />
                      </div>
                    </div>
                  </div>
                ))}
              </section>
            )}

            {prediction && prediction.status === 'success' && (
              <section className="insight-section">
                <MatchWorm scenarioA={prediction.scenario_a} scenarioB={prediction.scenario_b} />
              </section>
            )}
          </>
        ) : view === 'tournament' ? (
          <div className="tournament-dashboard">
            {tournamentResults ? (
              <>
                <div className="tournament-grid">
                  <StandingsTable standings={tournamentResults.standings} />
                  <PlayoffBracket playoffs={tournamentResults.playoffs} champion={tournamentResults.champion} />
                </div>
              </>
            ) : (
              <div className="glass-card empty-state">
                <h3>Tournament Simulation Needed</h3>
                <p>Run the background simulation script to view the predicted IPL 2026 table.</p>
              </div>
            )}
          </div>
        ) : (
          <div className="versus-dashboard">
            <section className="glass-card comparison-controls">
              <h1 className="neon-text">Matchup Duel</h1>
              <div className="comp-selectors">
                <div className="input-group">
                  <label>Search Player 1</label>
                  <SearchableSelect 
                    value={compSelection.p1} 
                    onChange={val => setCompSelection({...compSelection, p1: val})} 
                    options={uniquePlayers}
                    placeholder="e.g. Kohli"
                  />
                </div>
                <div className="vs-badge-mini">VS</div>
                <div className="input-group">
                  <label>Search Player 2</label>
                  <SearchableSelect 
                    value={compSelection.p2} 
                    onChange={val => setCompSelection({...compSelection, p2: val})} 
                    options={uniquePlayers}
                    placeholder="e.g. Bumrah"
                  />
                </div>
              </div>
              <button 
                className={`predict-btn ${compLoading ? 'loading' : ''}`} 
                onClick={handleCompare}
                disabled={!compSelection.p1 || !compSelection.p2 || compLoading}
              >
                {compLoading ? 'Simulating Duel...' : 'RUN MATCHUP SIMULATION'}
              </button>
            </section>

            {compResult && (
              <section className="comp-results-area">
                {compResult.type === 'matchup' ? (
                  <div className="glass-card matchup-result">
                    <div className="matchup-header">
                      <div className="m-player">
                        <span className="m-role">BATTER</span>
                        <h2 className="neon-text">{compResult.batter}</h2>
                      </div>
                      <div className="m-vs">VS</div>
                      <div className="m-player">
                        <span className="m-role">BOWLER</span>
                        <h2 className="neon-text">{compResult.bowler}</h2>
                      </div>
                    </div>

                    <div className="matchup-stats">
                      {compResult.data.map((ph, i) => (
                        <div key={i} className="phase-comp-card glass-card">
                          <h4>{ph.phase} Phase</h4>
                          <div className="m-grid">
                            <div className="m-item">
                              <span className="m-lab">Pred. Strike Rate</span>
                              <span className="m-val neon-text">{ph.sr}</span>
                            </div>
                            <div className="m-item">
                              <span className="m-lab">Expected Econ</span>
                              <span className="m-val">{ph.econ}</span>
                            </div>
                            <div className="m-item">
                              <span className="m-lab">Wkt Prob (%)</span>
                              <span className="m-val highlight">{Math.round(ph.wkt_prob * 100)}%</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="glass-card comparison-result">
                    <h3 className="neon-text">{compResult.role.toUpperCase()} COMPARISON</h3>
                    <div className="comp-matrix">
                      <div className="comp-col">
                        <h2>{compResult.player1.name}</h2>
                        {compResult.role === 'batter' ? (
                          <div className="comp-stats-list">
                            <div className="s-item"><label>Recent Avg</label><span>{compResult.player1.stats.recent_form_avg}</span></div>
                            <div className="s-item"><label>Recent SR</label><span>{compResult.player1.stats.recent_form_sr}</span></div>
                            <div className="s-item"><label>Global Avg</label><span>{compResult.player1.stats.global_avg}</span></div>
                          </div>
                        ) : (
                          <div className="comp-stats-list">
                            <div className="s-item"><label>Recent Econ</label><span>{compResult.player1.stats.recent_form_economy}</span></div>
                            <div className="s-item"><label>Global Econ</label><span>{compResult.player1.stats.global_econ}</span></div>
                            <div className="s-item"><label>Wkt Avg</label><span>{compResult.player1.stats.global_wkt_avg}</span></div>
                          </div>
                        )}
                      </div>
                      <div className="comp-vs-divider">VS</div>
                      <div className="comp-col">
                        <h2>{compResult.player2.name}</h2>
                        {compResult.role === 'batter' ? (
                          <div className="comp-stats-list">
                            <div className="s-item"><label>Recent Avg</label><span>{compResult.player2.stats.recent_form_avg}</span></div>
                            <div className="s-item"><label>Recent SR</label><span>{compResult.player2.stats.recent_form_sr}</span></div>
                            <div className="s-item"><label>Global Avg</label><span>{compResult.player2.stats.global_avg}</span></div>
                          </div>
                        ) : (
                          <div className="comp-stats-list">
                            <div className="s-item"><label>Recent Econ</label><span>{compResult.player2.stats.recent_form_economy}</span></div>
                            <div className="s-item"><label>Global Econ</label><span>{compResult.player2.stats.global_econ}</span></div>
                            <div className="s-item"><label>Wkt Avg</label><span>{compResult.player2.stats.global_wkt_avg}</span></div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </section>
            )}
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Verified against 2025 hold-out set | MAE: 16.3 runs</p>
      </footer>
    </div>
  )
}

export default App
