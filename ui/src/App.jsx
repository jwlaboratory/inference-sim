import { useState, useEffect, useRef } from 'react'
import { DEFAULT_CONFIG, POLICY_NAMES, TIERS, TIER_COLOR, fmtT } from './config.js'
import ConfigPanel from './components/ConfigPanel.jsx'
import Gantt from './components/Gantt.jsx'
import GpuCards from './components/GpuCards.jsx'

export default function App() {
  const [cfg, setCfg] = useState(DEFAULT_CONFIG)
  const [result, setResult] = useState(null)
  const [stale, setStale] = useState(false)
  const [running, setRunning] = useState(false)
  const [err, setErr] = useState(null)
  const [policy, setPolicy] = useState('cache_aware')
  const [t, setT] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(60)
  const [tip, setTip] = useState(null)
  const raf = useRef()

  const runAll = async () => {
    setRunning(true)
    setErr(null)
    try {
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cfg),
      })
      if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
      setResult(await res.json())
      setStale(false); setT(0); setPlaying(false)
    } catch (e) {
      setErr(`simulation failed: ${e.message}`)
    }
    setRunning(false)
  }
  useEffect(() => { runAll() }, [])
  const edit = (c) => { setCfg(c); setStale(true) }

  const run = result && result.runs[policy]
  useEffect(() => {
    if (!playing || !run) return
    let last = performance.now()
    const step = (now) => {
      const dt = (now - last) / 1000
      last = now
      setT((prev) => {
        const next = prev + dt * speed
        if (next >= run.span) { setPlaying(false); return run.span }
        return next
      })
      raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf.current)
  }, [playing, speed, run])

  if (!result) {
    return (
      <>
        <ConfigPanel cfg={cfg} setCfg={edit} onRun={runAll} stale={stale} running={running} />
        <div className="main">
          {err
            ? <div className="err">{err} — is the server running? <code>python3 server.py</code></div>
            : <div className="sub">running…</div>}
        </div>
      </>
    )
  }

  const m = run.metrics
  const done = run.events.filter((e) => e.finish <= t).length
  const tierCount = Object.fromEntries(TIERS.map((tier) =>
    [tier, run.events.filter((e) => e.tier === tier).length]))

  return (
    <>
      <ConfigPanel cfg={cfg} setCfg={edit} onRun={runAll} stale={stale} running={running} />
      <div className="main">
        {err && <div className="err">{err}</div>}
        <div className="row" style={{ justifyContent: 'space-between', marginBottom: 12 }}>
          <div className="legend">
            <span style={{ color: 'var(--muted)' }}>prefix served from:</span>
            {TIERS.map((tier) => (
              <span key={tier}><span className="chip" style={{ background: TIER_COLOR[tier] }} />
                {tier} ({tierCount[tier]})</span>
            ))}
            <span style={{ color: 'var(--muted)' }}>
              {result.requests.count} requests · mean in {Math.round(result.requests.mean_input)} tok
              · out {Math.round(result.requests.mean_output)} tok · span {fmtT(run.span)}</span>
          </div>
        </div>

        <div className="tiles">
          {[['mean latency', fmtT(m.mean_lat)], ['p95 latency', fmtT(m.p95_lat)],
            ['mean TTFT', fmtT(m.mean_ttft)], ['prefix reuse (any tier)', `${Math.round(m.cache_hit * 100)}%`],
            ['from local HBM', `${Math.round(m.hbm_hit * 100)}%`],
            ['GPU utilization', `${Math.round(m.util * 100)}%`]].map(([k, v]) => (
            <div className="card tile" key={k}><div className="v">{v}</div><div className="k">{k} · {policy}</div></div>
          ))}
        </div>

        <div className="card" style={{ marginBottom: 16 }}>
          <div className="row" style={{ marginBottom: 10 }}>
            <button className="btn" onClick={() => setPlaying((p) => !p)}>
              {playing ? '⏸ pause' : '▶ play'}</button>
            <button className="btn" onClick={() => { setT(0); setPlaying(false) }}>↺</button>
            <select className="btn" value={speed} onChange={(e) => setSpeed(+e.target.value)}>
              {[10, 60, 300, 1000].map((s) => <option key={s} value={s}>{s}×</option>)}
            </select>
            <select className="btn" value={policy}
                    onChange={(e) => { setPolicy(e.target.value); setT(0); setPlaying(false) }}>
              {POLICY_NAMES.map((p) => <option key={p}>{p}</option>)}
            </select>
            <input type="range" min="0" max={run.span} step={run.span / 500} value={t}
                   style={{ flex: 1 }} onChange={(e) => setT(+e.target.value)} />
            <span style={{ fontVariantNumeric: 'tabular-nums', color: 'var(--ink-2)', minWidth: 130 }}>
              {fmtT(t)} · {done}/{run.events.length} done</span>
          </div>
          <Gantt run={run} t={t}
                 onHover={(e, x, y) => setTip(e ? { e, x, y } : null)} />
          <GpuCards run={run} t={t} />
        </div>

        <div className="card">
          <div className="hd">Policy comparison — same trace</div>
          <table>
            <thead><tr><th>policy</th><th>mean lat</th><th>p95 lat</th><th>mean TTFT</th>
              <th>reuse</th><th>hbm hit</th><th>util</th></tr></thead>
            <tbody>
              {POLICY_NAMES.map((p) => {
                const mm = result.runs[p].metrics
                return (
                  <tr key={p} className={p === policy ? 'sel' : ''}
                      onClick={() => { setPolicy(p); setT(0); setPlaying(false) }}>
                    <td>{p}</td><td>{fmtT(mm.mean_lat)}</td><td>{fmtT(mm.p95_lat)}</td>
                    <td>{fmtT(mm.mean_ttft)}</td><td>{Math.round(mm.cache_hit * 100)}%</td>
                    <td>{Math.round(mm.hbm_hit * 100)}%</td>
                    <td>{Math.round(mm.util * 100)}%</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {tip && (
          <div className="tip" style={{ left: tip.x + 14, top: tip.y + 14 }}>
            <div><span className="chip" style={{ background: TIER_COLOR[tip.e.tier] }} />
              <b>#{tip.e.id}</b> {tip.e.group} → {tip.e.gpu}</div>
            <div style={{ color: 'var(--ink-2)', marginTop: 3 }}>
              in {tip.e.input_tokens} tok (prefix {tip.e.prefix_tokens}) · out {tip.e.output_tokens} tok<br />
              prefix: {tip.e.hit ? `${tip.e.hit} tok from ${tip.e.tier}` : 'miss'}<br />
              wait {fmtT(tip.e.start - tip.e.arrival)} · prefill {fmtT(tip.e.reuse + tip.e.prefill)} ·
              decode {fmtT(tip.e.decode)}
            </div>
          </div>
        )}
      </div>
    </>
  )
}
