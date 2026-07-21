import { SECTIONS, SPEC_FIELDS } from '../config.js'

function Field({ label, value, scale, step, onChange }) {
  return (
    <div className="field">
      <label>{label}</label>
      <input type="number" step={step} value={+(value / scale).toPrecision(6)}
             onChange={(e) => onChange(parseFloat(e.target.value) * scale || 0)} />
    </div>
  )
}

export default function ConfigPanel({ cfg, setCfg, onRun, stale, running }) {
  const set = (k, v) => setCfg({ ...cfg, config: { ...cfg.config, [k]: v } })
  const setSpec = (name, k, v) =>
    setCfg({ ...cfg, specs: { ...cfg.specs, [name]: { ...cfg.specs[name], [k]: v } } })
  const setCluster = (cluster) => setCfg({ ...cfg, cluster })

  return (
    <div className="panel">
      <h1>Inference Simulator</h1>
      <div className="sub">Replays a window of a real trace. Edit any tunable, then run.</div>
      <div className="row" style={{ marginBottom: 14 }}>
        <button className="btn primary" onClick={onRun} disabled={running}>
          {running ? 'running…' : 'Run simulation'}</button>
        {stale && !running && <span className="stale">config changed — re-run</span>}
      </div>
      <div className="sect">
        <div className="hd">Dataset</div>
        <div className="field">
          <label>HF dataset</label>
          <input style={{ width: 170 }} value={cfg.config.DATASET}
                 onChange={(e) => set('DATASET', e.target.value)} />
        </div>
      </div>
      {SECTIONS.map(([title, fields]) => (
        <div className="sect" key={title}>
          <div className="hd">{title}</div>
          {fields.map(([k, label, scale, step]) => (
            <Field key={k} label={label} value={cfg.config[k]} scale={scale} step={step}
                   onChange={(v) => set(k, v)} />
          ))}
        </div>
      ))}
      <div className="sect">
        <div className="hd">Cluster</div>
        {cfg.cluster.map((c, i) => (
          <div className="cluster-row" key={i}>
            <input value={c.name} onChange={(e) => {
              const cl = [...cfg.cluster]; cl[i] = { ...c, name: e.target.value }; setCluster(cl)
            }} />
            <select value={c.spec} onChange={(e) => {
              const cl = [...cfg.cluster]; cl[i] = { ...c, spec: e.target.value }; setCluster(cl)
            }}>
              {Object.keys(cfg.specs).map((s) => <option key={s}>{s}</option>)}
            </select>
            <button className="x" title="remove"
              onClick={() => setCluster(cfg.cluster.filter((_, j) => j !== i))}>✕</button>
          </div>
        ))}
        <button className="btn" style={{ marginTop: 4 }}
          onClick={() => setCluster([...cfg.cluster,
            { name: `gpu${cfg.cluster.length}`, spec: Object.keys(cfg.specs)[0] }])}>
          + add GPU
        </button>
      </div>
      {Object.entries(cfg.specs).map(([name, spec]) => (
        <div className="sect" key={name}>
          <div className="hd">{name} spec</div>
          {SPEC_FIELDS.map(([k, label, scale]) => (
            <Field key={k} label={label} value={spec[k]} scale={scale} step={1}
                   onChange={(v) => setSpec(name, k, v)} />
          ))}
        </div>
      ))}
      <div className="sect">
        <div className="hd">Cache tiers</div>
        <div className="field">
          <label>persist KV to disk</label>
          <input type="checkbox" style={{ width: 'auto' }} checked={cfg.config.DISK_CACHE}
                 onChange={(e) => set('DISK_CACHE', e.target.checked)} />
        </div>
      </div>
    </div>
  )
}
