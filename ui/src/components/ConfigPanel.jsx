import { SECTIONS, SPEC_FIELDS, MODEL_PRESETS, PRESET_KEYS, PRESET_MATCH_KEYS,
         QUANT_OPTIONS } from '../config.js'

function ModelPreset({ cfg, setCfg }) {
  const current = MODEL_PRESETS.find((p) => PRESET_MATCH_KEYS.every((k) => cfg.config[k] === p[k]))
  const apply = (name) => {
    const p = MODEL_PRESETS.find((m) => m.name === name)
    if (!p) return
    const patch = Object.fromEntries(PRESET_KEYS.map((k) => [k, p[k]]))
    setCfg({ ...cfg, config: { ...cfg.config, ...patch } })
  }
  const quant = QUANT_OPTIONS.find(([, b]) => b === cfg.config.DTYPE_BYTES)
  const weightGB = cfg.config.PARAMS * cfg.config.DTYPE_BYTES / 1e9
  const minHbm = Math.min(...cfg.cluster.map(
    (c) => (cfg.specs[c.spec]?.hbm_cap ?? Infinity) * (c.gpus || 1))) / 1e9
  return (
    <>
      <div className="field">
        <label>preset</label>
        <select style={{ width: 150 }} value={current?.name ?? 'custom'}
                onChange={(e) => apply(e.target.value)}>
          {!current && <option value="custom">custom</option>}
          {MODEL_PRESETS.map((p) => <option key={p.name}>{p.name}</option>)}
        </select>
      </div>
      <div className="field">
        <label>quantization</label>
        <select style={{ width: 150 }} value={quant?.[0] ?? 'custom'}
                onChange={(e) => {
                  const opt = QUANT_OPTIONS.find(([label]) => label === e.target.value)
                  if (opt) setCfg({ ...cfg, config: { ...cfg.config, DTYPE_BYTES: opt[1] } })
                }}>
          {!quant && <option value="custom">custom</option>}
          {QUANT_OPTIONS.map(([label]) => <option key={label}>{label}</option>)}
        </select>
      </div>
      <div className="sub" style={{ margin: '2px 0 6px' }}>
        weights {weightGB >= 100 ? weightGB.toFixed(0) : weightGB.toPrecision(3)} GB
        {quant ? ` ${quant[0].split(' ')[0]}` : ''}
        {weightGB > minHbm &&
          ` — exceeds smallest node's ${minHbm.toFixed(0)} GB HBM, run will error`}
      </div>
    </>
  )
}

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
          {title === 'Model' && <ModelPreset cfg={cfg} setCfg={setCfg} />}
          {fields.map(([k, label, scale, step]) => (
            <Field key={k} label={label} value={cfg.config[k]} scale={scale} step={step}
                   onChange={(v) => set(k, v)} />
          ))}
        </div>
      ))}
      <div className="sect">
        <div className="hd">Cluster</div>
        <div className="sub" style={{ margin: '2px 0 6px' }}>
          GPUs in a node serve together (tensor parallel); nodes are independent replicas.
        </div>
        {cfg.cluster.map((c, i) => (
          <div className="cluster-row" key={i}>
            <input value={c.name} onChange={(e) => {
              const cl = [...cfg.cluster]; cl[i] = { ...c, name: e.target.value }; setCluster(cl)
            }} />
            <input type="number" min="1" step="1" style={{ width: 44 }}
              title="GPUs in this node" value={c.gpus || 1} onChange={(e) => {
                const cl = [...cfg.cluster]
                cl[i] = { ...c, gpus: Math.max(1, Math.round(+e.target.value) || 1) }
                setCluster(cl)
              }} />
            <span style={{ color: 'var(--muted)' }}>×</span>
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
            { name: `node${cfg.cluster.length}`, spec: Object.keys(cfg.specs)[0], gpus: 1 }])}>
          + add node
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
