import { TIER_COLOR } from '../config.js'

export default function GpuCards({ run, t }) {
  return (
    <div className="gpu-cards">
      {run.gpus.map((g) => {
        const mine = run.events.filter((e) => e.gpu === g.name)
        const active = mine.find((e) => e.start <= t && t < e.finish)
        const queued = mine.filter((e) => e.arrival <= t && e.start > t).length
        const busy = mine.reduce((s, e) => s + Math.max(0, Math.min(t, e.finish) - e.start), 0)
        const util = t > 0 ? busy / t : 0
        const kvTok = mine.filter((e) => e.finish <= t)
          .reduce((m, e) => m.set(e.group, e.input_tokens + e.output_tokens), new Map())
        let kv = 0
        kvTok.forEach((v) => { kv += v })
        return (
          <div className="card tile" key={g.name}>
            <div style={{ fontWeight: 650 }}>{g.name}</div>
            <div className="k">util {Math.round(util * 100)}% · queue {queued}</div>
            <div className="bar"><div style={{ width: `${Math.min(100, util * 100)}%` }} /></div>
            <div className="k" style={{ marginTop: 6 }}>
              {active
                ? <span><span className="chip" style={{ background: TIER_COLOR[active.tier] }} />
                    serving #{active.id} (prefix: {active.tier})</span>
                : 'idle'}
            </div>
            <div className="k">KV cached ≈ {Math.min(100, Math.round(kv / g.budget * 100))}% of budget</div>
          </div>
        )
      })}
    </div>
  )
}
