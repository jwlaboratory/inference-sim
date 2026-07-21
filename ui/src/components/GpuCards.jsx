export default function GpuCards({ run, t }) {
  return (
    <div className="gpu-cards">
      {run.nodes.map((nd) => {
        const mine = run.events.filter((e) => e.node === nd.name)
        const active = mine.filter((e) => e.start <= t && t < e.finish)
        const queued = mine.filter((e) => e.arrival <= t && e.start > t).length
        // busy = union of serving intervals up to t (bars overlap when batched)
        const merged = []
        mine.slice().sort((a, b) => a.start - b.start).forEach((e) => {
          const last = merged[merged.length - 1]
          if (last && e.start <= last[1]) last[1] = Math.max(last[1], e.finish)
          else merged.push([e.start, e.finish])
        })
        const busy = merged.reduce((s, [a, b]) => s + Math.max(0, Math.min(t, b) - a), 0)
        const util = t > 0 ? busy / t : 0
        const kvNow = active.reduce((s, e) => s + e.input_tokens + e.output_tokens, 0)
        return (
          <div className="card tile" key={nd.name}>
            <div style={{ fontWeight: 650 }}>{nd.name} · {nd.gpus}×{nd.spec}</div>
            <div className="k">util {Math.round(util * 100)}% · batch {active.length} · queue {queued}</div>
            <div className="bar"><div style={{ width: `${Math.min(100, util * 100)}%` }} /></div>
            <div className="k" style={{ marginTop: 6 }}>
              {active.length ? `decoding ${active.length} seq${active.length > 1 ? 's' : ''}` : 'idle'}
            </div>
            <div className="k">KV reserved ≈ {Math.min(100, Math.round(kvNow / nd.kv_budget * 100))}% of budget</div>
          </div>
        )
      })}
    </div>
  )
}
