import { TIER_COLOR, fmtT } from '../config.js'

export default function Gantt({ run, t, onHover }) {
  const W = 1000, barH = 20, qH = 14, laneH = barH + qH + 10, axisH = 22
  const names = run.gpus.map((g) => g.name)
  const H = names.length * laneH + axisH
  const x = (s) => (s / run.span) * W
  const tickStep = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1800, 3600]
    .find((s) => run.span / s <= 10) || 7200
  const ticks = []
  for (let s = 0; s <= run.span; s += tickStep) ticks.push(s)

  // Queue depth per GPU: requests arrived but not yet started, as step segments.
  const series = names.map((n) => {
    const deltas = new Map()
    run.events.filter((e) => e.gpu === n).forEach((e) => {
      deltas.set(e.arrival, (deltas.get(e.arrival) || 0) + 1)
      deltas.set(e.start, (deltas.get(e.start) || 0) - 1)
    })
    const times = [...deltas.keys()].sort((a, b) => a - b)
    const segs = []
    let depth = 0, prev = 0
    for (const tt of times) {
      if (depth > 0 && tt > prev) segs.push({ t0: prev, t1: tt, depth })
      depth += deltas.get(tt)
      prev = tt
    }
    return segs
  })
  const maxQ = Math.max(1, ...series.flat().map((s) => s.depth))

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }}>
      {ticks.map((s) => (
        <g key={s}>
          <line x1={x(s)} x2={x(s)} y1={0} y2={H - axisH} stroke="var(--grid)" strokeWidth="1" />
          <text x={x(s)} y={H - 6} fontSize="10" fill="var(--muted)" textAnchor="middle"
                style={{ fontVariantNumeric: 'tabular-nums' }}>{fmtT(s)}</text>
        </g>
      ))}
      {names.map((n, i) => {
        const yBase = i * laneH + 4 + barH + 2 + qH   // queue strip baseline
        const peak = Math.max(0, ...series[i].map((s) => s.depth))
        return (
          <g key={n}>
            <text x={2} y={i * laneH + 4 + barH / 2 + 3} fontSize="10.5" fill="var(--ink-2)">{n}</text>
            <line x1={0} x2={W} y1={yBase} y2={yBase} stroke="var(--axis)" strokeWidth="1" />
            {peak > 0 &&
              <text x={W - 2} y={yBase - qH + 4} fontSize="9" fill="var(--muted)" textAnchor="end"
                    style={{ fontVariantNumeric: 'tabular-nums' }}>peak queue {peak}</text>}
            {series[i].map((s, j) => {
              const h = Math.max(2, (s.depth / maxQ) * qH)
              const done = Math.max(0, Math.min(t, s.t1) - s.t0)
              return (
                <g key={j}
                   onMouseMove={(ev) => onHover({ queue: true, gpu: n, depth: s.depth,
                                                  t0: s.t0, t1: s.t1 }, ev.clientX, ev.clientY)}
                   onMouseLeave={() => onHover(null)}>
                  <rect x={x(s.t0)} y={yBase - h} width={Math.max(x(s.t1) - x(s.t0), 1)}
                        height={h} fill="var(--ink)" opacity="0.12" />
                  {done > 0 &&
                    <rect x={x(s.t0)} y={yBase - h} width={Math.max(x(s.t0 + done) - x(s.t0), 1)}
                          height={h} fill="var(--ink)" opacity="0.5" />}
                </g>
              )
            })}
          </g>
        )
      })}
      {run.events.map((e) => {
        const lane = names.indexOf(e.gpu)
        const y = lane * laneH + 4, h = barH
        const done = Math.max(0, Math.min(t, e.finish) - e.start)
        return (
          <g key={e.id}
             onMouseMove={(ev) => onHover(e, ev.clientX, ev.clientY)}
             onMouseLeave={() => onHover(null)}>
            <rect x={x(e.start)} y={y} width={Math.max(x(e.finish) - x(e.start), 1.5)} height={h}
                  rx="2.5" fill={TIER_COLOR[e.tier]} opacity="0.18" />
            {done > 0 &&
              <rect x={x(e.start)} y={y} width={Math.max(x(e.start + done) - x(e.start), 1.5)}
                    height={h} rx="2.5" fill={TIER_COLOR[e.tier]} />}
          </g>
        )
      })}
      <line x1={x(t)} x2={x(t)} y1={0} y2={H - axisH} stroke="var(--ink)" strokeWidth="1.5" />
    </svg>
  )
}
