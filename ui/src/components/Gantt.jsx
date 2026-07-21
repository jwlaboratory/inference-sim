import { TIER_COLOR, fmtT } from '../config.js'

export default function Gantt({ run, t, onHover }) {
  const W = 1000, laneH = 30, axisH = 22
  const names = run.gpus.map((g) => g.name)
  const H = names.length * laneH + axisH
  const x = (s) => (s / run.span) * W
  const tickStep = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1800, 3600]
    .find((s) => run.span / s <= 10) || 7200
  const ticks = []
  for (let s = 0; s <= run.span; s += tickStep) ticks.push(s)

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }}>
      {ticks.map((s) => (
        <g key={s}>
          <line x1={x(s)} x2={x(s)} y1={0} y2={H - axisH} stroke="var(--grid)" strokeWidth="1" />
          <text x={x(s)} y={H - 6} fontSize="10" fill="var(--muted)" textAnchor="middle"
                style={{ fontVariantNumeric: 'tabular-nums' }}>{fmtT(s)}</text>
        </g>
      ))}
      {names.map((n, i) => (
        <text key={n} x={2} y={i * laneH + laneH / 2 + 3} fontSize="10.5" fill="var(--ink-2)">{n}</text>
      ))}
      {run.events.map((e) => {
        const lane = names.indexOf(e.gpu)
        const y = lane * laneH + 5, h = laneH - 10
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
