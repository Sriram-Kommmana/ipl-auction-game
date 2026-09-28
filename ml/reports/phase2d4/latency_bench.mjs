// Phase 2D.4 §24 — inference latency distribution of real OpenAI-ES exports
// (observation + planner + act-v3 mask + forward + selection, i.e. seat.decide),
// single seat and 9 seats deciding the same lot in turn. Every decision is
// timed individually; decisions over the 20 ms production budget are counted.
import { readFileSync, writeFileSync } from 'node:fs'
const base = new URL('../../../packages/shared/', import.meta.url)
const { createRng } = await import(new URL('src/sim.js', base))
const RL = await import(new URL('src/rl/index.js', base))
const { realStates } = await import(new URL('test/rlHelpers.js', base))
const states = realStates([7301, 7302, 7303], 5).filter((s) => s.mask.mask.some((ok, a) => ok && a !== RL.PASS))
const out = []
const q = (a, p) => a[Math.min(a.length - 1, Math.floor(p * a.length))]
for (const path of process.argv.slice(3)) {
    const policy = JSON.parse(readFileSync(path, 'utf8'))
    const now = () => 0 // the seat's own 20 ms guard is bypassed here so every decision is measured
    const seats = Array.from({ length: 9 }, () => RL.createRlSeat({ policy, fallbackPersona: 'moneyball', now }))
    for (let w = 0; w < 5; w++) for (const s of states) seats[0].decide(s.ctx, s.extras, createRng(1)) // warm-up
    const single = [], perDecision = [], perLot = []
    for (let rep = 0; rep < 60; rep++) for (const s of states) { const t0 = performance.now(); seats[0].decide(s.ctx, s.extras, createRng(1)); single.push(performance.now() - t0) }
    for (let rep = 0; rep < 30; rep++) for (const s of states) {
        const t0 = performance.now()
        for (const st of seats) { const t1 = performance.now(); st.decide(s.ctx, s.extras, createRng(1)); perDecision.push(performance.now() - t1) }
        perLot.push(performance.now() - t0)
    }
    for (const a of [single, perDecision, perLot]) a.sort((x, y) => x - y)
    const over = [...single, ...perDecision].filter((x) => x > 20)
    const row = { path, single: { n: single.length, p50: q(single, 0.5), p95: q(single, 0.95), p99: q(single, 0.99), max: single.at(-1) },
                  nineSeatDecisions: { n: perDecision.length, p50: q(perDecision, 0.5), p95: q(perDecision, 0.95), p99: q(perDecision, 0.99), max: perDecision.at(-1) },
                  nineSeatLots: { n: perLot.length, p50: q(perLot, 0.5), p95: q(perLot, 0.95), p99: q(perLot, 0.99), max: perLot.at(-1) },
                  over20ms: over.length, totalTimed: single.length + perDecision.length }
    out.push(row)
    const f = (o) => `p50 ${o.p50.toFixed(3)} p95 ${o.p95.toFixed(3)} p99 ${o.p99.toFixed(3)} max ${o.max.toFixed(3)} ms (n=${o.n})`
    console.log(`${path}\n  single seat: ${f(row.single)}\n  9 seats, per decision: ${f(row.nineSeatDecisions)}\n  9 seats, per lot (all nine): ${f(row.nineSeatLots)}\n  decisions over 20 ms: ${row.over20ms}/${row.totalTimed}`)
}
writeFileSync(process.argv[2], JSON.stringify(out, null, 1))
