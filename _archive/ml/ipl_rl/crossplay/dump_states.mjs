// Phase 2E.0 §14 — real Stage-B decision states for the production-parity
// audit. Plays S4 and C4 rooms (fixed clock, same harness as the experiment)
// and records, for every export, the states it actually decided in, both as
// learner and as an opponent seat: observation, act-v3 mask, the production
// JavaScript scores, the action the production path chose, and state tags
// (phase, purse, requirement needs, shield state, forced, final path).
//   node ml/ipl_rl/crossplay/dump_states.mjs <out.json> [entries=12] [perExport=600]
import { writeFileSync } from 'node:fs'
import { ALGOS, EXPORTS, SEEDS, exportKey, loadExports, loadPlayers, loadValidation } from './common.mjs'
import { playEpisode } from './harness.mjs'

const [out, E = '12', CAP = '600'] = process.argv.slice(2)
const ex = loadExports()
const players = loadPlayers()
const entries = loadValidation()
const byKey = Object.fromEntries(EXPORTS.map((e) => [e.key, []]))
const rooms = []
for (let k = 0; k < Number(E); k++) {
    const entry = entries[(k * 41) % 500]
    for (const a of ALGOS) for (const s of SEEDS) {
        rooms.push([entry, exportKey(a, s), { kind: 'S4', learnerAlgo: a, seed: s }])
        if (k < 3) rooms.push([entry, exportKey(a, s), { kind: 'C4', opp: exportKey(ALGOS[(ALGOS.indexOf(a) + 1 + k) % 5], s) }])
    }
}
for (const [entry, learner, cond] of rooms) {
    const dump = { max: Infinity, count: 0, states: [] }
    playEpisode({ players, baseEntry: entry, learnerKey: learner, cond, exportsByKey: ex, dump })
    for (const st of dump.states) byKey[st.key].push({ ...st, seed: entry.seed, cond: cond.kind })
}
// Keep every rare state (re-auction, low purse, forced, final path) and an
// even sample of the rest, up to CAP per export.
const rare = (s) => s.tags.phase === 'reauction' || s.tags.purse <= 200 || s.tags.forced || s.tags.finalPath
const keep = {}
for (const [key, states] of Object.entries(byKey)) {
    const r = states.filter(rare)
    // high-purse states (≥ ₹9,000L left) — up to 150, evenly spread
    const hp = states.filter((s) => !rare(s) && s.tags.purse >= 9000)
    const hpStep = Math.max(1, hp.length / 150)
    r.push(...Array.from({ length: Math.min(150, hp.length) }, (_, i) => hp[Math.floor(i * hpStep)]))
    const n = states.filter((s) => !rare(s) && s.tags.purse < 9000)
    const room = Math.max(0, Number(CAP) - r.length)
    const step = Math.max(1, n.length / Math.max(1, room))
    keep[key] = [...r, ...Array.from({ length: Math.min(room, n.length) }, (_, i) => n[Math.floor(i * step)])]
}
writeFileSync(out, JSON.stringify(keep))
console.log(Object.entries(keep).map(([k, v]) => `${k} ${v.length} (learner ${v.filter((s) => s.who === 'learner').length}, opponent ${v.filter((s) => s.who === 'opponent').length}, reauction ${v.filter((s) => s.tags.phase === 'reauction').length}, forced ${v.filter((s) => s.tags.forced).length}, finalPath ${v.filter((s) => s.tags.finalPath).length})`).join('\n'))
