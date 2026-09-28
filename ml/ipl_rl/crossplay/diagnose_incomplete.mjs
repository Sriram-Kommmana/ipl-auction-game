// Phase 2E.0 safety-stop investigation (read-only): replay one room exactly as
// the harness plays it, record every decision of every RL opponent seat with
// its completion-shield analysis, and explain any team that ends without a
// complete Best XI. Changes nothing.
//   node ml/ipl_rl/crossplay/diagnose_incomplete.mjs <k> <learnerKey> <C1|C4> <oppKey> [<seatToTrace>]
//   node ml/ipl_rl/crossplay/diagnose_incomplete.mjs <k> <learnerKey> S4 <seedIndex> [<seatToTrace>]
import { RL, ROOT, compose, loadExports, loadPlayers, loadValidation } from './common.mjs'
const { planBid } = await import(new URL('packages/shared/src/planning.js', ROOT))
const { RL_PERSONAS } = await import(new URL('packages/shared/src/personas.js', ROOT))
const { selectBestXI } = await import(new URL('packages/shared/src/scoring.js', ROOT))

const [kArg, learnerKey, kind, oppKey, traceArg] = process.argv.slice(2)
const ex = loadExports()
const players = loadPlayers()
const base = loadValidation()[Number(kArg)]
const cond = kind === 'S4' ? { kind, learnerAlgo: learnerKey.split(':')[0], seed: Number(oppKey) } : { kind, opp: oppKey }
const { entry, snapshotKeys } = compose(base, cond)
const ep = new RL.RlEpisode({ players, entry, snapshots: snapshotKeys.map((k) => ex[k].policy), tremble: 0, maskVersion: 'act-v3' })
const log = {}
entry.seats.forEach((seat, i) => {
    if (seat.type !== 'rlSnapshot') return
    const rt = RL.createRlSeat({ policy: ex[snapshotKeys[seat.snapshot]].policy, fallbackPersona: RL_PERSONAS[seat.rlSeat].fallback, now: () => 0 })
    log[i] = []
    ep.seats[i].runtime = {
        state: rt.state,
        decide(ctx, extras, rng) {
            const plan = planBid(ctx)
            const m = RL.rlActionMask(ctx, plan)
            const r = rt.decide(ctx, extras, rng)
            if (RL.hasBidAction(m.mask)) {
                const d = m.shield
                log[i].push({
                    lot: ctx.lot.slNo, role: ctx.lot.role, keeper: Boolean(ctx.lot.isWicketKeeper ?? ctx.lot.wicketKeeper), os: ctx.lot.nationality === 'Overseas', base: ctx.lot.basePrice,
                    phase: ctx.phase, progress: +ctx.progress.toFixed(3), purse: ctx.self.purseLeft, squad: ctx.self.playerCount,
                    need: Object.fromEntries(['keeper', 'bowling', 'indians', 'players'].map((q) => [q, plan.requirements[q]?.need])),
                    state: d.state, forced: d.forced, forcedBy: d.forcedBy, finalPath: d.finalPath, alreadyInfeasible: d.alreadyInfeasible,
                    legal: m.mask.flatMap((ok, a) => (ok ? [a] : [])), action: r.action, cap: r.cap, source: r.source
                })
            }
            return r
        }
    }
})
const rng = ep.learnerRng
ep.reset()
let step
log[entry.learnerSeat] = []
do {
    const { ctx, plan, mask } = ep.pending
    const a = RL.selectAction(ex[learnerKey].policy, RL.actionScores(ex[learnerKey].policy, ep.pending.obs), mask.mask, rng)
    const d = mask.shield
    log[entry.learnerSeat].push({
        lot: ctx.lot.slNo, role: ctx.lot.role, os: ctx.lot.nationality === 'Overseas', base: ctx.lot.basePrice, phase: ctx.phase, progress: +ctx.progress.toFixed(3),
        purse: ctx.self.purseLeft, squad: ctx.self.playerCount, need: Object.fromEntries(['keeper', 'bowling', 'indians', 'players'].map((q) => [q, plan.requirements[q]?.need])),
        state: d.state, forced: d.forced, forcedBy: d.forcedBy, finalPath: d.finalPath, alreadyInfeasible: d.alreadyInfeasible,
        legal: mask.mask.flatMap((ok, x) => (ok ? [x] : [])), action: a, cap: mask.caps[a], source: 'learner'
    })
    step = ep.step(a)
} while (!step.done)
const s = step.info.episode
console.log(`seed ${entry.seed} purse ${entry.purse} (${entry.stratum}) learner ${learnerKey} seat ${entry.learnerSeat}: XI ${s.xi.toFixed(3)} legal ${s.legalXI} invariantViolations ${s.invariantViolations}`)
const hist = ep.sim.history
entry.seats.forEach((seat, i) => {
    const t = ep.sim.teams[i]
    const xi = selectBestXI(t.squad)
    const roles = t.squad.reduce((acc, p) => ({ ...acc, [p.role]: (acc[p.role] ?? 0) + 1 }), {})
    const keepers = t.squad.filter((p) => p.role === 'WICKET KEEPER' || p.isWicketKeeper || p.wicketKeeper).length
    console.log(`seat ${i} ${i === entry.learnerSeat ? 'LEARNER ' + learnerKey : seat.type === 'rlSnapshot' ? snapshotKeys[seat.snapshot] + ' (' + seat.rlSeat + ')' : seat.type + ' ' + (seat.persona ?? seat.proxy ?? '')}: XI ${(xi.total / 11).toFixed(2)} empty ${xi.emptySlots} squad ${t.playerCount} overseas ${t.overseasCount} purseLeft ${t.purseLeft} roles ${JSON.stringify(roles)} keepers ${keepers}`)
})
const trace = traceArg !== undefined ? [Number(traceArg)] : Object.keys(log).map(Number).filter((i) => selectBestXI(ep.sim.teams[i].squad).emptySlots > 0)
const keeperLots = hist.filter((h) => (players.get?.(h.slNo) ?? ep.players.get(h.slNo)).role === 'WICKET KEEPER')
console.log(`keeper lots: ${keeperLots.length} — ${keeperLots.map((h) => `${h.slNo}/${h.phase}→${h.winner === null ? 'unsold' : 'seat ' + h.winner}@${h.price}`).join(' ')}`)
for (const i of trace) {
    const t = ep.sim.teams[i]
    const xi = selectBestXI(t.squad)
    console.log(`\n=== seat ${i} (${i === entry.learnerSeat ? 'LEARNER ' + learnerKey : snapshotKeys[entry.seats[i].snapshot]}) empty slots ${xi.emptySlots}; squad:`)
    console.log(t.squad.map((p) => `${p.slNo}:${p.role}${p.nationality === 'Overseas' ? '(OS)' : ''}:${p.rating}`).join(', '))
    const L = log[i]
    console.log(`${L.length} decisions; forced ${L.filter((x) => x.forced).length}; finalPath ${L.filter((x) => x.finalPath).length}; alreadyInfeasible ${L.filter((x) => x.alreadyInfeasible).length}; states ${JSON.stringify(L.reduce((a, x) => ({ ...a, [x.state]: (a[x.state] ?? 0) + 1 }), {}))}`)
    const firstBad = L.findIndex((x) => x.state === 'IMPOSSIBLE' || x.alreadyInfeasible)
    console.log(`first IMPOSSIBLE / alreadyInfeasible decision: ${firstBad}`)
    const interesting = L.filter((x, j) => x.forced || x.state !== 'SAFE' || x.role === 'WICKET KEEPER' || (firstBad >= 0 && j >= firstBad - 3 && j <= firstBad + 3))
    for (const x of interesting.slice(0, 80)) {
        const h = hist.find((hh) => hh.slNo === x.lot && hh.phase === x.phase)
        console.log(JSON.stringify({ ...x, legal: x.legal.length, result: h ? (h.winner === null ? 'unsold' : h.winner === i ? `WON@${h.price}` : `lost to seat ${h.winner}@${h.price}`) : '?' }))
    }
}
