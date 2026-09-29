// Phase 2E.0 — one Stage-B cross-play episode, evaluation only.
//
// Learner seat: the frozen export, played exactly as the Stage-A evaluator
// plays it (evaluate.policyController: act-v3 mask, the algorithm's frozen
// selection — PPO/A2C temperature 0.3, D3QN/QR-DQN/ES argmax — and the
// learner's own seed-derived random stream).
//
// Opponent RL seats: the frozen export through the PRODUCTION seat runtime
// (createRlSeat, unmodified). The only difference from production is the
// clock: `clock: 'fixed'` injects now: () => 0 through createRlSeat's
// existing `now` parameter, so an OS/GC pause can never trip the 20 ms guard
// and silently swap the model for its rule persona (decision D4). The guard
// code itself is untouched and is measured with the real clock separately
// (`clock: 'real'`, performance runs only).
//
// Every other runtime fallback is a SAFETY STOP here, never a silent rule-bot
// decision: any opponent decision whose source is not 'rl' (non-finite
// output, masked/invalid action, exception, disabled model) throws, as do a
// wrong model, a non-finite learner score, a masked learner action and an
// invariant violation.
// Incomplete XI of an RL-controlled team (learner or opponent) — approved
// Option A after the C4 safety stop: a recorded Stage-B behavioural FINDING,
// only after a defect screen passes (the room reproduces identically through
// the unmodified frozen path: RlEpisode's own production opponent runtimes,
// real clock, and the evaluate.runEpisode learner loop; 0 invariant
// violations; 0 fallbacks; every decision legal — the last two are already
// hard stops above). A failed screen is a SAFETY STOP.
// tremble = 0 (decision D5): no random opponent actions.

import { RL, compose, layerDigest, sha256 } from './common.mjs'
const { createRlSeat, actionScores, selectAction } = RL
const ROOT = new URL('../../../', import.meta.url)
const { planBid } = await import(new URL('packages/shared/src/planning.js', ROOT))
const { RL_PERSONAS } = await import(new URL('packages/shared/src/personas.js', ROOT))
const { selectBestXI } = await import(new URL('packages/shared/src/scoring.js', ROOT))
const { fairValue } = await import(new URL('packages/shared/src/valuation.js', ROOT))

export class SafetyStop extends Error {
    constructor(message, detail = {}) {
        super(message)
        this.name = 'SafetyStop'
        this.detail = detail
    }
}

const REQS = ['keeper', 'bowling', 'indians']
const STAR = 90
const perfNow = () => globalThis.performance.now()
// The production loader returns the validated object it was given, so the
// runtime's weights are the export's own object; hash each distinct object once.
const LAYER_CACHE = new WeakMap()
const verifiedLayers = (policy) => {
    if (!LAYER_CACHE.has(policy)) LAYER_CACHE.set(policy, layerDigest(policy))
    return LAYER_CACHE.get(policy)
}
const digestOf = (x) => sha256(RL.canonicalJson(x)).slice(0, 16)

// Outcome of any team from the finished simulator (read-only).
const teamOutcome = (ep, i, totals) => {
    const t = ep.sim.teams[i]
    const xi = selectBestXI(t.squad)
    const P = ep.rules.pursePerTeam
    const bought = ep.sim.history.filter((h) => h.winner === i)
    return {
        xi: xi.total / 11, emptySlots: xi.emptySlots, legalXI: xi.emptySlots === 0, strongXI: xi.emptySlots === 0 && xi.strength >= 85,
        rank: 1 + totals.filter((x) => x > xi.total).length, purseLeft: t.purseLeft, purseLeftShare: t.purseLeft / P, purseSpent: t.purseSpent,
        squadSize: t.playerCount, overseas: t.overseasCount, buys: bought.length,
        stars: bought.filter((h) => ep.players.get(h.slNo).rating >= STAR).length,
        reauctionBuys: bought.filter((h) => h.phase === 'reauction').length,
        priceToFair: bought.length ? bought.reduce((s, h) => s + h.price / fairValue(ep.players.get(h.slNo), P), 0) / bought.length : null
    }
}

// exportsByKey: loadExports() output. cond: see common.compose.
// Options: clock 'fixed' | 'real'; dump: { max } collects real decision states
// (obs, mask, JS scores, chosen action, state tags) for the parity audit.
export const playEpisode = ({ players, baseEntry, learnerKey, cond, exportsByKey, clock = 'fixed', latency = null, dump = null }) => {
    if (clock !== 'fixed' && clock !== 'real') throw new Error(`clock must be fixed or real (${clock})`)
    const learnerExport = exportsByKey[learnerKey]
    if (!learnerExport) throw new SafetyStop(`WRONG MODEL: unknown learner export ${learnerKey}`)
    const { entry, snapshotKeys } = compose(baseEntry, cond)
    const opponents = snapshotKeys.map((k) => {
        const e = exportsByKey[k]
        if (!e) throw new SafetyStop(`WRONG MODEL: unknown opponent export ${k}`)
        return e
    })
    const ep = new RL.RlEpisode({ players, entry, snapshots: opponents.map((o) => o.policy), tremble: 0, maskVersion: 'act-v3' })
    if (ep.tremble !== 0 || ep.maskVersion !== 'act-v3') throw new SafetyStop('evaluation mode violated (tremble / mask)')

    // Opponent RL seats: production runtime, deterministic clock, and a
    // wrapper that turns any fallback into a safety stop.
    const seatStats = []
    const wrappers = new Map()
    entry.seats.forEach((seat, i) => {
        if (seat.type !== 'rlSnapshot') return
        const exp = opponents[seat.snapshot]
        if (ep.seats[i]?.kind !== 'rl') throw new SafetyStop(`WRONG MODEL: seat ${i} is not an RL seat in the environment`)
        const rt = createRlSeat({ policy: exp.policy, fallbackPersona: RL_PERSONAS[seat.rlSeat].fallback, ...(clock === 'fixed' ? { now: () => 0 } : {}) })
        if (rt.state.disabled) throw new SafetyStop(`WRONG MODEL: runtime refused ${exp.key}: ${rt.state.disabledReason}`)
        if (rt.state.policy.algorithm !== exp.algo || verifiedLayers(rt.state.policy) !== exp.layers) throw new SafetyStop(`WRONG MODEL: seat ${i} runtime does not hold ${exp.key}`)
        const st = { seat: i, key: exp.key, algo: exp.algo, rlSeat: seat.rlSeat, calls: 0, noLegalBid: 0, decisions: 0, bidDecisions: 0, capToFairSum: 0, actions: new Array(RL.ACTION_COUNT).fill(0), fallbacks: [], actionLog: [] }
        seatStats.push(st)
        const wrapper = {
            state: rt.state,
            decide(ctx, extras, rng) {
                st.calls++
                let pre = null
                if (dump && dump.count < dump.max) {
                    const plan = planBid(ctx)
                    const m = RL.rlActionMask(ctx, plan)
                    if (RL.hasBidAction(m.mask)) pre = { plan, m, obs: RL.buildRlObservation(ctx, extras, plan) }
                }
                const t0 = latency ? perfNow() : 0
                const r = rt.decide(ctx, extras, rng)
                if (latency) latency.push({ algo: exp.algo, key: exp.key, seed: entry.seed, seat: i, ms: perfNow() - t0, source: r.source, reason: r.reason, disabledBefore: st.fallbacks.length > 0 })
                if (r.source !== 'rl') {
                    st.fallbacks.push(r.reason)
                    if (clock === 'fixed') throw new SafetyStop(`OPPONENT FALLBACK seat ${i} (${exp.key}) seed ${entry.seed}: ${r.reason}`, { seat: i, key: exp.key })
                    return r
                }
                if (r.reason === 'no legal bid') { st.noLegalBid++; return r }
                st.decisions++
                st.actions[r.action]++
                st.actionLog.push(r.action)
                if (r.cap >= ctx.lot.basePrice) {
                    st.bidDecisions++
                    st.capToFairSum += r.cap / fairValue(ctx.lot, ep.rules.pursePerTeam)
                }
                if (pre) {
                    const d = pre.m.shield
                    dump.count++
                    dump.states.push({
                        who: 'opponent', key: exp.key, algo: exp.algo, obs: pre.obs, mask: pre.m.mask.map(Number), scores: actionScores(exp.policy, pre.obs), action: r.action,
                        tags: tagsOf(ctx, pre.plan, d)
                    })
                }
                return r
            }
        }
        ep.seats[i].runtime = wrapper
        wrappers.set(i, wrapper)
    })

    // Learner: identical to evaluate.policyController on the learner stream,
    // plus the Stage-A requirement audit (audit_requirements.mjs, 2D.1–2D.4).
    const policy = learnerExport.policy
    const rng = ep.learnerRng
    ep.reset()
    const req = Object.fromEntries(REQS.map((r) => [r, { initialNeed: null, closed: null, minPurseUnmet: null }]))
    const aud = { forced: 0, forcedWon: 0, forcedLost: 0, finalPath: 0, finalPathWon: 0, reauctionForced: 0, forcedBy: {}, forcedKeeperBids: 0, forcedKeeperWon: 0, shieldStates: {}, minPurseAnyUnmet: null }
    const actions = []
    let step
    do {
        const { ctx, plan, mask, obs } = ep.pending
        const d = mask.shield
        const need = Object.fromEntries(REQS.map((r) => [r, plan.requirements[r].need]))
        for (const r of REQS) {
            if (req[r].initialNeed === null) req[r].initialNeed = need[r]
            if (need[r] > 0) req[r].minPurseUnmet = Math.min(req[r].minPurseUnmet ?? Infinity, ctx.self.purseLeft)
        }
        if (REQS.some((r) => need[r] > 0)) aud.minPurseAnyUnmet = Math.min(aud.minPurseAnyUnmet ?? Infinity, ctx.self.purseLeft)
        aud.shieldStates[d.state] = (aud.shieldStates[d.state] ?? 0) + 1
        const finalOpp = Object.fromEntries(REQS.map((r) => [r, Boolean(plan.requirements[r].finalOpportunity)]))
        const scores = actionScores(policy, obs)
        if (scores.some((s) => !Number.isFinite(s))) throw new SafetyStop(`LEARNER NON-FINITE output ${learnerKey} seed ${entry.seed}`)
        const a = selectAction(policy, scores, mask.mask, rng)
        if (!mask.mask[a]) throw new SafetyStop(`LEARNER MASKED ACTION ${a} ${learnerKey} seed ${entry.seed}`)
        if (dump && dump.count < dump.max) {
            dump.count++
            dump.states.push({ who: 'learner', key: learnerKey, algo: learnerExport.algo, obs, mask: mask.mask.map(Number), scores, action: a, tags: tagsOf(ctx, plan, d) })
        }
        actions.push(a)
        step = ep.step(a)
        const L = step.info.lastLot
        if (d.forced) {
            aud.forced++
            if (L.won) aud.forcedWon++
            else aud.forcedLost++
            if (d.finalPath) { aud.finalPath++; if (L.won) aud.finalPathWon++ }
            if (ctx.phase === 'reauction') aud.reauctionForced++
            for (const r of d.forcedBy) aud.forcedBy[r] = (aud.forcedBy[r] ?? 0) + 1
            if (d.forcedBy.some((r) => r === 'keeper' || r.startsWith('keeper'))) { aud.forcedKeeperBids++; if (L.won) aud.forcedKeeperWon++ }
        }
        if (L.won) {
            const after = step.done ? null : ep.pending.plan.requirements
            for (const r of REQS) {
                if (need[r] > 0 && req[r].closed === null && (after ? after[r].need === 0 : true)) {
                    req[r].closed = { phase: ctx.phase, progress: +ctx.progress.toFixed(3), price: L.price, forced: Boolean(d.forced), finalPath: Boolean(d.finalPath), finalOpportunity: finalOpp[r], atFinalStep: step.done }
                }
            }
        }
    } while (!step.done)
    const summary = step.info.episode
    for (const r of REQS) if (req[r].closed?.atFinalStep && !summary.legalXI) req[r].closed = null

    // Safety: every frozen invariant, every RL-controlled team complete.
    if (summary.invariantViolations) throw new SafetyStop(`INVARIANT ${learnerKey} seed ${entry.seed}: ${summary.violations.join('; ')}`)
    entry.seats.forEach((s, i) => {
        if (s.type === 'rlSnapshot' && ep.seats[i].runtime !== wrappers.get(i)) throw new SafetyStop(`WRONG MODEL: seat ${i} is not running its assigned runtime`)
    })
    if (wrappers.size !== snapshotKeys.length) throw new SafetyStop('WRONG MODEL: opponent seat count')
    const totals = ep.sim.teams.map((t) => selectBestXI(t.squad).total)
    const teams = entry.seats.map((s, i) => ({ type: i === entry.learnerSeat ? 'learner' : s.type, ...teamOutcome(ep, i, totals) }))
    const opp = seatStats.map((st) => {
        const o = teams[st.seat]
        const { actionLog, ...s } = st
        return { ...s, actionsDigest: digestOf(actionLog), bidRate: st.decisions ? st.bidDecisions / st.decisions : 0, capToFair: st.bidDecisions ? st.capToFairSum / st.bidDecisions : null, ...o }
    })
    // Incomplete XI of an RL-controlled team → defect screen → finding.
    const incomplete = teams.flatMap((t, i) => (t.legalXI ? [] : i === entry.learnerSeat ? [{ seat: i, type: 'learner', key: learnerKey }] : entry.seats[i].type === 'rlSnapshot' ? [{ seat: i, type: 'rlSnapshot', key: snapshotKeys[entry.seats[i].snapshot] }] : []))
    let findings = null
    if (incomplete.length && clock === 'fixed') {
        const screen = defectScreen({ players, entry, opponents, policy, summary, history: ep.sim.history, teams })
        if (!screen.pass) throw new SafetyStop(`DEFECT SCREEN FAILED: incomplete XI ${JSON.stringify(incomplete)} seed ${entry.seed} cond ${cond.kind}: ${screen.reason}`, { incomplete, screen })
        findings = incomplete.map((f) => ({
            ...f, rlSeat: entry.seats[f.seat].rlSeat ?? entry.learnerRlSeat, emptySlots: teams[f.seat].emptySlots, squadSize: teams[f.seat].squadSize,
            overseas: teams[f.seat].overseas, purseLeft: teams[f.seat].purseLeft, xi: teams[f.seat].xi, screen
        }))
    }
    const { violations, ...learner } = summary
    return {
        ...(findings ? { findings } : {}),
        cond: cond.kind, learner: learnerKey, opponents: snapshotKeys, seed: entry.seed, stratum: entry.stratum, purse: entry.purse,
        learnerSeat: entry.learnerSeat, learnerRlSeat: entry.learnerRlSeat,
        opponentSeats: seatStats.map((s) => ({ seat: s.seat, rlSeat: s.rlSeat, key: s.key })),
        summary: learner, req, audit: aud, opp,
        others: teams.map((t, i) => ({ i, type: t.type, xi: t.xi, legalXI: t.legalXI, rank: t.rank })),
        incompleteByType: teams.reduce((acc, t) => (t.legalXI ? acc : { ...acc, [t.type]: (acc[t.type] ?? 0) + 1 }), {}),
        digests: {
            learnerActions: digestOf(actions),
            auction: digestOf(ep.sim.history.map((h) => [h.slNo, h.phase, h.winner, h.price])),
            summary: digestOf(summary)
        },
        clock
    }
}

// Defect screen: replay the room through the unmodified frozen path —
// RlEpisode's OWN opponent runtimes (production createRlSeat, real clock,
// 20 ms guard) and the evaluate.runEpisode learner loop — and require the
// identical outcome. A real-clock guard trip (OS/GC pause) makes a replay
// inconclusive, so up to three attempts are made.
export function defectScreen({ players, entry, opponents, policy, summary, history, teams }) {
    for (let attempt = 1; attempt <= 3; attempt++) {
        const ep = new RL.RlEpisode({ players, entry, snapshots: opponents.map((o) => o.policy), tremble: 0, maskVersion: 'act-v3' })
        const ctrl = RL.policyController(policy)
        const rng = ep.learnerRng
        ep.reset()
        let step
        do { step = ep.step(ctrl.act(ep, rng)) } while (!step.done)
        const tripped = ep.seats.filter((s) => s?.kind === 'rl' && s.runtime.state.disabled).length
        if (tripped) continue
        const { violations, ...plain } = step.info.episode
        const { violations: _v, ...mine } = summary
        const totals = ep.sim.teams.map((t) => selectBestXI(t.squad).total)
        const sameSummary = RL.canonicalJson(plain) === RL.canonicalJson(mine)
        const sameHistory = digestOf(ep.sim.history.map((h) => [h.slNo, h.phase, h.winner, h.price])) === digestOf(history.map((h) => [h.slNo, h.phase, h.winner, h.price]))
        const sameTeams = ep.sim.teams.every((t, i) => selectBestXI(t.squad).emptySlots === teams[i].emptySlots && t.purseLeft === teams[i].purseLeft)
        const violationsZero = plain.invariantViolations === 0
        const pass = sameSummary && sameHistory && sameTeams && violationsZero
        return { pass, attempts: attempt, plainReplayIdentical: sameSummary && sameHistory && sameTeams, invariantViolations: plain.invariantViolations, reason: pass ? null : `summary ${sameSummary} history ${sameHistory} teams ${sameTeams} violations ${plain.invariantViolations}`, totalsDigest: digestOf(totals) }
    }
    return { pass: false, attempts: 3, reason: 'plain replay inconclusive: the real-clock guard tripped in 3 attempts' }
}

function tagsOf(ctx, plan, d) {
    return {
        phase: ctx.phase, purse: ctx.self.purseLeft,
        need: Object.fromEntries(REQS.map((r) => [r, plan.requirements[r].need])),
        forced: Boolean(d?.forced), finalPath: Boolean(d?.finalPath), shieldState: d?.state ?? null
    }
}
