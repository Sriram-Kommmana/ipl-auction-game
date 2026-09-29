// Gameplay audit of the trained RL exports as game opponents (productization).
//
// Plays complete Full Pool auctions (main round + re-auction) with the
// unchanged AuctionSim, the frozen rule bots and the production RL seat
// runtime (createRlSeat: act-v3 mask + shield, cap ≤ maxSafeBid, 20 ms guard,
// rule fallback, completionGuard as in production). Nothing is trained or
// written back; results are JSON lines for aggregate.mjs.
//
// Rooms are the product lineup (samplers.sampleEpisode): 1 human proxy, the
// 4 rule bots, 5 RL seats.
//
//   node audit.mjs --mode export --rooms 120 --shard 0 --of 8 --out f.jsonl
//       every candidate export (15 Stage-A + 15 Stage-B pilot) takes the
//       sampled RL seat of each room; the other four RL seats are Stage-A
//       exports of the other algorithms, drawn from the room seed only, so a
//       Stage-A and a Stage-B export of one algorithm meet identical rooms.
//   node audit.mjs --mode roster --roster roster.json --rooms 300 ...
//       the five RL seats play the given roster (RL persona id → export key).

import { readFileSync, writeFileSync, appendFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { performance } from 'node:perf_hooks'

const CP = new URL('../crossplay/', import.meta.url)
const { RL, ROOT, loadExports, loadPlayers, sha256 } = await import(new URL('common.mjs', CP))
const at = (rel) => fileURLToPath(new URL(rel, ROOT))
const { AuctionSim, agentCap, createRng } = await import(new URL('packages/shared/src/sim.js', ROOT))
const { DEFAULT_RULES } = await import(new URL('packages/shared/src/rules.js', ROOT))
const { planBid } = await import(new URL('packages/shared/src/planning.js', ROOT))
const { fairValue } = await import(new URL('packages/shared/src/valuation.js', ROOT))
const { selectBestXI, xiTotal } = await import(new URL('packages/shared/src/scoring.js', ROOT))
const { RL_PERSONAS } = await import(new URL('packages/shared/src/personas.js', ROOT))
const { sampleEpisode, deriveSeed, createRlSeat, rlActionMask, analyseCompletion, SHIELD_STATE, RECENT_WINDOW, PASS, ACTION_COUNT } = RL

const arg = (name, dflt) => {
    const i = process.argv.indexOf(`--${name}`)
    return i >= 0 ? process.argv[i + 1] : dflt
}
const MODE = arg('mode', 'export')
const ROOMS = Number(arg('rooms', 100))
const SHARD = Number(arg('shard', 0))
const OF = Number(arg('of', 1))
const OUT = arg('out', null)
const SEED0 = Number(arg('seed0', 90000)) // regression-split seeds: never train / validation / test
const GUARD = arg('guard', 'on') === 'on'
const ALGOS = ['ppo', 'a2c', 'd3qn', 'qrdqn', 'es']
const STAR = 90

// ── candidate exports ─────────────────────────────────────────────────────
const stageA = loadExports() // digest-checked frozen Stage-A exports
const models = {}
for (const [key, e] of Object.entries(stageA)) models[`A:${key}`] = { key: `A:${key}`, stage: 'A', algo: e.algo, seed: e.seed, policy: e.policy, sha256: e.sha256, path: e.path }
const repro = JSON.parse(readFileSync(at('ml/reports/phase2f/reproducibility.json'), 'utf8'))
for (const [k, v] of Object.entries(repro.checkpointPolicies)) {
    if (!k.endsWith('@c500')) continue
    const [algo, s] = k.replace('@c500', '').split(':s')
    const bytes = readFileSync(at(`ml/${v.policy}`))
    if (sha256(bytes) !== v.sha256) throw new Error(`Stage-B export ${k}: sha256 mismatch`)
    const loaded = RL.loadPolicy(bytes.toString('utf8'))
    if (!loaded.ok) throw new Error(`Stage-B export ${k} rejected: ${loaded.error}`)
    models[`B:${algo}:s${s}`] = { key: `B:${algo}:s${s}`, stage: 'B', algo, seed: Number(s), policy: loaded.policy, sha256: v.sha256, path: `ml/${v.policy}` }
}
// Baseline = today's production: no model file, so the RL seat plays its frozen rule fallback.
models['R:fallback'] = { key: 'R:fallback', stage: 'rule', algo: 'rule', seed: 0, policy: null, sha256: '-', path: '-' }
if (MODE === 'list') {
    for (const m of Object.values(models).filter((x) => x.policy)) console.log([m.key, m.algo, m.stage, m.seed, m.sha256.slice(0, 16), m.policy.architecture.hidden.join('x'), m.policy.architecture.activation, m.policy.architecture.head, m.policy.selection.mode, m.path].join('\t'))
    process.exit(0)
}

const players = loadPlayers()
const rosterCfg = MODE === 'roster' ? JSON.parse(readFileSync(arg('roster'), 'utf8')) : null

// ── one auction ───────────────────────────────────────────────────────────
const entropy = (counts) => {
    const n = counts.reduce((a, b) => a + b, 0)
    return n ? -counts.reduce((s, c) => (c ? s + (c / n) * Math.log2(c / n) : s), 0) : 0
}

const playRoom = (seed, rlAssign /* rlSeat persona id → model key */, target /* persona id or null */) => {
    const entry = sampleEpisode(seed)
    const rules = { ...DEFAULT_RULES, pursePerTeam: entry.purse }
    const P = rules.pursePerTeam
    const rng = createRng(seed)
    const sim = new AuctionSim({ players, teamCount: entry.seats.length, rules, rng })
    const decideRng = createRng(deriveSeed(seed, 'audit-rl'))
    const seats = entry.seats.map((s) => {
        if (s.type === 'rule') return { kind: 'rule', label: `rule:${s.persona}`, agent: { kind: 'rule', persona: s.persona } }
        if (s.type === 'human') return s.proxy === 'passive' ? { kind: 'passive', label: 'human:passive' } : { kind: 'rule', label: `human:${s.proxy}`, agent: { kind: 'rule', persona: s.persona, noise: s.noise } }
        const persona = s.rlSeat
        const m = models[rlAssign[persona]]
        return {
            kind: 'rl', label: m.key, persona, model: m,
            runtime: createRlSeat({ policy: m.policy, fallbackPersona: RL_PERSONAS[persona].fallback, completionGuard: GUARD }),
            st: { dec: 0, bid: 0, actions: new Array(ACTION_COUNT).fill(0), fallback: 0, guard: 0, forced: 0, finalPath: 0, capViol: 0, ms: [],
                critical: 0, criticalBid: 0, criticalPassAfford: 0, starEarly: 0, starEarlyPassRich: 0, lowPurse: 0, lowPurseBid: 0,
                capFairStar: [], capFairAll: [], reDec: 0, reBid: 0 }
        }
    })
    const spentAt = seats.map(() => ({}))
    const marks = [0.25, 0.5, 0.75]
    const lotLog = []
    while (!sim.done) {
        const lot = sim.currentLot()
        const progress = sim.progress()
        for (const q of marks) if (sim.phase === 'main' && progress >= q) sim.teams.forEach((t, i) => { if (spentAt[i][q] === undefined) spentAt[i][q] = t.purseSpent / P })
        if (sim.phase === 'reauction') sim.teams.forEach((t, i) => { for (const q of marks) if (spentAt[i][q] === undefined) spentAt[i][q] = t.purseSpent / P; if (spentAt[i].main === undefined) spentAt[i].main = t.purseSpent / P })
        const extras = {
            poolSize: sim.mainLength,
            recent: sim.history.slice(-RECENT_WINDOW).map((h) => ({ sold: h.winner !== null, price: h.price, fairValue: fairValue(sim.players.get(h.slNo), P), winnerTeamId: h.winner === null ? null : sim.teams[h.winner].teamId }))
        }
        const fair = fairValue(lot, P)
        const caps = seats.map((seat, i) => {
            const ctx = sim.contextFor(i)
            if (seat.kind === 'passive') return 0
            if (seat.kind === 'rule') return agentCap(seat.agent, ctx, rng)
            const t0 = performance.now()
            const d = seat.runtime.decide(ctx, extras, decideRng)
            const ms = performance.now() - t0
            const st = seat.st
            if (d.reason === 'no legal bid') return d.cap
            st.ms.push(ms)
            st.dec++
            if (d.source === 'fallback') st.fallback++
            if (d.source === 'guard') st.guard++
            if (d.action !== null) st.actions[d.action]++
            const plan = planBid(ctx)
            const mr = rlActionMask(ctx, plan)
            if (mr.shield?.forced) st.forced++
            if (d.cap > 0 && plan.allowed && d.cap > plan.budget.maxSafeBid) st.capViol++
            if (d.cap > 0 && !plan.allowed) st.capViol++
            const bids = d.cap >= lot.basePrice
            if (bids) st.bid++
            const purseShare = ctx.self.purseLeft / P
            const cf = Math.min(5, d.cap / fair)
            st.capFairAll.push(bids ? cf : 0)
            if (lot.rating >= STAR && sim.phase === 'main') st.capFairStar.push(bids ? cf : 0)
            if (sim.phase === 'reauction') { st.reDec++; if (bids) st.reBid++ }
            if (purseShare < 0.15) { st.lowPurse++; if (bids) st.lowPurseBid++ }
            if (lot.rating >= STAR && sim.phase === 'main' && progress < 0.3) { st.starEarly++; if (!bids && purseShare > 0.5) st.starEarlyPassRich++ }
            if (plan.allowed) {
                const crit = Object.values(analyseCompletion(ctx, plan).requirements).some((a) => a.fillsNow && a.state === SHIELD_STATE.CRITICAL)
                if (crit) { st.critical++; if (bids) st.criticalBid++; else if (plan.budget.maxSafeBid >= lot.basePrice) st.criticalPassAfford++ }
            }
            return d.cap
        })
        const opening = lot.basePrice
        const contenders = caps.filter((c) => c >= opening).length
        const outcome = sim.resolveLot(caps)
        lotLog.push({ phase: sim.history.at(-1).phase, winner: outcome.winner, price: outcome.price, bids: outcome.bids, fair, contenders, star: lot.rating >= STAR, progress, caps })
    }
    sim.teams.forEach((t, i) => { if (spentAt[i].main === undefined) spentAt[i].main = t.purseSpent / P })
    const violations = RL.auditAuction(sim)

    const rows = []
    seats.forEach((seat, i) => {
        const t = sim.teams[i]
        const xi = selectBestXI(t.squad)
        const wins = lotLog.filter((l) => l.winner === i)
        const others = (l) => l.caps.filter((c, j) => j !== i && c >= l.price).length
        const row = {
            seed, purse: P, stratum: entry.stratum, seat: i, kind: seat.kind, label: seat.label, persona: seat.persona ?? null, target: seat.persona !== undefined && seat.persona === target,
            xi: xiTotal(t.squad) / 11, legalXI: xi.emptySlots === 0 ? 1 : 0, emptySlots: xi.emptySlots,
            squad: t.playerCount, overseas: t.overseasCount,
            keepers: t.squad.filter((p) => p.role === 'WICKET KEEPER').length,
            bowlers: t.squad.filter((p) => p.role === 'BOWLER').length, allRounders: t.squad.filter((p) => p.role === 'ALL ROUNDER').length,
            batsmen: t.squad.filter((p) => p.role === 'BATSMAN').length,
            purseLeftShare: t.purseLeft / P, spent25: spentAt[i][0.25], spent50: spentAt[i][0.5], spent75: spentAt[i][0.75], spentMain: spentAt[i].main,
            stars: t.squad.filter((p) => p.rating >= STAR).length,
            starsBy30: wins.filter((l) => l.star && l.phase === 'main' && l.progress < 0.3).length,
            buys: wins.length, reBuys: wins.filter((l) => l.phase === 'reauction').length,
            contestedWins: wins.filter((l) => l.contenders >= 2).length,
            warWins: wins.filter((l) => l.bids >= 6).length,
            winPriceFair: wins.length ? wins.reduce((s, l) => s + l.price / l.fair, 0) / wins.length : null,
            overbid2: wins.filter((l) => l.price / l.fair >= 2).length, overbid3: wins.filter((l) => l.price / l.fair >= 3).length,
            lostAtCap: lotLog.filter((l) => l.winner !== null && l.winner !== i && l.caps[i] >= l.price).length, // was still in when the lot went elsewhere at that price
            violations: violations.filter((v) => v.startsWith(`team ${i}:`)).length
        }
        if (seat.kind === 'rl') {
            const s = seat.st
            const ms = s.ms
            const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null)
            Object.assign(row, {
                model: seat.model.key, algo: seat.model.algo, stage: seat.model.stage,
                decisions: s.dec, bidShare: s.dec ? s.bid / s.dec : null, fallback: s.fallback, guard: s.guard, forced: s.forced, capViol: s.capViol,
                topAction: Math.max(...s.actions) / Math.max(1, s.actions.reduce((a, b) => a + b, 0)), actionEntropy: entropy(s.actions), actions: s.actions,
                critical: s.critical, criticalBid: s.criticalBid, criticalPassAfford: s.criticalPassAfford,
                starEarly: s.starEarly, starEarlyPassRich: s.starEarlyPassRich, lowPurse: s.lowPurse, lowPurseBid: s.lowPurseBid,
                reDec: s.reDec, reBid: s.reBid, capFairStar: mean(s.capFairStar), capFairAll: mean(s.capFairAll),
                msMax: ms.length ? Math.max(...ms) : 0, msMean: mean(ms),
                disabled: seat.runtime.state.disabled ? seat.runtime.state.disabledReason : null
            })
            row.msSample = ms.filter((_, k) => k % 7 === 0).map((x) => Math.round(x * 1000) / 1000)
        }
        rows.push(row)
    })
    const sold = lotLog.filter((l) => l.winner !== null)
    const room = {
        type: 'room', seed, purse: P, lots: lotLog.length, unsold: lotLog.length - sold.length, reLots: lotLog.filter((l) => l.phase === 'reauction').length,
        contested: lotLog.filter((l) => l.contenders >= 2).length, wars: lotLog.filter((l) => l.bids >= 6).length,
        starContested: lotLog.filter((l) => l.star && l.contenders >= 2).length, stars: lotLog.filter((l) => l.star).length,
        meanBidsSold: sold.length ? sold.reduce((s, l) => s + l.bids, 0) / sold.length : 0,
        soldPriceFair: sold.length ? sold.reduce((s, l) => s + l.price / l.fair, 0) / sold.length : 0,
        violations
    }
    return { rows, room }
}

// ── jobs ──────────────────────────────────────────────────────────────────
const jobs = []
if (MODE === 'export') {
    const only = arg('only', null)
    const cands = only ? only.split(',') : Object.keys(models).filter((k) => k !== 'R:fallback')
    for (const c of cands) for (let r = 0; r < ROOMS; r++) jobs.push({ cand: c, seed: SEED0 + r })
} else {
    for (let r = 0; r < ROOMS; r++) jobs.push({ seed: SEED0 + r })
}
const RL_IDS = Object.keys(RL_PERSONAS)
const stageAKeys = (algo) => [1, 2, 3].map((s) => `A:${algo}:s${s}`)
if (OUT) writeFileSync(OUT, '')
let n = 0
const t0 = Date.now()
for (let j = SHARD; j < jobs.length; j += OF) {
    const job = jobs[j]
    let assign, target = null
    if (MODE === 'export') {
        const cand = models[job.cand]
        const entry = sampleEpisode(job.seed)
        target = entry.learnerRlSeat
        const orng = createRng(deriveSeed(job.seed, 'audit-opponents'))
        // the fallback baseline meets the rooms the PPO candidates meet
        const otherAlgos = ALGOS.filter((a) => a !== (cand.algo === 'rule' ? 'ppo' : cand.algo))
        assign = {}
        let k = 0
        for (const id of RL_IDS) {
            if (id === target) { assign[id] = cand.key; continue }
            const algo = otherAlgos[k++ % otherAlgos.length]
            assign[id] = stageAKeys(algo)[Math.floor(orng() * 3)]
        }
    } else {
        assign = rosterCfg.rl
    }
    const { rows, room } = playRoom(job.seed, assign, target)
    const lines = rows.filter((r) => MODE !== 'export' || r.target).map((r) => JSON.stringify({ type: 'seat', cand: job.cand ?? null, ...r }))
    lines.push(JSON.stringify({ ...room, cand: job.cand ?? null }))
    if (OUT) appendFileSync(OUT, lines.join('\n') + '\n')
    n++
    if (n % 20 === 0) console.error(`shard ${SHARD}: ${n} rooms, ${((Date.now() - t0) / n).toFixed(0)} ms/room`)
}
console.error(`shard ${SHARD} done: ${n} rooms in ${((Date.now() - t0) / 1000).toFixed(0)} s`)
