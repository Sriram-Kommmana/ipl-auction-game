// Phase 2E.1 — READ-ONLY reproduction of recorded Phase 2E.0 episodes to
// extract data the raw records do not contain (per-lot trajectories, role-level
// purchases of every team, per-decision shield/planner state). Nothing here
// changes the frozen experiment: every replayed episode is played exactly as
// the Phase 2E.0 harness played it (same composition, production opponent
// runtime with the deterministic clock, learner = policyController on the
// learner stream, tremble 0) and MUST reproduce the recorded learner-action,
// auction and summary digests — any mismatch aborts the replay.
//
//   node replay.mjs traj   <out.jsonl> <cond> <records.jsonl> [--filter same-s1]
//   node replay.mjs keeper <out.json>  <findings.json>
import { createReadStream, readFileSync, writeFileSync, createWriteStream } from 'node:fs'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads'

const CP = new URL('../crossplay/', import.meta.url)
const { RL, ROOT, compose, loadExports, loadPlayers, loadValidation, sha256 } = await import(new URL('common.mjs', CP))
const { planBid } = await import(new URL('packages/shared/src/planning.js', ROOT))
const { RL_PERSONAS } = await import(new URL('packages/shared/src/personas.js', ROOT))
const { selectBestXI } = await import(new URL('packages/shared/src/scoring.js', ROOT))
const { fairValue } = await import(new URL('packages/shared/src/valuation.js', ROOT))

const digestOf = (x) => sha256(RL.canonicalJson(x)).slice(0, 16)
const BINS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] // main-round progress; bin 11 = end of re-auction
const isKeeper = (p) => p.role === 'WICKET KEEPER'
const isBowl = (p) => p.role === 'BOWLER' || p.role === 'ALL ROUNDER'
const isOS = (p) => p.nationality === 'Overseas'
const r3 = (x) => Math.round(x * 1000) / 1000

// Rebuild a room exactly as harness.playEpisode does; `onOpp(i, ctx, extras, r)` observes opponent decisions.
function play({ players, ex, baseEntry, learnerKey, cond, onOpp = null, onLearner = null, onCreate = null }) {
    const { entry, snapshotKeys } = compose(baseEntry, cond)
    const ep = new RL.RlEpisode({ players, entry, snapshots: snapshotKeys.map((k) => ex[k].policy), tremble: 0, maskVersion: 'act-v3' })
    if (onCreate) onCreate(ep)
    entry.seats.forEach((seat, i) => {
        if (seat.type !== 'rlSnapshot') return
        const exp = ex[snapshotKeys[seat.snapshot]]
        const rt = RL.createRlSeat({ policy: exp.policy, fallbackPersona: RL_PERSONAS[seat.rlSeat].fallback, now: () => 0 })
        ep.seats[i].runtime = {
            state: rt.state,
            decide(ctx, extras, rng) {
                const pre = onOpp ? onOpp.before(i, ctx, extras) : null
                const r = rt.decide(ctx, extras, rng)
                if (r.source !== 'rl') throw new Error(`replay: opponent fallback seat ${i}: ${r.reason}`)
                if (onOpp) onOpp.after(i, ctx, pre, r)
                return r
            }
        }
    })
    const policy = ex[learnerKey].policy
    const rng = ep.learnerRng
    ep.reset()
    const actions = []
    let step
    do {
        const scores = RL.actionScores(policy, ep.pending.obs)
        const a = RL.selectAction(policy, scores, ep.pending.mask.mask, rng)
        if (onLearner) onLearner.before(ep, a)
        actions.push(a)
        step = ep.step(a)
        if (onLearner) onLearner.after(ep, step)
    } while (!step.done)
    return { ep, entry, snapshotKeys, actions, summary: step.info.episode }
}

const checkDigests = (res, rec) => {
    const d = { learnerActions: digestOf(res.actions), auction: digestOf(res.ep.sim.history.map((h) => [h.slNo, h.phase, h.winner, h.price])), summary: digestOf(res.summary) }
    for (const k of Object.keys(d)) if (d[k] !== rec.digests[k]) throw new Error(`REPLAY MISMATCH ${rec.cond} ${rec.learner} ${rec.opponents} k=${rec.k}: ${k} ${d[k]} ≠ ${rec.digests[k]}`)
}

// Per-team purchase trajectory from the auction history (read-only).
function teamTrajectories(ep, entry, snapshotKeys, learnerKey) {
    const P = ep.rules.pursePerTeam
    const main = ep.sim.mainLength
    const T = ep.sim.teams.map(() => ({ buys: [] }))
    let mainIdx = 0
    for (const h of ep.sim.history) {
        const prog = h.phase === 'main' ? (++mainIdx) / main : 1.0001
        if (h.winner === null) continue
        const p = ep.players.get(h.slNo)
        T[h.winner].buys.push({ prog, phase: h.phase, keeper: isKeeper(p), bowl: isBowl(p), os: isOS(p), star: p.rating >= 90, rating: p.rating, price: h.price, fair: fairValue(p, P) })
    }
    const totals = ep.sim.teams.map((t) => selectBestXI(t.squad).total)
    return ep.sim.teams.map((t, i) => {
        const seat = entry.seats[i]
        const b = T[i].buys
        const at = (edge) => b.filter((x) => x.prog <= edge + 1e-9)
        const bins = [...BINS, 1.0002].map(at)
        const xi = selectBestXI(t.squad)
        return {
            seat: i, type: i === entry.learnerSeat ? 'learner' : seat.type, key: i === entry.learnerSeat ? learnerKey : seat.type === 'rlSnapshot' ? snapshotKeys[seat.snapshot] : (seat.persona ?? seat.proxy ?? null),
            xi: r3(xi.total / 11), legal: xi.emptySlots === 0, rank: 1 + totals.filter((x) => x > xi.total).length, purse: P,
            spendShare: bins.map((s) => r3(s.reduce((a, x) => a + x.price, 0) / P)),
            squad: bins.map((s) => s.length),
            keepers: bins.map((s) => s.filter((x) => x.keeper).length),
            bowl: bins.map((s) => s.filter((x) => x.bowl).length),
            indians: bins.map((s) => s.filter((x) => !x.os).length),
            overseas: bins.map((s) => s.filter((x) => x.os).length),
            stars: bins.map((s) => s.filter((x) => x.star).length),
            keeperBuys: b.filter((x) => x.keeper).map((x) => [r3(x.prog), x.price, r3(x.price / x.fair)]),
            pf: b.length ? r3(b.reduce((a, x) => a + x.price / x.fair, 0) / b.length) : null,
            reauctionBuys: b.filter((x) => x.phase === 'reauction').length
        }
    })
}

function trajJob(ctx, rec) {
    const { players, ex, entries } = ctx
    const cond = rec.cond === 'S4' ? { kind: 'S4', learnerAlgo: rec.learner.split(':')[0], seed: Number(rec.learner.split(':s')[1]) } : rec.cond === 'A' ? { kind: 'A' } : { kind: rec.cond, opp: rec.opponents[0] }
    // learner main-round decisions: [slNo, cap/fair] for same-lot pairing across conditions
    const lots = []
    const onLearner = {
        before(ep, a) {
            const { ctx: c, mask } = ep.pending
            if (c.phase === 'main') lots.push([c.lot.slNo, r3(mask.caps[a] / fairValue(c.lot, ep.rules.pursePerTeam)), r3(c.self.purseLeft / ep.rules.pursePerTeam)])
        },
        after() {}
    }
    const res = play({ players, ex, baseEntry: entries[rec.k], learnerKey: rec.learner, cond, onLearner })
    checkDigests(res, rec)
    return { cond: rec.cond, learner: rec.learner, opponents: rec.opponents, k: rec.k, seed: rec.seed, stratum: rec.stratum, replayVerified: true, teams: teamTrajectories(res.ep, res.entry, res.snapshotKeys, rec.learner), lots }
}

// Keeper forensics: every decision of the affected seat, with the planner's
// maxSafeBid and the shield's keeper analysis, plus every keeper lot.
function keeperJob(ctx, f, rec) {
    const { players, ex, entries } = ctx
    const cond = { kind: rec.cond, opp: rec.opponents[0] }
    const rows = []
    const snap = (ep, c, plan, m, seatIdx) => {
        const kr = m.shield?.requirements ?? {}
        const keeperReqs = Object.entries(kr).filter(([r]) => r === 'keeper' || r.startsWith('class:W')).map(([r, v]) => ({ r, ...v }))
        const rivals = ep.sim.teams.filter((_, j) => j !== seatIdx)
        return {
            lot: c.lot.slNo, role: c.lot.role, os: isOS(c.lot), rating: c.lot.rating, base: c.lot.basePrice, phase: c.phase, progress: r3(c.progress),
            purse: c.self.purseLeft, squad: c.self.playerCount, overseas: c.self.overseasCount, keepersOwned: c.self.squad.filter(isKeeper).length,
            need: Object.fromEntries(['keeper', 'bowling', 'indians', 'players'].map((q) => [q, plan.requirements[q]?.need])),
            keeperAnalysis: keeperReqs.map((x) => ({ r: x.r, state: x.state, viableAfterLot: x.viableAfterLot, spare: x.spare, rivalsNeeding: x.rivalsNeeding, contestants: x.contestants, cheapestBase: x.cheapestBase })),
            maxSafeBid: plan.budget.maxSafeBid, shieldState: m.shield?.state, forced: Boolean(m.shield?.forced), forcedBy: m.shield?.forcedBy ?? [], finalPath: Boolean(m.shield?.finalPath),
            alreadyInfeasible: Boolean(m.shield?.alreadyInfeasible), legal: m.mask.flatMap((ok, a) => (ok ? [a] : [])).length, hasBid: RL.hasBidAction(m.mask),
            maxRivalPurse: Math.max(...rivals.map((t) => t.purseLeft)), rivalsAbleToPayBase: rivals.filter((t) => t.purseLeft >= c.lot.basePrice && t.playerCount < ep.rules.maxPlayers).length
        }
    }
    let epRef = null
    const onOpp = f.type === 'rlSnapshot' ? {
        before(i, c, extras) {
            if (i !== f.seat) return null
            const plan = planBid(c)
            return { plan, m: RL.rlActionMask(c, plan) }
        },
        after(i, c, pre, r) {
            if (i !== f.seat || !pre) return
            if (RL.hasBidAction(pre.m.mask) || isKeeper(c.lot)) rows.push({ ...snap(epRef, c, pre.plan, pre.m, i), action: r.action, cap: r.cap })
        }
    } : null
    const onLearner = f.type === 'learner' ? {
        before(ep, a) {
            const { ctx: c, plan, mask } = ep.pending
            rows.push({ ...snap(ep, c, plan, mask, f.seat), action: a, cap: mask.caps[a] })
        },
        after() {}
    } : null
    const res = play({ players, ex, baseEntry: entries[rec.k], learnerKey: rec.learner, cond, onOpp, onLearner, onCreate: (ep) => { epRef = ep } })
    checkDigests(res, rec)
    const ep = res.ep
    // resolve each row's lot result and list every keeper lot
    let mainIdx = 0
    const hist = ep.sim.history.map((h) => ({ ...h, prog: h.phase === 'main' ? r3(++mainIdx / ep.sim.mainLength) : 1 }))
    for (const row of rows) {
        const h = hist.find((x) => x.slNo === row.lot && x.phase === row.phase)
        row.result = h ? (h.winner === null ? 'unsold' : h.winner === f.seat ? 'won' : `lost:seat${h.winner}`) : '?'
        row.soldPrice = h?.price ?? null
    }
    const seatLabel = (j) => (j === null ? null : j === res.entry.learnerSeat ? `learner ${rec.learner}` : res.entry.seats[j].type === 'rlSnapshot' ? `rl ${res.snapshotKeys[res.entry.seats[j].snapshot]}` : `${res.entry.seats[j].type} ${res.entry.seats[j].persona ?? res.entry.seats[j].proxy}`)
    const keeperLots = hist.filter((h) => isKeeper(ep.players.get(h.slNo))).map((h) => {
        const row = rows.find((x) => x.lot === h.slNo && x.phase === h.phase)
        return { lot: h.slNo, phase: h.phase, progress: h.prog, base: ep.players.get(h.slNo).basePrice, os: isOS(ep.players.get(h.slNo)), winner: seatLabel(h.winner), winnerSeat: h.winner, price: h.price,
            affectedDecision: row ? { action: row.action, cap: row.cap, state: row.shieldState, forced: row.forced, finalPath: row.finalPath, purse: row.purse, maxSafeBid: row.maxSafeBid, hasBid: row.hasBid } : null }
    })
    const team = ep.sim.teams[f.seat]
    return { finding: f, room: { seed: rec.seed, stratum: rec.stratum, purse: rec.purse, learner: rec.learner, opponents: rec.opponents, seats: res.entry.seats.map((s, j) => seatLabel(j)) },
        final: { squad: team.playerCount, overseas: team.overseasCount, purseLeft: team.purseLeft, keepers: team.squad.filter(isKeeper).length, emptySlots: selectBestXI(team.squad).emptySlots },
        teamsKeepers: ep.sim.teams.map((t, j) => ({ seat: j, who: seatLabel(j), keepers: t.squad.filter(isKeeper).length, purseLeft: t.purseLeft, squad: t.playerCount })),
        decisions: rows, keeperLots, replayVerified: true }
}
if (!isMainThread) {
    const ctx = { players: loadPlayers(), ex: loadExports(), entries: loadValidation() }
    const out = []
    try {
        for (const job of workerData.jobs) out.push(workerData.mode === 'traj' ? trajJob(ctx, job) : keeperJob(ctx, job.f, job.rec))
        parentPort.postMessage({ ok: true, out })
    } catch (err) {
        parentPort.postMessage({ ok: false, error: err.message, stack: err.stack })
    }
} else {
    const [mode, out, ...rest] = process.argv.slice(2)
    const W = 14
    let jobs = []
    const readRecords = async (path, keep) => {
        const recs = []
        for await (const line of createInterface({ input: createReadStream(path), crlfDelay: Infinity })) {
            if (!line) continue
            const r = JSON.parse(line)
            if (keep(r)) recs.push({ cond: r.cond, learner: r.learner, opponents: r.opponents, k: r.k, seed: r.seed, stratum: r.stratum, purse: r.purse, digests: r.digests })
        }
        return recs
    }
    if (mode === 'traj') {
        const [cond, records, filter] = rest
        const keep = filter === '--same-s1' ? (r) => r.learner.endsWith(':s1') && r.opponents[0].endsWith(':s1') : () => true
        jobs = await readRecords(records, keep)
        console.log(`${cond}: replaying ${jobs.length} recorded episodes`)
    } else if (mode === 'keeper') {
        const [findingsPath, runDir] = rest
        const F = JSON.parse(readFileSync(findingsPath, 'utf8')).findings
        const recs = await readRecords(`${runDir}/C4.jsonl`, (r) => F.some((f) => f.k === r.k && f.learner === r.learner && f.opponents[0] === r.opponents[0]))
        jobs = F.map((f) => ({ f, rec: recs.find((r) => r.k === f.k && r.learner === f.learner && r.opponents[0] === f.opponents[0]) }))
        if (jobs.some((j) => !j.rec)) throw new Error('finding without a recorded episode')
    } else throw new Error('mode traj|keeper')
    const t0 = Date.now()
    const parts = await Promise.all(Array.from({ length: Math.min(W, jobs.length) }, (_, w) => new Promise((res, rej) => {
        const wk = new Worker(fileURLToPath(import.meta.url), { workerData: { mode, jobs: jobs.filter((_, i) => i % W === w) } })
        wk.once('message', (m) => (m.ok ? res(m.out) : rej(new Error(m.error + '\n' + m.stack))))
        wk.once('error', rej)
    })))
    const all = parts.flat()
    if (mode === 'traj') {
        const s = createWriteStream(out)
        for (const r of all) s.write(`${JSON.stringify(r)}\n`)
        s.end()
    } else writeFileSync(out, JSON.stringify(all, null, 1))
    console.log(`REPLAYDONE ${mode} ${all.length} episodes, all digests verified, ${((Date.now() - t0) / 1000).toFixed(0)}s`)
}
