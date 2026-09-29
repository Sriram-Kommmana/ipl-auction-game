// Phase 2E.3 — CANDIDATE-SIGNAL replay (design analysis only).
//
// Replays the same recorded Phase 2E.0 episodes and produces the SAME decision
// rows as Phase 2E.2 replay_obs.mjs (same hooks, same selection, same order),
// computing for each row candidate signals from information available at that
// exact decision point: the current public state (every squad and purse, the
// player card, the catalogue minus auctioned players) and the history of lots
// ALREADY resolved. No row uses a future lot, price, winner or squad.
// obs-v2 is untouched; the values exist only in ml/runs/_2e3 for the
// counterfactual statistics of Phase 2E.3.
//
// Replay mechanics as Phase 2E.1/2E.2 (production opponent runtime with the
// fixed clock, learner = policyController on the learner stream, tremble 0);
// every episode MUST reproduce its recorded learner-action, auction and summary
// digests — any mismatch aborts. The simulator instance's resolveLot is wrapped
// pass-through to read the caps of each resolved lot (the per-lot ladder
// participants — public in a live auction, internal in the simulator).
//
//   node replay_cand.mjs <outDir> <runDir> <findings.json> <k,k,...>
import { createReadStream, readFileSync, writeFileSync, openSync, writeSync, closeSync } from 'node:fs'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads'

const CP = new URL('../crossplay/', import.meta.url)
const { RL, ROOT, loadExports, loadPlayers, loadValidation, sha256, compose } = await import(new URL('common.mjs', CP))
const { planBid, teamComposition, legalCompletionCost, supplyOf } = await import(new URL('packages/shared/src/planning.js', ROOT))
const { bidBlocker } = await import(new URL('packages/shared/src/rules.js', ROOT))
const { fairValue } = await import(new URL('packages/shared/src/valuation.js', ROOT))
const { RL_PERSONAS } = await import(new URL('packages/shared/src/personas.js', ROOT))

export const META = [
    'ep', 'who', 'seat', 'dec', 'slNo', 'phase',
    // keeper competition (current public state)
    'kContestAll', 'kContestNeed', 'kContestNoNeed', 'kContestNoNeedFair', 'kStockExcess', 'kStock2',
    // keeper purchase history (results so far): buys by rivals already holding a keeper
    'kHistExtraBuys', 'kHistExtraBuyers', 'kW20Extra', 'kW20Buys',
    // rival capacity, unclipped (the quantity riv_capacity_1-3 clip at 5)
    'capRaw1', 'capRaw2', 'capRaw3', 'capGe1_5', 'capGe2', 'capGe3', 'capGe5',
    // per-rival history (results so far)
    'overpay13', 'overpay15', 'starBuyers2', 'buyersSoFar',
    // pooled history beyond the 20-lot window; bidding breadth (ladder participants)
    'cumPF', 'cumStarPF', 'cumSales', 'partic20', 'partic5', 'particAll', 'bids20', 'bids5',
    // unclipped scarcity (the quantities mkt_scarcity_* clip at 5)
    'scarKeeperRaw', 'scarBowlRaw', 'scarIndRaw', 'needBowl', 'needInd', 'rivNeedBowl', 'rivNeedInd',
    // outcome of a keeper lot (evaluation only): -1 unsold, 0 this seat, 1 rival without a keeper, 2 rival already holding one
    'kWinnerType'
]
const IDX = Object.fromEntries(META.map((m, i) => [m, i]))
const isKeeper = (p) => p.role === 'WICKET KEEPER'
const digestOf = (x) => sha256(RL.canonicalJson(x)).slice(0, 16)

// Identical to replay.mjs / replay_obs.mjs `play`.
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

function episodeJob(ctx, job) {
    const { players, ex, entries } = ctx
    const { rec, ep: epId, findingSeats } = job
    const cond = rec.cond === 'S4' ? { kind: 'S4', learnerAlgo: rec.learner.split(':')[0], seed: Number(rec.learner.split(':s')[1]) } : rec.cond === 'A' ? { kind: 'A' } : { kind: rec.cond, opp: rec.opponents[0] }
    const meta = []
    const lotLog = [] // per resolved lot, in order: { caps, base }
    const decCount = new Array(10).fill(0)
    let E = null
    const composed = compose(entries[rec.k], cond)
    const learnerSeat = composed.entry.learnerSeat

    const rowFor = (ep, seat, c, plan) => {
        const P = ep.rules.pursePerTeam
        const lot = c.lot
        const fv = fairValue(lot, P)
        const r = new Array(META.length).fill(NaN)
        const set = (k, v) => { r[IDX[k]] = v }
        set('ep', epId); set('who', seat === learnerSeat ? 0 : 1); set('seat', seat); set('dec', decCount[seat]++)
        set('slNo', lot.slNo); set('phase', c.phase === 'main' ? 0 : 1)
        const rivals = ep.sim.teams.map((t, j) => ({ t, j })).filter((x) => x.j !== seat)
        const supply = supplyOf(c)
        const view = rivals.map(({ t, j }) => {
            const comp = teamComposition(t, ep.rules)
            const blocked = bidBlocker({ team: t, lot: { ...lot, currentBidderId: '' }, rules: ep.rules, amount: lot.basePrice }) !== null
            const cost = legalCompletionCost(comp.counts, { slots: comp.slotsLeft, overseasSlots: comp.overseasSlots, supply }).cost
            const capacity = blocked ? 0 : Math.max(0, t.purseLeft - (Number.isFinite(cost) ? cost : 0)) / fv
            return { j, blocked, capacity, keepers: comp.keepers, needKeeper: comp.missing.keeper > 0 }
        })
        const open = view.filter((v) => !v.blocked)
        set('kContestAll', open.length)
        set('kContestNeed', open.filter((v) => v.needKeeper).length)
        set('kContestNoNeed', open.filter((v) => !v.needKeeper).length)
        set('kContestNoNeedFair', open.filter((v) => !v.needKeeper && v.capacity >= 1).length)
        set('kStockExcess', view.reduce((a, v) => a + Math.max(0, v.keepers - 1), 0))
        set('kStock2', view.filter((v) => v.keepers >= 2).length)
        // results so far (all strictly before this decision)
        const H = ep.sim.history
        const kept = new Array(ep.sim.teams.length).fill(0)
        const per = ep.sim.teams.map(() => ({ n: 0, pf: 0, stars: 0 }))
        const rivalSet = new Set(rivals.map((x) => x.j))
        const extraIdx = []
        const keeperIdx = []
        const extraBuyers = new Set()
        let pfSum = 0, pfN = 0, sSum = 0, sN = 0
        H.forEach((h, idx) => {
            if (h.winner === null) return
            const p = ep.players.get(h.slNo)
            const pf = h.price / fairValue(p, P)
            pfSum += pf; pfN++
            if (p.rating >= 90) { sSum += pf; sN++ }
            const w = per[h.winner]
            w.n++; w.pf += pf
            if (p.rating >= 90) w.stars++
            if (isKeeper(p)) {
                if (rivalSet.has(h.winner)) {
                    keeperIdx.push(idx)
                    if (kept[h.winner] >= 1) { extraIdx.push(idx); extraBuyers.add(h.winner) }
                }
                kept[h.winner]++
            }
        })
        const w0 = H.length - RL.RECENT_WINDOW
        set('kHistExtraBuys', extraIdx.length); set('kHistExtraBuyers', extraBuyers.size)
        set('kW20Extra', extraIdx.filter((i) => i >= w0).length); set('kW20Buys', keeperIdx.filter((i) => i >= w0).length)
        const caps = open.map((v) => v.capacity).sort((a, b) => b - a)
        set('capRaw1', caps[0] ?? 0); set('capRaw2', caps[1] ?? 0); set('capRaw3', caps[2] ?? 0)
        for (const [k, t] of [['capGe1_5', 1.5], ['capGe2', 2], ['capGe3', 3], ['capGe5', 5]]) set(k, caps.filter((x) => x >= t).length)
        const riv = rivals.map(({ j }) => per[j])
        set('overpay13', riv.filter((w) => w.n >= 2 && w.pf / w.n >= 1.3).length)
        set('overpay15', riv.filter((w) => w.n >= 2 && w.pf / w.n >= 1.5).length)
        set('starBuyers2', riv.filter((w) => w.stars >= 2).length)
        set('buyersSoFar', riv.filter((w) => w.n >= 1).length)
        set('cumPF', pfN ? pfSum / pfN : NaN); set('cumStarPF', sN ? sSum / sN : NaN); set('cumSales', pfN)
        const part = (L) => (L.length ? L.reduce((a, e) => a + e.caps.filter((x, j) => j !== seat && x >= e.base).length, 0) / L.length : NaN)
        set('partic20', part(lotLog.slice(-20))); set('partic5', part(lotLog.slice(-5))); set('particAll', part(lotLog))
        const bidsOf = (L) => (L.length ? L.reduce((a, h) => a + h.bids, 0) / L.length : NaN)
        set('bids20', bidsOf(H.slice(-20))); set('bids5', bidsOf(H.slice(-5)))
        const q = plan.requirements
        const scar = (x) => x.remainingAfterLot / (x.need + x.rivalsNeeding + 1)
        set('scarKeeperRaw', scar(q.keeper)); set('scarBowlRaw', scar(q.bowling)); set('scarIndRaw', scar(q.indians))
        set('needBowl', q.bowling.need); set('needInd', q.indians.need); set('rivNeedBowl', q.bowling.rivalsNeeding); set('rivNeedInd', q.indians.rivalsNeeding)
        meta.push(r)
    }

    const onCreate = (ep) => {
        E = ep
        const orig = ep.sim.resolveLot.bind(ep.sim)
        ep.sim.resolveLot = (caps) => {
            const base = ep.sim.currentLot().basePrice
            const out = orig(caps)
            lotLog.push({ caps: caps.slice(), base })
            return out
        }
    }
    // Row selection identical to replay_obs.mjs (Phase 2E.2).
    const onOpp = {
        before(i, c) {
            if (!findingSeats.includes(i) && !isKeeper(c.lot)) return null
            const plan = planBid(c)
            const m = RL.rlActionMask(c, plan)
            if (!RL.hasBidAction(m.mask)) return null
            return { plan }
        },
        after(i, c, pre) {
            if (pre) rowFor(E, i, c, pre.plan)
        }
    }
    const onLearner = {
        before(ep) {
            const { ctx: c, plan } = ep.pending
            rowFor(ep, learnerSeat, c, plan)
        },
        after() {}
    }
    const res = play({ players, ex, baseEntry: entries[rec.k], learnerKey: rec.learner, cond, onOpp, onLearner, onCreate })
    const d = { learnerActions: digestOf(res.actions), auction: digestOf(res.ep.sim.history.map((h) => [h.slNo, h.phase, h.winner, h.price])), summary: digestOf(res.summary) }
    for (const k of Object.keys(d)) if (d[k] !== rec.digests[k]) throw new Error(`REPLAY MISMATCH ${rec.cond} ${rec.learner} ${rec.opponents} k=${rec.k}: ${k}`)
    // keeper-lot outcome (evaluation label only): did the winner already hold a keeper?
    const kc = new Array(res.ep.sim.teams.length).fill(0)
    const kOut = new Map()
    for (const h of res.ep.sim.history) {
        if (!isKeeper(res.ep.players.get(h.slNo))) continue
        kOut.set(`${h.phase}:${h.slNo}`, h.winner === null ? { w: null } : { w: h.winner, had: kc[h.winner] })
        if (h.winner !== null) kc[h.winner]++
    }
    for (const r of meta) {
        const o = kOut.get(`${r[IDX.phase] === 0 ? 'main' : 'reauction'}:${r[IDX.slNo]}`)
        if (o) r[IDX.kWinnerType] = o.w === null ? -1 : o.w === r[IDX.seat] ? 0 : o.had === 0 ? 1 : 2
    }
    return { episode: { ep: epId, cond: rec.cond, learner: rec.learner, opp: rec.opponents[0] ?? '', k: rec.k, rows: meta.length }, meta: Float32Array.from(meta.flat()) }
}

if (!isMainThread) {
    const ctx = { players: loadPlayers(), ex: loadExports(), entries: loadValidation() }
    try {
        const eps = []
        const metas = []
        for (const job of workerData.jobs) {
            const r = episodeJob(ctx, job)
            eps.push(r.episode); metas.push(r.meta)
        }
        const n = metas.reduce((a, x) => a + x.length, 0)
        const meta = new Float32Array(n)
        let o = 0
        for (const x of metas) { meta.set(x, o); o += x.length }
        parentPort.postMessage({ ok: true, eps, meta }, [meta.buffer])
    } catch (err) {
        parentPort.postMessage({ ok: false, error: err.message, stack: err.stack })
    }
} else {
    const [outDir, runDir, findingsPath, kList] = process.argv.slice(2)
    const K = new Set(kList.split(',').map(Number))
    const F = JSON.parse(readFileSync(findingsPath, 'utf8')).findings
    const jobs = []
    for (const cond of ['A', 'C1', 'C4', 'S4']) {
        for await (const line of createInterface({ input: createReadStream(`${runDir}/${cond}.jsonl`), crlfDelay: Infinity })) {
            if (!line) continue
            const r = JSON.parse(line)
            if (!K.has(r.k)) continue
            const rec = { cond: r.cond, learner: r.learner, opponents: r.opponents, k: r.k, seed: r.seed, stratum: r.stratum, purse: r.purse, digests: r.digests }
            const findingSeats = cond === 'C4' ? F.filter((f) => f.k === r.k && f.learner === r.learner && f.opponents[0] === r.opponents[0]).map((f) => f.seat) : []
            jobs.push({ rec, ep: jobs.length, findingSeats })
        }
    }
    console.log(`replaying ${jobs.length} recorded episodes on ${K.size} entries`)
    const W = 14
    const t0 = Date.now()
    const B = 60
    const batches = []
    for (let i = 0; i < jobs.length; i += B) batches.push(jobs.slice(i, i + B))
    const fm = openSync(`${outDir}/cand.f32`, 'w')
    const episodes = []
    let next = 0
    let done = 0
    let rows = 0
    const results = new Map()
    let written = 0
    const flush = () => {
        while (results.has(written)) {
            const m = results.get(written)
            results.delete(written)
            writeSync(fm, Buffer.from(m.meta.buffer))
            episodes.push(...m.eps)
            rows += m.meta.length / META.length
            written++
        }
    }
    await Promise.all(Array.from({ length: W }, () => (async () => {
        while (next < batches.length) {
            const b = next++
            const m = await new Promise((res, rej) => {
                const wk = new Worker(fileURLToPath(import.meta.url), { workerData: { jobs: batches[b] } })
                wk.once('message', (msg) => (msg.ok ? res(msg) : rej(new Error(msg.error + '\n' + msg.stack))))
                wk.once('error', rej)
            })
            results.set(b, m)
            flush()
            done += batches[b].length
            if (b % 20 === 0) console.log(`  ${done}/${jobs.length} episodes, ${((Date.now() - t0) / 1000).toFixed(0)}s`)
        }
    })()))
    flush()
    closeSync(fm)
    writeFileSync(`${outDir}/cand-episodes.json`, JSON.stringify({ meta: META, entries: [...K].sort((a, b) => a - b), episodes }))
    console.log(`REPLAYDONE ${episodes.length} episodes, ${rows} rows, all digests verified, ${((Date.now() - t0) / 1000).toFixed(0)}s`)
}
