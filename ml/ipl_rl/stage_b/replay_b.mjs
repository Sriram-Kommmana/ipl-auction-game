// Phase 2F — per-decision replay of Stage-B pilot EVALUATION episodes (copy of the Phase 2E.2
// replay_obs.mjs; only the learner source and the record source differ: learners are
// stage_b_pilot exports, records come from eval_b.mjs outputs). Original header:
//
// Phase 2E.2 — READ-ONLY reproduction of recorded Phase 2E.0 episodes that
// captures the exact 80-feature observation at decision points, next to the
// hidden state the observation may or may not encode. Rooms are rebuilt with
// replay.mjs `play` (Phase 2E.1: production opponent runtime with the fixed
// clock, learner = policyController on the learner stream, tremble 0) and
// every episode MUST reproduce its recorded learner-action, auction and
// summary digests — any mismatch aborts.
//
// Captured rows
//   learner      every decision (the obs is ep.pending.obs — the exact vector
//                the frozen learner policy received)
//   rl opponent  every decision on a keeper lot, and every decision of a seat
//                named in a Phase 2E.0 finding. The obs is rebuilt with the
//                frozen buildRlObservation on the seat's own ctx/extras, and
//                for argmax algorithms the rebuilt obs must reproduce the
//                runtime's action (checked; mismatch aborts).
// Observation only: the simulator's resolveLot is wrapped on the INSTANCE to
// read the caps it is given (pass-through, unchanged arguments and result —
// the digest check proves the auction is identical).
//
//   node replay_obs.mjs <outDir> <runDir> <findings.json> <k,k,...>
import { createReadStream, readFileSync, writeFileSync, openSync, writeSync, closeSync } from 'node:fs'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads'

const CP = new URL('../crossplay/', import.meta.url)
const { RL, ROOT, loadExports, loadPlayers, loadValidation, sha256, layerDigest, FROZEN_OBS_HASH, FROZEN_ACT_HASH } = await import(new URL('common.mjs', CP))
const loadLearners = (specs) => Object.fromEntries(specs.map((sp) => {
    const bytes = readFileSync(sp.policy)
    const loaded = RL.loadPolicy(bytes.toString('utf8'))
    if (!loaded.ok) throw new Error(`learner ${sp.key} rejected: ${loaded.error}`)
    const p = loaded.policy
    if (p.algorithm !== sp.algo || p.obsSpec.hash !== FROZEN_OBS_HASH || p.actSpec.hash !== FROZEN_ACT_HASH || p.meta?.config?.stage_b?.tag !== 'stage_b_pilot') throw new Error(`bad stage-b learner ${sp.key}`)
    return [sp.key, { key: sp.key, algo: sp.algo, policy: p, sha256: sha256(bytes), layers: layerDigest(p) }]
}))
const { compose } = await import(new URL('common.mjs', CP))
const { planBid, teamComposition } = await import(new URL('packages/shared/src/planning.js', ROOT))
const { fairValue } = await import(new URL('packages/shared/src/valuation.js', ROOT))
const { RL_PERSONAS } = await import(new URL('packages/shared/src/personas.js', ROOT))

// Identical to replay.mjs `play` (Phase 2E.1), which is not importable because
// that file runs its CLI at import time.
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

export const META = [
    'ep', 'who', 'seat', 'dec', 'slNo', 'phase', 'progress', 'lotIdx',
    'lotRole', 'lotRating', 'lotOS', 'lotStar', 'fair', 'base',
    'purse', 'purseShare', 'squad', 'overseas', 'ownKeepers', 'ownStars', 'keeperNeed',
    'action', 'cap', 'capFair', 'maxSafe', 'plannerAllowed',
    'shieldState', 'forced', 'forcedKeeper', 'finalPath', 'kState', 'kViable', 'kSpare', 'kRivalsNeeding', 'kContestants',
    // hidden rival state at decision time (relative to the deciding seat)
    'rivKeepersTot', 'rivKeepersMax', 'rivKeep2', 'rivKeep3', 'rivNoNeedCanBuy', 'rivNeedKeeper',
    'rivStarsTot', 'rivStarsMax', 'rlRivals', 'rlRivPurse', 'ruleRivPurse', 'rlRivStars', 'rlRivKeepers',
    'rp0', 'rp1', 'rp2', 'rp3', 'rp4', 'rp5', 'rp6', 'rp7', 'rp8',
    // hidden history (same 20-lot window as the observation, but split by buyer)
    'wRivKeepers', 'wRivStars', 'wRlBuys', 'wRlSpend', 'wRuleSpend', 'wMaxRivSpend', 'wBuyers', 'wLen',
    'cumRlSpend', 'cumRuleSpend', 'cumRlStars', 'cumRuleStars',
    // this lot's outcome and the caps the other nine seats brought
    'winner', 'price', 'priceFair', 'othCap1', 'othCap2', 'rlCapMax', 'ruleCapMax', 'othBidders', 'rlBidders'
]
const IDX = Object.fromEntries(META.map((m, i) => [m, i]))
const ROLE = { BATSMAN: 0, BOWLER: 1, 'ALL ROUNDER': 2, 'WICKET KEEPER': 3 }
const SHIELD = { SAFE: 0, WARNING: 1, CRITICAL: 2, IMPOSSIBLE: 3 }
const isKeeper = (p) => p.role === 'WICKET KEEPER'
const isOS = (p) => p.nationality === 'Overseas'
const digestOf = (x) => sha256(RL.canonicalJson(x)).slice(0, 16)
const ARGMAX = new Set(['d3qn', 'qrdqn', 'es'])

function episodeJob(ctx, job) {
    const { players, ex, entries } = ctx
    const { rec, ep: epId, findingSeats } = job
    const cond = rec.cond === 'S4' ? { kind: 'S4', learnerAlgo: rec.learner.split(':')[0], seed: job.stageASeed } : rec.cond === 'A' ? { kind: 'A' } : { kind: rec.cond, opp: rec.opponents[0] }
    const meta = []
    const obs = []
    const capsByLot = new Map()
    let E = null
    let seatKind = null // per seat: 0 learner, 1 rlSnapshot, 2 rule/rlFallback, 3 human
    const decCount = new Array(10).fill(0)
    let learnerSeat = -1

    const rowFor = (ep, seat, c, plan, m, action, cap, o) => {
        const P = ep.rules.pursePerTeam
        const lot = c.lot
        const fv = fairValue(lot, P)
        const r = new Array(META.length).fill(NaN)
        const set = (k, v) => { r[IDX[k]] = v }
        const rivals = ep.sim.teams.map((t, j) => ({ t, j })).filter((x) => x.j !== seat)
        const comp = rivals.map(({ t }) => teamComposition(t, ep.rules))
        const keepers = rivals.map(({ t }) => t.squad.filter(isKeeper).length)
        const stars = rivals.map(({ t }) => t.squad.filter((p) => p.rating >= 90).length)
        const rl = rivals.filter((x) => seatKind[x.j] === 1 || seatKind[x.j] === 0)
        const rule = rivals.filter((x) => seatKind[x.j] === 2)
        const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN)
        set('ep', epId); set('who', seat === learnerSeat ? 0 : 1); set('seat', seat); set('dec', decCount[seat]++)
        set('slNo', lot.slNo); set('phase', c.phase === 'main' ? 0 : 1); set('progress', c.progress)
        set('lotIdx', c.phase === 'main' ? ep.sim.index : ep.sim.mainLength + ep.sim.index)
        set('lotRole', ROLE[lot.role]); set('lotRating', lot.rating); set('lotOS', isOS(lot) ? 1 : 0); set('lotStar', lot.rating >= 90 ? 1 : 0)
        set('fair', fv); set('base', lot.basePrice)
        set('purse', c.self.purseLeft); set('purseShare', c.self.purseLeft / P); set('squad', c.self.playerCount); set('overseas', c.self.overseasCount)
        set('ownKeepers', c.self.squad.filter(isKeeper).length); set('ownStars', c.self.squad.filter((p) => p.rating >= 90).length)
        set('keeperNeed', plan.requirements.keeper.need)
        set('action', action); set('cap', cap); set('capFair', cap / fv); set('maxSafe', plan.budget.maxSafeBid); set('plannerAllowed', plan.allowed ? 1 : 0)
        const d = m.shield
        set('shieldState', SHIELD[d?.state] ?? -1); set('forced', d?.forced ? 1 : 0)
        set('forcedKeeper', d?.forcedBy?.some((x) => x.startsWith('keeper')) ? 1 : 0); set('finalPath', d?.finalPath ? 1 : 0)
        const kr = d?.requirements?.keeper
        if (kr) { set('kState', SHIELD[kr.state]); set('kViable', kr.viableAfterLot); set('kSpare', kr.spare); set('kRivalsNeeding', kr.rivalsNeeding); set('kContestants', kr.contestants) }
        set('rivKeepersTot', keepers.reduce((a, b) => a + b, 0)); set('rivKeepersMax', Math.max(...keepers))
        set('rivKeep2', keepers.filter((x) => x >= 2).length); set('rivKeep3', keepers.filter((x) => x >= 3).length)
        set('rivNoNeedCanBuy', rivals.filter(({ t }, i) => comp[i].missing.keeper === 0 && t.playerCount < ep.rules.maxPlayers && t.purseLeft >= lot.basePrice).length)
        set('rivNeedKeeper', comp.filter((x) => x.missing.keeper > 0).length)
        set('rivStarsTot', stars.reduce((a, b) => a + b, 0)); set('rivStarsMax', Math.max(...stars))
        set('rlRivals', rl.length); set('rlRivPurse', mean(rl.map(({ t }) => t.purseLeft / P))); set('ruleRivPurse', mean(rule.map(({ t }) => t.purseLeft / P)))
        set('rlRivStars', rl.reduce((a, { t }) => a + t.squad.filter((p) => p.rating >= 90).length, 0))
        set('rlRivKeepers', rl.reduce((a, { t }) => a + t.squad.filter(isKeeper).length, 0))
        rivals.map(({ t }) => t.purseLeft / P).sort((a, b) => b - a).forEach((v, i) => set(`rp${i}`, v))
        const win = ep.sim.history.slice(-RL.RECENT_WINDOW)
        const rivalSet = new Set(rivals.map((x) => x.j))
        const sold = win.filter((h) => h.winner !== null && rivalSet.has(h.winner))
        set('wLen', win.length)
        set('wRivKeepers', sold.filter((h) => isKeeper(ep.players.get(h.slNo))).length)
        set('wRivStars', sold.filter((h) => ep.players.get(h.slNo).rating >= 90).length)
        set('wRlBuys', sold.filter((h) => seatKind[h.winner] <= 1).length)
        set('wRlSpend', sold.filter((h) => seatKind[h.winner] <= 1).reduce((a, h) => a + h.price, 0) / P)
        set('wRuleSpend', sold.filter((h) => seatKind[h.winner] === 2).reduce((a, h) => a + h.price, 0) / P)
        const per = new Map()
        for (const h of sold) per.set(h.winner, (per.get(h.winner) ?? 0) + h.price)
        set('wMaxRivSpend', per.size ? Math.max(...per.values()) / P : 0); set('wBuyers', per.size)
        const all = ep.sim.history.filter((h) => h.winner !== null && rivalSet.has(h.winner))
        set('cumRlSpend', all.filter((h) => seatKind[h.winner] <= 1).reduce((a, h) => a + h.price, 0) / P)
        set('cumRuleSpend', all.filter((h) => seatKind[h.winner] === 2).reduce((a, h) => a + h.price, 0) / P)
        set('cumRlStars', all.filter((h) => seatKind[h.winner] <= 1 && ep.players.get(h.slNo).rating >= 90).length)
        set('cumRuleStars', all.filter((h) => seatKind[h.winner] === 2 && ep.players.get(h.slNo).rating >= 90).length)
        meta.push(r)
        obs.push(o)
        return r
    }

    const onCreate = (ep) => {
        E = ep
        const orig = ep.sim.resolveLot.bind(ep.sim)
        ep.sim.resolveLot = (caps) => {
            capsByLot.set(`${ep.sim.phase}:${ep.sim.currentLot().slNo}`, caps.slice())
            return orig(caps)
        }
    }
    const pendingOpp = new Map()
    const onOpp = {
        before(i, c, extras) {
            if (!seatKind) return null
            const wantAll = findingSeats.includes(i)
            if (!wantAll && !isKeeper(c.lot)) return null
            const plan = planBid(c)
            const m = RL.rlActionMask(c, plan)
            if (!RL.hasBidAction(m.mask)) return null
            return { plan, m, o: RL.buildRlObservation(c, extras, plan) }
        },
        after(i, c, pre, r) {
            if (!pre) return
            const key = seatKeys[i]
            const algo = key.split(':')[0]
            if (ARGMAX.has(algo)) {
                const sc = RL.actionScores(ex[key].policy, pre.o)
                let best = -1
                for (let a = 0; a < sc.length; a++) if (pre.m.mask[a] && (best < 0 || sc[a] > sc[best])) best = a
                if (best !== r.action) throw new Error(`OBS REBUILD MISMATCH seat ${i} ${key} lot ${c.lot.slNo}: argmax ${best} ≠ runtime ${r.action}`)
            }
            rowFor(E, i, c, pre.plan, pre.m, r.action, r.cap, pre.o)
        }
    }
    let seatKeys = null
    const onLearner = {
        before(ep, a) {
            const { ctx: c, plan, mask, obs: o } = ep.pending
            rowFor(ep, learnerSeat, c, plan, mask, a, mask.caps[a], o)
        },
        after() {}
    }
    // seat kinds must be known before the first decision: derive from the composed entry
    const composed = compose(entries[rec.k], cond)
    learnerSeat = composed.entry.learnerSeat
    seatKind = composed.entry.seats.map((s, i) => (i === learnerSeat ? 0 : s.type === 'rlSnapshot' ? 1 : s.type === 'human' ? 3 : 2))
    seatKeys = composed.entry.seats.map((s, i) => (i === learnerSeat ? rec.learner : s.type === 'rlSnapshot' ? composed.snapshotKeys[s.snapshot] : null))
    const res = play({ players, ex, baseEntry: entries[rec.k], learnerKey: rec.learner, cond, onOpp, onLearner, onCreate })
    const d = { learnerActions: digestOf(res.actions), auction: digestOf(res.ep.sim.history.map((h) => [h.slNo, h.phase, h.winner, h.price])), summary: digestOf(res.summary) }
    for (const k of Object.keys(d)) if (d[k] !== rec.digests[k]) throw new Error(`REPLAY MISMATCH ${rec.cond} ${rec.learner} ${rec.opponents} k=${rec.k}: ${k}`)
    // resolve outcomes
    const hist = new Map(res.ep.sim.history.map((h) => [`${h.phase}:${h.slNo}`, h]))
    const P = res.ep.rules.pursePerTeam
    for (const r of meta) {
        const key = `${r[IDX.phase] === 0 ? 'main' : 'reauction'}:${r[IDX.slNo]}`
        const h = hist.get(key)
        const caps = capsByLot.get(key)
        const seat = r[IDX.seat]
        const fv = r[IDX.fair]
        r[IDX.winner] = h.winner === null ? -1 : h.winner === seat ? 0 : seatKind[h.winner] <= 1 ? 1 : seatKind[h.winner] === 2 ? 2 : 3
        r[IDX.price] = h.price ?? NaN
        r[IDX.priceFair] = h.price ? h.price / fv : NaN
        const others = caps.map((x, j) => ({ x, j })).filter((y) => y.j !== seat)
        const sorted = others.map((y) => y.x).sort((a, b) => b - a)
        r[IDX.othCap1] = sorted[0] / fv
        r[IDX.othCap2] = sorted[1] / fv
        const rlC = others.filter((y) => seatKind[y.j] <= 1).map((y) => y.x)
        const ruleC = others.filter((y) => seatKind[y.j] === 2).map((y) => y.x)
        r[IDX.rlCapMax] = rlC.length ? Math.max(...rlC) / fv : NaN
        r[IDX.ruleCapMax] = ruleC.length ? Math.max(...ruleC) / fv : NaN
        r[IDX.othBidders] = others.filter((y) => y.x >= r[IDX.base]).length
        r[IDX.rlBidders] = others.filter((y) => seatKind[y.j] <= 1 && y.x >= r[IDX.base]).length
    }
    const teams = res.ep.sim.teams.map((t, j) => {
        const comp = teamComposition(t, res.ep.rules)
        return { seat: j, kind: seatKind[j], key: seatKeys[j], keepers: comp.keepers, legal: comp.missing.players === 0 && comp.missing.keeper === 0 && comp.missing.bowling === 0 && comp.missing.indians === 0, purseLeftShare: t.purseLeft / P, squad: t.playerCount }
    })
    return {
        episode: { ep: epId, cond: rec.cond, learner: rec.learner, opp: rec.opponents[0] ?? '', opponents: rec.opponents, k: rec.k, seed: rec.seed, stratum: rec.stratum, purse: rec.purse, learnerSeat, findingSeats, xi: res.summary.xi, legalXI: res.summary.legalXI, teams, rows: meta.length },
        meta: Float32Array.from(meta.flat()), obs: Float32Array.from(obs.flat())
    }
}

if (!isMainThread) {
    const ctx = { players: loadPlayers(), ex: { ...loadExports(), ...loadLearners(workerData.specs) }, entries: loadValidation() }
    try {
        const eps = []
        const metas = []
        const obss = []
        for (const job of workerData.jobs) {
            const r = episodeJob(ctx, job)
            eps.push(r.episode); metas.push(r.meta); obss.push(r.obs)
        }
        const cat = (arrs) => {
            const n = arrs.reduce((a, x) => a + x.length, 0)
            const out = new Float32Array(n)
            let o = 0
            for (const x of arrs) { out.set(x, o); o += x.length }
            return out
        }
        const meta = cat(metas)
        const obs = cat(obss)
        parentPort.postMessage({ ok: true, eps, meta, obs }, [meta.buffer, obs.buffer])
    } catch (err) {
        parentPort.postMessage({ ok: false, error: err.message, stack: err.stack })
    }
} else {
    const [outDir, specsPath, runDir, kList] = process.argv.slice(2)
    const K = new Set(kList.split(',').map(Number))
    const SPECS = JSON.parse(readFileSync(specsPath, 'utf8'))
    const seedOf = Object.fromEntries(SPECS.map((sp) => [sp.key, sp.stageASeed]))
    const jobs = []
    for (const cond of ['A', 'C1', 'C4', 'S4']) {
        for await (const line of createInterface({ input: createReadStream(`${runDir}/${cond}.jsonl`), crlfDelay: Infinity })) {
            if (!line) continue
            const r = JSON.parse(line)
            if (!K.has(r.k)) continue
            const rec = { cond: r.cond, learner: r.learner, opponents: r.opponents, k: r.k, seed: r.seed, stratum: r.stratum, purse: r.purse, digests: r.digests }
            const findingSeats = (r.findings || []).map((f) => f.seat)
            jobs.push({ rec, ep: jobs.length, findingSeats, stageASeed: seedOf[r.learner] })
        }
    }
    console.log(`replaying ${jobs.length} recorded episodes on ${K.size} entries`)
    const W = 14
    const t0 = Date.now()
    // many small batches so progress is visible and memory stays bounded
    const B = 60
    const batches = []
    for (let i = 0; i < jobs.length; i += B) batches.push(jobs.slice(i, i + B))
    const fo = openSync(`${outDir}/obs.f32`, 'w')
    const fm = openSync(`${outDir}/meta.f32`, 'w')
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
            writeSync(fo, Buffer.from(m.obs.buffer)); writeSync(fm, Buffer.from(m.meta.buffer))
            episodes.push(...m.eps)
            rows += m.meta.length / META.length
            written++
        }
    }
    await Promise.all(Array.from({ length: W }, () => (async () => {
        while (next < batches.length) {
            const b = next++
            const m = await new Promise((res, rej) => {
                const wk = new Worker(fileURLToPath(import.meta.url), { workerData: { jobs: batches[b], specs: SPECS } })
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
    closeSync(fo); closeSync(fm)
    writeFileSync(`${outDir}/episodes.json`, JSON.stringify({ meta: META, obsSize: RL.OBS_SIZE, entries: [...K].sort((a, b) => a - b), episodes }))
    console.log(`REPLAYDONE ${episodes.length} episodes, ${rows} rows, all digests verified, ${((Date.now() - t0) / 1000).toFixed(0)}s`)
}
