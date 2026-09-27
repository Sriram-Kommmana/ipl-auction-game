// Phase 2C.3 behaviour audit (evaluation only): replay validation episodes of a
// checkpoint policy exactly as the evaluator does (act-v3, temperature 0.3,
// learner stream) and record requirement-related behaviour per episode:
// keeper purchase (phase, price, purse before, forced?), the minimum purse
// while a keeper was still needed, forced bids by requirement and whether they
// were won, and whether a requirement was only ever filled on a final path.
import { readFileSync, writeFileSync } from 'node:fs'
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads'
import { fileURLToPath } from 'node:url'
const base = 'file:///C:/web%20dev%20projects/ipl-auction-game/packages/shared/'
const { parsePlayersCsv } = await import(base + 'src/playersCsv.js')
const RL = await import(base + 'src/rl/index.js')

if (!isMainThread) {
    const players = parsePlayersCsv(readFileSync('C:/web dev projects/ipl-auction-game/apps/server/src/db/players.csv', 'utf8'))
    const entries = JSON.parse(readFileSync('C:/web dev projects/ipl-auction-game/packages/shared/data/rl-manifests/validation.json', 'utf8')).entries
    const out = []
    for (const [policyPath, tag, k] of workerData.jobs) {
        const policy = RL.loadPolicy(readFileSync(policyPath, 'utf8'))
        if (!policy.ok) throw new Error(policy.error)
        const ctrl = RL.policyController(policy.policy)
        const ep = new RL.RlEpisode({ players, entry: entries[k], tremble: 0 })
        const rng = ep.learnerRng
        ep.reset()
        const rec = { tag, seed: entries[k].seed, stratum: entries[k].stratum, keeper: null, minPurseKeeperNeeded: null, forced: 0, forcedWon: 0, finalPath: 0, finalPathWon: 0, forcedBy: {}, keeperForcedLots: 0 }
        let step
        do {
            const { ctx, plan, mask } = ep.pending
            const keeperNeeded = plan.requirements.keeper.need > 0
            if (keeperNeeded) rec.minPurseKeeperNeeded = Math.min(rec.minPurseKeeperNeeded ?? Infinity, ctx.self.purseLeft)
            const purseBefore = ctx.self.purseLeft
            const d = mask.shield
            const a = ctrl.act(ep, rng)
            step = ep.step(a)
            const L = step.info.lastLot
            if (d.forced) {
                rec.forced++
                if (L.won) rec.forcedWon++
                if (d.finalPath) { rec.finalPath++; if (L.won) rec.finalPathWon++ }
                for (const r of d.forcedBy) rec.forcedBy[r] = (rec.forcedBy[r] ?? 0) + 1
                if (ctx.lot.role === 'WICKET KEEPER' && keeperNeeded) rec.keeperForcedLots++
            }
            if (L.won && ctx.lot.role === 'WICKET KEEPER' && keeperNeeded) {
                rec.keeper = { phase: ctx.phase, price: L.price, purseBefore, forced: Boolean(d.forced), finalPath: Boolean(d.finalPath), progress: +ctx.progress.toFixed(3) }
            }
        } while (!step.done)
        const s = step.info.episode
        Object.assign(rec, { legalXI: s.legalXI, xi: s.xi, purseLeft: s.purseLeft, decisions: s.decisions })
        out.push(rec)
    }
    parentPort.postMessage(out)
} else {
    // argv: <out.json> <tag=policyPath> ...
    const [outPath, ...specs] = process.argv.slice(2)
    const jobs = specs.flatMap((spec) => {
        const i = spec.indexOf('=')
        return Array.from({ length: 500 }, (_, k) => [spec.slice(i + 1), spec.slice(0, i), k])
    })
    const W = 14
    const parts = await Promise.all(Array.from({ length: W }, (_, w) => new Promise((res, rej) => {
        const wk = new Worker(fileURLToPath(import.meta.url), { workerData: { jobs: jobs.filter((_, i) => i % W === w) }, resourceLimits: { maxYoungGenerationSizeMb: 96 } })
        wk.once('message', res); wk.once('error', rej)
    })))
    writeFileSync(outPath, JSON.stringify(parts.flat()))
    console.log(`audited ${parts.flat().length} episodes`)
}
