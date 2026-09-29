// Phase 2D.1 requirement / feasibility audit (evaluation only). Replays the
// validation episodes of checkpoint policies exactly as the evaluator does
// (act-v3, temperature 0.3, learner stream — cross-checked against the
// evaluator's episodes) and records, per episode:
//   - per requirement (keeper / bowling / indians): initial need, the purchase
//     that closed it (phase, progress, price, forced?, final path?, at the
//     requirement's final opportunity?), min purse while it was unmet
//   - min purse while ANY requirement was unmet
//   - forced bids by reason, won, final-path forced bids, re-auction forced bids
//   - decisions by shield state and by the planner's whole-XI completion status
//   node audit_requirements.mjs <out.json> <tag=policy.json> ...
import { readFileSync, writeFileSync } from 'node:fs'
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads'
import { fileURLToPath } from 'node:url'
const ROOT = new URL('../../../', import.meta.url)
const { parsePlayersCsv } = await import(new URL('packages/shared/src/playersCsv.js', ROOT))
const RL = await import(new URL('packages/shared/src/rl/index.js', ROOT))
const REQS = ['keeper', 'bowling', 'indians']

if (!isMainThread) {
    const players = parsePlayersCsv(readFileSync(new URL('apps/server/src/db/players.csv', ROOT), 'utf8'))
    const entries = JSON.parse(readFileSync(new URL('packages/shared/data/rl-manifests/validation.json', ROOT), 'utf8')).entries
    const out = []
    const policies = new Map()
    for (const [policyPath, tag, k] of workerData.jobs) {
        if (!policies.has(policyPath)) {
            const loaded = RL.loadPolicy(readFileSync(policyPath, 'utf8'))
            if (!loaded.ok) throw new Error(loaded.error)
            policies.set(policyPath, loaded.policy)
        }
        const ctrl = RL.policyController(policies.get(policyPath))
        const ep = new RL.RlEpisode({ players, entry: entries[k], tremble: 0 })
        const rng = ep.learnerRng
        ep.reset()
        const rec = {
            tag, seed: entries[k].seed, stratum: entries[k].stratum,
            req: Object.fromEntries(REQS.map((r) => [r, { initialNeed: null, closed: null, minPurseUnmet: null }])),
            minPurseAnyUnmet: null, forced: 0, forcedWon: 0, finalPath: 0, finalPathWon: 0, reauctionForced: 0, forcedBy: {},
            shieldStates: {}, completion: {}, alreadyInfeasible: 0
        }
        let step
        do {
            const { ctx, plan, mask } = ep.pending
            const d = mask.shield
            const need = Object.fromEntries(REQS.map((r) => [r, plan.requirements[r].need]))
            for (const r of REQS) {
                if (rec.req[r].initialNeed === null) rec.req[r].initialNeed = need[r]
                if (need[r] > 0) rec.req[r].minPurseUnmet = Math.min(rec.req[r].minPurseUnmet ?? Infinity, ctx.self.purseLeft)
            }
            if (REQS.some((r) => need[r] > 0)) rec.minPurseAnyUnmet = Math.min(rec.minPurseAnyUnmet ?? Infinity, ctx.self.purseLeft)
            rec.shieldStates[d.state] = (rec.shieldStates[d.state] ?? 0) + 1
            const cs = plan.teamNeeds.completion
            rec.completion[cs] = (rec.completion[cs] ?? 0) + 1
            if (d.alreadyInfeasible) rec.alreadyInfeasible++
            const finalOpp = Object.fromEntries(REQS.map((r) => [r, Boolean(plan.requirements[r].finalOpportunity)]))
            const a = ctrl.act(ep, rng)
            step = ep.step(a)
            const L = step.info.lastLot
            if (d.forced) {
                rec.forced++
                if (L.won) rec.forcedWon++
                if (d.finalPath) { rec.finalPath++; if (L.won) rec.finalPathWon++ }
                if (ctx.phase === 'reauction') rec.reauctionForced++
                for (const r of d.forcedBy) rec.forcedBy[r] = (rec.forcedBy[r] ?? 0) + 1
            }
            if (L.won) {
                const after = step.done ? null : ep.pending.plan.requirements
                for (const r of REQS) {
                    if (need[r] > 0 && rec.req[r].closed === null && (after ? after[r].need === 0 : true)) {
                        rec.req[r].closed = {
                            phase: ctx.phase, progress: +ctx.progress.toFixed(3), price: L.price, forced: Boolean(d.forced),
                            finalPath: Boolean(d.finalPath), finalOpportunity: finalOpp[r], atFinalStep: step.done
                        }
                    }
                }
            }
        } while (!step.done)
        const s = step.info.episode
        Object.assign(rec, { legalXI: s.legalXI, strongXI: s.strongXI, xi: s.xi, purseLeft: s.purseLeft, decisions: s.decisions, evalForced: s.shield?.forced })
        // A requirement closed on the very last decision is only confirmed by the final legal XI.
        for (const r of REQS) if (rec.req[r].closed?.atFinalStep && !s.legalXI) rec.req[r].closed = null
        out.push(rec)
    }
    parentPort.postMessage(out)
} else {
    const [outPath, ...specs] = process.argv.slice(2)
    const jobs = specs.flatMap((spec) => {
        const i = spec.indexOf('=')
        return Array.from({ length: Number(process.env.AUDIT_LIMIT || 500) }, (_, k) => [spec.slice(i + 1), spec.slice(0, i), k])
    })
    const W = Math.min(14, jobs.length)
    const parts = await Promise.all(Array.from({ length: W }, (_, w) => new Promise((res, rej) => {
        const wk = new Worker(fileURLToPath(import.meta.url), { workerData: { jobs: jobs.filter((_, i) => i % W === w) }, resourceLimits: { maxYoungGenerationSizeMb: 96 } })
        wk.once('message', res); wk.once('error', rej)
    })))
    writeFileSync(outPath, JSON.stringify(parts.flat()))
    console.log(`audited ${parts.flat().length} episodes`)
}
