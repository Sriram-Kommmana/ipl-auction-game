// Phase 2E.0 — parallel, deterministic cross-play runner (evaluation only).
//
//   node ml/ipl_rl/crossplay/run.mjs --cond A|C1|C4|S4 --out <file.jsonl>
//        [--limit N]          first N validation entries (default 500)
//        [--workers W]        worker threads (default 14); results do not depend on W
//        [--seeds 1,2,3]      training seeds used (learner and opponent)
//        [--seed-pairs same]  C1/C4: only s_i vs s_i instead of all 3 × 3
//        [--hashes <file>]    frozen-hash record to verify before and after (required for full runs)
//
// Jobs: A  = 15 exports × entries (Stage-A control, the manifest entry unchanged)
//       C1 = C4 = 20 ordered algorithm pairs × 3 × 3 seed pairs × entries
//       S4 = 15 learner exports × entries (the other four algorithms, same seed index)
// Every job depends only on (condition, learner export, opponent export(s),
// manifest entry), so the output is identical for any worker count or
// scheduling. Output: one JSON line per episode, tagged with its job id; a
// safety stop terminates every worker and writes <out>.STOP.json.

import { createWriteStream, existsSync, readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads'
import { ALGOS, SEEDS, exportKey, frozenHashes, loadExports, loadPlayers, loadValidation } from './common.mjs'
import { playEpisode } from './harness.mjs'

export const buildJobs = ({ cond, limit = 500, seeds = SEEDS, seedPairs = 'all' }) => {
    const entries = Array.from({ length: limit }, (_, k) => k)
    const jobs = []
    const add = (learner, c) => { for (const k of entries) jobs.push({ id: jobs.length, k, learner, cond: c }) }
    if (cond === 'A') {
        for (const a of ALGOS) for (const s of seeds) add(exportKey(a, s), { kind: 'A' })
    } else if (cond === 'C1' || cond === 'C4') {
        for (const a of ALGOS) for (const b of ALGOS) {
            if (a === b) continue
            for (const sa of seeds) for (const sb of seeds) {
                if (seedPairs === 'same' && sa !== sb) continue
                add(exportKey(a, sa), { kind: cond, opp: exportKey(b, sb) })
            }
        }
    } else if (cond === 'S4') {
        for (const a of ALGOS) for (const s of seeds) add(exportKey(a, s), { kind: 'S4', learnerAlgo: a, seed: s })
    } else throw new Error(`unknown --cond ${cond}`)
    return jobs
}

const sameHashes = (a, b) => JSON.stringify(a) === JSON.stringify(b)

if (!isMainThread) {
    const { jobs } = workerData
    const exportsByKey = loadExports()
    const players = loadPlayers()
    const entries = loadValidation()
    for (const job of jobs) {
        try {
            const rec = playEpisode({ players, baseEntry: entries[job.k], learnerKey: job.learner, cond: job.cond, exportsByKey })
            parentPort.postMessage({ type: 'rec', rec: { job: job.id, k: job.k, ...rec } })
        } catch (err) {
            parentPort.postMessage({ type: 'stop', job, name: err.name, message: err.message, stack: err.stack, detail: err.detail ?? null })
            break
        }
    }
    parentPort.postMessage({ type: 'done' })
} else if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
    const args = process.argv.slice(2)
    const opt = (n, d) => (args.includes(`--${n}`) ? args[args.indexOf(`--${n}`) + 1] : d)
    const cond = opt('cond')
    const out = opt('out')
    if (!cond || !out) throw new Error('--cond and --out are required')
    const limit = Number(opt('limit', 500))
    const W = Math.max(1, Number(opt('workers', 14)))
    const seeds = opt('seeds', '1,2,3').split(',').map(Number)
    const seedPairs = opt('seed-pairs', 'all')
    const hashFile = opt('hashes')
    const stopFile = `${out}.STOP.json`
    const fail = (payload) => {
        writeFileSync(stopFile, JSON.stringify(payload, null, 2))
        console.log(`SAFETY STOP: ${payload.message}`)
        process.exit(2)
    }

    // Frozen files and exports must match the record, before and after.
    let recorded = null
    if (hashFile) {
        recorded = JSON.parse(readFileSync(hashFile, 'utf8'))
        if (!sameHashes(frozenHashes(), recorded)) fail({ message: 'CHANGED FROZEN FILE or EXPORT before the run', stage: 'pre' })
    }
    loadExports() // wrong model / spec hash → throws before any episode
    const jobs = buildJobs({ cond, limit, seeds, seedPairs })
    const t0 = Date.now()
    const stream = createWriteStream(out)
    let done = 0
    let finished = 0
    let lastPct = -1
    const workers = []
    console.log(`${cond}: ${jobs.length} episodes on ${W} workers → ${out}`)
    for (let w = 0; w < W; w++) {
        const mine = jobs.filter((_, i) => i % W === w)
        const wk = new Worker(fileURLToPath(import.meta.url), { workerData: { jobs: mine } })
        workers.push(wk)
        wk.on('message', (msg) => {
            if (msg.type === 'rec') {
                stream.write(`${JSON.stringify(msg.rec)}\n`)
                done++
                const pct = Math.floor((100 * done) / jobs.length)
                if (pct !== lastPct && pct % 5 === 0) {
                    lastPct = pct
                    const s = (Date.now() - t0) / 1000
                    console.log(`progress ${cond} ${done}/${jobs.length} (${pct}%) ${(done / s).toFixed(1)} eps/s elapsed ${s.toFixed(0)}s`)
                }
            } else if (msg.type === 'stop') {
                for (const x of workers) x.terminate()
                stream.end(() => fail({ message: msg.message, name: msg.name, job: msg.job, stack: msg.stack, detail: msg.detail }))
            } else if (msg.type === 'done') {
                finished++
                if (finished === W) {
                    stream.end(() => {
                        if (done !== jobs.length) fail({ message: `only ${done}/${jobs.length} episodes finished` })
                        if (recorded && !sameHashes(frozenHashes(), recorded)) fail({ message: 'CHANGED FROZEN FILE or EXPORT after the run', stage: 'post' })
                        const s = (Date.now() - t0) / 1000
                        writeFileSync(`${out}.meta.json`, JSON.stringify({ cond, episodes: done, workers: W, seconds: s, episodesPerSecond: done / s, limit, seeds, seedPairs, hashesVerified: Boolean(recorded) }, null, 2))
                        console.log(`ALLDONE ${cond} ${done} episodes in ${s.toFixed(0)}s (${(done / s).toFixed(1)} eps/s)`)
                    })
                }
            }
        })
        wk.on('error', (err) => {
            for (const x of workers) x.terminate()
            fail({ message: `worker crashed: ${err.message}`, stack: err.stack })
        })
    }
    if (existsSync(stopFile)) console.log(`note: an older ${stopFile} exists`)
}
