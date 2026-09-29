// Phase 2F — deterministic cross-play evaluation of Stage-B pilot checkpoints.
// Reuses the frozen Phase 2E.0 harness (harness.playEpisode: composition,
// learner = evaluate.policyController semantics, opponents = production
// runtime with the fixed clock, safety stops, defect screen) and the Phase
// 2E.0 conditions exactly; only the LEARNER is a Stage-B checkpoint.
//
//   node eval_b.mjs --learners <learners.json> --cond A|C1|C4|S4 --limit N --out <file.jsonl> [--workers W]
// learners.json: [{ key, algo, stageASeed, policy }]  (policy = path to a stage_b_pilot policy.json)
// Jobs: A  = learner × entries[0..N)
//       C1 = C4 = learner × (other four algorithms × Stage-A seeds 1–3) × entries   (the Phase 2E.0 cells)
//       S4 = learner × entries (the other four algorithms at the learner's Stage-A seed index)
import { createWriteStream, readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads'

const CP = new URL('../crossplay/', import.meta.url)
const { RL, ALGOS, SEEDS, exportKey, layerDigest, loadExports, loadPlayers, loadValidation, sha256, FROZEN_OBS_HASH, FROZEN_ACT_HASH } = await import(new URL('common.mjs', CP))
const { playEpisode } = await import(new URL('harness.mjs', CP))

const loadLearners = (specs) => Object.fromEntries(specs.map((s) => {
    const bytes = readFileSync(s.policy)
    const loaded = RL.loadPolicy(bytes.toString('utf8'))
    if (!loaded.ok) throw new Error(`stage-b learner ${s.key} rejected: ${loaded.error}`)
    const p = loaded.policy
    if (p.algorithm !== s.algo) throw new Error(`WRONG MODEL ${s.key}: ${p.algorithm}`)
    if (p.obsSpec.hash !== FROZEN_OBS_HASH || p.actSpec.hash !== FROZEN_ACT_HASH) throw new Error(`spec hash mismatch ${s.key}`)
    const tag = p.meta?.config?.stage_b?.tag
    if (tag !== 'stage_b_pilot') throw new Error(`${s.key} is not a stage_b_pilot export (tag ${tag})`)
    return [s.key, { key: s.key, algo: s.algo, seed: s.stageASeed, policy: p, sha256: sha256(bytes), layers: layerDigest(p) }]
}))

export const buildJobs = ({ cond, limit, specs }) => {
    const jobs = []
    const add = (learner, c) => { for (let k = 0; k < limit; k++) jobs.push({ id: jobs.length, k, learner, cond: c }) }
    for (const s of specs) {
        if (cond === 'A') add(s.key, { kind: 'A' })
        else if (cond === 'C1' || cond === 'C4') {
            for (const b of ALGOS) { if (b === s.algo) continue; for (const sb of SEEDS) add(s.key, { kind: cond, opp: exportKey(b, sb) }) }
        } else if (cond === 'S4') add(s.key, { kind: 'S4', learnerAlgo: s.algo, seed: s.stageASeed })
        else throw new Error(`unknown cond ${cond}`)
    }
    return jobs
}

if (!isMainThread) {
    const { jobs, specs } = workerData
    const exportsByKey = { ...loadExports(), ...loadLearners(specs) }
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
    const specs = JSON.parse(readFileSync(opt('learners'), 'utf8'))
    const cond = opt('cond')
    const out = opt('out')
    const limit = Number(opt('limit', 500))
    const W = Math.max(1, Number(opt('workers', 14)))
    loadLearners(specs) // validate before spawning
    const jobs = buildJobs({ cond, limit, specs })
    console.log(`${cond}: ${jobs.length} episodes (${specs.length} learners × ${limit} entries), ${W} workers`)
    const stream = createWriteStream(out)
    const t0 = Date.now()
    let done = 0
    let stopped = null
    await Promise.all(Array.from({ length: Math.min(W, jobs.length) }, (_, w) => new Promise((res, rej) => {
        const wk = new Worker(fileURLToPath(import.meta.url), { workerData: { jobs: jobs.filter((_, i) => i % W === w), specs } })
        wk.on('message', (m) => {
            if (m.type === 'rec') {
                stream.write(`${JSON.stringify(m.rec)}\n`)
                if (++done % 2000 === 0) console.log(`  ${done}/${jobs.length}  ${((Date.now() - t0) / 1000).toFixed(0)}s`)
            } else if (m.type === 'stop') { stopped = m; res() } else if (m.type === 'done') res()
        })
        wk.once('error', rej)
    })))
    stream.end()
    if (stopped) {
        writeFileSync(`${out}.STOP.json`, JSON.stringify(stopped, null, 2))
        console.log(`SAFETY STOP: ${stopped.message}`)
        process.exit(2)
    }
    console.log(`EVALDONE ${cond} ${done} episodes ${((Date.now() - t0) / 1000).toFixed(0)}s`)
}
