#!/usr/bin/env node
// Baseline / policy evaluation on the committed manifests (deterministic Node).
//
//   node packages/shared/bin/rl-evaluate.js --split validation --limit 50
//   node packages/shared/bin/rl-evaluate.js --split test --baselines moneyball,randomLegal --out report.json
//   node packages/shared/bin/rl-evaluate.js --split validation --policy path/to/policy.json --baselines none \
//        --compare packages/shared/data/rl-baselines/validation.episodes.json --workers 12
//   node packages/shared/bin/rl-evaluate.js --split validation --workers 12 --lock   (writes the locked baselines)
//
// Every controller plays the SAME auctions (paired by seed). Reports mean and
// bootstrap 95% CI per metric, per purse stratum, and the paired XI
// difference against the reference (default moneyball).
//
// --workers N   run episodes on N worker threads. Every episode depends only
//               on its manifest entry and controller, so results are
//               identical for any N.
// --compare F   add the locked baseline episodes in F (not re-run) to the
//               report, for paired comparisons against a policy.
// --episodes-out F   also write every episode summary.
// --lock        write data/rl-baselines/<split>.{report,episodes}.json (the
//               permanent baseline reference; refuses to overwrite).

import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { execSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads'
import { parsePlayersCsv } from '../src/playersCsv.js'
import {
    ACT_SPEC_HASH, ADVERSARIAL, BASELINES, OBS_SPEC_HASH, SHIELD_V2_PARAMS, buildReport, canonicalJson, fnv1a64, generateManifest, loadPolicy, policyController, runEpisode
} from '../src/rl/index.js'

const loadPlayers = () => parsePlayersCsv(readFileSync(fileURLToPath(new URL('../../../apps/server/src/db/players.csv', import.meta.url)), 'utf8'))
const loadManifest = (split) => JSON.parse(readFileSync(fileURLToPath(new URL(`../data/rl-manifests/${split}.json`, import.meta.url)), 'utf8'))
const CONTROLLERS = { ...BASELINES, ...ADVERSARIAL }
const controllerFor = (name, policy) => (name.startsWith('policy:') ? policyController(policy) : CONTROLLERS[name])

if (!isMainThread) {
    // Worker: play the assigned (controller, entry) jobs.
    const { split, jobs, policyText, shieldOpts } = workerData
    const players = loadPlayers()
    const entries = loadManifest(split).entries
    const policy = policyText ? loadPolicy(policyText).policy : null
    const rows = jobs.map(([name, k]) => runEpisode({ players, entry: entries[k], controller: controllerFor(name, policy), ...shieldOpts }))
    parentPort.postMessage(rows)
} else {
    const args = process.argv.slice(2)
    const opt = (name, fallback) => (args.includes(`--${name}`) ? args[args.indexOf(`--${name}`) + 1] : fallback)
    const flag = (name) => args.includes(`--${name}`)
    const split = opt('split', 'validation')
    const limit = Number(opt('limit', Infinity))
    const baselineArg = opt('baselines', Object.keys(BASELINES).join(','))
    const names = baselineArg === 'none' ? [] : baselineArg.split(',').filter(Boolean)
    const reference = opt('reference', 'moneyball')
    const workers = Math.max(1, Number(opt('workers', 1)))
    const quiet = flag('quiet')
    // --shield v2: the CANDIDATE completion shield (not adopted; default v1 = frozen).
    // --shield-diagnostics: under v1, also report what v2 would have done.
    // --shield-params '{"marginIncrements":2}': candidate-shield sensitivity runs.
    const shieldOpts = { shield: opt('shield', 'v1'), shieldDiagnostics: flag('shield-diagnostics'), shieldTrace: flag('shield-trace'),
        ...(opt('shield-params') ? { shieldParams: { ...SHIELD_V2_PARAMS, ...JSON.parse(opt('shield-params')) } } : {}) }

    const manifest = loadManifest(split)
    // Guard against a stale committed manifest.
    if (JSON.stringify(generateManifest(split, manifest.entries.length).entries) !== JSON.stringify(manifest.entries)) {
        throw new Error(`${split}.json no longer matches the sampler — regenerate with bin/rl-manifests.js`)
    }
    const entries = manifest.entries.slice(0, limit)
    const manifestHash = fnv1a64(canonicalJson(entries))

    for (const n of names) if (!CONTROLLERS[n]) throw new Error(`unknown controller ${n} (have ${Object.keys(CONTROLLERS).join(', ')})`)
    const controllerNames = [...names]
    let policyText = null
    let policyInfo = null
    if (opt('policy')) {
        policyText = readFileSync(opt('policy'), 'utf8')
        const loaded = loadPolicy(policyText)
        if (!loaded.ok) throw new Error(loaded.error)
        policyInfo = { path: opt('policy'), algorithm: loaded.policy.algorithm, meta: loaded.policy.meta }
        controllerNames.push(`policy:${loaded.policy.algorithm}`)
    }

    const t0 = Date.now()
    const jobs = controllerNames.flatMap((name) => entries.map((_, k) => [name, k]))
    let rows
    if (workers === 1) {
        const players = loadPlayers()
        const policy = policyText ? loadPolicy(policyText).policy : null
        rows = jobs.map(([name, k]) => runEpisode({ players, entry: entries[k], controller: controllerFor(name, policy), ...shieldOpts }))
    } else {
        // Interleaved chunks balance slow and fast controllers across workers.
        const chunks = Array.from({ length: Math.min(workers, jobs.length) }, (_, w) => jobs.map((j, i) => [j, i]).filter(([, i]) => i % workers === w))
        const results = await Promise.all(chunks.map((chunk) => new Promise((resolve, reject) => {
            // A larger young generation cuts GC in the allocation-heavy simulator
            // (runtime setting only; results are identical).
            const worker = new Worker(fileURLToPath(import.meta.url), { workerData: { split, jobs: chunk.map(([j]) => j), policyText, shieldOpts }, resourceLimits: { maxYoungGenerationSizeMb: 96 } })
            worker.once('message', resolve)
            worker.once('error', reject)
        })))
        rows = new Array(jobs.length)
        chunks.forEach((chunk, w) => chunk.forEach(([, i], k) => { rows[i] = results[w][k] }))
    }
    const episodes = Object.fromEntries(controllerNames.map((name, c) => [name, rows.slice(c * entries.length, (c + 1) * entries.length)]))

    let locked = null
    if (opt('compare')) {
        locked = JSON.parse(readFileSync(opt('compare'), 'utf8'))
        if (locked.split !== split) throw new Error(`--compare file is for ${locked.split}, not ${split}`)
        for (const [name, list] of Object.entries(locked.episodes)) {
            if (episodes[name]) continue
            const bySeed = new Map(list.map((r) => [r.seed, r]))
            const paired = entries.map((e) => bySeed.get(e.seed))
            if (paired.some((r) => !r)) throw new Error(`--compare: ${name} lacks some of these seeds`)
            episodes[name] = paired
        }
    }

    const report = buildReport(episodes, episodes[reference] ? reference : controllerNames[0])
    const secs = (Date.now() - t0) / 1000
    const invariantViolations = rows.reduce((s, r) => s + r.invariantViolations, 0)
    if (!quiet) {
        const fmt = (s) => `${s.mean.toFixed(2)} [${s.ci95[0].toFixed(2)}, ${s.ci95[1].toFixed(2)}]`
        console.log(`${split}: ${entries.length} auctions × ${controllerNames.length} controllers played in ${secs.toFixed(0)}s (${workers} workers); invariant violations: ${invariantViolations}`)
        for (const [name, r] of Object.entries(report)) {
            console.log(`${name.padEnd(18)} XI ${fmt(r.xi)}  legal ${(100 * r.legalXI.mean).toFixed(0)}%  strong ${(100 * r.strongXI.mean).toFixed(0)}%  rank ${r.rank.mean.toFixed(2)}  purse left ${r.purseLeftShare.mean.toFixed(2)}  XI/₹1000L ${r.xiGainPer1000.mean.toFixed(2)}  price/fair ${r.priceToFair.mean?.toFixed(2)}` +
                (r.paired ? `  ΔXI vs ${r.paired.reference} ${r.paired.xiDiffMean.toFixed(2)} [${r.paired.ci95[0].toFixed(2)}, ${r.paired.ci95[1].toFixed(2)}]` : ''))
        }
    }

    const meta = {
        split, n: entries.length, manifestHash, obsSpecHash: OBS_SPEC_HASH, actSpecHash: ACT_SPEC_HASH,
        controllers: controllerNames, policy: policyInfo, lockedBaselines: locked ? { file: opt('compare'), rowsHash: locked.rowsHash } : null,
        workers, seconds: secs, invariantViolations, shield: shieldOpts
    }
    if (opt('out')) writeFileSync(opt('out'), JSON.stringify({ ...meta, report }, null, 2))
    if (opt('episodes-out')) writeFileSync(opt('episodes-out'), JSON.stringify({ ...meta, episodes: Object.fromEntries(controllerNames.map((n) => [n, episodes[n]])) }))

    if (flag('lock')) {
        if (policyText || limit !== Infinity || shieldOpts.shield !== 'v1' || shieldOpts.shieldDiagnostics || names.length !== Object.keys(BASELINES).length || names.some((n) => !BASELINES[n])) throw new Error('--lock needs every baseline, the full manifest and no policy')
        if (invariantViolations) throw new Error(`refusing to lock: ${invariantViolations} invariant violations`)
        const dir = fileURLToPath(new URL('../data/rl-baselines/', import.meta.url))
        const base = `${dir}${split}`
        if (existsSync(`${base}.report.json`) || existsSync(`${base}.episodes.json`)) throw new Error(`${base}.* already locked — baselines are permanent`)
        mkdirSync(dir, { recursive: true })
        const baselineEpisodes = Object.fromEntries(names.map((n) => [n, episodes[n]]))
        const rowsHash = fnv1a64(canonicalJson(baselineEpisodes))
        let commit = null
        let dirty = null
        try {
            commit = execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim()
            dirty = execSync('git status --porcelain', { encoding: 'utf8' }).trim().length > 0
        } catch { /* not a checkout */ }
        // Hash of every shared source file, so the exact code is identifiable
        // even from an uncommitted working tree.
        const srcDir = fileURLToPath(new URL('../src/', import.meta.url))
        const files = readdirSync(srcDir, { recursive: true }).filter((f) => f.endsWith('.js')).map((f) => f.replaceAll('\\', '/')).sort()
        const sourceHash = fnv1a64(files.map((f) => `${f}\n${readFileSync(srcDir + f, 'utf8').replaceAll('\r\n', '\n')}`).join('\n'))
        const lock = {
            format: 'rl-baselines-v1', lockedAt: new Date().toISOString(), commit, workingTreeDirty: dirty, sourceHash, ...meta, rowsHash,
            note: 'Permanent baseline reference for Phase 2C RL experiments. Do not regenerate or edit.'
        }
        writeFileSync(`${base}.episodes.json`, JSON.stringify({ ...lock, episodes: baselineEpisodes }))
        writeFileSync(`${base}.report.json`, JSON.stringify({ ...lock, report }, null, 2))
        console.log(`locked ${base}.{report,episodes}.json  rowsHash ${rowsHash}`)
    }
}
