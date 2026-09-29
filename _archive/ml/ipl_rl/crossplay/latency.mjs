// Phase 2E.0 §15 — real-clock performance of the production seat runtime in
// mixed Stage-B rooms. Every room is S4: the learner plus the other four
// algorithms, each opponent through createRlSeat with the REAL clock and the
// unmodified 20 ms guard, so a slow decision behaves exactly as in production
// (the seat falls back for the rest of the room). Measured per decision: the
// whole runtime decide() call (plan, mask, observation, inference, checks) —
// the span the guard protects. Guard trips are counted, not hidden; results
// of these rooms are NOT used for any cross-play metric.
//
//   node ml/ipl_rl/crossplay/latency.mjs --entries 30 --workers 1  --out single.json
//   node ml/ipl_rl/crossplay/latency.mjs --entries 30 --workers 14 --out concurrent.json
import { writeFileSync } from 'node:fs'
import { cpus } from 'node:os'
import { fileURLToPath } from 'node:url'
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads'
import { ALGOS, exportKey, loadExports, loadPlayers, loadValidation } from './common.mjs'
import { playEpisode } from './harness.mjs'

if (!isMainThread) {
    const { ks, w } = workerData
    const ex = loadExports()
    const players = loadPlayers()
    const entries = loadValidation()
    const samples = []
    const t0 = performance.now()
    let episodes = 0
    for (const k of ks) {
        const algo = ALGOS[(k + w) % 5]
        const seed = 1 + (k % 3)
        playEpisode({ players, baseEntry: entries[k], learnerKey: exportKey(algo, seed), cond: { kind: 'S4', learnerAlgo: algo, seed }, exportsByKey: ex, clock: 'real', latency: samples })
        episodes++
    }
    parentPort.postMessage({ samples, seconds: (performance.now() - t0) / 1000, episodes })
} else {
    const args = process.argv.slice(2)
    const opt = (n, d) => (args.includes(`--${n}`) ? args[args.indexOf(`--${n}`) + 1] : d)
    const E = Number(opt('entries', 30))
    const W = Number(opt('workers', 1))
    const out = opt('out')
    const offset = Number(opt('offset', 0))
    const parts = await Promise.all(Array.from({ length: W }, (_, w) => new Promise((res, rej) => {
        const ks = Array.from({ length: E }, (_, i) => (offset + i + w * E) % 500)
        const wk = new Worker(fileURLToPath(import.meta.url), { workerData: { ks, w } })
        wk.once('message', res)
        wk.once('error', rej)
    })))
    const q = (xs, p) => xs[Math.min(xs.length - 1, Math.floor(p * xs.length))]
    const stats = (xs) => {
        const s = [...xs].sort((a, b) => a - b)
        return { n: s.length, p50: q(s, 0.5), p95: q(s, 0.95), p99: q(s, 0.99), max: s.at(-1), mean: s.reduce((a, b) => a + b, 0) / s.length, over20ms: s.filter((x) => x > 20).length, over5ms: s.filter((x) => x > 5).length }
    }
    const all = parts.flatMap((p) => p.samples)
    const rl = all.filter((s) => s.source === 'rl')
    const byAlgo = Object.fromEntries(ALGOS.map((a) => [a, stats(rl.filter((s) => s.algo === a).map((s) => s.ms))]))
    // A trip = the first fallback of a seat in a room; later decisions of that
    // seat are the rule persona (the production behaviour after a trip).
    const trips = all.filter((s) => s.source !== 'rl' && !s.disabledBefore && /inference took/.test(s.reason ?? ''))
    const afterTrip = all.filter((s) => s.source !== 'rl' && s.disabledBefore)
    const otherFallbacks = all.filter((s) => s.source !== 'rl' && !s.disabledBefore && !/inference took/.test(s.reason ?? ''))
    const wall = Math.max(...parts.map((p) => p.seconds))
    const res = {
        clock: 'real (production default), 20 ms guard unmodified', workers: W, cpus: cpus().length, entriesPerWorker: E,
        episodes: parts.reduce((s, p) => s + p.episodes, 0), wallSeconds: wall,
        rlDecisions: rl.length, rlDecisionsPerSecond: rl.length / wall, episodesPerSecond: parts.reduce((s, p) => s + p.episodes, 0) / wall,
        all: stats(rl.map((s) => s.ms)), byAlgo,
        guardTrips: trips.map((s) => ({ algo: s.algo, key: s.key, seed: s.seed, seat: s.seat, reason: s.reason })),
        ruleDecisionsAfterTrips: afterTrip.length,
        otherFallbacks: otherFallbacks.length,
        otherFallbackReasons: [...new Set(otherFallbacks.map((s) => s.reason))].slice(0, 10)
    }
    writeFileSync(out, JSON.stringify(res, null, 2))
    console.log(JSON.stringify({ workers: W, episodes: res.episodes, rlDecisions: res.rlDecisions, perSec: res.rlDecisionsPerSecond.toFixed(0), all: res.all, trips: res.guardTrips.length, otherFallbackReasons: res.otherFallbackReasons }))
}
