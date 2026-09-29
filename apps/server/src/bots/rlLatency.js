import { PerformanceObserver, constants, performance } from 'node:perf_hooks'

// Ops visibility for the 20 ms RL room guard (createRlSeat in
// packages/shared/src/rl/runtime.js). The guard itself is unchanged; this
// only measures. Every RL decision is timed, and recent garbage-collection
// pauses are kept, so a guard trip in the log says whether a GC pause
// overlapped the decision (a server hiccup) or not (the process didn't get
// the CPU, or the decision itself was slow).

const GC_KEEP_MS = 15 * 60 * 1000
const GC_KINDS = {
    [constants.NODE_PERFORMANCE_GC_MINOR]: 'minor',
    [constants.NODE_PERFORMANCE_GC_MAJOR]: 'major',
    [constants.NODE_PERFORMANCE_GC_INCREMENTAL]: 'incremental',
    [constants.NODE_PERFORMANCE_GC_WEAKCB]: 'weakcb'
}

const gcPauses = [] // { start, end, kind } in performance.now() time
let observer = null

const watchGc = () => {
    if (observer) return
    observer = new PerformanceObserver((list) => {
        for (const e of list.getEntries()) {
            gcPauses.push({ start: e.startTime, end: e.startTime + e.duration, kind: GC_KINDS[e.detail?.kind] ?? 'gc' })
        }
        const cutoff = performance.now() - GC_KEEP_MS
        while (gcPauses.length && gcPauses[0].end < cutoff) gcPauses.shift()
    })
    observer.observe({ entryTypes: ['gc'] })
}

const gcDuring = (start, end) => gcPauses.filter((g) => g.end > start && g.start < end)

const describeGc = (pauses) => pauses.length
    ? pauses.map((g) => `${g.kind} GC ${(g.end - g.start).toFixed(1)} ms`).join(', ')
    : 'no GC pause — the process was likely waiting for the CPU'

const latencySummary = (ms) => {
    if (!ms.length) return 'no model decisions'
    const s = [...ms].sort((a, b) => a - b)
    const at = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))].toFixed(2)
    return `latency p50 ${at(0.5)} p99 ${at(0.99)} max ${s[s.length - 1].toFixed(2)} ms`
}

// Slowest GC pause since `since` (performance.now() time).
const maxGcSince = (since) => gcPauses.reduce((m, g) => (g.start >= since ? Math.max(m, g.end - g.start) : m), 0)

export { watchGc, gcDuring, describeGc, latencySummary, maxGcSince }
