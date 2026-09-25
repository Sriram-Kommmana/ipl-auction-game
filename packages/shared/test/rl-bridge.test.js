// Phase 2B — rl-bridge-v2 protocol (I), driven exactly as Python drives it:
// a child process, one JSON request per line, one JSON reply per line.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'
import { ACTION_NAMES, ACT_SPEC_HASH, OBS_FEATURES, OBS_SPEC_HASH, PASS, RlEpisode, fnv1a64, sampleEpisode } from '../src/rl/index.js'
import { players, randomPolicy } from './rlHelpers.js'

const BRIDGE = fileURLToPath(new URL('../bin/rl-bridge-v2.js', import.meta.url))

const openBridge = () => {
    const proc = spawn(process.execPath, [BRIDGE], { stdio: ['pipe', 'pipe', 'inherit'] })
    const lines = createInterface({ input: proc.stdout })
    const queue = []
    lines.on('line', (line) => queue.shift()(JSON.parse(line)))
    const call = (msg) => new Promise((resolve) => {
        queue.push(resolve)
        proc.stdin.write(`${JSON.stringify(msg)}\n`)
    })
    return { call, close: () => { proc.stdin.end(); proc.kill() } }
}

// A fixed, deterministic policy: cycle through the legal actions.
const chooser = (mask, t) => {
    const legal = mask.flatMap((ok, a) => (ok ? [a] : []))
    return legal[t % legal.length]
}

const playThroughBridge = async (bridge, reset) => {
    const replies = [await bridge.call({ cmd: 'reset', ...reset })]
    let t = 0
    for (;;) {
        const last = replies.at(-1)
        const r = await bridge.call({ cmd: 'step', action: chooser(last.mask, t++) })
        replies.push(r)
        if (r.done) return replies
    }
}

test('I1. info: protocol, obs-v2 / act-v2 names and hashes, γ = 1, λ_rel = 0, splits', async () => {
    const b = openBridge()
    try {
        const info = await b.call({ cmd: 'info' })
        assert.equal(info.ok, true)
        assert.equal(info.protocol, 'rl-bridge-v2')
        assert.deepEqual(info.obsSpec.features, [...OBS_FEATURES])
        assert.equal(info.obsSpec.hash, OBS_SPEC_HASH)
        assert.equal(info.obsSpec.size, 80)
        assert.deepEqual(info.actSpec.actions, [...ACTION_NAMES])
        assert.equal(info.actSpec.hash, ACT_SPEC_HASH)
        assert.equal(info.actSpec.count, 20)
        assert.equal(info.gamma, 1)
        assert.equal(info.lambdaRel, 0)
        assert.deepEqual(info.splits.validation, { start: 100000, count: 500 })
    } finally { b.close() }
})

test('I2. reset/step over a full episode: shapes, shieldActive, done, episode statistics, reward sum', async () => {
    const b = openBridge()
    try {
        const replies = await playThroughBridge(b, { seed: 1_000_030, split: 'train' })
        assert.ok(replies.every((r) => r.ok))
        const first = replies[0]
        assert.equal(first.info.entry.seed, 1_000_030)
        for (const r of replies.slice(0, -1)) {
            assert.equal(r.obs.length, 80)
            assert.equal(r.mask.length, 20)
            assert.equal(r.info.shieldActive, r.mask[PASS] === 0)
        }
        const last = replies.at(-1)
        assert.equal(last.done, true)
        assert.equal(last.obs, null)
        const ep = last.info.episode
        const sum = replies.slice(1).reduce((s, r) => s + r.reward, 0)
        assert.ok(Math.abs(sum - ep.return) < 1e-9)
        assert.ok(Math.abs(sum - (ep.xiTotal / 110 + (ep.emptySlots > 0 ? -2 : 0))) < 1e-9)
        assert.ok(ep.reauctionLots > 0)
        assert.ok(['xi', 'legalXI', 'purseLeft', 'squadSize', 'overseas', 'shieldActivations', 'decisions'].every((k) => k in ep))
    } finally { b.close() }
})

test('I3. determinism: two bridge processes given the same requests reply identically', async () => {
    const a = openBridge()
    const c = openBridge()
    try {
        const ra = await playThroughBridge(a, { seed: 1_000_031 })
        const rc = await playThroughBridge(c, { seed: 1_000_031 })
        assert.equal(fnv1a64(JSON.stringify(ra)), fnv1a64(JSON.stringify(rc)))
    } finally { a.close(); c.close() }
})

test('I4. JS/JS parity: the bridge replays exactly what the in-process environment does', async () => {
    const b = openBridge()
    try {
        const replies = await playThroughBridge(b, { seed: 1_000_032 })
        const ep = new RlEpisode({ players, entry: sampleEpisode(1_000_032), tremble: 0.01 })
        let local = ep.reset()
        assert.deepEqual(replies[0].obs, local.obs)
        for (let t = 0; t < replies.length - 1; t++) {
            local = ep.step(chooser(replies[t].mask, t))
            assert.deepEqual(replies[t + 1].obs, local.obs)
            assert.deepEqual(replies[t + 1].mask, local.mask)
            assert.equal(replies[t + 1].reward, local.reward)
        }
    } finally { b.close() }
})

test('I5. errors are replies, not crashes: step before reset, masked / out-of-range action, wrong split, unknown command', async () => {
    const b = openBridge()
    try {
        assert.match((await b.call({ cmd: 'step', action: 0 })).error, /reset/)
        assert.match((await b.call({ cmd: 'reset', seed: 5, split: 'train' })).error, /not in the train split/)
        assert.match((await b.call({ cmd: 'nope' })).error, /unknown command/)
        const r = await b.call({ cmd: 'reset', seed: 1_000_033, split: 'train' })
        const masked = r.mask.findIndex((ok) => !ok)
        assert.match((await b.call({ cmd: 'step', action: masked })).error, /masked/)
        assert.match((await b.call({ cmd: 'step', action: 20 })).error, /integer/)
        assert.equal((await b.call({ cmd: 'step', action: r.mask.findIndex((ok) => ok) })).ok, true, 'still usable')
    } finally { b.close() }
})

test('I6. league: snapshots load through validation; Stage B episodes seat them; bad snapshots are refused', async () => {
    const b = openBridge()
    try {
        assert.equal((await b.call({ cmd: 'addSnapshot', policy: randomPolicy('es', 41) })).snapshots, 1)
        assert.match((await b.call({ cmd: 'addSnapshot', policy: { ...randomPolicy('ppo', 42), actSpec: { version: 'act-v2', hash: 'x', count: 20 } } })).error, /rejected/)
        assert.equal((await b.call({ cmd: 'configure', snapshotShare: 1 })).ok, true)
        const replies = await playThroughBridge(b, { seed: 1_000_034 })
        assert.equal(replies[0].info.entry.stage, 'B')
        assert.equal(replies[0].info.entry.seats.filter((s) => s.type === 'rlSnapshot').length, 4)
        assert.equal(replies.at(-1).done, true)
        assert.match((await b.call({ cmd: 'configure', snapshotShare: 2 })).error, /\[0, 1\]/)
    } finally { b.close() }
})
