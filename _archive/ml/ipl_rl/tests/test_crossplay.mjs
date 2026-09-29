// Phase 2E.0 cross-play harness tests (evaluation infrastructure only).
//   node --test ml/ipl_rl/tests/test_crossplay.mjs
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { ALGOS, RL, compose, c1Seat, exportKey, layerDigest, loadExports, loadPlayers, loadValidation, s4Permutation } from '../crossplay/common.mjs'
import { SafetyStop, defectScreen, playEpisode } from '../crossplay/harness.mjs'
import { buildJobs } from '../crossplay/run.mjs'

const EX = loadExports()
const PLAYERS = loadPlayers()
const V = loadValidation()
const stripStage = ({ stage, stageB, seats, ...rest }) => rest
const clone = (x) => JSON.parse(JSON.stringify(x))

test('15 frozen exports: right algorithm, spec hashes, production loader accepts', () => {
    assert.equal(Object.keys(EX).length, 15)
    for (const e of Object.values(EX)) {
        assert.equal(e.policy.algorithm, e.algo)
        assert.equal(e.policy.obsSpec.hash, '629b25783f833af7')
        assert.equal(e.policy.actSpec.hash, '5f72f510c48b1f46')
        assert.ok(RL.loadPolicy(e.policy).ok)
    }
    assert.deepEqual(EX['ppo:s1'].policy.selection, { mode: 'sample', temperature: 0.3 })
    assert.deepEqual(EX['a2c:s1'].policy.selection, { mode: 'sample', temperature: 0.3 })
    for (const a of ['d3qn', 'qrdqn', 'es']) assert.deepEqual(EX[`${a}:s2`].policy.selection, { mode: 'argmax' })
})

test('Stage-A control composition is the manifest entry, unchanged', () => {
    for (const e of V.slice(0, 50)) assert.deepEqual(compose(e, { kind: 'A' }).entry, e)
})

test('C1: exactly one rlFallback seat becomes the opponent; nothing else changes; same seat for every opponent', () => {
    const pos = [0, 0, 0, 0]
    for (const e of V) {
        const fb = e.seats.flatMap((s, i) => (s.type === 'rlFallback' ? [i] : []))
        const seat = c1Seat(e)
        pos[fb.indexOf(seat)]++
        for (const opp of ['ppo:s1', 'es:s3']) {
            const { entry, snapshotKeys } = compose(e, { kind: 'C1', opp })
            assert.deepEqual(snapshotKeys, [opp])
            assert.deepEqual(stripStage(entry), stripStage(e))
            entry.seats.forEach((s, i) => {
                if (i === seat) assert.deepEqual(s, { ...e.seats[i], type: 'rlSnapshot', snapshot: 0 })
                else assert.deepEqual(s, e.seats[i])
            })
        }
    }
    for (const n of pos) assert.ok(n > 90 && n < 160, `C1 seat position balance ${pos}`)
})

test('C4: all four other RL seats hold the opponent; rule bots stay 4 of 9 (≥ 40%)', () => {
    for (const e of V.slice(0, 100)) {
        const { entry, snapshotKeys } = compose(e, { kind: 'C4', opp: 'd3qn:s2' })
        assert.deepEqual(snapshotKeys, ['d3qn:s2', 'd3qn:s2', 'd3qn:s2', 'd3qn:s2'])
        assert.equal(entry.seats.filter((s) => s.type === 'rlSnapshot').length, 4)
        assert.equal(entry.seats.filter((s) => s.type === 'rlFallback').length, 0)
        assert.equal(entry.seats.filter((s) => s.type === 'rule').length, 4)
        assert.deepEqual(stripStage(entry), stripStage(e))
    }
})

test('S4: the four other algorithms, one each, balanced over seats by a seed-derived permutation', () => {
    for (const learnerAlgo of ALGOS) {
        const count = {}
        for (const e of V) {
            const { entry, snapshotKeys } = compose(e, { kind: 'S4', learnerAlgo, seed: 2 })
            const algos = snapshotKeys.map((k) => k.split(':')[0])
            assert.deepEqual([...algos].sort(), ALGOS.filter((a) => a !== learnerAlgo).sort())
            assert.ok(snapshotKeys.every((k) => k.endsWith(':s2')))
            assert.deepEqual(algos, s4Permutation(e, learnerAlgo))
            entry.seats.forEach((s) => { if (s.type === 'rlSnapshot') count[`${algos[s.snapshot]}@${s.rlSeat}`] = (count[`${algos[s.snapshot]}@${s.rlSeat}`] ?? 0) + 1 })
        }
        // 500 rooms × 4 seats over 4 algorithms × 5 RL-seat names (the learner takes one name).
        const values = Object.values(count)
        assert.equal(values.length, 20)
        for (const n of values) assert.ok(n > 60 && n < 145, `${learnerAlgo}: ${JSON.stringify(count)}`)
    }
})

test('Stage-A control reproduces the frozen evaluator exactly (runEpisode + policyController)', () => {
    for (const key of ['ppo:s1', 'd3qn:s3', 'es:s2']) {
        for (const k of [0, 7, 311]) {
            const ref = RL.runEpisode({ players: PLAYERS, entry: V[k], controller: RL.policyController(EX[key].policy) })
            const { violations, ...want } = ref
            const got = playEpisode({ players: PLAYERS, baseEntry: V[k], learnerKey: key, cond: { kind: 'A' }, exportsByKey: EX })
            assert.deepEqual(got.summary, want)
        }
    }
})

test('Stage-A control matches the stored Stage-A validation episodes', () => {
    const stored = JSON.parse(readFileSync(new URL('../../runs/qr-dqn-2d3-s2/checkpoints/update_0325/validation/episodes.json', import.meta.url), 'utf8')).episodes['policy:qrdqn']
    for (const k of [3, 250]) {
        const { violations, ...want } = stored[k]
        assert.deepEqual(playEpisode({ players: PLAYERS, baseEntry: V[k], learnerKey: 'qrdqn:s2', cond: { kind: 'A' }, exportsByKey: EX }).summary, want)
    }
})

test('deterministic: the same cross-play episode twice gives identical records', () => {
    for (const cond of [{ kind: 'C1', opp: 'ppo:s3' }, { kind: 'C4', opp: 'a2c:s2' }, { kind: 'S4', learnerAlgo: 'qrdqn', seed: 1 }]) {
        const learner = cond.kind === 'S4' ? 'qrdqn:s1' : 'es:s1'
        const a = playEpisode({ players: PLAYERS, baseEntry: V[42], learnerKey: learner, cond, exportsByKey: EX })
        const b = playEpisode({ players: PLAYERS, baseEntry: V[42], learnerKey: learner, cond, exportsByKey: EX })
        assert.deepEqual(a, b)
        assert.ok(a.opp.every((o) => o.decisions > 0 && o.fallbacks.length === 0))
    }
})

test('opponents really are the assigned models: every decision comes from the RL policy', () => {
    const r = playEpisode({ players: PLAYERS, baseEntry: V[5], learnerKey: 'ppo:s2', cond: { kind: 'S4', learnerAlgo: 'ppo', seed: 3 }, exportsByKey: EX })
    assert.equal(r.opp.length, 4)
    for (const o of r.opp) {
        assert.ok(o.key.endsWith(':s3'))
        assert.equal(o.decisions, o.actions.reduce((s, x) => s + x, 0))
        assert.equal(o.fallbacks.length, 0)
    }
})

test('safety stop: an opponent whose output is non-finite (would silently fall back) stops the run', () => {
    const broken = clone(EX['es:s1'].policy)
    const last = broken.layers.at(-1)
    last.weight = last.weight.map((row) => row.map(() => 1e308))
    last.bias = last.bias.map(() => 1e308)
    const ex = { ...EX, 'es:s1': { ...EX['es:s1'], policy: broken, layers: layerDigest(broken) } }
    assert.throws(() => playEpisode({ players: PLAYERS, baseEntry: V[1], learnerKey: 'ppo:s1', cond: { kind: 'C4', opp: 'es:s1' }, exportsByKey: ex }),
        (err) => err instanceof SafetyStop && /OPPONENT FALLBACK/.test(err.message) && /non-finite/.test(err.message))
})

test('safety stop: a runtime holding different weights than the assigned export (wrong model)', () => {
    const ex = { ...EX, 'd3qn:s1': { ...EX['d3qn:s1'], policy: EX['d3qn:s2'].policy } }
    assert.throws(() => playEpisode({ players: PLAYERS, baseEntry: V[1], learnerKey: 'ppo:s1', cond: { kind: 'C1', opp: 'd3qn:s1' }, exportsByKey: ex }),
        (err) => err instanceof SafetyStop && /WRONG MODEL/.test(err.message))
    assert.throws(() => playEpisode({ players: PLAYERS, baseEntry: V[1], learnerKey: 'ppo:s9', cond: { kind: 'A' }, exportsByKey: EX }),
        (err) => err instanceof SafetyStop && /WRONG MODEL/.test(err.message))
})

test('safety stop: a non-finite learner output', () => {
    const broken = clone(EX['a2c:s1'].policy)
    broken.layers.at(-1).bias = broken.layers.at(-1).bias.map(() => 1e308)
    broken.layers.at(-1).weight = broken.layers.at(-1).weight.map((row) => row.map(() => 1e308))
    const ex = { ...EX, 'a2c:s1': { ...EX['a2c:s1'], policy: broken, layers: layerDigest(broken) } }
    assert.throws(() => playEpisode({ players: PLAYERS, baseEntry: V[2], learnerKey: 'a2c:s1', cond: { kind: 'A' }, exportsByKey: ex }),
        (err) => err instanceof SafetyStop && /LEARNER NON-FINITE/.test(err.message))
})

test('the production 20 ms guard is untouched: a slow decision still disables the seat', () => {
    let t = 0
    const seat = RL.createRlSeat({ policy: EX['es:s1'].policy, fallbackPersona: 'moneyball', now: () => (t += 25) })
    const ep = new RL.RlEpisode({ players: PLAYERS, entry: V[0], tremble: 0 })
    ep.reset()
    const r = seat.decide(ep.pending.ctx, ep.extras(), () => 0.5)
    assert.equal(r.source, 'fallback')
    assert.match(r.reason, /inference took/)
    assert.equal(seat.state.disabled, true)
    assert.equal(RL.INFERENCE_BUDGET_MS, 20)
})

test('job lists: sizes, unique ids, every ordered pair and seed pair', () => {
    assert.equal(buildJobs({ cond: 'A' }).length, 7500)
    const c1 = buildJobs({ cond: 'C1' })
    assert.equal(c1.length, 90000)
    assert.equal(new Set(c1.map((j) => j.id)).size, 90000)
    const pairs = new Set(c1.map((j) => `${j.learner}>${j.cond.opp}`))
    assert.equal(pairs.size, 180)
    for (const a of ALGOS) for (const b of ALGOS) if (a !== b) assert.ok(pairs.has(`${exportKey(a, 1)}>${exportKey(b, 3)}`))
    assert.ok(c1.every((j) => j.learner.split(':')[0] !== j.cond.opp.split(':')[0]))
    assert.equal(buildJobs({ cond: 'C4' }).length, 90000)
    assert.equal(buildJobs({ cond: 'S4' }).length, 7500)
})

test('Option A: an incomplete RL-opponent XI from legal policy behaviour is a screened finding, not a stop (C4 seed 100490)', () => {
    const r = playEpisode({ players: PLAYERS, baseEntry: V[490], learnerKey: 'ppo:s2', cond: { kind: 'C4', opp: 'd3qn:s1' }, exportsByKey: EX })
    assert.equal(r.findings.length, 1)
    const [f] = r.findings
    assert.equal(f.seat, 4); assert.equal(f.type, 'rlSnapshot'); assert.equal(f.key, 'd3qn:s1'); assert.equal(f.emptySlots, 1)
    assert.equal(f.screen.pass, true); assert.equal(f.screen.plainReplayIdentical, true); assert.equal(f.screen.invariantViolations, 0)
    assert.equal(r.summary.invariantViolations, 0)
    assert.ok(r.opp.every((o) => o.fallbacks.length === 0))
    assert.equal(r.incompleteByType.rlSnapshot, 1)
})

test('Option A: the defect screen fails when the harness outcome differs from the unmodified frozen path', () => {
    const { entry, snapshotKeys } = compose(V[490], { kind: 'C4', opp: 'd3qn:s1' })
    const good = playEpisode({ players: PLAYERS, baseEntry: V[490], learnerKey: 'ppo:s2', cond: { kind: 'C4', opp: 'd3qn:s1' }, exportsByKey: EX })
    const teams = good.others.map((t) => ({ emptySlots: t.legalXI ? 0 : 1, purseLeft: null }))
    const tampered = { ...good.summary, xi: good.summary.xi + 0.01 }
    const s = defectScreen({ players: PLAYERS, entry, opponents: snapshotKeys.map((k) => EX[k]), policy: EX['ppo:s2'].policy, summary: tampered, history: [], teams })
    assert.equal(s.pass, false)
})

test('records of rooms without findings carry no findings field (earlier C1 records stay comparable)', () => {
    const r = playEpisode({ players: PLAYERS, baseEntry: V[3], learnerKey: 'es:s1', cond: { kind: 'C1', opp: 'qrdqn:s2' }, exportsByKey: EX })
    assert.equal('findings' in r, false)
})
