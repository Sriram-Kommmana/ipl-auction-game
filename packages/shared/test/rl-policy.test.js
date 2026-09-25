// Phase 2B — rl-policy-v2 validation / inference (L, JS side) and the
// production fallback runtime (O). Networks here are RANDOMLY INITIALISED:
// forward passes only, no training.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { agentCap, createRng } from '../src/sim.js'
import { planBid } from '../src/planning.js'
import {
    ACTION_COUNT, ALGORITHMS, FAILURE_LIMIT, PASS,
    actionScores, createRlSeat, forwardRaw, loadPolicy, rlActionMask, selectAction, validatePolicy
} from '../src/rl/index.js'
import { randomPolicy, realStates } from './rlHelpers.js'

const states = realStates([7201, 7202], 4)
const clone = (x) => JSON.parse(JSON.stringify(x))

// ── L. format, validation, inference ──────────────────────────────────────
test('L1. a random-init policy of each of the five algorithms validates and runs', () => {
    for (const algo of Object.keys(ALGORITHMS)) {
        const p = validatePolicy(randomPolicy(algo))
        const scores = actionScores(p, states[0].obs)
        assert.equal(scores.length, ACTION_COUNT)
        assert.ok(scores.every(Number.isFinite))
    }
})

test('L2. validation rejects spec/hash/shape/value mismatches', () => {
    const good = randomPolicy('ppo')
    const bad = {
        'wrong format': (p) => { p.format = 'mlp-v1' },
        'unknown algorithm': (p) => { p.algorithm = 'sac' },
        'obs hash': (p) => { p.obsSpec.hash = '0000000000000000' },
        'obs version': (p) => { p.obsSpec.version = 'obs-v1' },
        'act hash': (p) => { p.actSpec.hash = 'ffffffffffffffff' },
        'act count': (p) => { p.actSpec.count = 16 },
        'input width': (p) => { p.layers[0].weight[3].pop() },
        'bias size': (p) => { p.layers[1].bias.push(0) },
        'output width': (p) => { p.layers[2].weight.pop(); p.layers[2].bias.pop() },
        'NaN weight': (p) => { p.layers[1].weight[0][0] = NaN },
        'string weight': (p) => { p.layers[0].weight[0][0] = '0.1' },
        'layer count': (p) => { p.layers.pop() },
        'head for algorithm': (p) => { p.architecture.head = 'q' },
        'selection for algorithm': (p) => { p.selection = { mode: 'argmax' } },
        'temperature': (p) => { p.selection = { mode: 'sample', temperature: 1 } },
        'activation': (p) => { p.architecture.activation = 'gelu' }
    }
    for (const [name, mutate] of Object.entries(bad)) {
        const p = clone(good)
        mutate(p)
        const r = loadPolicy(p)
        assert.equal(r.ok, false, name)
        assert.match(r.error, /rl-policy-v2/)
    }
    assert.equal(loadPolicy('{not json').ok, false)
    assert.equal(loadPolicy(JSON.stringify(good)).ok, true, 'accepts a JSON string')
})

test('L3. heads: quantile head scores = mean over 32 quantiles; forward matches a hand computation', () => {
    const q = validatePolicy(randomPolicy('qrdqn', 5))
    const raw = forwardRaw(q, states[1].obs)
    assert.equal(raw.length, ACTION_COUNT * 32)
    const scores = actionScores(q, states[1].obs)
    for (let a = 0; a < ACTION_COUNT; a++) {
        const m = raw.slice(a * 32, a * 32 + 32).reduce((s, x) => s + x, 0) / 32
        assert.ok(Math.abs(scores[a] - m) < 1e-12)
    }
    // The ES network (80 → 64 → 64 → 20, tanh) computed by hand.
    const tiny = randomPolicy('es', 9)
    const x = states[2].obs
    const h1 = tiny.layers[0].weight.map((row, r) => Math.tanh(row.reduce((s, w, c) => s + w * x[c], tiny.layers[0].bias[r])))
    const h2 = tiny.layers[1].weight.map((row, r) => Math.tanh(row.reduce((s, w, c) => s + w * h1[c], tiny.layers[1].bias[r])))
    const out = tiny.layers[2].weight.map((row, r) => row.reduce((s, w, c) => s + w * h2[c], tiny.layers[2].bias[r]))
    forwardRaw(validatePolicy(tiny), x).forEach((v, i) => assert.ok(Math.abs(v - out[i]) < 1e-12))
})

test('L4. selection: argmax for DQN/QR-DQN/ES, temperature-0.3 sampling for PPO/A2C — always a legal action', () => {
    const scores = Array.from({ length: ACTION_COUNT }, (_, a) => a) // best = 19
    const mask = scores.map((_, a) => (a === 19 || a === 7 ? 0 : 1)) // best legal = 18
    for (const algo of ['d3qn', 'qrdqn', 'es']) assert.equal(selectAction(randomPolicy(algo), scores, mask), 18)
    const rng = createRng(4)
    const ppo = randomPolicy('ppo')
    const picks = Array.from({ length: 4000 }, () => selectAction(ppo, scores, mask, rng))
    assert.ok(picks.every((a) => mask[a] === 1))
    const top = picks.filter((a) => a === 18).length / picks.length
    const expected = 1 / (1 + Math.exp(-1 / 0.3) + Math.exp(-2 / 0.3) + Math.exp(-3 / 0.3) + Math.exp(-4 / 0.3))
    assert.ok(Math.abs(top - expected) < 0.03, `top-action share ${top} vs softmax(T=0.3) ${expected}`)
})

// ── O. production fallback ────────────────────────────────────────────────
const decisionStates = states.filter((s) => s.mask.mask.some((ok, a) => ok && a !== PASS)).slice(0, 25)

test('O1. no model / invalid / mismatched model → the frozen rule persona decides, for the whole room', () => {
    for (const policy of [null, { format: 'junk' }, { ...randomPolicy('ppo'), obsSpec: { version: 'obs-v2', hash: 'x', size: 80 } }]) {
        const seat = createRlSeat({ policy, fallbackPersona: 'moneyball' })
        assert.equal(seat.state.disabled, true)
        for (const s of decisionStates.slice(0, 5)) {
            const out = seat.decide(s.ctx, s.extras, createRng(11))
            assert.equal(out.source, 'fallback')
            assert.equal(out.cap, agentCap({ kind: 'rule', persona: 'moneyball' }, s.ctx, createRng(11)), 'exactly the frozen rule bot')
        }
    }
})

test('O2. a healthy policy decides; its cap is always the masked, clamped act-v2 cap (≤ maxSafeBid)', () => {
    for (const algo of Object.keys(ALGORITHMS)) {
        const seat = createRlSeat({ policy: randomPolicy(algo, 21), fallbackPersona: 'opportunist' })
        for (const s of decisionStates) {
            const out = seat.decide(s.ctx, s.extras, createRng(3))
            assert.equal(out.source, 'rl', `${algo}: ${out.reason}`)
            const m = rlActionMask(s.ctx)
            assert.equal(m.mask[out.action], 1)
            assert.equal(out.cap, m.caps[out.action])
            assert.ok(out.cap <= planBid(s.ctx).budget.maxSafeBid)
        }
        assert.equal(seat.state.failures, 0)
    }
})

test('O3. NaN/Infinity output → fallback for that lot; after 3 failures, fallback for the rest of the room', () => {
    const p = randomPolicy('d3qn', 2)
    p.layers[2].weight[0] = p.layers[2].weight[0].map(() => 1e308)
    p.layers[2].bias[0] = 1e308 // finite weights, but the output overflows to Infinity
    const seat = createRlSeat({ policy: p, fallbackPersona: 'balancedBuilder' })
    assert.equal(seat.state.disabled, false)
    for (let i = 0; i < FAILURE_LIMIT; i++) {
        const out = seat.decide(decisionStates[i].ctx, decisionStates[i].extras, createRng(1))
        assert.equal(out.source, 'fallback')
        assert.match(out.reason, /non-finite/)
    }
    assert.equal(seat.state.disabled, true)
    assert.equal(seat.decide(decisionStates[5].ctx, decisionStates[5].extras, createRng(1)).source, 'fallback')
})

test('O4. masked or invalid action from the model → fallback for the lot', () => {
    const seat = createRlSeat({ policy: randomPolicy('es', 4), fallbackPersona: 'starChaser', select: (_p, _s, mask) => mask.findIndex((ok) => !ok) })
    const s = decisionStates.find((x) => x.mask.mask.some((ok) => !ok))
    const out = seat.decide(s.ctx, s.extras, createRng(2))
    assert.equal(out.source, 'fallback')
    assert.match(out.reason, /masked/)
    assert.equal(seat.state.failures, 1)
    const outOfRange = createRlSeat({ policy: randomPolicy('es', 4), fallbackPersona: 'starChaser', select: () => 42 })
    assert.match(outOfRange.decide(s.ctx, s.extras, createRng(2)).reason, /out of range/)
})

test('O5. an inference slower than 20 ms → fallback for the rest of the room', () => {
    let t = 0
    const seat = createRlSeat({ policy: randomPolicy('a2c', 6), fallbackPersona: 'moneyball', now: () => (t += 25) })
    const out = seat.decide(decisionStates[0].ctx, decisionStates[0].extras, createRng(1))
    assert.equal(out.source, 'fallback')
    assert.match(out.reason, /ms/)
    assert.equal(seat.state.disabled, true)
})

test('O6. real inference latency is far inside the 20 ms budget (observation + forward + mask)', () => {
    const seat = createRlSeat({ policy: randomPolicy('qrdqn', 8), fallbackPersona: 'moneyball' })
    const t0 = performance.now()
    for (const s of decisionStates) seat.decide(s.ctx, s.extras, createRng(1))
    const perDecision = (performance.now() - t0) / decisionStates.length
    assert.ok(perDecision < 5, `${perDecision.toFixed(2)} ms per decision`)
    assert.equal(seat.state.disabled, false)
})
