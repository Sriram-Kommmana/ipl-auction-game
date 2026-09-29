// Phase 2D.3 §29 — production-inference contract of a REAL exported QR-DQN
// policy (rl-policy-v2): healthy decisions are legal act-v3 caps ≤ maxSafeBid,
// and every fault path falls back to the frozen rule persona.
//   node ml/runs/_2d3/qrdqn_fallback.mjs <policy.json> [...]
import { readFileSync } from 'node:fs'
import assert from 'node:assert/strict'
const base = new URL('../../../packages/shared/', import.meta.url)
const { agentCap, createRng } = await import(new URL('src/sim.js', base))
const { planBid } = await import(new URL('src/planning.js', base))
const RL = await import(new URL('src/rl/index.js', base))
const { realStates } = await import(new URL('test/rlHelpers.js', base))

const states = realStates([7301, 7302, 7303], 5).filter((s) => s.mask.mask.some((ok, a) => ok && a !== RL.PASS))
const clone = (x) => JSON.parse(JSON.stringify(x))
const persona = 'moneyball'
const ruleCap = (s, seed) => agentCap({ kind: 'rule', persona }, s.ctx, createRng(seed))
const results = []
const check = (name, fn) => { fn(); results.push(name) }

for (const path of process.argv.slice(2)) {
    const good = JSON.parse(readFileSync(path, 'utf8'))
    assert.equal(good.algorithm, 'qrdqn'); assert.equal(good.meta.algorithmName, 'qr_dqn'); assert.equal(good.meta.quantileFractions.length, 32)
    assert.equal(good.actSpec.hash, RL.ACT_SPEC_HASH)
    assert.equal(good.obsSpec.hash, RL.OBS_SPEC_HASH)
    assert.deepEqual(good.selection, { mode: 'argmax' })
    assert.deepEqual(good.architecture, { input: 80, hidden: [128, 128], activation: 'tanh', head: 'quantiles', actions: 20, quantiles: 32 })

    check('healthy: legal act-v3 cap ≤ maxSafeBid, completionGuard off', () => {
        const seat = RL.createRlSeat({ policy: good, fallbackPersona: persona, now: () => 0 })
        for (const s of states) {
            const out = seat.decide(s.ctx, s.extras, createRng(3))
            assert.equal(out.source, 'rl', out.reason)
            const m = RL.rlActionMask(s.ctx)
            assert.equal(m.mask[out.action], 1)
            assert.equal(out.cap, m.caps[out.action])
            assert.ok(out.cap <= planBid(s.ctx).budget.maxSafeBid)
        }
        assert.equal(seat.state.failures, 0)
        assert.equal(seat.state.guardInterventions, 0)
    })
    const wholeRoomFallback = (policy, why) => {
        const seat = RL.createRlSeat({ policy, fallbackPersona: persona })
        assert.equal(seat.state.disabled, true, why)
        for (const s of states.slice(0, 6)) {
            const out = seat.decide(s.ctx, s.extras, createRng(11))
            assert.equal(out.source, 'fallback', why)
            assert.equal(out.cap, ruleCap(s, 11), `${why}: exactly the frozen rule bot`)
        }
        return seat.state.disabledReason
    }
    check('missing model → fallback', () => wholeRoomFallback(null, 'missing'))
    check('invalid model → fallback', () => {
        wholeRoomFallback({ format: 'junk' }, 'junk')
        const p = clone(good); p.layers.pop(); wholeRoomFallback(p, 'layer count')
        wholeRoomFallback('{"format": "rl-policy-v2", ', 'truncated JSON')
        wholeRoomFallback('not json at all', 'invalid JSON')
    })
    check('malformed architecture → fallback', () => {
        const h = clone(good); h.architecture.hidden = [128, 128, 256]; wholeRoomFallback(h, 'hidden sizes vs layers')
        const w = clone(good); w.layers[2].weight.pop(); w.layers[2].bias.pop(); wholeRoomFallback(w, 'layer width')
        const c = clone(good); c.layers[1].weight[3] = c.layers[1].weight[3].slice(1); wholeRoomFallback(c, 'row width')
        const hd = clone(good); hd.architecture.head = 'logits'; wholeRoomFallback(hd, 'head for algorithm')
        const s = clone(good); s.selection = { mode: 'sample', temperature: 0.3 }; wholeRoomFallback(s, 'selection for algorithm')
        const a = clone(good); a.architecture.activation = 'gelu'; wholeRoomFallback(a, 'activation')
        const o = clone(good); o.architecture.actions = 16; wholeRoomFallback(o, 'architecture action count')
        const q16 = clone(good); q16.architecture.quantiles = 16; wholeRoomFallback(q16, 'quantile count 16 (≠ 32)')
        const q0 = clone(good); delete q0.architecture.quantiles; wholeRoomFallback(q0, 'quantile count missing')
        const o639 = clone(good); o639.layers[2].weight.pop(); o639.layers[2].bias.pop(); wholeRoomFallback(o639, 'output 639 (≠ 640)')
        const o660 = clone(good); o660.layers[2].weight.push(o660.layers[2].weight[0]); o660.layers[2].bias.push(0); wholeRoomFallback(o660, 'output 641 (≠ 640)')
    })
    check('healthy decisions are the masked argmax of the mean of the 32 exported quantiles', () => {
        const seat = RL.createRlSeat({ policy: good, fallbackPersona: persona, now: () => 0 })
        const pol = RL.loadPolicy(good).policy
        for (const s of states) {
            const out = seat.decide(s.ctx, s.extras, createRng(9))
            const raw = RL.forwardRaw(pol, s.obs)
            assert.equal(raw.length, 640)
            const q = RL.actionScores(pol, s.obs)
            for (let x = 0; x < 20; x++) assert.ok(Math.abs(q[x] - raw.slice(x * 32, x * 32 + 32).reduce((a, b) => a + b, 0) / 32) < 1e-12, 'mean of the 32 quantiles')
            const m = RL.rlActionMask(s.ctx).mask
            let best = -1
            for (let x = 0; x < q.length; x++) if (m[x] && (best < 0 || q[x] > q[best])) best = x
            assert.equal(out.action, best)
        }
    })
    check('spec mismatch → fallback', () => {
        const a = clone(good); a.actSpec.hash = 'ffffffffffffffff'; wholeRoomFallback(a, 'act hash')
        const v2 = clone(good); v2.actSpec = { version: 'act-v2', hash: 'a0f075207054eaf7', count: 20 }; wholeRoomFallback(v2, 'act-v2 stamp (the real act-v2 hash)')
        const o = clone(good); o.obsSpec.hash = '0000000000000000'; wholeRoomFallback(o, 'obs hash')
        const c = clone(good); c.actSpec.count = 16; wholeRoomFallback(c, 'action count')
    })
    check('NaN / Infinity → fallback', () => {
        const n = clone(good); n.layers[1].weight[0][0] = NaN; wholeRoomFallback(n, 'NaN weight')
        const inf = clone(good)
        const last = inf.layers.length - 1 // the linear output layer (no tanh to squash the overflow)
        inf.layers[last].weight[0] = inf.layers[last].weight[0].map(() => 1e308); inf.layers[last].bias[0] = 1e308
        const seat = RL.createRlSeat({ policy: inf, fallbackPersona: persona })
        assert.equal(seat.state.disabled, false)
        for (let i = 0; i < RL.FAILURE_LIMIT; i++) {
            const out = seat.decide(states[i].ctx, states[i].extras, createRng(1))
            assert.equal(out.source, 'fallback'); assert.match(out.reason, /non-finite/)
            assert.equal(out.cap, ruleCap(states[i], 1))
        }
        assert.equal(seat.state.disabled, true)
        // NaN quantile mean: finite weights only. The last hidden layer is pinned to
        // tanh(100) = 1, so quantile 0 of action 0 overflows to +Infinity and quantile 1
        // to −Infinity — their mean over the 32 quantiles is NaN.
        const nan = clone(good)
        const L = nan.layers.length - 1
        nan.layers[L - 1].weight = nan.layers[L - 1].weight.map((row) => row.map(() => 0)); nan.layers[L - 1].bias = nan.layers[L - 1].bias.map(() => 100)
        nan.layers[L].weight[0] = nan.layers[L].weight[0].map(() => 1e307); nan.layers[L].bias[0] = 0
        nan.layers[L].weight[1] = nan.layers[L].weight[1].map(() => -1e307); nan.layers[L].bias[1] = 0
        const nanPol = RL.loadPolicy(nan)
        assert.equal(nanPol.ok, true)
        const rawNaN = RL.forwardRaw(nanPol.policy, states[0].obs)
        assert.equal(rawNaN[0], Infinity); assert.equal(rawNaN[1], -Infinity)
        assert.ok(Number.isNaN(RL.actionScores(nanPol.policy, states[0].obs)[0]), 'action 0 mean is NaN')
        const ns = RL.createRlSeat({ policy: nan, fallbackPersona: persona })
        const nout = ns.decide(states[0].ctx, states[0].extras, createRng(1))
        assert.equal(nout.source, 'fallback'); assert.match(nout.reason, /non-finite/)
        assert.equal(nout.cap, ruleCap(states[0], 1))
    })
    check('masked / out-of-range action → fallback', () => {
        const s = states.find((x) => x.mask.mask.some((ok) => !ok))
        const seat = RL.createRlSeat({ policy: good, fallbackPersona: persona, select: (_p, _s, mask) => mask.findIndex((ok) => !ok) })
        const out = seat.decide(s.ctx, s.extras, createRng(2))
        assert.equal(out.source, 'fallback'); assert.match(out.reason, /masked/)
        const oor = RL.createRlSeat({ policy: good, fallbackPersona: persona, select: () => 42 })
        assert.match(oor.decide(s.ctx, s.extras, createRng(2)).reason, /out of range/)
    })
    check('timeout (> 20 ms) → fallback for the room', () => {
        let t = 0
        const seat = RL.createRlSeat({ policy: good, fallbackPersona: persona, now: () => (t += 25) })
        const out = seat.decide(states[0].ctx, states[0].extras, createRng(1))
        assert.equal(out.source, 'fallback'); assert.match(out.reason, /ms/)
        assert.equal(seat.state.disabled, true)
    })
    check('real latency inside the budget', () => {
        const seat = RL.createRlSeat({ policy: good, fallbackPersona: persona })
        const t0 = performance.now()
        for (const s of states) seat.decide(s.ctx, s.extras, createRng(1))
        const per = (performance.now() - t0) / states.length
        assert.ok(per < 5, `${per.toFixed(2)} ms`)
        assert.equal(seat.state.disabled, false)
        console.log(`  ${path}: ${states.length} healthy decisions, ${per.toFixed(3)} ms/decision`)
    })
}
console.log(`QR-DQN production contract: ${results.length}/${results.length} checks passed over ${process.argv.length - 2} polic${process.argv.length - 2 === 1 ? 'y' : 'ies'} (${states.length} real decision states)`)
