// Derived unchanged from ml/reports/phase2d1/a2c_fallback.mjs (PPO and A2C share the export format: 80-128-128-20 logits, temperature 0.3); only the algorithm id differs.
// Phase 2E.0 §14 — production-inference contract of a REAL exported PPO
// policy (rl-policy-v2): healthy decisions are legal act-v3 caps ≤ maxSafeBid,
// and every fault path falls back to the frozen rule persona.
//   node ml/ipl_rl/crossplay/ppo_fallback.mjs <policy.json> [...]
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
    assert.equal(good.algorithm, 'ppo')
    assert.equal(good.actSpec.hash, RL.ACT_SPEC_HASH)
    assert.equal(good.obsSpec.hash, RL.OBS_SPEC_HASH)
    assert.deepEqual(good.selection, { mode: 'sample', temperature: 0.3 })
    assert.deepEqual(good.architecture, { input: 80, hidden: [128, 128], activation: 'tanh', head: 'logits', actions: 20 })

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
        inf.layers[2].weight[0] = inf.layers[2].weight[0].map(() => 1e308); inf.layers[2].bias[0] = 1e308
        const seat = RL.createRlSeat({ policy: inf, fallbackPersona: persona })
        assert.equal(seat.state.disabled, false)
        for (let i = 0; i < RL.FAILURE_LIMIT; i++) {
            const out = seat.decide(states[i].ctx, states[i].extras, createRng(1))
            assert.equal(out.source, 'fallback'); assert.match(out.reason, /non-finite/)
            assert.equal(out.cap, ruleCap(states[i], 1))
        }
        assert.equal(seat.state.disabled, true)
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
console.log(`PPO production contract: ${results.length}/${results.length} checks passed over ${process.argv.length - 2} polic${process.argv.length - 2 === 1 ? 'y' : 'ies'} (${states.length} real decision states)`)
