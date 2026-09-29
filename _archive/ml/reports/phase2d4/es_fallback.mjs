// Phase 2D.4 §21–24 — production-inference contract of a REAL exported OpenAI-ES
// policy (rl-policy-v2): healthy decisions are legal act-v3 caps ≤ maxSafeBid,
// and every fault path falls back to the frozen rule persona.
//   node ml/runs/_2d4/es_fallback.mjs <policy.json> [...]
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
const latency = []
const check = (name, fn) => { fn(); results.push(name) }

for (const path of process.argv.slice(2)) {
    const good = JSON.parse(readFileSync(path, 'utf8'))
    assert.equal(good.algorithm, 'es'); assert.equal(good.meta.algorithmName, 'openai_es')
    assert.equal(good.actSpec.hash, RL.ACT_SPEC_HASH)
    assert.equal(good.obsSpec.hash, RL.OBS_SPEC_HASH)
    assert.deepEqual(good.selection, { mode: 'argmax' })
    assert.deepEqual(good.architecture, { input: 80, hidden: [64, 64], activation: 'tanh', head: 'logits', actions: 20 })

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
        const h = clone(good); h.architecture.hidden = [128, 128]; wholeRoomFallback(h, 'wrong architecture: hidden 128-128')
        const w = clone(good); w.layers[2].weight.pop(); w.layers[2].bias.pop(); wholeRoomFallback(w, 'layer width')
        const c = clone(good); c.layers[1].weight[3] = c.layers[1].weight[3].slice(1); wholeRoomFallback(c, 'row width')
        const hd = clone(good); hd.architecture.head = 'q'; wholeRoomFallback(hd, 'head for algorithm')
        const s = clone(good); s.selection = { mode: 'sample', temperature: 0.3 }; wholeRoomFallback(s, 'selection for algorithm')
        const a = clone(good); a.architecture.activation = 'gelu'; wholeRoomFallback(a, 'activation')
        const o = clone(good); o.architecture.actions = 16; wholeRoomFallback(o, 'architecture action count')
        const o19 = clone(good); o19.layers[2].weight.pop(); o19.layers[2].bias.pop(); wholeRoomFallback(o19, 'output count 19')
        const o21 = clone(good); o21.layers[2].weight.push(o21.layers[2].weight[0]); o21.layers[2].bias.push(0); wholeRoomFallback(o21, 'output count 21')
    })
    check('healthy decisions are the masked argmax of the exported logits', () => {
        const seat = RL.createRlSeat({ policy: good, fallbackPersona: persona, now: () => 0 })
        const pol = RL.loadPolicy(good).policy
        for (const s of states) {
            const out = seat.decide(s.ctx, s.extras, createRng(9))
            const q = RL.actionScores(pol, s.obs)
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
        const n = clone(good); n.layers[1].weight[0][0] = NaN; wholeRoomFallback(n, 'NaN parameter')
        const inf1 = clone(good); inf1.layers[0].bias[3] = Infinity; wholeRoomFallback(inf1, 'Infinity parameter (in memory)')
        wholeRoomFallback(JSON.stringify(inf1), 'Infinity parameter (JSON: serialises to null)')
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
        // NaN / Infinity OUTPUT with a policy that passed validation: the seat keeps the
        // validated object, so corrupt a head bias in memory afterwards (as a fault would).
        for (const [bad, label] of [[NaN, 'NaN output'], [Infinity, 'Infinity output']]) {
            const obj = clone(good)
            const s2 = RL.createRlSeat({ policy: obj, fallbackPersona: persona })
            assert.equal(s2.state.disabled, false)
            s2.state.policy.layers[2].bias[5] = bad
            const out = s2.decide(states[0].ctx, states[0].extras, createRng(4))
            assert.equal(out.source, 'fallback', label); assert.match(out.reason, /non-finite/)
            assert.equal(out.cap, ruleCap(states[0], 4))
        }
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
    check('real latency inside the budget (distribution; 9 seats in turn; see latency_bench.mjs for outliers)', () => {
        // measurement seats: the 20 ms guard is exercised separately above; here every decision is timed
        const seat = RL.createRlSeat({ policy: good, fallbackPersona: persona, now: () => 0 })
        const lat = []
        for (let rep = 0; rep < 40; rep++) for (const s of states) { const t0 = performance.now(); seat.decide(s.ctx, s.extras, createRng(1)); lat.push(performance.now() - t0) }
        lat.sort((a, b) => a - b)
        const q = (p) => lat[Math.min(lat.length - 1, Math.floor(p * lat.length))]
        assert.equal(seat.state.disabled, false)
        assert.ok(q(0.99) < 20, `p99 ${q(0.99)} ms`)
        // nine RL seats deciding the same lot one after another (one Node thread, as in a room)
        const seats = Array.from({ length: 9 }, () => RL.createRlSeat({ policy: good, fallbackPersona: persona, now: () => 0 }))
        const lot = []
        for (let rep = 0; rep < 20; rep++) for (const s of states) { const t0 = performance.now(); for (const st of seats) st.decide(s.ctx, s.extras, createRng(1)); lot.push(performance.now() - t0) }
        lot.sort((a, b) => a - b)
        const ql = (p) => lot[Math.min(lot.length - 1, Math.floor(p * lot.length))]
        assert.ok(seats.every((st) => !st.state.disabled))
        latency.push({ path, decisions: lat.length, p50: q(0.5), p95: q(0.95), p99: q(0.99), max: lat[lat.length - 1],
                       nineSeats: { lots: lot.length, p50: ql(0.5), p95: ql(0.95), p99: ql(0.99), max: lot[lot.length - 1] } })
        console.log(`  ${path}: per decision p50 ${q(0.5).toFixed(3)} p95 ${q(0.95).toFixed(3)} p99 ${q(0.99).toFixed(3)} max ${lat[lat.length - 1].toFixed(3)} ms (n=${lat.length}); `
            + `9 seats per lot p50 ${ql(0.5).toFixed(3)} p95 ${ql(0.95).toFixed(3)} p99 ${ql(0.99).toFixed(3)} max ${lot[lot.length - 1].toFixed(3)} ms (n=${lot.length})`)
    })
}
console.log(`OpenAI-ES production contract: ${results.length}/${results.length} checks passed over ${process.argv.length - 2} polic${process.argv.length - 2 === 1 ? 'y' : 'ies'} (${states.length} real decision states)`)

if (process.env.LATENCY_OUT) (await import('node:fs')).writeFileSync(process.env.LATENCY_OUT, JSON.stringify(latency, null, 1))
