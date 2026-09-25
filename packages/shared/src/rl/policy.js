// rl-policy-v2 — the exported model format and its plain-JavaScript
// inference. No ONNX, no native code: a small MLP forward pass.
//
// {
//   format: 'rl-policy-v2',
//   algorithm: 'ppo' | 'a2c' | 'd3qn' | 'qrdqn' | 'es',
//   modelType: 'mlp',
//   architecture: { input: 80, hidden: [h1, h2], activation: 'tanh'|'relu',
//                   head: 'logits' | 'q' | 'quantiles', actions: 20, quantiles?: 32 },
//   obsSpec: { version: 'obs-v2', hash, size: 80 },
//   actSpec: { version: 'act-v2', hash, count: 20 },
//   selection: { mode: 'sample', temperature: 0.3 } | { mode: 'argmax' },
//   layers: [{ weight: [[out × in]], bias: [out] }, ...],
//   meta: { trainedSteps, seed, config, createdAt, gitSha, ... }
// }
//
// Heads: 'logits' (PPO/A2C/ES), 'q' (Dueling Double DQN — the dueling
// V + A − mean(A) is folded into one linear layer at export), 'quantiles'
// (QR-DQN: the last layer outputs actions × quantiles, action-major; the
// action value is the mean over quantiles).
//
// Selection is fixed by the algorithm: PPO/A2C sample with temperature 0.3,
// DQN/QR-DQN/ES take the argmax — always over the legal actions only.

import { OBS_SIZE, OBS_SPEC, OBS_SPEC_HASH } from './obsSpec.js'
import { ACTION_COUNT, ACT_SPEC, ACT_SPEC_HASH } from './actionSpec.js'

export const POLICY_FORMAT = 'rl-policy-v2'
export const MAX_PARAMETERS = 1_000_000
export const ALGORITHMS = Object.freeze({
    ppo: Object.freeze({ head: 'logits', selection: Object.freeze({ mode: 'sample', temperature: 0.3 }) }),
    a2c: Object.freeze({ head: 'logits', selection: Object.freeze({ mode: 'sample', temperature: 0.3 }) }),
    d3qn: Object.freeze({ head: 'q', selection: Object.freeze({ mode: 'argmax' }) }),
    qrdqn: Object.freeze({ head: 'quantiles', selection: Object.freeze({ mode: 'argmax' }) }),
    es: Object.freeze({ head: 'logits', selection: Object.freeze({ mode: 'argmax' }) })
})
const ACTIVATIONS = { tanh: Math.tanh, relu: (x) => (x > 0 ? x : 0) }

const fail = (msg) => { throw new Error(`rl-policy-v2: ${msg}`) }

// Throws on anything that doesn't match the current specs exactly.
export const validatePolicy = (p) => {
    if (!p || typeof p !== 'object') fail('not an object')
    if (p.format !== POLICY_FORMAT) fail(`format is ${p.format}`)
    const algo = ALGORITHMS[p.algorithm]
    if (!algo) fail(`unknown algorithm ${p.algorithm}`)
    if (p.modelType !== 'mlp') fail(`unsupported modelType ${p.modelType}`)
    if (p.obsSpec?.version !== OBS_SPEC.version || p.obsSpec?.hash !== OBS_SPEC_HASH || p.obsSpec?.size !== OBS_SIZE) {
        fail(`observation spec mismatch (model ${p.obsSpec?.version}/${p.obsSpec?.hash}, runtime ${OBS_SPEC.version}/${OBS_SPEC_HASH})`)
    }
    if (p.actSpec?.version !== ACT_SPEC.version || p.actSpec?.hash !== ACT_SPEC_HASH || p.actSpec?.count !== ACTION_COUNT) {
        fail(`action spec mismatch (model ${p.actSpec?.version}/${p.actSpec?.hash}, runtime ${ACT_SPEC.version}/${ACT_SPEC_HASH})`)
    }
    const arch = p.architecture || fail('missing architecture')
    if (arch.input !== OBS_SIZE) fail(`architecture.input ${arch.input} != ${OBS_SIZE}`)
    if (arch.actions !== ACTION_COUNT) fail(`architecture.actions ${arch.actions} != ${ACTION_COUNT}`)
    if (!ACTIVATIONS[arch.activation]) fail(`unknown activation ${arch.activation}`)
    if (arch.head !== algo.head) fail(`${p.algorithm} must use head '${algo.head}', got '${arch.head}'`)
    const outputs = arch.head === 'quantiles' ? ACTION_COUNT * arch.quantiles : ACTION_COUNT
    if (arch.head === 'quantiles' && !(Number.isInteger(arch.quantiles) && arch.quantiles > 0)) fail('quantiles head needs a positive integer quantile count')
    const sel = p.selection || fail('missing selection')
    if (sel.mode !== algo.selection.mode || (sel.mode === 'sample' && sel.temperature !== algo.selection.temperature)) {
        fail(`${p.algorithm} must select by ${JSON.stringify(algo.selection)}`)
    }
    if (!Array.isArray(p.layers) || p.layers.length !== (arch.hidden?.length ?? -1) + 1) fail('layer count does not match architecture.hidden')
    let width = OBS_SIZE
    let params = 0
    p.layers.forEach((layer, i) => {
        const out = i < arch.hidden.length ? arch.hidden[i] : outputs
        if (!Array.isArray(layer.weight) || layer.weight.length !== out) fail(`layer ${i}: weight has ${layer.weight?.length} rows, expected ${out}`)
        if (!Array.isArray(layer.bias) || layer.bias.length !== out) fail(`layer ${i}: bias size ${layer.bias?.length}, expected ${out}`)
        for (const row of layer.weight) {
            if (!Array.isArray(row) || row.length !== width) fail(`layer ${i}: a weight row has ${row?.length} columns, expected ${width}`)
            for (const w of row) if (typeof w !== 'number' || !Number.isFinite(w)) fail(`layer ${i}: non-finite weight`)
        }
        for (const b of layer.bias) if (typeof b !== 'number' || !Number.isFinite(b)) fail(`layer ${i}: non-finite bias`)
        params += out * width + out
        width = out
    })
    if (params > MAX_PARAMETERS) fail(`${params} parameters exceeds the ${MAX_PARAMETERS} limit`)
    return p
}

// Safe load: { ok: true, policy } or { ok: false, error } — never throws.
export const loadPolicy = (json) => {
    try {
        return { ok: true, policy: validatePolicy(typeof json === 'string' ? JSON.parse(json) : json) }
    } catch (err) {
        return { ok: false, error: err.message }
    }
}

// Raw network output (logits, Q-values, or actions × quantiles).
export const forwardRaw = (policy, input) => {
    const act = ACTIVATIONS[policy.architecture.activation]
    const last = policy.layers.length - 1
    let x = input
    for (let i = 0; i <= last; i++) {
        const { weight, bias } = policy.layers[i]
        const out = new Array(weight.length)
        for (let r = 0; r < weight.length; r++) {
            const row = weight[r]
            let sum = bias[r]
            for (let c = 0; c < row.length; c++) sum += row[c] * x[c]
            out[r] = i === last ? sum : act(sum)
        }
        x = out
    }
    return x
}

// One score per action: logits, Q-values, or mean quantile value.
export const actionScores = (policy, obs) => {
    const raw = forwardRaw(policy, obs)
    if (policy.architecture.head !== 'quantiles') return raw
    const n = policy.architecture.quantiles
    return Array.from({ length: ACTION_COUNT }, (_, a) => {
        let s = 0
        for (let k = 0; k < n; k++) s += raw[a * n + k]
        return s / n
    })
}

// Pick a legal action from the scores (mask: 1 = legal).
export const selectAction = (policy, scores, mask, rng = Math.random) => {
    const legal = []
    for (let a = 0; a < mask.length; a++) if (mask[a]) legal.push(a)
    if (!legal.length) throw new Error('no legal action')
    let best = legal[0]
    for (const a of legal) if (scores[a] > scores[best]) best = a
    const sel = ALGORITHMS[policy.algorithm].selection
    if (sel.mode === 'argmax') return best
    const max = scores[best]
    const weights = legal.map((a) => Math.exp((scores[a] - max) / sel.temperature))
    const total = weights.reduce((s, w) => s + w, 0)
    let r = rng() * total
    for (let i = 0; i < legal.length; i++) {
        r -= weights[i]
        if (r <= 0) return legal[i]
    }
    return best
}
