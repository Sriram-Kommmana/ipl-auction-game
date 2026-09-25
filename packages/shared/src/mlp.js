// Runs a trained policy network in plain JavaScript — no ONNX runtime, no
// native dependency. A 2×64 MLP is ~7k weights; a forward pass is a few
// thousand multiply-adds, microseconds even on a 1 GiB VPS.
//
// Model file format ("mlp-v1"), written by ml/export.py:
//   {
//     "format": "mlp-v1",
//     "observationSize": 38, "actionCount": 8,
//     "hiddenActivation": "tanh",
//     "features": [...],                // must equal OBSERVATION_FEATURES
//     "layers": [ { "weight": [[...]], "bias": [...] }, ... ]   // weight is out × in
//   }
// This mirrors Stable-Baselines3's default actor for PPO: Linear→tanh→
// Linear→tanh→Linear (the last layer outputs one logit per action).

import { ACTION_COUNT, OBSERVATION_FEATURES, OBSERVATION_SIZE } from './observation.js'

export const validateModel = (model) => {
    if (!model || model.format !== 'mlp-v1') throw new Error('Model is not mlp-v1')
    if (model.observationSize !== OBSERVATION_SIZE) {
        throw new Error(`Model expects ${model.observationSize} observation values, runtime produces ${OBSERVATION_SIZE}`)
    }
    if (model.actionCount !== ACTION_COUNT) {
        throw new Error(`Model has ${model.actionCount} actions, runtime has ${ACTION_COUNT}`)
    }
    if (Array.isArray(model.features)) {
        const mismatch = model.features.findIndex((f, i) => f !== OBSERVATION_FEATURES[i])
        if (mismatch !== -1 || model.features.length !== OBSERVATION_FEATURES.length) {
            throw new Error(`Model was trained on a different observation layout (first difference at index ${mismatch})`)
        }
    }
    let width = model.observationSize
    for (const [i, layer] of model.layers.entries()) {
        if (layer.weight.some((row) => row.length !== width)) {
            throw new Error(`Layer ${i} weight rows don't match input width ${width}`)
        }
        if (layer.bias.length !== layer.weight.length) throw new Error(`Layer ${i} bias size mismatch`)
        width = layer.weight.length
    }
    if (width !== model.actionCount) throw new Error('Final layer width must equal actionCount')
    return model
}

const ACTIVATIONS = {
    tanh: Math.tanh,
    relu: (x) => (x > 0 ? x : 0)
}

export const forward = (model, input) => {
    const act = ACTIVATIONS[model.hiddenActivation || 'tanh']
    let x = input
    const last = model.layers.length - 1
    model.layers.forEach((layer, i) => {
        const out = new Array(layer.weight.length)
        for (let r = 0; r < layer.weight.length; r++) {
            const row = layer.weight[r]
            let sum = layer.bias[r]
            for (let c = 0; c < row.length; c++) sum += row[c] * x[c]
            out[r] = i === last ? sum : act(sum)
        }
        x = out
    })
    return x
}

// Illegal actions get -Infinity (Huang & Ontañón 2022: masking logits is
// the correct way to handle invalid actions in policy-gradient methods —
// and it's exactly what MaskablePPO did during training).
export const maskLogits = (logits, mask) => logits.map((l, i) => (mask[i] ? l : -Infinity))

export const argmax = (values) => values.reduce((best, v, i) => (v > values[best] ? i : best), 0)

// Sample with a temperature: 0 → always the top action, 1 → the policy's own
// distribution. A little randomness keeps RL bots from being perfectly
// predictable to a human who plays them repeatedly.
export const sampleAction = (logits, mask, { temperature = 0.3, rng = Math.random } = {}) => {
    const masked = maskLogits(logits, mask)
    if (temperature <= 0) return argmax(masked)
    const max = Math.max(...masked)
    const weights = masked.map((l) => (l === -Infinity ? 0 : Math.exp((l - max) / temperature)))
    const total = weights.reduce((s, w) => s + w, 0)
    let r = rng() * total
    for (let i = 0; i < weights.length; i++) {
        r -= weights[i]
        if (r <= 0 && weights[i] > 0) return i
    }
    return argmax(masked)
}
