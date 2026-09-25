import { readFileSync, existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { validateModel } from '@ipl-auction/shared'

// The trained RL policy, exported by ml/export.py. One network plays all
// five RL personalities — each seat feeds it a different persona vector.
//
// Until a model has been trained and exported, the file doesn't exist and
// RL seats play their fallback rule personality (see RL_PERSONAS in
// packages/shared/src/personas.js), so solo mode works from day one.
const MODEL_PATH = fileURLToPath(new URL('./models/rl-policy.json', import.meta.url))

let model = null
let loadError = null

const loadRlPolicy = () => {
    if (!existsSync(MODEL_PATH)) {
        console.log('[bots] No RL model at bots/models/rl-policy.json — RL seats will use their fallback rule personalities')
        return null
    }
    try {
        model = validateModel(JSON.parse(readFileSync(MODEL_PATH, 'utf8')))
        console.log(`[bots] Loaded RL policy (${model.meta?.trainedSteps ?? 'unknown'} training steps)`)
    } catch (err) {
        loadError = err
        model = null
        console.error('[bots] RL model is invalid — RL seats will use fallbacks:', err.message)
    }
    return model
}

const getRlPolicy = () => model
const getRlPolicyError = () => loadError

export { loadRlPolicy, getRlPolicy, getRlPolicyError }
