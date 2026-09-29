import { OBS_SIZE, actionScores } from '@ipl-auction/shared/rl'
import { loadRoster } from './rlRoster.js'

// The trained RL opponents, loaded once at startup from bots/models/
// (registry.json + the frozen rl-policy-v2 exports it pins by sha256).
//
// A persona whose model is missing, corrupted, hash-mismatched or rejected by
// the obs/act spec checks plays its fallback rule personality (RL_PERSONAS in
// packages/shared/src/personas.js), so solo mode always works.
let roster = null

// RL_MODELS_DIR (optional) points at another models folder — e.g. a staged
// roster, or a deliberately broken copy for fallback drills.
const loadRlRoster = () => {
    roster = process.env.RL_MODELS_DIR ? loadRoster(process.env.RL_MODELS_DIR) : loadRoster()
    const summary = Object.entries(roster.byPersona)
        .map(([persona, m]) => `${persona}=${m ? m.key : 'rule fallback'}`)
        .join(', ')
    console.log(`[bots] RL roster: ${summary}`)
    for (const err of roster.errors) console.error(`[bots] RL model problem — seat uses its rule fallback: ${err}`)
    // Warm the forward pass once at startup, so a room's first RL decision is
    // not also the process's first (cold) one — the 20 ms room guard counts it.
    const zeros = new Array(OBS_SIZE).fill(0)
    for (const m of Object.values(roster.byPersona)) if (m) for (let i = 0; i < 20; i++) actionScores(m.policy, zeros)
    return roster
}

const getRlRoster = () => roster

export { loadRlRoster, getRlRoster }
