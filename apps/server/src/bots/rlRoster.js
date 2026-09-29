import { createHash } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { RL_PERSONAS, fairValue } from '@ipl-auction/shared'
import { RECENT_WINDOW, createRlSeat, loadPolicy } from '@ipl-auction/shared/rl'

// The trained RL opponents of a solo room.
//
// models/registry.json names one frozen rl-policy-v2 export per RL
// personality (RL_PERSONAS) and pins each file's sha256:
//   { "format": "rl-roster-v1",
//     "roster": { "<personaId>": "<modelKey>", ... },
//     "models": { "<modelKey>": { "file": "x.json", "sha256": "...", "algorithm": "ppo", ... } } }
//
// Every seat then plays through the shared production runtime (createRlSeat):
// the act-v3 mask and completion shield, cap ≤ maxSafeBid, the 20 ms room
// guard and the completion guard. A persona whose model is missing, corrupted,
// hash-mismatched or rejected by the spec checks plays its frozen rule
// fallback (RL_PERSONAS[id].fallback) — the game never depends on a model.

const MODELS_DIR = fileURLToPath(new URL('./models/', import.meta.url))

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex')

// → { byPersona: { personaId: { key, algorithm, policy } | null }, errors: [string] }
export const loadRoster = (dir = MODELS_DIR) => {
    const byPersona = Object.fromEntries(Object.keys(RL_PERSONAS).map((id) => [id, null]))
    const errors = []
    const registryPath = `${dir}/registry.json`
    if (!existsSync(registryPath)) {
        errors.push('no models/registry.json')
        return { byPersona, errors }
    }
    let registry
    try {
        registry = JSON.parse(readFileSync(registryPath, 'utf8'))
        if (registry.format !== 'rl-roster-v1') throw new Error(`format is ${registry.format}`)
    } catch (err) {
        errors.push(`registry unreadable: ${err.message}`)
        return { byPersona, errors }
    }

    const loaded = new Map()
    const loadModel = (key) => {
        if (loaded.has(key)) return loaded.get(key)
        let result = null
        const entry = registry.models?.[key]
        try {
            if (!entry) throw new Error('not in registry')
            const bytes = readFileSync(`${dir}/${entry.file}`)
            const digest = sha256(bytes)
            if (digest !== entry.sha256) throw new Error(`sha256 ${digest.slice(0, 12)} ≠ registry ${String(entry.sha256).slice(0, 12)}`)
            const res = loadPolicy(bytes.toString('utf8'))
            if (!res.ok) throw new Error(res.error)
            if (entry.algorithm && res.policy.algorithm !== entry.algorithm) throw new Error(`file is ${res.policy.algorithm}, registry says ${entry.algorithm}`)
            result = { key, algorithm: res.policy.algorithm, policy: res.policy }
        } catch (err) {
            errors.push(`${key}: ${err.message}`)
        }
        loaded.set(key, result)
        return result
    }

    for (const personaId of Object.keys(RL_PERSONAS)) {
        const key = registry.roster?.[personaId]
        if (!key) {
            errors.push(`${personaId}: no model assigned`)
            continue
        }
        byPersona[personaId] = loadModel(key)
    }
    return { byPersona, errors }
}

// One runtime per RL seat per room (it keeps the room's failure count).
export const createSeatRuntime = (roster, personaId) => {
    const persona = RL_PERSONAS[personaId]
    if (!persona) throw new Error(`unknown RL persona ${personaId}`)
    return createRlSeat({
        policy: roster?.byPersona?.[personaId]?.policy ?? null,
        fallbackPersona: persona.fallback,
        completionGuard: true
    })
}

// The observation's public market window, from the room history list
// (entries written by timerManager / onSkip) — the same fields the training
// environment passes (RlEpisode.extras): the last RECENT_WINDOW lots.
export const buildExtras = ({ history, poolSize, players, pursePerTeam }) => ({
    poolSize,
    recent: history.slice(-RECENT_WINDOW).flatMap((raw) => {
        const h = typeof raw === 'string' ? JSON.parse(raw) : raw
        const player = players.get(Number(h.iplPlayerId))
        if (!player) return [] // never expected: every history entry is a pool player
        const sold = h.status === 'sold'
        return [{
            sold,
            price: sold ? Number(h.soldFor) : null,
            fairValue: fairValue(player, pursePerTeam),
            winnerTeamId: sold ? h.soldTo : null
        }]
    })
})
