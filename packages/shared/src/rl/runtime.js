// One RL seat at run time: the trained policy when it is present and
// healthy, the seat's frozen rule persona otherwise. Never lets a model
// decision through that the canonical mask rejects, and never returns a cap
// above maxSafeBid.
//
//   missing / invalid / mismatched model  → frozen rule persona, whole room
//   NaN / Infinity output, masked action,
//   invalid cap, any exception             → rule persona for that lot;
//                                           after FAILURE_LIMIT, for the room
//   a decision slower than 20 ms          → rule persona for the rest of the room
//
// One instance per seat per room (the failure count is room state).

import { agentCap } from '../sim.js'
import { planBid } from '../planning.js'
import { buildRlObservation } from './obsSpec.js'
import { hasBidAction, rlActionMask, validateRlDecision } from './mask.js'
import { actionScores, loadPolicy, selectAction } from './policy.js'
import { PASS } from './actionSpec.js'

export const FAILURE_LIMIT = 3
export const INFERENCE_BUDGET_MS = 20
const defaultNow = () => globalThis.performance?.now?.() ?? Date.now()

// `select` and `now` are injectable only so tests can simulate faults.
export const createRlSeat = ({ policy = null, fallbackPersona, now = defaultNow, select = selectAction }) => {
    if (!fallbackPersona) throw new Error('createRlSeat: fallbackPersona is required')
    const loaded = policy ? loadPolicy(policy) : { ok: false, error: 'no model' }
    const state = {
        policy: loaded.ok ? loaded.policy : null,
        disabled: !loaded.ok,
        disabledReason: loaded.ok ? null : loaded.error,
        failures: 0,
        log: []
    }
    const fallback = (ctx, rng, reason) => ({ cap: agentCap({ kind: 'rule', persona: fallbackPersona }, ctx, rng), action: null, source: 'fallback', reason })
    const failure = (ctx, rng, reason, { wholeRoom = false } = {}) => {
        state.failures++
        state.log.push(reason)
        if (wholeRoom || state.failures >= FAILURE_LIMIT) {
            state.disabled = true
            state.disabledReason = wholeRoom ? reason : `${FAILURE_LIMIT} failed decisions (last: ${reason})`
        }
        return fallback(ctx, rng, reason)
    }

    return {
        state,
        // extras: as for buildRlObservation ({ poolSize, recent }).
        decide(ctx, extras, rng = Math.random) {
            if (state.disabled) return fallback(ctx, rng, state.disabledReason)
            const started = now()
            let plan, maskResult, action
            try {
                plan = planBid(ctx)
                maskResult = rlActionMask(ctx, plan)
                if (!hasBidAction(maskResult.mask)) return { cap: 0, action: PASS, source: 'rl', reason: 'no legal bid' }
                const obs = buildRlObservation(ctx, extras, plan)
                const scores = actionScores(state.policy, obs)
                if (scores.some((s) => !Number.isFinite(s))) return failure(ctx, rng, 'non-finite model output')
                action = select(state.policy, scores, maskResult.mask, rng)
            } catch (err) {
                return failure(ctx, rng, `inference error: ${err.message}`)
            }
            const cap = maskResult.caps[action]
            const invalid = validateRlDecision(ctx, action, cap, maskResult)
            if (invalid) return failure(ctx, rng, `invalid decision: ${invalid}`)
            const elapsed = now() - started
            if (elapsed > INFERENCE_BUDGET_MS) return failure(ctx, rng, `inference took ${elapsed.toFixed(1)} ms`, { wholeRoom: true })
            return { cap, action, source: 'rl', reason: null }
        }
    }
}
