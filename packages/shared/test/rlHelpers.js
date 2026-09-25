// Shared fixtures for the RL infrastructure tests (not a test file itself).
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { parsePlayersCsv } from '../src/playersCsv.js'
import { createRng } from '../src/sim.js'
import { ACTION_COUNT, ACT_SPEC, ACT_SPEC_HASH, ALGORITHMS, OBS_SIZE, OBS_SPEC, OBS_SPEC_HASH, RlEpisode, sampleEpisode } from '../src/rl/index.js'

export const players = parsePlayersCsv(readFileSync(fileURLToPath(new URL('../../../apps/server/src/db/players.csv', import.meta.url)), 'utf8'))

const copyTeam = (t) => ({ ...t, squad: [...t.squad] })
// A frozen copy of a context: the simulator mutates its teams as it goes.
export const snapshotCtx = (ctx) => ({ ...ctx, self: copyTeam(ctx.self), rivals: ctx.rivals.map(copyTeam), upcoming: [...ctx.upcoming], returning: [...ctx.returning] })

// Real decision states from full episodes (random legal learner), every
// `every`-th decision: { ctx, extras, plan, mask, obs, entry }.
export const realStates = (seeds, every = 7) => {
    const out = []
    for (const seed of seeds) {
        const entry = sampleEpisode(seed)
        const ep = new RlEpisode({ players, entry })
        const rng = ep.learnerRng
        ep.reset()
        let i = 0
        for (;;) {
            const p = ep.pending
            if (i++ % every === 0) out.push({ ctx: snapshotCtx(p.ctx), extras: ep.extras(), plan: p.plan, mask: p.mask, obs: p.obs, entry })
            const legal = p.mask.mask.flatMap((ok, a) => (ok ? [a] : []))
            const step = ep.step(legal[Math.floor(rng() * legal.length)])
            if (step.done) break
        }
    }
    return out
}

// Randomly initialised rl-policy-v2 networks (forward passes only — no training).
const ARCH = {
    ppo: { hidden: [128, 128], activation: 'tanh', head: 'logits' },
    a2c: { hidden: [128, 128], activation: 'tanh', head: 'logits' },
    d3qn: { hidden: [128, 128], activation: 'tanh', head: 'q' },
    qrdqn: { hidden: [128, 128], activation: 'tanh', head: 'quantiles', quantiles: 32 },
    es: { hidden: [64, 64], activation: 'tanh', head: 'logits' }
}

export const randomPolicy = (algorithm, seed = 1, scale = 0.2) => {
    const rng = createRng(seed)
    const a = ARCH[algorithm]
    const outputs = a.head === 'quantiles' ? ACTION_COUNT * a.quantiles : ACTION_COUNT
    const widths = [OBS_SIZE, ...a.hidden, outputs]
    const layers = widths.slice(1).map((out, i) => ({
        weight: Array.from({ length: out }, () => Array.from({ length: widths[i] }, () => (rng() * 2 - 1) * scale)),
        bias: Array.from({ length: out }, () => (rng() * 2 - 1) * scale)
    }))
    return {
        format: 'rl-policy-v2',
        algorithm,
        modelType: 'mlp',
        architecture: { input: OBS_SIZE, actions: ACTION_COUNT, ...a },
        obsSpec: { version: OBS_SPEC.version, hash: OBS_SPEC_HASH, size: OBS_SIZE },
        actSpec: { version: ACT_SPEC.version, hash: ACT_SPEC_HASH, count: ACTION_COUNT },
        selection: ALGORITHMS[algorithm].selection,
        layers,
        meta: { trainedSteps: 0, seed, config: { note: 'random init — test only' }, createdAt: '2026-09-26T00:00:00Z' }
    }
}

