#!/usr/bin/env node
// Bridge between the Python RL trainer (ml/) and the JavaScript auction
// simulator. Python never re-implements the game: it sends actions, this
// process runs the real rules, scoring, rule bots and observation builder
// from this package, and sends back observations, masks and rewards.
//
// Protocol: one JSON object per line on stdin, one JSON reply per line on
// stdout. Commands:
//
//   {"cmd":"info"}
//       → { observationSize, actionCount, features, personaDims, capMultipliers, rlPersonas }
//   {"cmd":"configure", "tremble":0.01, "snapshotShare":0.4, "humanProxyNoise":0.25}
//   {"cmd":"addSnapshot", "path":"…/snap.json"}  → load a frozen policy as a future opponent
//   {"cmd":"reset", "seed":123, "persona":[5 numbers] (optional)}
//       → { obs, mask, info }
//   {"cmd":"step", "action":3}
//       → { obs, mask, reward, done, info }
//
// One episode = one whole auction. The learner controls one of 10 seats; it
// is only asked to act on lots where it has a real choice (lots where it can
// only pass are resolved automatically). The other nine seats are rule bots,
// frozen snapshots of earlier policies, and one noisy "human proxy".

import { readFileSync } from 'node:fs'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'
import { parsePlayersCsv } from '../src/playersCsv.js'
import { AuctionSim, agentCap, createRng } from '../src/sim.js'
import {
    ACTION_COUNT, CAP_MULTIPLIERS, OBSERVATION_FEATURES, OBSERVATION_SIZE, PERSONA_DIMS,
    actionMask, buildObservation, capForAction, deriveLotFacts
} from '../src/observation.js'
import { RL_PERSONAS } from '../src/personas.js'
import { RULE_BOT_IDS } from '../src/ruleBots.js'
import { validateModel } from '../src/mlp.js'
import { slotBudget } from '../src/valuation.js'
import { episodeReward, lotShaping } from '../src/rewards.js'
import { DEFAULT_RULES } from '../src/rules.js'

const csvPath = process.argv[2] ||
    fileURLToPath(new URL('../../../apps/server/src/db/players.csv', import.meta.url))
const PLAYERS = parsePlayersCsv(readFileSync(csvPath, 'utf8'))
const TEAM_COUNT = 10
const MAX_SNAPSHOTS = 20

const config = { tremble: 0.01, snapshotShare: 0.4, humanProxyNoise: 0.25 }
const snapshots = []

let episode = null

const clip01 = (x) => Math.min(1, Math.max(0, x))
const pick = (rng, list) => list[Math.floor(rng() * list.length)]
const presetVectors = Object.values(RL_PERSONAS).map((p) => p.vector)

// Personas the learner trains under: the five presets, jittered, so the one
// network learns to respond to the persona input rather than memorising five points.
const samplePersona = (rng) => pick(rng, presetVectors).map((v) => clip01(v + (rng() - 0.5) * 0.3))

const buildOpponent = (rng, isHumanProxy) => {
    if (isHumanProxy) return { kind: 'rule', persona: pick(rng, RULE_BOT_IDS), noise: config.humanProxyNoise }
    if (snapshots.length && rng() < config.snapshotShare) {
        return { kind: 'mlp', model: pick(rng, snapshots), persona: pick(rng, presetVectors), temperature: 0.5 }
    }
    return { kind: 'rule', persona: pick(rng, RULE_BOT_IDS) }
}

const observe = () => {
    const ctx = episode.sim.contextFor(episode.learner)
    const facts = deriveLotFacts(ctx)
    return { ctx, facts, obs: buildObservation(ctx, episode.persona), mask: actionMask(ctx, facts) }
}

// Resolve lots until the learner has a real decision (or the auction ends).
const advanceToDecision = () => {
    let shaping = 0
    while (!episode.sim.done) {
        const view = observe()
        if (view.mask.slice(1).some(Boolean)) {
            episode.view = view
            return shaping
        }
        shaping += resolveLot(0, view)
    }
    episode.view = null
    return shaping
}

// Every seat names its cap; the simulator runs the bidding ladder.
const resolveLot = (learnerCap, view) => {
    const { sim, learner, agents, rng } = episode
    const caps = agents.map((agent, i) =>
        i === learner ? learnerCap : agentCap(agent, sim.contextFor(i), rng, { tremble: config.tremble })
    )
    const outcome = sim.resolveLot(caps)
    return lotShaping({
        won: outcome.winner === learner,
        price: outcome.price,
        xiGainAtOpen: view.facts.xiGain,
        slotBudget: slotBudget(sim.rules)
    })
}

const handlers = {
    info: () => ({
        observationSize: OBSERVATION_SIZE,
        actionCount: ACTION_COUNT,
        features: OBSERVATION_FEATURES,
        personaDims: PERSONA_DIMS,
        capMultipliers: CAP_MULTIPLIERS,
        rlPersonas: RL_PERSONAS
    }),

    configure: (msg) => {
        for (const key of Object.keys(config)) if (msg[key] !== undefined) config[key] = msg[key]
        return { config }
    },

    addSnapshot: ({ path }) => {
        snapshots.push(validateModel(JSON.parse(readFileSync(path, 'utf8'))))
        if (snapshots.length > MAX_SNAPSHOTS) snapshots.shift()
        return { snapshots: snapshots.length }
    },

    reset: ({ seed = Date.now(), persona = null }) => {
        const rng = createRng(seed)
        const sim = new AuctionSim({ players: PLAYERS, teamCount: TEAM_COUNT, rules: DEFAULT_RULES, rng })
        const learner = Math.floor(rng() * TEAM_COUNT)
        const humanProxy = (learner + 1 + Math.floor(rng() * (TEAM_COUNT - 1))) % TEAM_COUNT
        const agents = Array.from({ length: TEAM_COUNT }, (_, i) =>
            i === learner ? null : buildOpponent(rng, i === humanProxy)
        )
        episode = { sim, learner, agents, rng, persona: persona || samplePersona(rng) }
        advanceToDecision()
        if (!episode.view) throw new Error('Learner never gets a decision in this auction')
        return { obs: episode.view.obs, mask: episode.view.mask, info: { seat: learner, persona: episode.persona } }
    },

    step: ({ action }) => {
        if (!episode?.view) throw new Error('Call reset before step')
        if (!episode.view.mask[action]) throw new Error(`Action ${action} is masked out`)
        const cap = capForAction(episode.view.ctx, action, episode.view.facts)
        let reward = resolveLot(cap, episode.view)
        reward += advanceToDecision()

        if (!episode.view) {
            const result = episodeReward(episode.sim, episode.learner, episode.persona)
            return {
                obs: new Array(OBSERVATION_SIZE).fill(0),
                mask: [1, ...new Array(ACTION_COUNT - 1).fill(0)],
                reward: reward + result.reward,
                done: true,
                info: { strength: result.strength, rank: result.rank, strengths: result.strengths, terms: result.terms }
            }
        }
        return { obs: episode.view.obs, mask: episode.view.mask, reward, done: false, info: {} }
    }
}

const reply = (payload) => process.stdout.write(`${JSON.stringify(payload)}\n`)

createInterface({ input: process.stdin, terminal: false }).on('line', (line) => {
    if (!line.trim()) return
    let msg
    try {
        msg = JSON.parse(line)
        const handler = handlers[msg.cmd]
        if (!handler) throw new Error(`Unknown command: ${msg.cmd}`)
        reply({ ok: true, ...handler(msg) })
    } catch (err) {
        reply({ ok: false, error: err.message })
    }
})
