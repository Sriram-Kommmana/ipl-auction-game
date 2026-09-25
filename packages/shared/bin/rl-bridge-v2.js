#!/usr/bin/env node
// rl-bridge-v2 — Python ↔ JavaScript bridge for IplAuctionEnv-v2. Python
// never implements game rules: it sends actions, this process runs the real
// AuctionSim, frozen opponents, obs-v2, the canonical mask and the reward.
//
// Protocol: one JSON object per line on stdin → one JSON reply per line on
// stdout. Every reply has "ok"; failures carry "error". Deterministic: the
// same sequence of requests always produces the same replies.
//
//   {"cmd":"info"}
//     → protocol, obsSpec {version, hash, size, features}, actSpec {version, hash,
//       count, actions}, gamma, lambdaRel, splits
//   {"cmd":"configure", "tremble":0.01, "snapshotShare":0.4}
//   {"cmd":"addSnapshot", "policy":{rl-policy-v2}}    → { snapshots }
//   {"cmd":"reset", "seed":1000123, "split":"train"}  → { obs, mask, info }
//   {"cmd":"reset", "entry":{manifest entry}}         (replay a manifest entry)
//   {"cmd":"step", "action":7}                        → { obs, mask, reward, done, info }
//
// At done, obs and mask are null and info.episode holds the episode summary.

import { readFileSync } from 'node:fs'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'
import { parsePlayersCsv } from '../src/playersCsv.js'
import {
    ACTIONS, ACTION_COUNT, ACT_SPEC, ACT_SPEC_HASH, GAMMA, LAMBDA_REL, OBS_FEATURES, OBS_SIZE, OBS_SPEC, OBS_SPEC_HASH,
    RlEpisode, SPLITS, loadPolicy, sampleEpisode, splitOfSeed
} from '../src/rl/index.js'

export const PROTOCOL = 'rl-bridge-v2'
const csvPath = process.argv[2] || fileURLToPath(new URL('../../../apps/server/src/db/players.csv', import.meta.url))
const PLAYERS = parsePlayersCsv(readFileSync(csvPath, 'utf8'))

const config = { tremble: 0.01, snapshotShare: 0 }
const snapshots = []
let episode = null

const handlers = {
    info: () => ({
        protocol: PROTOCOL,
        obsSpec: { version: OBS_SPEC.version, hash: OBS_SPEC_HASH, size: OBS_SIZE, features: OBS_FEATURES },
        actSpec: { version: ACT_SPEC.version, hash: ACT_SPEC_HASH, count: ACTION_COUNT, actions: ACTIONS.map((a) => a.name) },
        gamma: GAMMA,
        lambdaRel: LAMBDA_REL,
        splits: Object.fromEntries(Object.entries(SPLITS).map(([k, v]) => [k, { start: v.start, count: Number.isFinite(v.count) ? v.count : null }]))
    }),

    configure: (msg) => {
        if (msg.tremble !== undefined) {
            if (!(msg.tremble >= 0 && msg.tremble <= 1)) throw new Error('tremble must be in [0, 1]')
            config.tremble = msg.tremble
        }
        if (msg.snapshotShare !== undefined) {
            if (!(msg.snapshotShare >= 0 && msg.snapshotShare <= 1)) throw new Error('snapshotShare must be in [0, 1]')
            config.snapshotShare = msg.snapshotShare
        }
        return { config }
    },

    addSnapshot: ({ policy }) => {
        const loaded = loadPolicy(policy)
        if (!loaded.ok) throw new Error(`snapshot rejected: ${loaded.error}`)
        snapshots.push(loaded.policy)
        return { snapshots: snapshots.length }
    },

    reset: ({ seed, split, entry }) => {
        if (!entry) {
            if (!Number.isInteger(seed)) throw new Error('reset needs an integer seed or an entry')
            if (split && splitOfSeed(seed) !== split) throw new Error(`seed ${seed} is not in the ${split} split`)
            const league = config.snapshotShare > 0 && snapshots.length ? { snapshotShare: config.snapshotShare, poolSize: snapshots.length } : null
            entry = sampleEpisode(seed, { league })
        }
        episode = new RlEpisode({ players: PLAYERS, entry, snapshots, tremble: config.tremble })
        const first = episode.reset()
        return { ...first, info: { ...first.info, entry } }
    },

    step: ({ action }) => {
        if (!episode) throw new Error('call reset before step')
        if (!Number.isInteger(action) || action < 0 || action >= ACTION_COUNT) throw new Error(`action must be an integer in [0, ${ACTION_COUNT})`)
        return episode.step(action)
    }
}

const reply = (payload) => process.stdout.write(`${JSON.stringify(payload)}\n`)

createInterface({ input: process.stdin, terminal: false }).on('line', (line) => {
    if (!line.trim()) return
    try {
        const msg = JSON.parse(line)
        const handler = handlers[msg.cmd]
        if (!handler) throw new Error(`unknown command: ${msg.cmd}`)
        reply({ ok: true, ...handler(msg) })
    } catch (err) {
        reply({ ok: false, error: err.message })
    }
})
