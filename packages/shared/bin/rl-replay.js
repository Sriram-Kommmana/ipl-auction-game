#!/usr/bin/env node
// Parity helper: replays one episode in a fresh process from stdin
//   { "entry": {manifest entry}, "actions": [..], "tremble": 0.01 }
// and prints { obs: [...], masks: [...], rewards: [...], shield: [...] } — the
// JavaScript ground truth the Python environment is compared against.

import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { parsePlayersCsv } from '../src/playersCsv.js'
import { RlEpisode } from '../src/rl/index.js'

const players = parsePlayersCsv(readFileSync(fileURLToPath(new URL('../../../apps/server/src/db/players.csv', import.meta.url)), 'utf8'))
const { entry, actions, tremble = 0.01 } = JSON.parse(readFileSync(0, 'utf8'))
const ep = new RlEpisode({ players, entry, tremble })
const first = ep.reset()
const out = { obs: [first.obs], masks: [first.mask], rewards: [], shield: [first.info.shieldActive] }
for (const action of actions) {
    const step = ep.step(action)
    out.rewards.push(step.reward)
    if (step.done) break
    out.obs.push(step.obs)
    out.masks.push(step.mask)
    out.shield.push(step.info.shieldActive)
}
process.stdout.write(JSON.stringify(out))
