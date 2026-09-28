#!/usr/bin/env node
// Test helper (Phase 2D.3): replays one episode in a fresh process from stdin
//   { "entry": {manifest entry}, "actions": [..], "tremble": 0.01 }
// and prints the JavaScript ground truth per decision — obs, mask, reward,
// terminal — plus what each decision state IS (phase, purse, requirement
// needs, act-v3 shield forced / final path / state), so tests can prove
// they covered main-auction, re-auction, low/high-purse, keeper, Indian,
// bowling and final-path states. Read-only use of the shared package.
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const shared = new URL('../../../packages/shared/', import.meta.url)
const { parsePlayersCsv } = await import(new URL('src/playersCsv.js', shared))
const { RlEpisode } = await import(new URL('src/rl/index.js', shared))
const players = parsePlayersCsv(readFileSync(fileURLToPath(new URL('../../apps/server/src/db/players.csv', shared)), 'utf8'))
const { entry, actions, tremble = 0.01 } = JSON.parse(readFileSync(0, 'utf8'))
const ep = new RlEpisode({ players, entry, tremble })
const first = ep.reset()
const describe = () => {
    const { ctx, plan, mask } = ep.pending
    const d = mask.shield
    return {
        phase: ctx.phase, purse: ctx.self.purseLeft,
        need: { keeper: plan.requirements.keeper.need, bowling: plan.requirements.bowling.need, indians: plan.requirements.indians.need },
        forced: Boolean(d.forced), finalPath: Boolean(d.finalPath), shieldState: d.state
    }
}
const out = { obs: [first.obs], masks: [first.mask], rewards: [], dones: [], states: [describe()] }
for (const action of actions) {
    const step = ep.step(action)
    out.rewards.push(step.reward)
    out.dones.push(step.done)
    if (step.done) break
    out.obs.push(step.obs)
    out.masks.push(step.mask)
    out.states.push(describe())
}
process.stdout.write(JSON.stringify(out))
