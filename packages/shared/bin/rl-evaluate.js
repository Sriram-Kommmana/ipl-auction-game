#!/usr/bin/env node
// Baseline / policy evaluation on the committed manifests (deterministic Node).
//
//   node packages/shared/bin/rl-evaluate.js --split validation --limit 50
//   node packages/shared/bin/rl-evaluate.js --split test --baselines moneyball,randomLegal --out report.json
//   node packages/shared/bin/rl-evaluate.js --split validation --policy path/to/policy.json
//
// Every controller plays the SAME auctions (paired by seed). Reports mean and
// bootstrap 95% CI per metric, per purse stratum, and the paired XI
// difference against the reference (default moneyball).

import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { parsePlayersCsv } from '../src/playersCsv.js'
import { BASELINES, evaluate, generateManifest, loadPolicy, policyController } from '../src/rl/index.js'

const args = process.argv.slice(2)
const opt = (name, fallback) => (args.includes(`--${name}`) ? args[args.indexOf(`--${name}`) + 1] : fallback)
const split = opt('split', 'validation')
const limit = Number(opt('limit', Infinity))
const names = opt('baselines', Object.keys(BASELINES).join(',')).split(',').filter(Boolean)
const reference = opt('reference', 'moneyball')

const players = parsePlayersCsv(readFileSync(fileURLToPath(new URL('../../../apps/server/src/db/players.csv', import.meta.url)), 'utf8'))
const manifest = JSON.parse(readFileSync(fileURLToPath(new URL(`../data/rl-manifests/${split}.json`, import.meta.url)), 'utf8'))
// Guard against a stale committed manifest.
if (JSON.stringify(generateManifest(split, manifest.entries.length).entries) !== JSON.stringify(manifest.entries)) {
    throw new Error(`${split}.json no longer matches the sampler — regenerate with bin/rl-manifests.js`)
}
const entries = manifest.entries.slice(0, limit)

const controllers = {}
for (const n of names) {
    if (!BASELINES[n]) throw new Error(`unknown baseline ${n} (have ${Object.keys(BASELINES).join(', ')})`)
    controllers[n] = BASELINES[n]
}
if (opt('policy')) {
    const loaded = loadPolicy(readFileSync(opt('policy'), 'utf8'))
    if (!loaded.ok) throw new Error(loaded.error)
    controllers[`policy:${loaded.policy.algorithm}`] = policyController(loaded.policy)
}

const t0 = Date.now()
const { report } = evaluate({ players, entries, controllers, reference })
const fmt = (s) => `${s.mean.toFixed(2)} [${s.ci95[0].toFixed(2)}, ${s.ci95[1].toFixed(2)}]`
console.log(`${split}: ${entries.length} auctions × ${Object.keys(controllers).length} controllers in ${((Date.now() - t0) / 1000).toFixed(0)}s`)
for (const [name, r] of Object.entries(report)) {
    console.log(`${name.padEnd(18)} XI ${fmt(r.xi)}  legal ${(100 * r.legalXI.mean).toFixed(0)}%  strong ${(100 * r.strongXI.mean).toFixed(0)}%  rank ${r.rank.mean.toFixed(2)}  purse left ${r.purseLeftShare.mean.toFixed(2)}  XI/₹1000L ${r.xiGainPer1000.mean.toFixed(2)}  shield ${r.shieldActivations.mean.toFixed(2)}` +
        (r.paired ? `  ΔXI vs ${r.paired.reference} ${r.paired.xiDiffMean.toFixed(2)} [${r.paired.ci95[0].toFixed(2)}, ${r.paired.ci95[1].toFixed(2)}]` : ''))
}
if (opt('out')) writeFileSync(opt('out'), JSON.stringify({ split, n: entries.length, report }, null, 2))
