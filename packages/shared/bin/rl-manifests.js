#!/usr/bin/env node
// Writes the committed RL seed manifests (Phase 2A, frozen split):
//   validation  seeds 100,000–100,499   (500)   checkpoint selection
//   test        seeds 200,000–200,999   (1,000) final reports only
//   train       seeds 1,000,000+        first TRAIN_PINNED pinned for audit;
//               training streams further seeds through the same sampler
// Each entry is sampleEpisode(seed): purse + stratum, seating, learner seat,
// human proxy. Tests regenerate these and fail on any drift.
//
//   node packages/shared/bin/rl-manifests.js          → write
//   node packages/shared/bin/rl-manifests.js --check  → verify only

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { generateManifest } from '../src/rl/samplers.js'

export const MANIFEST_DIR = fileURLToPath(new URL('../data/rl-manifests/', import.meta.url))
export const TRAIN_PINNED = 2000
const COUNTS = { validation: 500, test: 1000, train: TRAIN_PINNED }

// One entry per line: small diffs, still valid JSON.
export const serialiseManifest = (m) =>
    `{"format":${JSON.stringify(m.format)},"split":${JSON.stringify(m.split)},"sampler":${JSON.stringify(m.sampler)},"entries":[\n` +
    m.entries.map((e) => JSON.stringify(e)).join(',\n') + '\n]}\n'

const check = process.argv.includes('--check')
mkdirSync(MANIFEST_DIR, { recursive: true })
let drift = 0
for (const [split, count] of Object.entries(COUNTS)) {
    const text = serialiseManifest(generateManifest(split, count))
    const path = `${MANIFEST_DIR}${split}.json`
    if (check) {
        let current = null
        try { current = readFileSync(path, 'utf8') } catch { /* missing */ }
        if (current !== text) { drift++; console.log(`✖ ${split}.json differs from the sampler`) } else console.log(`✔ ${split}.json (${count})`)
    } else {
        writeFileSync(path, text)
        console.log(`wrote ${path} (${count} entries)`)
    }
}
if (check && drift) process.exitCode = 1
