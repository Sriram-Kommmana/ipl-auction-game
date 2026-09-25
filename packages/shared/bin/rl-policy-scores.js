#!/usr/bin/env node
// Parity helper: validates an rl-policy-v2 with the production loader and
// scores observations with the production JavaScript inference.
//   stdin:  { "policy": {rl-policy-v2}, "observations": [[80 numbers], ...] }
//   stdout: { ok, error?, raw: [[...]], scores: [[20]] }

import { readFileSync } from 'node:fs'
import { actionScores, forwardRaw, loadPolicy } from '../src/rl/index.js'

const { policy, observations } = JSON.parse(readFileSync(0, 'utf8'))
const loaded = loadPolicy(policy)
process.stdout.write(JSON.stringify(loaded.ok
    ? { ok: true, raw: observations.map((o) => forwardRaw(loaded.policy, o)), scores: observations.map((o) => actionScores(loaded.policy, o)) }
    : { ok: false, error: loaded.error }))
