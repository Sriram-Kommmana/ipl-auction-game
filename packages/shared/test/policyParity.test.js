// After `python ml/export.py …`, proves the JavaScript forward pass produces
// the same logits PyTorch did for the same observations — i.e. the bot the
// server runs is the bot you trained. Skipped until a model has been exported.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { forward, validateModel } from '../src/mlp.js'

const fixturePath = fileURLToPath(new URL('./fixtures/policy-parity.json', import.meta.url))
const repoRoot = fileURLToPath(new URL('../../../', import.meta.url))

test('exported RL policy matches PyTorch', { skip: !existsSync(fixturePath) && 'no exported policy yet (run ml/export.py)' }, () => {
    const fixture = JSON.parse(readFileSync(fixturePath, 'utf8'))
    const model = validateModel(JSON.parse(readFileSync(repoRoot + fixture.model, 'utf8')))
    fixture.observations.forEach((obs, i) => {
        const js = forward(model, obs)
        js.forEach((logit, a) =>
            assert.ok(Math.abs(logit - fixture.logits[i][a]) < 1e-4, `observation ${i}, action ${a}: ${logit} vs ${fixture.logits[i][a]}`)
        )
    })
})
