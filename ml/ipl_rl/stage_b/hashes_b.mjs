// Phase 2F — frozen-artifact record for the Stage-B pilot (§36).
//   node ml/ipl_rl/stage_b/hashes_b.mjs --record <file>   (once, before any training)
//   node ml/ipl_rl/stage_b/hashes_b.mjs --check  <file>   (exit 1 on any difference)
// Covers the Phase 2E.0 frozen set (obs/act spec hashes, 25 frozen sources,
// 15 Stage-A exports + checkpoints) plus the rule-bot / planner support files,
// the production bot runtime, the bridges and evaluator, and the Phase 2D
// trainer configs. Stage-A artifacts must never change; production files must
// never change.
import { createHash } from 'node:crypto'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const CP = new URL('../crossplay/', import.meta.url)
const { ROOT, RL, frozenHashes, loadExports, FROZEN_SOURCES } = await import(new URL('common.mjs', CP))
const at = (rel) => fileURLToPath(new URL(rel, ROOT))
const sha = (rel) => createHash('sha256').update(readFileSync(at(rel))).digest('hex')

export const EXTRA_FROZEN = Object.freeze([
    // rule bots, planner support, legacy shared modules used by them
    'packages/shared/src/ruleBots.js', 'packages/shared/src/botSignals.js', 'packages/shared/src/pool.js', 'packages/shared/src/rewards.js',
    'packages/shared/src/observation.js', 'packages/shared/src/mlp.js', 'packages/shared/src/index.js',
    // production inference / bot runtime (must not change)
    'apps/server/src/bots/botManager.js', 'apps/server/src/bots/context.js', 'apps/server/src/bots/playerCache.js',
    'apps/server/src/bots/rlPolicy.js', 'apps/server/src/bots/seatBots.js',
    // bridges and evaluator
    'packages/shared/bin/rl-bridge-v2.js', 'packages/shared/bin/rl-evaluate.js',
    // Phase 2D / 2C trainer configs (Stage-A anchors)
    'ml/ipl_rl/configs/ppo_2c3.json', 'ml/ipl_rl/configs/a2c_2d1.json', 'ml/ipl_rl/configs/d3qn_2d2.json',
    'ml/ipl_rl/configs/qr_dqn_2d3.json', 'ml/ipl_rl/configs/openai_es_2d4.json'
])

const record = () => {
    const base = frozenHashes()
    return { ...base, rewardSpec: { gamma: RL.GAMMA, lambdaRel: RL.LAMBDA_REL, file: base.sources['packages/shared/src/rl/reward.js'] }, extra: Object.fromEntries(EXTRA_FROZEN.map((f) => [f, sha(f)])) }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
    const [mode, file] = process.argv.slice(2)
    loadExports() // digest + loader + spec-hash checks on all 15 Stage-A exports
    const now = record()
    const dirty = execFileSync('git', ['status', '--porcelain', '--', ...FROZEN_SOURCES, ...EXTRA_FROZEN.filter((f) => !f.startsWith('ml/'))], { cwd: at('.'), encoding: 'utf8' }).trim()
    if (dirty) { console.log(`FROZEN FILES CHANGED vs git HEAD:\n${dirty}`); process.exit(1) }
    if (mode === '--record') {
        if (existsSync(file)) throw new Error(`${file} exists — refusing to overwrite`)
        writeFileSync(file, JSON.stringify(now, null, 2))
        console.log(`recorded obs ${now.obsSpecHash} act ${now.actSpecHash} gamma ${now.rewardSpec.gamma} · ${Object.keys(now.sources).length} frozen + ${Object.keys(now.extra).length} extra files · ${Object.keys(now.exports).length} exports → ${file}`)
    } else if (mode === '--check') {
        const rec = JSON.parse(readFileSync(file, 'utf8'))
        const diffs = []
        for (const k of ['obsSpecHash', 'actSpecHash']) if (rec[k] !== now[k]) diffs.push(k)
        if (JSON.stringify(rec.rewardSpec) !== JSON.stringify(now.rewardSpec)) diffs.push('rewardSpec')
        for (const [f, h] of Object.entries(rec.sources)) if (now.sources[f] !== h) diffs.push(f)
        for (const [f, h] of Object.entries(rec.extra)) if (now.extra[f] !== h) diffs.push(f)
        for (const [k, v] of Object.entries(rec.exports)) if (JSON.stringify(now.exports[k]) !== JSON.stringify(v)) diffs.push(k)
        if (now.obsSpecHash !== '629b25783f833af7' || now.actSpecHash !== '5f72f510c48b1f46') diffs.push('spec hash ≠ frozen value')
        if (diffs.length) { console.log(`HASH MISMATCH: ${diffs.join(', ')}`); process.exit(1) }
        console.log(`HASHES OK: obs ${now.obsSpecHash} · act ${now.actSpecHash} · gamma ${now.rewardSpec.gamma} · ${Object.keys(rec.sources).length} frozen + ${Object.keys(rec.extra).length} extra files unchanged · ${Object.keys(rec.exports).length} Stage-A exports + checkpoints unchanged`)
    } else throw new Error('usage: hashes_b.mjs --record|--check <file>')
}
