// Phase 2E.0 — Stage B frozen cross-play evaluation: shared pieces.
//
// Evaluation only. Nothing here trains, updates or rewrites a model: the 15
// frozen Stage-A exports are read, checked against their recorded digests and
// played through the canonical JavaScript environment (RlEpisode) and the
// production seat runtime (createRlSeat). No frozen file is modified — the
// Stage-B rooms are built by editing a COPY of each validation-manifest entry.
//
//   EXPORTS            the 15 frozen exports (5 algorithms × 3 training seeds)
//   loadExports()      read + validate + digest-check every export
//   frozenHashes()     sha256 of every frozen source / data file and export
//   compose(entry, c)  a Stage-B room from a Stage-A manifest entry

import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

export const ROOT = new URL('../../../', import.meta.url)
const at = (rel) => fileURLToPath(new URL(rel, ROOT))
const { parsePlayersCsv } = await import(new URL('packages/shared/src/playersCsv.js', ROOT))
export const RL = await import(new URL('packages/shared/src/rl/index.js', ROOT))
const { createRng } = await import(new URL('packages/shared/src/sim.js', ROOT))

export const FROZEN_OBS_HASH = '629b25783f833af7'
export const FROZEN_ACT_HASH = '5f72f510c48b1f46'
export const ALGOS = Object.freeze(['ppo', 'a2c', 'd3qn', 'qrdqn', 'es'])
export const ALGO_NAMES = Object.freeze({ ppo: 'PPO', a2c: 'A2C', d3qn: 'D3QN', qrdqn: 'QR-DQN', es: 'OpenAI-ES' })
export const SEEDS = Object.freeze([1, 2, 3])

// Final frozen exports (Phase 2C.3, 2D.1, 2D.2, 2D.3, 2D.4) and the sha256
// digests recorded in the Phase 2E.0 gap report before any Stage-B code ran
// (first 12 hex; the full digests are written to frozen-hashes.json).
const RUN_DIR = { ppo: 'ppo-2c3', a2c: 'a2c-2d1', d3qn: 'd3qn-2d2', qrdqn: 'qr-dqn-2d3', es: 'openai-es-2d4' }
const FINAL_CKPT = { ppo: 'update_0325', a2c: 'update_0325', d3qn: 'update_0325', qrdqn: 'update_0325', es: 'gen_2000' }
export const EXPECTED_DIGEST12 = Object.freeze({
    'ppo:s1': '1484f74db7ab', 'ppo:s2': 'ffa4a6bbdb18', 'ppo:s3': '9cdd49ed48c9',
    'a2c:s1': '9482565a9322', 'a2c:s2': '8d54a091f8e4', 'a2c:s3': '338ed33cb8a4',
    'd3qn:s1': '92d77f58a27a', 'd3qn:s2': 'e4b54c2a8b27', 'd3qn:s3': '7d938abb7cb5',
    'qrdqn:s1': '957d8847988e', 'qrdqn:s2': 'd4089e713f6d', 'qrdqn:s3': 'a5bdbe2fd1dd',
    'es:s1': 'cccc7996e636', 'es:s2': '37bb9702b090', 'es:s3': '76bcd8ff0638'
})
export const exportKey = (algo, seed) => `${algo}:s${seed}`
export const exportDir = (algo, seed) => `ml/runs/${RUN_DIR[algo]}-s${seed}/checkpoints/${FINAL_CKPT[algo]}`
export const EXPORTS = Object.freeze(ALGOS.flatMap((algo) => SEEDS.map((seed) => Object.freeze({
    key: exportKey(algo, seed), algo, seed, path: `${exportDir(algo, seed)}/policy.json`, checkpoint: `${exportDir(algo, seed)}/checkpoint.pt`,
    stageAEpisodes: `${exportDir(algo, seed)}/validation/episodes.json`
}))))

export const sha256 = (buf) => createHash('sha256').update(buf).digest('hex')
// Identity of the weights a runtime actually holds (after the production loader).
export const layerDigest = (policy) => sha256(RL.canonicalJson({ algorithm: policy.algorithm, architecture: policy.architecture, selection: policy.selection, layers: policy.layers }))

// Every export: exact bytes → sha256 (must match the gap-report digest),
// production loader (must accept), identifier, head, spec hashes.
export const loadExports = () => {
    const out = {}
    for (const e of EXPORTS) {
        const bytes = readFileSync(at(e.path))
        const digest = sha256(bytes)
        if (digest.slice(0, 12) !== EXPECTED_DIGEST12[e.key]) throw new Error(`EXPORT MISMATCH ${e.key}: sha256 ${digest.slice(0, 12)} ≠ recorded ${EXPECTED_DIGEST12[e.key]}`)
        const loaded = RL.loadPolicy(bytes.toString('utf8'))
        if (!loaded.ok) throw new Error(`EXPORT REJECTED ${e.key}: ${loaded.error}`)
        const p = loaded.policy
        if (p.algorithm !== e.algo) throw new Error(`WRONG MODEL ${e.key}: file says algorithm ${p.algorithm}`)
        if (p.obsSpec.hash !== FROZEN_OBS_HASH || p.obsSpec.hash !== RL.OBS_SPEC_HASH) throw new Error(`WRONG OBS HASH ${e.key}: ${p.obsSpec.hash}`)
        if (p.actSpec.hash !== FROZEN_ACT_HASH || p.actSpec.hash !== RL.ACT_SPEC_HASH) throw new Error(`WRONG ACT HASH ${e.key}: ${p.actSpec.hash}`)
        out[e.key] = { ...e, policy: p, sha256: digest, layers: layerDigest(p) }
    }
    if (RL.OBS_SPEC_HASH !== FROZEN_OBS_HASH) throw new Error(`WRONG OBS HASH in code: ${RL.OBS_SPEC_HASH}`)
    if (RL.ACT_SPEC_HASH !== FROZEN_ACT_HASH) throw new Error(`WRONG ACT HASH in code: ${RL.ACT_SPEC_HASH}`)
    return out
}

// Frozen specification, rules, bots, data and manifests (none may change).
export const FROZEN_SOURCES = Object.freeze([
    'packages/shared/src/rl/obsSpec.js', 'packages/shared/src/rl/actionSpec.js', 'packages/shared/src/rl/mask.js',
    'packages/shared/src/rl/reward.js', 'packages/shared/src/rl/samplers.js', 'packages/shared/src/rl/env.js',
    'packages/shared/src/rl/runtime.js', 'packages/shared/src/rl/policy.js', 'packages/shared/src/rl/evaluate.js',
    'packages/shared/src/rl/invariants.js', 'packages/shared/src/rl/hash.js', 'packages/shared/src/rl/index.js',
    'packages/shared/src/sim.js', 'packages/shared/src/planning.js', 'packages/shared/src/personas.js',
    'packages/shared/src/rules.js', 'packages/shared/src/scoring.js', 'packages/shared/src/valuation.js',
    'packages/shared/src/playersCsv.js', 'apps/server/src/db/players.csv',
    'packages/shared/data/rl-manifests/train.json', 'packages/shared/data/rl-manifests/validation.json',
    'packages/shared/data/rl-manifests/test.json',
    'packages/shared/data/rl-baselines/validation.report.json', 'packages/shared/data/rl-baselines/validation.episodes.json'
])
export const frozenHashes = () => ({
    obsSpecHash: RL.OBS_SPEC_HASH,
    actSpecHash: RL.ACT_SPEC_HASH,
    sources: Object.fromEntries(FROZEN_SOURCES.map((f) => [f, sha256(readFileSync(at(f)))])),
    exports: Object.fromEntries(EXPORTS.map((e) => [e.key, { path: e.path, sha256: sha256(readFileSync(at(e.path))), checkpointSha256: sha256(readFileSync(at(e.checkpoint))) }]))
})

export const loadPlayers = () => parsePlayersCsv(readFileSync(at('apps/server/src/db/players.csv'), 'utf8'))
export const loadValidation = () => {
    const m = JSON.parse(readFileSync(at('packages/shared/data/rl-manifests/validation.json'), 'utf8'))
    if (m.split !== 'validation' || m.entries.length !== 500) throw new Error('validation manifest is not the frozen 500-entry split')
    return m.entries
}

// ── Stage-B room composition ─────────────────────────────────────────────
// Conditions:
//   { kind: 'A' }                           Stage-A control: the manifest entry unchanged
//   { kind: 'C1', opp: 'd3qn:s2' }          opponent in ONE of the four other RL seats
//   { kind: 'C4', opp: 'd3qn:s2' }          opponent in ALL FOUR other RL seats
//   { kind: 'S4', learnerAlgo, seed: k }    the other four algorithms (seed k), one each
// Only 'rlFallback' seats change (to 'rlSnapshot', keeping rlSeat and persona,
// exactly the transformation samplers.js makes in league configuration);
// purse, seating, learner seat, rule bots and human proxy are untouched.
// Seat choices use the existing seed-derived mechanism
// (createRng(deriveSeed(seed, stream))) and depend only on the auction seed,
// so every pairing on an entry uses the same seat(s).
export const C1_STREAM = 'phase2e0/c1-opponent-seat'
export const S4_STREAM = 'phase2e0/s4-algorithm-permutation'
const rlFallbackSeats = (entry) => {
    const idx = entry.seats.flatMap((s, i) => (s.type === 'rlFallback' ? [i] : []))
    if (idx.length !== 4) throw new Error(`seed ${entry.seed}: expected 4 rlFallback seats, found ${idx.length}`)
    return idx
}
export const c1Seat = (entry) => rlFallbackSeats(entry)[Math.floor(createRng(RL.deriveSeed(entry.seed, C1_STREAM))() * 4)]
export const s4Permutation = (entry, learnerAlgo) => {
    const rest = ALGOS.filter((a) => a !== learnerAlgo)
    const rng = createRng(RL.deriveSeed(entry.seed, S4_STREAM))
    for (let i = rest.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1))
        ;[rest[i], rest[j]] = [rest[j], rest[i]]
    }
    return rest
}

export const compose = (entry, cond) => {
    const out = JSON.parse(JSON.stringify(entry))
    const snapshotKeys = []
    const put = (i, key) => {
        const seat = out.seats[i]
        if (seat.type !== 'rlFallback') throw new Error(`seed ${entry.seed}: seat ${i} is ${seat.type}, not rlFallback`)
        out.seats[i] = { ...seat, type: 'rlSnapshot', snapshot: snapshotKeys.length }
        snapshotKeys.push(key)
    }
    switch (cond.kind) {
        case 'A':
            return { entry: out, snapshotKeys }
        case 'C1':
            put(c1Seat(entry), cond.opp)
            break
        case 'C4':
            for (const i of rlFallbackSeats(entry)) put(i, cond.opp)
            break
        case 'S4': {
            const perm = s4Permutation(entry, cond.learnerAlgo)
            rlFallbackSeats(entry).forEach((i, k) => put(i, exportKey(perm[k], cond.seed)))
            break
        }
        default:
            throw new Error(`unknown condition ${cond.kind}`)
    }
    out.stage = 'B'
    out.stageB = { composition: cond.kind }
    // Frozen league rule: rule bots stay ≥ 40% of the nine opponents.
    const opponents = out.seats.filter((_, i) => i !== out.learnerSeat)
    const ruleShare = opponents.filter((s) => s.type === 'rule').length / opponents.length
    if (ruleShare < RL.MIN_RULE_OPPONENT_SHARE) throw new Error(`seed ${entry.seed}: rule bots ${ruleShare} < ${RL.MIN_RULE_OPPONENT_SHARE}`)
    return { entry: out, snapshotKeys }
}
