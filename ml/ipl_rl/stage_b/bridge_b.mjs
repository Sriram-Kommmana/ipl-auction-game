// Phase 2F — Stage-B TRAINING bridge (protocol rl-bridge-v2-stage-b).
//
// Speaks the rl-bridge-v2 protocol (info / configure / reset / step) so the
// unchanged Python vector environment can drive it, and plays exactly the
// frozen environment: the frozen sampler's league draw
// (sampleEpisode(seed, { league: { snapshotShare, poolSize } })), the frozen
// RlEpisode (AuctionSim, act-v3 mask, completion shield, reward), the frozen
// rule bots and human proxy, trembling 1% on RL snapshot opponents only.
//
// Differences from packages/shared/bin/rl-bridge-v2.js (none change mechanics):
//   · the snapshot pool is FIXED: the 15 frozen Stage-A exports (5 algorithms
//     × 3 seeds), loaded through crossplay/common.loadExports (sha256 digest,
//     production loader, spec-hash checks); addSnapshot is refused;
//   · snapshot seats run the production runtime (createRlSeat) with the fixed
//     clock now: () => 0 exactly as the Phase 2E.0 harness (decision D4), so an
//     OS pause can never trip the 20 ms guard and silently swap an opponent for
//     its rule persona; any other fallback (non-finite, masked, exception) is a
//     HARD STOP — never a silent rule-bot decision;
//   · train seeds only; manifest entries are refused;
//   · every 250 episodes the pool's weights are re-hashed (uncached) against
//     the load-time digest — any change is a HARD STOP (opponent mutation);
//   · one JSON line per finished episode is appended to $STAGE_B_LOG_DIR
//     (seed, which export sits in which seat, rule-bot share, learner summary).
import { appendFileSync, mkdirSync, readFileSync } from 'node:fs'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'
import { createHash } from 'node:crypto'

const CP = new URL('../crossplay/', import.meta.url)
const { RL, ROOT, EXPORTS, loadExports, loadPlayers } = await import(new URL('common.mjs', CP))
const { RL_PERSONAS } = await import(new URL('packages/shared/src/personas.js', ROOT))

export const PROTOCOL = 'rl-bridge-v2-stage-b'
const PLAYERS = loadPlayers()
const EX = loadExports()
const POOL = EXPORTS.map((e) => EX[e.key]) // fixed order: ppo s1-3, a2c s1-3, d3qn s1-3, qrdqn s1-3, es s1-3
const POOL_POLICIES = POOL.map((e) => e.policy)
const freshDigest = (p) => createHash('sha256').update(RL.canonicalJson({ algorithm: p.algorithm, architecture: p.architecture, selection: p.selection, layers: p.layers })).digest('hex')
const LOAD_DIGESTS = POOL_POLICIES.map(freshDigest)
POOL.forEach((e, i) => { if (LOAD_DIGESTS[i] !== e.layers) throw new Error(`HARD STOP pool digest ${e.key}`) })
const LOG_DIR = process.env.STAGE_B_LOG_DIR || null
if (LOG_DIR) mkdirSync(LOG_DIR, { recursive: true })
const LOG_FILE = LOG_DIR ? `${LOG_DIR}/bridge-${process.pid}.jsonl` : null

const config = { tremble: 0.01, snapshotShare: 0 }
let episode = null
let meta = null
let episodesStarted = 0

class HardStop extends Error {}

const handlers = {
    info: () => ({
        protocol: PROTOCOL,
        obsSpec: { version: RL.OBS_SPEC.version, hash: RL.OBS_SPEC_HASH, size: RL.OBS_SIZE, features: RL.OBS_FEATURES },
        actSpec: { version: RL.ACT_SPEC.version, hash: RL.ACT_SPEC_HASH, count: RL.ACTION_COUNT, actions: RL.ACTIONS.map((a) => a.name), shield: { version: RL.SHIELD_VERSION, params: RL.SHIELD_PARAMS } },
        gamma: RL.GAMMA,
        lambdaRel: RL.LAMBDA_REL,
        splits: Object.fromEntries(Object.entries(RL.SPLITS).map(([k, v]) => [k, { start: v.start, count: Number.isFinite(v.count) ? v.count : null }])),
        pool: POOL.map((e) => ({ key: e.key, sha256: e.sha256, layers: e.layers }))
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

    addSnapshot: () => { throw new Error('stage-b bridge: the snapshot pool is fixed (15 frozen Stage-A exports)') },

    reset: ({ seed, split, entry }) => {
        if (entry) throw new Error('stage-b bridge: training seeds only (manifest entries refused)')
        if (!Number.isInteger(seed) || RL.splitOfSeed(seed) !== 'train' || (split && split !== 'train')) throw new Error(`stage-b bridge: seed ${seed} is not a train seed`)
        if (++episodesStarted % 250 === 0) {
            POOL_POLICIES.forEach((p, i) => { if (freshDigest(p) !== LOAD_DIGESTS[i]) throw new HardStop(`HARD STOP opponent-policy mutation: ${POOL[i].key}`) })
        }
        const league = config.snapshotShare > 0 ? { snapshotShare: config.snapshotShare, poolSize: POOL.length } : null
        const e = RL.sampleEpisode(seed, { league })
        episode = new RL.RlEpisode({ players: PLAYERS, entry: e, snapshots: POOL_POLICIES, tremble: config.tremble, maskVersion: 'act-v3' })
        const seats = []
        e.seats.forEach((seat, i) => {
            if (seat.type !== 'rlSnapshot') { seats.push({ seat: i, type: seat.type, key: seat.persona ?? seat.proxy ?? null }); return }
            const exp = POOL[seat.snapshot]
            const rt = RL.createRlSeat({ policy: exp.policy, fallbackPersona: RL_PERSONAS[seat.rlSeat].fallback, now: () => 0 })
            if (rt.state.disabled) throw new HardStop(`HARD STOP runtime refused ${exp.key}: ${rt.state.disabledReason}`)
            episode.seats[i].runtime = {
                state: rt.state,
                decide(ctx, extras, rng) {
                    const r = rt.decide(ctx, extras, rng)
                    if (r.source !== 'rl') throw new HardStop(`HARD STOP opponent fallback seat ${i} (${exp.key}) seed ${seed} lot ${ctx.lot?.slNo}: ${r.reason}`)
                    return r
                }
            }
            seats.push({ seat: i, type: 'rlSnapshot', key: exp.key })
        })
        const ruleShare = RL.ruleOpponentShare(e)
        if (ruleShare < RL.MIN_RULE_OPPONENT_SHARE) throw new HardStop(`HARD STOP rule-bot share ${ruleShare} seed ${seed}`)
        meta = { seed, purse: e.purse, stratum: e.stratum, learnerSeat: e.learnerSeat, snapshots: seats.filter((s) => s.type === 'rlSnapshot').map((s) => s.key), ruleShare }
        const first = episode.reset()
        return { ...first, info: { ...first.info, entry: e, stageB: meta } }
    },

    step: ({ action }) => {
        if (!episode) throw new Error('call reset before step')
        if (!Number.isInteger(action) || action < 0 || action >= RL.ACTION_COUNT) throw new Error(`action must be an integer in [0, ${RL.ACTION_COUNT})`)
        const out = episode.step(action)
        if (out.done) {
            const s = out.info.episode
            if (s.invariantViolations) throw new HardStop(`HARD STOP invariant violation seed ${meta.seed}: ${s.violations.join('; ')}`)
            if (LOG_FILE) appendFileSync(LOG_FILE, `${JSON.stringify({ ...meta, xi: s.xi, legalXI: s.legalXI, emptySlots: s.emptySlots, decisions: s.decisions })}\n`)
            out.info.stageB = meta
        }
        return out
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
        reply({ ok: false, error: `${err instanceof HardStop ? '' : ''}${err.message}` })
        if (err instanceof HardStop) process.stderr.write(`${err.message}\n${err.stack}\n`)
    }
})
