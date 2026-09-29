// Phase 2E.2 — exact inventory of the frozen obs-v2 observation (read-only).
// Reads OBS_SPEC straight from the frozen obsSpec.js, re-derives its hash and
// checks it (and the action hash) against the frozen values. The per-feature
// annotations (meaning, internal clipping, information source, how opponents
// can move it, temporal scope) are transcribed from obsSpec.js / planning.js /
// botSignals.js and are keyed by feature NAME — any drift in names or order
// aborts. Nothing here changes the observation.
//
//   node obs_inventory.mjs <out.json>
import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const CP = new URL('../crossplay/', import.meta.url)
const { RL, ROOT, FROZEN_OBS_HASH, FROZEN_ACT_HASH } = await import(new URL('common.mjs', CP))
const O = await import(new URL('packages/shared/src/rl/obsSpec.js', ROOT))
const { specHash } = await import(new URL('packages/shared/src/rl/hash.js', ROOT))

// source: where the information comes from
//   lot      the player card on the block            own     the seat's own squad / purse
//   config   room constants                          supply  catalogue minus auctioned (+ unsold returning)
//   planner  planBid on the seat's own ctx (own squad/purse + supply; `+rivals` = also rival composition)
//   rivals   aggregate over the 9 rival teams' public squad / purse
//   window   public results of the last 20 lots
// opp: how opponent behaviour can change the value at a fixed lot
//   none      fixed by the lot / room
//   own       only through the seat's own purchases (i.e. via contests it won or lost)
//   supply    through which players went unsold (returning pool)
//   direct    a direct function of rival state or recent public results
// scope: current | cumulative (whole auction so far) | window20
const A = {
    phase_reauction: ['1 in the re-auction, 0 in the main round', null, 'config', 'none', 'current'],
    main_progress: ['share of the main-round pool already auctioned (1 in re-auction)', '[0,1]', 'config', 'none', 'current'],
    lots_left_in_phase: ['players still to come in this phase / main-pool size', null, 'supply', 'none', 'current'],
    returning_count: ['players unsold so far (back in the re-auction) / main-pool size', null, 'supply', 'supply', 'cumulative'],
    set_index: ['lot set number / 10', null, 'lot', 'none', 'current'],
    purse_scale: ['log2(room purse / 12500) / 2 (room purse stratum)', null, 'config', 'none', 'current'],
    lot_is_batsman: ['lot role = batsman', null, 'lot', 'none', 'current'],
    lot_is_bowler: ['lot role = bowler', null, 'lot', 'none', 'current'],
    lot_is_all_rounder: ['lot role = all-rounder', null, 'lot', 'none', 'current'],
    lot_is_keeper: ['lot role = wicket keeper', null, 'lot', 'none', 'current'],
    lot_is_overseas: ['lot is an overseas player', null, 'lot', 'none', 'current'],
    lot_rating: ['(rating − 60) / 40', null, 'lot', 'none', 'current'],
    lot_is_star: ['rating ≥ 90', null, 'lot', 'none', 'current'],
    lot_base_price: ['base price / slot budget (purse / 25)', null, 'lot', 'none', 'current'],
    lot_fair_value: ['fair value / slot budget', 'max 4', 'lot', 'none', 'current'],
    self_purse: ['own purse left / room purse', null, 'own', 'own', 'current'],
    self_slots_left: ['own squad slots left / 25', null, 'own', 'own', 'current'],
    self_overseas_slots_left: ['own overseas slots left / 8', null, 'own', 'own', 'current'],
    self_class_Wi: ['own Indian keepers / 11', null, 'own', 'own', 'current'],
    self_class_Wo: ['own overseas keepers / 11', null, 'own', 'own', 'current'],
    self_class_Bi: ['own Indian bowling options / 11', null, 'own', 'own', 'current'],
    self_class_Bo: ['own overseas bowling options / 11', null, 'own', 'own', 'current'],
    self_class_Oi: ['own other Indian players / 11', null, 'own', 'own', 'current'],
    self_class_Oo: ['own other overseas players / 11', null, 'own', 'own', 'current'],
    self_xi_strength: ['own best-XI total / 1100', null, 'own', 'own', 'current'],
    self_xi_empty: ['own empty XI slots / 11', null, 'own', 'own', 'current'],
    self_xi_floor: ['(weakest own XI rating − 60) / 40; 0 while the XI has empty slots', null, 'own', 'own', 'current'],
    self_xi_overseas: ['overseas players in own best XI / 4', null, 'own', 'own', 'current'],
    self_status_keeper: ['planner keeper status (SAFE 0 = need met, NEED 1/3, CRITICAL 2/3, IMPOSSIBLE 1); CRITICAL uses rivals needing a keeper', null, 'planner+rivals', 'direct', 'current'],
    self_status_bowling: ['planner bowling status (as above)', null, 'planner+rivals', 'direct', 'current'],
    self_status_indians: ['planner Indians status (as above)', null, 'planner+rivals', 'direct', 'current'],
    self_status_players: ['planner squad-count status (as above)', null, 'planner+rivals', 'direct', 'current'],
    self_need_keeper: ['keepers still needed / 1', null, 'own', 'own', 'current'],
    self_need_bowling: ['bowling options still needed / 5', null, 'own', 'own', 'current'],
    self_need_indians: ['Indians still needed / 11', null, 'own', 'own', 'current'],
    self_need_players: ['XI players still needed / 11', null, 'own', 'own', 'current'],
    self_reserve_if_passed: ['cheapest legal completion (at base prices of remaining supply) if this lot is passed / purse; 1 if unreachable', null, 'planner', 'own', 'current'],
    self_max_safe_purse: ['maxSafeBid / room purse', null, 'planner', 'own', 'current'],
    self_max_safe_fv: ['maxSafeBid / fair value', null, 'planner', 'own', 'current'],
    self_xi_gain_share: ['own XI gain from this lot / (rating / 11)', '[0,1]', 'planner', 'own', 'current'],
    self_xi_gain: ['own XI gain from this lot / 10', null, 'planner', 'own', 'current'],
    self_fills_keeper: ['lot fills an unmet own keeper requirement', null, 'planner', 'own', 'current'],
    self_fills_bowling: ['lot fills an unmet own bowling requirement', null, 'planner', 'own', 'current'],
    self_fills_indians: ['lot fills an unmet own Indians requirement', null, 'planner', 'own', 'current'],
    self_unlocks: ['buying reaches a better XI than passing', null, 'planner', 'own', 'current'],
    self_final_opportunity: ['planner final-opportunity flag (after this lot fewer suitable players remain than needed, or unlocks)', null, 'planner', 'own', 'current'],
    self_pace_gap: ['share of own purse spent − share of premium value already auctioned', null, 'own+supply', 'own', 'cumulative'],
    self_upgrades_left: ['players still to come rated above the own XI floor / 50', null, 'supply', 'supply', 'current'],
    self_typical_upgrade: ['mean XI gain of those upgrades / 2 (0 while XI incomplete)', null, 'supply', 'supply', 'current'],
    self_bench: ['(squad − XI players) / 14', null, 'own', 'own', 'current'],
    mkt_equivalent_left: ['same role within 3 rating points still to come / 10', null, 'supply', 'supply', 'current'],
    mkt_better_left: ['same role, higher rating still to come / 10', null, 'supply', 'supply', 'current'],
    mkt_same_role_left: ['same role still to come / 40', null, 'supply', 'supply', 'current'],
    mkt_supply_Wi: ['Indian keepers still to come (incl. returning) / 40', null, 'supply', 'supply', 'current'],
    mkt_supply_Wo: ['overseas keepers still to come (incl. returning) / 40', null, 'supply', 'supply', 'current'],
    mkt_supply_Bi: ['Indian bowling options still to come / 40', null, 'supply', 'supply', 'current'],
    mkt_supply_Bo: ['overseas bowling options still to come / 40', null, 'supply', 'supply', 'current'],
    mkt_supply_Oi: ['other Indians still to come / 40', null, 'supply', 'supply', 'current'],
    mkt_supply_Oo: ['other overseas still to come / 40', null, 'supply', 'supply', 'current'],
    mkt_scarcity_keeper: ['keepers affordable to self after this lot / (own need + rivals NEEDING a keeper + 1)', null, 'planner+rivals', 'direct', 'current'],
    mkt_scarcity_bowling: ['bowling options after this lot / (own need + rivals needing + 1)', null, 'planner+rivals', 'direct', 'current'],
    mkt_scarcity_indians: ['Indians after this lot / (own need + rivals needing + 1)', null, 'planner+rivals', 'direct', 'current'],
    mkt_overseas_contested: ['overseas lot and more better overseas players still to come than own overseas slots after buying', null, 'supply', 'own', 'current'],
    mkt_premium_passed: ['share of premium value (fair − base) already auctioned', null, 'supply', 'none', 'cumulative'],
    mkt_recent_price_ratio: ['mean price / fair value of SALES in the last 20 lots (1 if none) — any buyer incl. self', null, 'window', 'direct', 'window20'],
    mkt_recent_sold_share: ['share of the last 20 lots that sold (1 if none)', null, 'window', 'direct', 'window20'],
    riv_purse_max: ['max rival purse / room purse', null, 'rivals', 'direct', 'current'],
    riv_purse_mean: ['mean rival purse / room purse', null, 'rivals', 'direct', 'current'],
    riv_purse_min: ['min rival purse / room purse', null, 'rivals', 'direct', 'current'],
    riv_purse_std: ['std of rival purse / room purse', null, 'rivals', 'direct', 'current'],
    riv_capacity_1: ['highest rival (purse − own-completion cost) / fair value, rivals able to bid', null, 'rivals', 'direct', 'current'],
    riv_capacity_2: ['2nd-highest rival capacity (as above)', null, 'rivals', 'direct', 'current'],
    riv_capacity_3: ['3rd-highest rival capacity (as above)', null, 'rivals', 'direct', 'current'],
    riv_able_share: ['share of rivals whose capacity ≥ fair value', null, 'rivals', 'direct', 'current'],
    riv_fills_share: ['share of rivals for whom the lot fills ANY unmet requirement (keeper/bowling/indians/players pooled)', null, 'rivals', 'direct', 'current'],
    riv_gain_mean: ['mean rival XI gain from the lot / (rating / 11); 0 for rivals who cannot bid', '[0,1] per rival', 'rivals', 'direct', 'current'],
    riv_free_slots_share: ['share of rivals with squad slots left', null, 'rivals', 'direct', 'current'],
    riv_recent_spend: ['rival spend in the last 20 lots / (rivals × purse) × (pool / window) — pooled over all rivals', null, 'window', 'direct', 'window20'],
    riv_xi_mean: ['mean rival best-XI total / 1100', null, 'rivals', 'direct', 'current'],
    riv_xi_max: ['max rival best-XI total / 1100', null, 'rivals', 'direct', 'current']
}

const out = process.argv[2]
const spec = O.OBS_SPEC
const recomputed = specHash(spec)
const checks = {
    obsSpecHashCode: O.OBS_SPEC_HASH, obsSpecHashRecomputed: recomputed, obsSpecHashFrozen: FROZEN_OBS_HASH,
    obsHashOk: O.OBS_SPEC_HASH === FROZEN_OBS_HASH && recomputed === FROZEN_OBS_HASH,
    actSpecHash: RL.ACT_SPEC_HASH, actHashOk: RL.ACT_SPEC_HASH === FROZEN_ACT_HASH,
    size: O.OBS_SIZE, sizeOk: O.OBS_SIZE === 80, blocks: O.OBS_BLOCKS, version: O.OBS_VERSION, clip: O.CLIP, recentWindow: O.RECENT_WINDOW
}
const file = fileURLToPath(new URL('packages/shared/src/rl/obsSpec.js', ROOT))
checks.obsSpecFileSha256 = createHash('sha256').update(readFileSync(file)).digest('hex')
const rec = JSON.parse(readFileSync(fileURLToPath(new URL('ml/reports/phase2e0/frozen-hashes.json', ROOT)), 'utf8'))
checks.obsSpecFileSha256AtPhase2E0 = rec.sources['packages/shared/src/rl/obsSpec.js']
checks.unchangedSincePhase2E0 = checks.obsSpecFileSha256 === checks.obsSpecFileSha256AtPhase2E0 && rec.obsSpecHash === O.OBS_SPEC_HASH
checks.gitDiffVsE31fd21 = execFileSync('git', ['diff', '--stat', 'e31fd21', '--', 'packages/shared/src'], { cwd: fileURLToPath(ROOT), encoding: 'utf8' }).trim() || '(none)'
const names = spec.features.map((f) => f.name)
const missing = names.filter((n) => !A[n])
const extra = Object.keys(A).filter((n) => !names.includes(n))
if (missing.length || extra.length) throw new Error(`annotation drift: missing ${missing} extra ${extra}`)
if (!checks.obsHashOk || !checks.actHashOk || !checks.sizeOk || !checks.unchangedSincePhase2E0) throw new Error(`FROZEN CHECK FAILED ${JSON.stringify(checks)}`)
const features = spec.features.map((f, i) => {
    const [meaning, internalClip, source, opponentChannel, scope] = A[f.name]
    return { index: i, name: f.name, block: f.block, normalisation: f.norm, meaning, internalClip, clip: spec.clip, source, opponentChannel, temporalScope: scope }
})
const count = (k) => features.reduce((m, f) => ({ ...m, [f[k]]: (m[f[k]] ?? 0) + 1 }), {})
writeFileSync(out, JSON.stringify({
    phase: '2E.2', note: 'Read-only transcription of the frozen obs-v2 (packages/shared/src/rl/obsSpec.js). Nothing was modified.',
    environment: 'IplAuctionEnv-v2', checks,
    legend: {
        source: { lot: 'player card', config: 'room constants', own: "seat's own squad/purse", supply: 'catalogue minus auctioned (+ returning unsold)', planner: 'planBid on own ctx', 'planner+rivals': 'planBid terms that count rivals NEEDING a requirement', 'own+supply': 'own purse vs auctioned premium', rivals: 'aggregate over the 9 rival teams (public squads/purses)', window: 'public results of the last 20 lots' },
        opponentChannel: { none: 'fixed by lot/room', own: "only via the seat's own purchases", supply: 'via which players went unsold', direct: 'direct function of rival state or recent public results' },
        temporalScope: { current: 'state now', cumulative: 'whole auction so far (aggregate)', window20: 'last 20 lots, pooled over teams' }
    },
    summary: { bySource: count('source'), byOpponentChannel: count('opponentChannel'), byScope: count('temporalScope'), byBlock: count('block') },
    features
}, null, 1))
console.log('INVENTORY OK', JSON.stringify(checks), JSON.stringify(count('opponentChannel')))
