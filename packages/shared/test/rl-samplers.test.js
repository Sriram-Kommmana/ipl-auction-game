// Phase 2B — deterministic samplers and committed seed manifests (H).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { RL_PERSONAS, RULE_PERSONAS } from '../src/personas.js'
import { createRng } from '../src/sim.js'
import {
    MIN_RULE_OPPONENT_SHARE, PURSE_NORMAL, SPLITS, generateManifest, ruleOpponentShare, sampleEpisode, samplePurse,
    splitOfSeed, validateLeague
} from '../src/rl/index.js'

const N = 12000
const train = Array.from({ length: N }, (_, i) => sampleEpisode(1_000_000 + i))

test('H1. same seed → identical episode; different seeds vary', () => {
    for (const seed of [1_000_000, 1_234_567, 100_123, 200_999]) assert.deepEqual(sampleEpisode(seed), sampleEpisode(seed))
    const distinct = new Set(train.slice(0, 200).map((e) => JSON.stringify({ ...e, seed: 0, split: null })))
    assert.ok(distinct.size > 190, `${distinct.size} distinct configurations in 200 seeds`)
})

test('H2. episode composition: 10 seats = 1 human proxy + 4 frozen rule bots + 5 RL seats, learner in one RL seat', () => {
    for (const e of train.slice(0, 2000)) {
        assert.equal(e.seats.length, 10)
        assert.equal(e.seats.filter((s) => s.type === 'human').length, 1)
        assert.deepEqual(e.seats.filter((s) => s.type === 'rule').map((s) => s.persona).sort(), Object.keys(RULE_PERSONAS).sort())
        const rl = e.seats.filter((s) => s.type === 'learner' || s.type === 'rlFallback')
        assert.deepEqual(rl.map((s) => s.rlSeat).sort(), Object.keys(RL_PERSONAS).sort())
        assert.equal(e.seats[e.learnerSeat].type, 'learner')
        assert.equal(e.seats[e.learnerSeat].rlSeat, e.learnerRlSeat)
        for (const s of e.seats.filter((x) => x.type === 'rlFallback')) assert.equal(s.persona, RL_PERSONAS[s.rlSeat].fallback)
        assert.equal(e.stage, 'A')
    }
})

test('H3. purse strata ≈ ⅓ low / ⅓ normal / ⅓ high; ₹12,500L ≈ ⅙; bounds and ₹100L rounding respected', () => {
    const rng = createRng(3)
    const samples = Array.from({ length: 60000 }, () => samplePurse(rng))
    const share = (f) => samples.filter(f).length / samples.length
    for (const s of ['low', 'normal', 'high']) assert.ok(Math.abs(share((x) => x.stratum === s) - 1 / 3) < 0.01, s)
    assert.ok(Math.abs(share((x) => x.purse === PURSE_NORMAL) - 1 / 6) < 0.01)
    for (const { purse, stratum } of samples) {
        assert.equal(purse % 100, 0)
        const [lo, hi] = { low: [5000, 9000], normal: [9000, 18000], high: [18000, 50000] }[stratum]
        assert.ok(purse >= lo && purse <= hi, `${stratum} ${purse}`)
    }
    // Log-uniform within a stratum: the geometric midpoint splits it in half.
    const high = samples.filter((x) => x.stratum === 'high').map((x) => x.purse)
    assert.ok(Math.abs(high.filter((p) => p < Math.sqrt(18000 * 50000)).length / high.length - 0.5) < 0.02)
    // Episodes use the same distribution.
    assert.ok(Math.abs(train.filter((e) => e.stratum === 'low').length / N - 1 / 3) < 0.02)
})

test('H4. learner seat is balanced over the 10 positions and the 5 RL seats', () => {
    for (let seat = 0; seat < 10; seat++) {
        const share = train.filter((e) => e.learnerSeat === seat).length / N
        assert.ok(Math.abs(share - 0.1) < 0.015, `position ${seat}: ${share}`)
    }
    for (const id of Object.keys(RL_PERSONAS)) {
        const share = train.filter((e) => e.learnerRlSeat === id).length / N
        assert.ok(Math.abs(share - 0.2) < 0.02, `${id}: ${share}`)
    }
})

test('H5. human proxy: passive 40% / noisy 40% / star-chasing 20%, noise in [0.1, 0.5]', () => {
    const humans = train.map((e) => e.seats.find((s) => s.type === 'human'))
    const share = (p) => humans.filter((h) => h.proxy === p).length / N
    assert.ok(Math.abs(share('passive') - 0.4) < 0.02)
    assert.ok(Math.abs(share('noisy') - 0.4) < 0.02)
    assert.ok(Math.abs(share('starChaser') - 0.2) < 0.02)
    for (const h of humans.filter((x) => x.proxy !== 'passive')) {
        assert.ok(h.noise >= 0.1 && h.noise <= 0.5)
        assert.ok(Object.keys(RULE_PERSONAS).includes(h.persona))
    }
    assert.ok(humans.filter((h) => h.proxy === 'starChaser').every((h) => h.persona === 'starChaser'))
})

test('H6. league (Stage B): snapshots only replace RL-fallback seats; frozen rule bots stay ≥ 40% of opponents', () => {
    for (const share of [0, 0.5, 1]) {
        let snaps = 0
        for (let i = 0; i < 500; i++) {
            const e = sampleEpisode(1_500_000 + i, { league: { snapshotShare: share, poolSize: 6 } })
            assert.equal(e.stage, 'B')
            assert.ok(ruleOpponentShare(e) >= MIN_RULE_OPPONENT_SHARE, `share ${ruleOpponentShare(e)}`)
            assert.equal(e.seats.filter((s) => s.type === 'rule').length, 4)
            for (const s of e.seats.filter((x) => x.type === 'rlSnapshot')) assert.ok(s.snapshot >= 0 && s.snapshot < 6)
            snaps += e.seats.filter((s) => s.type === 'rlSnapshot').length
        }
        if (share === 0) assert.equal(snaps, 0)
        if (share === 1) assert.equal(snaps, 500 * 4)
    }
    assert.throws(() => validateLeague({ snapshotShare: 1.5, poolSize: 2 }))
    assert.throws(() => validateLeague({ snapshotShare: 0.5, poolSize: -1 }))
})

test('H7. split ranges: validation 100,000–100,499, test 200,000–200,999, train ≥ 1,000,000, < 100,000 reserved', () => {
    assert.equal(splitOfSeed(100000), 'validation')
    assert.equal(splitOfSeed(100499), 'validation')
    assert.equal(splitOfSeed(100500), null)
    assert.equal(splitOfSeed(200000), 'test')
    assert.equal(splitOfSeed(200999), 'test')
    assert.equal(splitOfSeed(1_000_000), 'train')
    assert.equal(splitOfSeed(99_999), 'regression')
    assert.deepEqual([SPLITS.validation.count, SPLITS.test.count], [500, 1000])
})

test('H8. committed manifests equal the sampler exactly (no drift) and cover their full splits', () => {
    for (const [split, count] of [['validation', 500], ['test', 1000], ['train', 2000]]) {
        const committed = JSON.parse(readFileSync(fileURLToPath(new URL(`../data/rl-manifests/${split}.json`, import.meta.url)), 'utf8'))
        assert.equal(committed.format, 'rl-manifest-v2')
        assert.equal(committed.entries.length, count)
        assert.deepEqual(committed.entries, generateManifest(split, count).entries, `${split} drifted`)
        assert.ok(committed.entries.every((e) => e.split === split))
    }
})
