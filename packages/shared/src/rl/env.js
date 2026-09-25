// IplAuctionEnv-v2 core (JavaScript — the only implementation of the game).
// The Python Gymnasium environment drives this through bin/rl-bridge-v2.js;
// the Node evaluator drives it directly.
//
// One episode = one complete Full Pool auction (main round + re-auction),
// run by the unchanged AuctionSim. One learner seat; nine frozen seats from
// the episode entry (samplers.js):
//   rule / rlFallback  → frozen rule bot (agentCap, exactly as in production)
//   human passive      → never bids
//   human noisy / star → frozen rule persona with the entry's noise
//   rlSnapshot         → frozen rl-policy-v2 through the canonical runtime,
//                        with `tremble` probability of a random LEGAL action
// A step is a lot where the learner has at least one legal bid; every
// other lot is resolved with the learner passing.
//
// Random streams: the simulator and opponents share createRng(seed) (as in
// AuctionSim.playAuction); the learner seat has its own stream, so what the
// learner does never shifts opponents' random draws — paired comparisons
// between learners stay paired until their choices differ.

import { AuctionSim, agentCap, createRng } from '../sim.js'
import { DEFAULT_RULES } from '../rules.js'
import { planBid } from '../planning.js'
import { fairValue } from '../valuation.js'
import { selectBestXI } from '../scoring.js'
import { RL_PERSONAS } from '../personas.js'
import { deriveSeed } from './hash.js'
import { RECENT_WINDOW, buildRlObservation } from './obsSpec.js'
import { PASS } from './actionSpec.js'
import { hasBidAction, rlActionMask } from './mask.js'
import { stepReward, terminalReward, xiPotential } from './reward.js'
import { createRlSeat } from './runtime.js'

const STAR_RATING = 90
const MARGINAL = [0.05, 0.5]

export class RlEpisode {
    // entry: a manifest / sampleEpisode entry. snapshots: rl-policy-v2 objects
    // for 'rlSnapshot' seats (entry.seats[i].snapshot indexes this list).
    constructor({ players, entry, snapshots = [], tremble = 0.01 }) {
        this.entry = entry
        this.rules = { ...DEFAULT_RULES, pursePerTeam: entry.purse }
        this.rng = createRng(entry.seed)
        this.learnerRng = createRng(deriveSeed(entry.seed, 'learner'))
        this.sim = new AuctionSim({ players, teamCount: entry.seats.length, rules: this.rules, rng: this.rng })
        this.players = this.sim.players
        this.learner = entry.learnerSeat
        this.tremble = tremble
        this.seats = entry.seats.map((seat, i) => this.#opponent(seat, i, snapshots))
        this.pending = null
        this.done = false
        this.stats = { decisions: 0, shieldActivations: 0, shieldWins: 0, buys: [], contests: 0, lots: 0, reauctionLots: 0, rewardSum: 0 }
    }

    #opponent(seat, i, snapshots) {
        if (i === this.learner) return null
        switch (seat.type) {
            case 'rule':
            case 'rlFallback':
                return { kind: 'rule', agent: { kind: 'rule', persona: seat.persona } }
            case 'human':
                return seat.proxy === 'passive'
                    ? { kind: 'passive' }
                    : { kind: 'rule', agent: { kind: 'rule', persona: seat.persona, noise: seat.noise } }
            case 'rlSnapshot': {
                const policy = snapshots[seat.snapshot]
                if (!policy) throw new Error(`seat ${i}: snapshot ${seat.snapshot} not loaded`)
                return { kind: 'rl', runtime: createRlSeat({ policy, fallbackPersona: RL_PERSONAS[seat.rlSeat].fallback }) }
            }
            default:
                throw new Error(`seat ${i}: unknown seat type ${seat.type}`)
        }
    }

    // Public results of the last RECENT_WINDOW lots, for the observation.
    extras() {
        const P = this.rules.pursePerTeam
        return {
            poolSize: this.sim.mainLength,
            recent: this.sim.history.slice(-RECENT_WINDOW).map((h) => ({
                sold: h.winner !== null,
                price: h.price,
                fairValue: fairValue(this.players.get(h.slNo), P),
                winnerTeamId: h.winner === null ? null : this.sim.teams[h.winner].teamId
            }))
        }
    }

    #opponentCap(i, ctx, extras) {
        const seat = this.seats[i]
        if (seat.kind === 'passive') return 0
        if (seat.kind === 'rule') return agentCap(seat.agent, ctx, this.rng)
        // Frozen RL snapshot: occasionally a random legal action (trembling).
        if (this.tremble > 0 && this.rng() < this.tremble) {
            const m = rlActionMask(ctx)
            const legal = m.mask.flatMap((ok, a) => (ok ? [a] : []))
            return m.caps[legal[Math.floor(this.rng() * legal.length)]]
        }
        return seat.runtime.decide(ctx, extras, this.rng).cap
    }

    // Resolve the lot on the block with the learner's cap. Returns the reward.
    #resolve(learnerCap) {
        const sim = this.sim
        const lot = sim.currentLot()
        const phase = sim.phase
        const extras = this.extras()
        const ctxs = this.seats.map((_, i) => sim.contextFor(i))
        const caps = this.seats.map((_, i) => (i === this.learner ? learnerCap : this.#opponentCap(i, ctxs[i], extras)))
        const squad = sim.teams[this.learner].squad
        const before = xiPotential(squad)
        const gainAtOpen = this.pending?.plan.playerImpact.xiGain ?? 0
        const out = sim.resolveLot(caps)
        const won = out.winner === this.learner
        this.stats.lots++
        if (phase === 'reauction') this.stats.reauctionLots++
        if (learnerCap >= lot.basePrice && caps.some((c, i) => i !== this.learner && c >= lot.basePrice)) this.stats.contests++
        if (won) this.stats.buys.push({ slNo: lot.slNo, phase, price: out.price, rating: lot.rating, gain: gainAtOpen, shield: Boolean(this.pending?.mask.shieldActive) })
        return { out, won, lot, phase, reward: stepReward(before, squad) }
    }

    // Advance to the learner's next real decision; lots in between are
    // resolved with the learner passing. Returns the reward they produced (0).
    #advance() {
        let reward = 0
        this.pending = null
        while (!this.sim.done) {
            const ctx = this.sim.contextFor(this.learner)
            const plan = planBid(ctx)
            const mask = rlActionMask(ctx, plan)
            if (hasBidAction(mask.mask)) {
                const obs = buildRlObservation(ctx, this.extras(), plan)
                this.pending = { ctx, plan, mask, obs }
                this.stats.decisions++
                if (mask.shieldActive) this.stats.shieldActivations++
                return reward
            }
            reward += this.#resolve(0).reward
        }
        this.done = true
        return reward
    }

    reset() {
        this.#advance()
        if (!this.pending) throw new Error(`seed ${this.entry.seed}: the learner never has a decision`)
        return { obs: this.pending.obs, mask: this.pending.mask.mask, info: this.#info(null) }
    }

    // Learner acts with an act-v2 action.
    step(action) {
        if (!this.pending) throw new Error(this.done ? 'episode is over — call reset' : 'call reset before step')
        if (!this.pending.mask.mask[action]) throw new Error(`action ${action} is masked`)
        return this.#finish(action, this.pending.mask.caps[action])
    }

    // Evaluation only: the learner seat acts with a raw cap (a frozen rule
    // bot sitting in the learner's chair). Not used in training.
    stepCap(cap) {
        if (!this.pending) throw new Error('no pending decision')
        return this.#finish(null, cap)
    }

    #finish(action, cap) {
        const shield = this.pending.mask.shieldActive
        const r = this.#resolve(cap)
        if (shield && r.won) this.stats.shieldWins++
        let reward = r.reward
        const lotInfo = { slNo: r.lot.slNo, phase: r.phase, action, cap, won: r.won, price: r.won ? r.out.price : null, soldFor: r.out.price, xiReward: r.reward }
        reward += this.#advance()
        let terminal = 0
        if (this.done) {
            terminal = terminalReward(this.sim.teams[this.learner].squad)
            reward += terminal
        }
        this.stats.rewardSum += reward
        return {
            obs: this.pending ? this.pending.obs : null,
            mask: this.pending ? this.pending.mask.mask : null,
            reward,
            done: this.done,
            info: { ...this.#info(lotInfo), terminalReward: terminal, ...(this.done ? { episode: this.summary() } : {}) }
        }
    }

    #info(lastLot) {
        return {
            seed: this.entry.seed,
            decision: this.stats.decisions,
            shieldActive: Boolean(this.pending?.mask.shieldActive),
            phase: this.sim.done ? 'done' : this.sim.phase,
            lastLot
        }
    }

    // Learner-seat outcome (reported when the episode ends).
    summary() {
        const team = this.sim.teams[this.learner]
        const xi = selectBestXI(team.squad)
        const totals = this.sim.teams.map((t) => selectBestXI(t.squad).total)
        const spent = team.purseSpent
        const gained = this.stats.buys.reduce((s, b) => s + Math.max(0, b.gain), 0)
        return {
            seed: this.entry.seed,
            purse: this.rules.pursePerTeam,
            stratum: this.entry.stratum,
            xi: xi.total / 11,
            xiTotal: xi.total,
            strength: xi.strength,
            emptySlots: xi.emptySlots,
            legalXI: xi.emptySlots === 0,
            strongXI: xi.emptySlots === 0 && xi.strength >= 85,
            rank: 1 + totals.filter((t) => t > xi.total).length,
            purseLeft: team.purseLeft,
            purseLeftShare: team.purseLeft / this.rules.pursePerTeam,
            squadSize: team.playerCount,
            overseas: team.overseasCount,
            buys: this.stats.buys.length,
            stars: this.stats.buys.filter((b) => b.rating >= STAR_RATING).length,
            marginalBuys: this.stats.buys.filter((b) => b.gain > MARGINAL[0] && b.gain < MARGINAL[1]).length,
            reauctionBuys: this.stats.buys.filter((b) => b.phase === 'reauction').length,
            xiGainPer1000: spent > 0 ? (1000 * gained) / spent : 0,
            contests: this.stats.contests,
            decisions: this.stats.decisions,
            shieldActivations: this.stats.shieldActivations,
            shieldWins: this.stats.shieldWins,
            lots: this.stats.lots,
            reauctionLots: this.stats.reauctionLots,
            return: this.stats.rewardSum
        }
    }
}

export { PASS }
