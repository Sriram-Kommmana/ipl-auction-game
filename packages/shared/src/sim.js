// Headless auction — the same rules as the live server with no sockets,
// Redis or timers. Used to train the RL bots (through ml/ via
// bin/rl-bridge.js) and to test the bots end to end.
//
// Each lot is resolved as a bidding ladder. Every team has a cap (decided
// when the lot opened). While any non-leading team can afford the next
// increment within its cap, one of them — picked at random, just as in the
// live game whichever bot's delay fires first — raises by one increment.
// That is exactly what the live runtime does, minus the waiting.

import { DEFAULT_RULES, bidBlocker, nextBidAmount } from './rules.js'
import { buildAuctionPool, shuffle } from './pool.js'
import { ACTION_COUNT, actionMask, buildObservation, capForAction, deriveLotFacts } from './observation.js'
import { ruleBotCap } from './ruleBots.js'
import { forward, sampleAction } from './mlp.js'

// Small seedable PRNG (mulberry32) so a training episode or a failing test
// can be replayed exactly.
export const createRng = (seed = Date.now()) => {
    let a = seed >>> 0
    return () => {
        a = (a + 0x6d2b79f5) >>> 0
        let t = a
        t = Math.imul(t ^ (t >>> 15), t | 1)
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    }
}

export class AuctionSim {
    constructor({ players, teamCount = 10, rules = DEFAULT_RULES, rng = Math.random, teamIds = null }) {
        this.rules = rules
        this.rng = rng
        this.players = new Map(players.map((p) => [p.slNo, p]))
        this.pool = buildAuctionPool(players, { rng })
        this.mainLength = this.pool.length
        this.phase = 'main'
        this.index = 0
        this.unsold = []
        this.done = this.pool.length === 0
        this.history = []
        this.teams = Array.from({ length: teamCount }, (_, i) => ({
            teamId: teamIds?.[i] ?? `T${i}`,
            purseLeft: rules.pursePerTeam,
            purseSpent: 0,
            playerCount: 0,
            overseasCount: 0,
            squad: []
        }))
    }

    currentLot() {
        return this.done ? null : this.players.get(this.pool[this.index])
    }

    progress() {
        return this.phase === 'main' ? this.index / this.mainLength : 1
    }

    contextFor(teamIndex) {
        const lot = this.currentLot()
        return {
            rules: this.rules,
            lot,
            self: this.teams[teamIndex],
            rivals: this.teams.filter((_, i) => i !== teamIndex),
            upcoming: this.pool.slice(this.index + 1).map((slNo) => this.players.get(slNo)),
            progress: this.progress()
        }
    }

    // caps[i] = most team i will pay (0 = out). Returns what happened.
    resolveLot(caps) {
        const lot = this.currentLot()
        let price = lot.basePrice
        let leader = -1
        let bids = 0

        for (;;) {
            const amount = nextBidAmount(price, leader !== -1)
            const eligible = []
            for (let i = 0; i < this.teams.length; i++) {
                if (i === leader || !(caps[i] >= amount)) continue
                const blocked = bidBlocker({ team: this.teams[i], lot: { ...lot, currentBidderId: '' }, rules: this.rules, amount })
                if (!blocked) eligible.push(i)
            }
            if (eligible.length === 0) break
            leader = eligible[Math.floor(this.rng() * eligible.length)]
            price = amount
            bids++
        }

        const outcome = { slNo: lot.slNo, winner: leader === -1 ? null : leader, price: leader === -1 ? null : price, bids }

        if (leader !== -1) {
            const team = this.teams[leader]
            team.purseLeft -= price
            team.purseSpent += price
            team.playerCount += 1
            if (lot.nationality === 'Overseas') team.overseasCount += 1
            team.squad.push(lot)
        } else {
            this.unsold.push(lot.slNo)
        }

        this.history.push({ ...outcome, phase: this.phase })
        this.advance()
        return outcome
    }

    advance() {
        this.index++
        if (this.index < this.pool.length) return
        if (this.phase === 'main' && this.unsold.length > 0) {
            this.pool = shuffle(this.unsold, this.rng)
            this.unsold = []
            this.phase = 'reauction'
            this.index = 0
            return
        }
        this.done = true
    }
}

// ── Agents ────────────────────────────────────────────────────────────────
// An agent is plain data:
//   { kind: 'rule', persona: 'moneyball', noise? }
//   { kind: 'mlp',  model, persona: [5 numbers], temperature? }
// `tremble` (0..1) makes it play a random legal action that often —
// d'Eon et al. 2024 found this keeps self-play policies from overfitting
// to exactly one set of opponents.

export const agentCap = (agent, ctx, rng = Math.random, { tremble = 0 } = {}) => {
    const facts = deriveLotFacts(ctx)
    if (!facts.eligible) return 0

    if (tremble > 0 && rng() < tremble) {
        const legal = actionMask(ctx, facts).flatMap((m, a) => (m ? [a] : []))
        return capForAction(ctx, legal[Math.floor(rng() * legal.length)], facts)
    }

    if (agent.kind === 'rule') {
        return ruleBotCap(agent.persona, ctx, rng, agent.noise !== undefined ? { noise: agent.noise } : undefined)
    }
    if (agent.kind === 'mlp') {
        const obs = buildObservation(ctx, agent.persona)
        const logits = forward(agent.model, obs)
        const action = sampleAction(logits, actionMask(ctx, facts), { temperature: agent.temperature ?? 0.3, rng })
        return capForAction(ctx, action, facts)
    }
    throw new Error(`Unknown agent kind: ${agent.kind}`)
}

// Play a whole auction with every seat controlled by an agent.
export const playAuction = ({ players, agents, rules = DEFAULT_RULES, seed = 1, tremble = 0 }) => {
    const rng = createRng(seed)
    const sim = new AuctionSim({ players, teamCount: agents.length, rules, rng })
    const capsLog = []
    while (!sim.done) {
        const caps = agents.map((agent, i) => agentCap(agent, sim.contextFor(i), rng, { tremble }))
        const outcome = sim.resolveLot(caps)
        capsLog.push({ ...outcome, caps })
    }
    return { sim, capsLog }
}

export { ACTION_COUNT }
