import { RL_PERSONAS, bidBlocker, nextBidAmount } from '@ipl-auction/shared'
import { agentCap } from '@ipl-auction/shared/sim'
import redis from '../redis/client.js'
import { roomEvents } from '../services/roomEvents.js'
import { placeBid } from '../services/placeBid.js'
import { closeLotNow } from '../room/timerManager.js'
import { loadPlayers } from './playerCache.js'
import { contextBuilder, readAuctionState, rulesOf } from './context.js'
import { getRlPolicy, loadRlPolicy } from './rlPolicy.js'

// ── How bots play a lot ───────────────────────────────────────────────────
// 1. When a lot's timer first starts, every bot decides ONCE the most it will
//    pay (its cap) — rule bots from their hand-written logic, RL bots from
//    the trained policy. Same decision model as the training simulator.
// 2. Whenever the price changes (any bid, human or bot), one bot that isn't
//    leading and whose cap covers the next increment is picked to raise,
//    after a human-like pause. Bots bid fast while the price is far below
//    their cap and hesitate as it approaches it.
// 3. Bots never bid in the last 1.5s of the countdown — no sniping; every
//    bid resets the countdown, so there's always time to respond.
//    Fast-forward: once the human has passed (or can't bid on this lot at
//    all) nobody is waiting to jump in, so the bots race through the rest of
//    the bidding war at ~0.15s a raise — same engine path, every bid still
//    shown in the feed, just without the human-speed pauses.
// 4. Solo auto-close: once no bot will bid again AND the human has either
//    passed, is leading, or can't bid, the lot closes after 1s instead of
//    waiting out the countdown.
//
// Everything for a room runs through one promise chain, so decisions never
// interleave, and a generation counter discards work made stale by a newer event.
//
// Redis cost: team purses and squads only change when a lot is SOLD, so the
// full auction state is read once per lot (when caps are decided) and cached.
// After that, each re-evaluation reads just the current-lot hash — one
// command, instead of re-reading every team on every bid.

const LATE_BID_GUARD_MS = 1500
const AUTO_CLOSE_MS = 1000
const FIRST_LOOK_MS = 700 // extra pause before the opening bid so the human sees the card

let io = null
const rooms = new Map()
const notSolo = new Set()

const randomInt = (lo, hi) => lo + Math.floor(Math.random() * (hi - lo + 1))

// Eager far below the cap, hesitant near it.
const bidDelay = (amount, cap, opening) => {
    const closeness = Math.min(1, amount / Math.max(cap, 1))
    return 300 + Math.round(closeness * closeness * 1800) + randomInt(0, 300) + (opening ? FIRST_LOOK_MS : 0)
}

const FAST_FORWARD_DELAY = () => randomInt(100, 220)

class RoomBots {
    constructor(roomId, { bots, humanPlayerId, humanTeamId }) {
        this.roomId = roomId
        this.bots = bots // [{ botId, teamId, kind, persona }]
        this.humanPlayerId = humanPlayerId
        this.humanTeamId = humanTeamId
        this.lot = null // { slNo, caps, teams, rules, mode } — cached for the life of one lot
        this.humanPassed = false
        this.pending = null
        this.generation = 0
        this.scheduleQueued = false
        this.chain = Promise.resolve()
    }

    // Coalesce bursts: a bid emits timerStarted AND bidPlaced back to back;
    // one re-evaluation covers both.
    requestSchedule() {
        if (this.scheduleQueued) return
        this.scheduleQueued = true
        this.enqueue(() => this.schedule())
    }

    enqueue(task) {
        this.chain = this.chain.then(task).catch((err) =>
            console.error(`[bots] room ${this.roomId}:`, err)
        )
        return this.chain
    }

    cancelPending() {
        if (this.pending) clearTimeout(this.pending)
        this.pending = null
    }

    dispose() {
        this.cancelPending()
        this.generation++
    }

    agentFor(bot) {
        if (bot.kind === 'rule') return { kind: 'rule', persona: bot.persona }
        const persona = RL_PERSONAS[bot.persona]
        const model = getRlPolicy()
        return model
            ? { kind: 'mlp', model, persona: persona.vector, temperature: 0.3 }
            : { kind: 'rule', persona: persona.fallback }
    }

    async decideCaps(state) {
        const players = await loadPlayers()
        const ctxOf = contextBuilder(state, players)
        const caps = new Map()
        for (const bot of this.bots) {
            const ctx = ctxOf(bot.teamId)
            caps.set(bot.teamId, ctx.self && ctx.lot ? agentCap(this.agentFor(bot), ctx) : 0)
        }
        return caps
    }

    // Re-evaluate who (if anyone) raises next. Called on every price or timer change.
    async schedule() {
        this.scheduleQueued = false
        this.cancelPending()
        const gen = ++this.generation

        const current = await redis.hgetall(`room:${this.roomId}:current`)
        if (gen !== this.generation) return
        if (current.timerState !== 'RUNNING' || !current.iplPlayerId) return

        // First look at this lot: read everything once and decide caps.
        // (A new lot always resets this.lot via the nextPlayer/sold events.)
        if (!this.lot || this.lot.slNo !== current.iplPlayerId) {
            const state = await readAuctionState(this.roomId)
            if (gen !== this.generation || state.room.status !== 'active') return
            const caps = await this.decideCaps(state)
            if (gen !== this.generation) return
            this.lot = {
                slNo: current.iplPlayerId,
                caps,
                teams: new Map(state.teams.map((t) => [t.teamId, t])),
                rules: rulesOf(state.room),
                mode: state.room.mode
            }
        }

        const { rules, teams } = this.lot
        const price = Number(current.currentBid)
        const leader = current.currentBidderId || ''
        const amount = nextBidAmount(price, leader !== '')
        const lot = { nationality: current.nationality, currentBidderId: leader }

        // Is the human still in this lot? Decides pacing and auto-close.
        const human = teams.get(this.humanTeamId)
        const humanLeads = leader !== '' && leader === this.humanTeamId
        const humanCanBid = Boolean(human) && !humanLeads && !bidBlocker({ team: human, lot, rules, amount })
        const humanOut = this.humanPassed || (!humanLeads && !humanCanBid)

        const eligible = this.bots.filter((bot) => {
            const cap = this.lot.caps.get(bot.teamId) ?? 0
            const team = teams.get(bot.teamId)
            return bot.teamId !== leader && team && cap >= amount &&
                !bidBlocker({ team, lot, rules, amount })
        })

        if (eligible.length === 0) {
            this.scheduleAutoClose({ current, humanLeads, humanOut, gen })
            return
        }

        const bot = eligible[Math.floor(Math.random() * eligible.length)]
        let delay = humanOut && this.lot.mode === 'solo'
            ? FAST_FORWARD_DELAY()
            : bidDelay(amount, this.lot.caps.get(bot.teamId), leader === '')
        const timeLeft = Number(current.timerEndsAt) * 1000 - Date.now()
        if (timeLeft - delay < LATE_BID_GUARD_MS) {
            if (timeLeft < LATE_BID_GUARD_MS + 400) return
            delay = timeLeft - LATE_BID_GUARD_MS - 200
        }

        this.pending = setTimeout(() => {
            this.pending = null
            this.enqueue(() => this.fire(bot, amount, gen))
        }, delay)
    }

    async fire(bot, amount, gen) {
        if (gen !== this.generation) return
        const result = await placeBid(io, { roomId: this.roomId, teamId: bot.teamId, expectedAmount: amount })
        // On success the bid's own events reschedule everyone. On failure
        // (price moved, timer closed), look again.
        if (!result.success && gen === this.generation) await this.schedule()
    }

    scheduleAutoClose({ current, humanLeads, humanOut, gen }) {
        if (this.lot.mode !== 'solo') return
        if (!humanLeads && !humanOut) return // human still deciding — the countdown is the backstop

        const { iplPlayerId, currentBid: priceAtClose } = current
        this.pending = setTimeout(() => {
            this.pending = null
            this.enqueue(async () => {
                if (gen !== this.generation) return
                const now = await redis.hgetall(`room:${this.roomId}:current`)
                if (now.iplPlayerId !== iplPlayerId || now.currentBid !== priceAtClose || now.timerState !== 'RUNNING') return
                await closeLotNow(io, this.roomId)
            })
        }, AUTO_CLOSE_MS)
    }

    onEvent(event, payload) {
        switch (event) {
            case 'bidPlaced':
                if (payload?.teamId === this.humanTeamId) this.humanPassed = false
                return this.requestSchedule()
            case 'timerStarted':
            case 'timerResumed':
                return this.requestSchedule()
            case 'timerPaused':
                this.cancelPending()
                this.generation++
                return
            case 'auctionStarted':
            case 'nextPlayer':
            case 'playerSold':
            case 'playerUnsold':
            case 'playerSkipped':
                this.cancelPending()
                this.generation++
                this.lot = null
                this.humanPassed = false
                return
            case 'auctionCompleted':
                this.dispose()
                rooms.delete(this.roomId)
                return
            default:
        }
    }

    onHumanPass() {
        this.humanPassed = true
        return this.requestSchedule()
    }
}

// Finds (or rebuilds, e.g. after a restart) the bots for a room. Returns
// null for multiplayer rooms and for solo rooms whose bots aren't seated yet.
// Concurrent calls for the same room share one load, so two events arriving
// together can never create two competing sets of bots.
const loading = new Map()

const ensureRoom = (roomId) => {
    if (rooms.has(roomId)) return Promise.resolve(rooms.get(roomId))
    if (notSolo.has(roomId)) return Promise.resolve(null)
    if (!loading.has(roomId)) {
        loading.set(roomId, loadRoom(roomId).finally(() => loading.delete(roomId)))
    }
    return loading.get(roomId)
}

const loadRoom = async (roomId) => {
    const room = await redis.hgetall(`room:${roomId}`)
    if (!room || room.mode !== 'solo') {
        if (room && Object.keys(room).length) notSolo.add(roomId)
        return null
    }

    const seatsRaw = await redis.hgetall(`room:${roomId}:bots`)
    const bots = Object.entries(seatsRaw || {}).map(([botId, raw]) => ({ botId, ...JSON.parse(raw) }))
    if (bots.length === 0) return null

    const managerData = await redis.hget(`room:${roomId}:players`, room.managerPlayerId)
    const roomBots = new RoomBots(roomId, {
        bots,
        humanPlayerId: room.managerPlayerId,
        humanTeamId: managerData ? managerData.split(':')[2] : null
    })
    rooms.set(roomId, roomBots)
    return roomBots
}

const dispatch = async ({ roomId, event, payload }) => {
    const roomBots = await ensureRoom(roomId)
    if (roomBots) roomBots.onEvent(event, payload)
}

// Solo "Pass" button: the human is done with this lot.
const humanPass = async (roomId, playerId) => {
    const roomBots = await ensureRoom(roomId)
    if (!roomBots) return { ok: false, error: 'Pass is only available in single-player rooms.' }
    if (playerId !== roomBots.humanPlayerId) return { ok: false, error: 'Only the room owner can pass.' }
    roomBots.onHumanPass()
    return { ok: true }
}

const initBots = (socketServer) => {
    io = socketServer
    loadRlPolicy()
    loadPlayers().catch((err) => console.error('[bots] Failed to preload players:', err))
    roomEvents.on('event', (evt) => {
        // Never block or break the engine: defer to the next tick.
        setImmediate(() => dispatch(evt).catch((err) => console.error('[bots] dispatch failed:', err)))
    })
}

export { initBots, humanPass, ensureRoom }
