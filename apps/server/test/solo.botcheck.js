// Live rule-bot validation against a RUNNING server (pnpm --filter server start),
// run from apps/server so it shares the server's .env (Redis + MongoDB).
//
//   node test/solo.botcheck.js --human passive   → the human passes every lot
//   node test/solo.botcheck.js --human mixed     → early bids, late bids, star bidding wars
//   node test/solo.botcheck.js --lots 40         → stop after 40 lots (default: whole auction)
//
// For every lot it compares two independent views of the same game:
//   server — contextBuilder() over the live Redis state and MongoDB players
//            (exactly what botManager decides from);
//   sim    — the training simulator's AuctionSim, replaying the live results
//            (players from players.csv).
// Contexts must match; every bot's planner output and noise-free cap must
// match. Each bot's actual live bids must stay inside the band its cap can
// take (±8% noise, never above maxSafeBid), every increment must be exact,
// no team may outbid itself, and when a lot closes no bot may be left that
// was certain to raise. At the end the Redis state is checked against the
// replay and the persisted results.

import { io } from 'socket.io-client'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import mongoose from 'mongoose'
import redis from '../src/redis/client.js'
import connectMongoDB from '../src/db/client.js'
import { loadPlayers } from '../src/bots/playerCache.js'
import { contextBuilder, readAuctionState } from '../src/bots/context.js'
import { RL_PERSONAS, bidBlocker, fairValue, nextBidAmount, planBid, ruleBotCap, selectBestXI } from '@ipl-auction/shared'
import { parsePlayersCsv } from '@ipl-auction/shared/players-csv'
import { AuctionSim } from '@ipl-auction/shared/sim'

const SERVER = process.env.SERVER_URL || 'http://localhost:3001'
const args = process.argv.slice(2)
const HUMAN = args.includes('--human') ? args[args.indexOf('--human') + 1] : 'passive'
const LOTS = args.includes('--lots') ? Number(args[args.indexOf('--lots') + 1]) : Infinity
const HUMAN_TEAM = args.includes('--team') ? args[args.indexOf('--team') + 1] : 'MI'

const failures = []
const check = (ok, message) => {
    if (!ok) {
        failures.push(message)
        if (failures.length <= 60) console.log(`  ✖ ${message}`)
    }
}
const stats = { lots: 0, ctxCompared: 0, decisionsCompared: 0, botBids: 0, humanBids: 0, humanErrors: 0, sold: 0, unsold: 0, reLots: 0 }
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const post = async (path, body) => {
    const res = await fetch(`${SERVER}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    const json = await res.json()
    if (!res.ok) throw new Error(`${path} → ${res.status}: ${json.message}`)
    return json.data
}
const once = (socket, event, ms = 15000) => new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`timed out waiting for ${event}`)), ms)
    socket.once(event, (p) => { clearTimeout(t); resolve(p) })
})

// ── Player data: MongoDB (server) vs players.csv (simulator) ──────────────
await connectMongoDB()
const mongoPlayers = await loadPlayers()
const csvPlayers = parsePlayersCsv(readFileSync(fileURLToPath(new URL('../src/db/players.csv', import.meta.url)), 'utf8'))
check(mongoPlayers.size === csvPlayers.length, `MongoDB has ${mongoPlayers.size} players, players.csv ${csvPlayers.length}`)
for (const p of csvPlayers) {
    const m = mongoPlayers.get(p.slNo)
    for (const k of ['playerName', 'role', 'nationality', 'basePrice', 'rating', 'setNo']) {
        check(m && m[k] === p[k], `player ${p.slNo} ${k}: mongo ${m?.[k]} vs csv ${p[k]}`)
    }
}

// ── Room ──────────────────────────────────────────────────────────────────
const room = await post('/room/create', { managerNickname: `BotCheck-${HUMAN}`, managerPin: '1234', mode: 'solo' })
const roomId = room.roomId
console.log(`● room ${roomId}, human ${HUMAN} on ${HUMAN_TEAM}`)
const socket = io(SERVER, { transports: ['websocket'] })
const synced = once(socket, 'stateSync')
socket.on('connect', () => socket.emit('reconnect', { playerId: room.managerId }))
await synced
socket.emit('selectTeam', { playerId: room.managerId, teamId: HUMAN_TEAM })
await once(socket, 'teamSelected')
const serverErrors = []
for (const e of ['startAuctionError', 'passError', 'pauseError', 'resumeError', 'auctionError']) socket.on(e, (p) => serverErrors.push(`${e}: ${p.message}`))
// A human bid can legitimately lose the race with the lot closing ("Bidding is
// not currently open") — counted, not a failure. Anything else is.
socket.on('bidError', (p) => {
    stats.humanErrors++
    if (!/not currently open/i.test(p.message)) serverErrors.push(`bidError: ${p.message}`)
})
const started = once(socket, 'auctionStarted')
socket.emit('startAuction', { playerId: room.managerId })
await started

const seatsRaw = await redis.hgetall(`room:${roomId}:bots`)
const seats = Object.values(seatsRaw).map((raw) => JSON.parse(raw)).map((s) => ({
    ...s, persona: s.kind === 'rule' ? s.persona : RL_PERSONAS[s.persona].fallback
}))
check(seats.length === 9, `9 bots seated (${seats.length})`)
console.log(`  seats: ${seats.map((s) => `${s.teamId}=${s.persona}${s.kind === 'rl' ? '(rl fallback)' : ''}`).join(', ')}`)
const roomHash = await redis.hgetall(`room:${roomId}`)
const rules = { pursePerTeam: Number(roomHash.pursePerTeam), maxPlayers: Number(roomHash.maxPlayers), maxOverseas: Number(roomHash.maxOverseas) }
const teamIds = Object.keys(await redis.hgetall(`room:${roomId}:teams`)).sort()

// Simulator replay of the live game.
const sim = new AuctionSim({ players: csvPlayers, teamCount: teamIds.length, rules, teamIds, rng: () => 0.5 })
const livePool = (await redis.lrange(`room:${roomId}:pool`, 0, -1)).map(Number)
check(livePool.length === csvPlayers.length && new Set(livePool).size === livePool.length, `main pool: every player exactly once (${livePool.length})`)
const setOrder = livePool.map((sl) => mongoPlayers.get(sl).setNo)
check(setOrder.every((s, i) => i === 0 || s >= setOrder[i - 1]), 'main pool is in set order')
sim.pool = [...livePool]
sim.mainLength = livePool.length
const idx = (teamId) => teamIds.indexOf(teamId)

// Bands a rule bot's cap can fall in for a context: [−8% noise, +8% noise].
const band = (persona, ctx, plan) => ({
    low: ruleBotCap(persona, ctx, () => 0, { plan }),
    high: ruleBotCap(persona, ctx, () => 0.999999, { plan }),
    exact: ruleBotCap(persona, ctx, Math.random, { noise: 0, plan })
})
const slNos = (list) => list.map((p) => Number(p.slNo)).join(',')
const teamSig = (t) => `${t.teamId}:${t.purseLeft}:${t.playerCount}:${t.overseasCount}:${t.squad.map((p) => Number(p.slNo)).sort((a, b) => a - b).join('.')}`

// ── Per-lot bookkeeping ───────────────────────────────────────────────────
let lot = null // { slNo, phase, bands: Map(teamId → band), teams: Map(teamId → team view), bids: [], closed }
let queue = Promise.resolve()
const serial = (fn) => { queue = queue.then(fn).catch((e) => check(false, `checker error: ${e.stack}`)) }
let reauctionSynced = false

const humanDecision = (lotPlayer, humanTeam) => {
    if (HUMAN === 'passive') return 'pass'
    if (humanTeam.playerCount >= 18 || humanTeam.purseLeft < 2500) return 'pass'
    const r = Math.random()
    if (lotPlayer.rating >= 90 && r < 0.5) return 'war'
    if (r < 0.08) return 'early'
    if (r < 0.14) return 'late'
    return 'pass'
}

const openLot = async () => {
    const state = await readAuctionState(roomId)
    const slNo = Number(state.current.iplPlayerId)
    if (lot && lot.slNo === slNo && !lot.closed) return
    const phase = state.room.auctionPhase === 'main' ? 'main' : 'reauction'
    if (phase === 'reauction' && !reauctionSynced) {
        reauctionSynced = true
        const livePoolRe = state.pool.map(Number)
        const expectRe = [...sim.pool].sort((a, b) => a - b).join()
        check([...livePoolRe].sort((a, b) => a - b).join() === expectRe, `re-auction pool = main-round unsold (${livePoolRe.length} vs ${sim.pool.length})`)
        sim.pool = livePoolRe
        sim.index = 0
    }
    check(sim.currentLot()?.slNo === slNo, `sim replay is on lot ${sim.currentLot()?.slNo}, server on ${slNo}`)
    check(Number(state.room.currentPlayerIndex) === sim.index, `index ${state.room.currentPlayerIndex} vs replay ${sim.index}`)
    const serverCtxOf = contextBuilder(state, mongoPlayers)
    lot = { slNo, phase, player: mongoPlayers.get(slNo), bands: new Map(), teams: new Map(), bids: [], closed: false, openedAt: Date.now() }
    stats.lots++
    if (phase === 'reauction') stats.reLots++
    for (const teamId of teamIds) {
        const s = serverCtxOf(teamId)
        const m = sim.contextFor(idx(teamId))
        lot.teams.set(teamId, s.self)
        // Context equality.
        check(Number(s.lot.slNo) === m.lot.slNo, `lot ${slNo}: lot differs`)
        check(teamSig(s.self) === teamSig(m.self), `lot ${slNo} ${teamId}: self differs\n    server ${teamSig(s.self)}\n    sim    ${teamSig(m.self)}`)
        check(s.rivals.map(teamSig).sort().join('|') === m.rivals.map(teamSig).sort().join('|'), `lot ${slNo} ${teamId}: rivals differ`)
        check(slNos(s.upcoming) === slNos(m.upcoming), `lot ${slNo}: upcoming differs (${s.upcoming.length} vs ${m.upcoming.length})`)
        check(slNos(s.returning) === slNos(m.returning), `lot ${slNo}: returning differs (${s.returning.length} vs ${m.returning.length})`)
        check(s.phase === m.phase && Math.abs(s.progress - m.progress) < 1e-9, `lot ${slNo}: phase/progress ${s.phase}/${s.progress} vs ${m.phase}/${m.progress}`)
        stats.ctxCompared++
        const seat = seats.find((b) => b.teamId === teamId)
        if (!seat) continue
        const ps = planBid(s)
        const pm = planBid(m)
        const bs = band(seat.persona, s, ps)
        const bm = band(seat.persona, m, pm)
        check(ps.allowed === pm.allowed && ps.budget.maxSafeBid === pm.budget.maxSafeBid, `lot ${slNo} ${teamId}: planner differs (${ps.budget.maxSafeBid} vs ${pm.budget.maxSafeBid})`)
        check(bs.low === bm.low && bs.high === bm.high && bs.exact === bm.exact, `lot ${slNo} ${teamId} ${seat.persona}: decision differs server ${JSON.stringify(bs)} vs sim ${JSON.stringify(bm)}`)
        check(bs.high <= ps.budget.maxSafeBid && bs.high <= s.self.purseLeft, `lot ${slNo} ${teamId}: cap band above safe maximum`)
        lot.bands.set(teamId, bs)
        stats.decisionsCompared++
    }
    // Human behaviour for this lot.
    lot.humanPlan = humanDecision(lot.player, lot.teams.get(HUMAN_TEAM))
    lot.humanLimit = Math.floor(fairValue(lot.player, rules.pursePerTeam) * (lot.humanPlan === 'war' ? 1.2 : 1))
}

const humanCanBid = (amount) => {
    const t = lot.teams.get(HUMAN_TEAM)
    const leader = lot.bids.at(-1)?.teamId ?? ''
    return leader !== HUMAN_TEAM && amount <= lot.humanLimit && amount <= t.purseLeft - 1500 &&
        !bidBlocker({ team: t, lot: { nationality: lot.player.nationality, currentBidderId: leader }, rules, amount })
}
const humanAct = (timerEndsAt) => {
    const leader = lot.bids.at(-1)
    if (leader?.teamId === HUMAN_TEAM) return
    const amount = nextBidAmount(leader ? leader.newBid : lot.player.basePrice, Boolean(leader))
    const plan = lot.humanPlan
    // One pending human bid at a time, re-checked when it fires: the price may
    // have moved or the lot closed in between.
    const bid = () => {
        if (lot.humanPending) return
        lot.humanPending = true
        const forLot = lot.slNo
        serial(async () => {
            lot.humanPending = false
            const lead = lot.bids.at(-1)
            if (lot.closed || lot.slNo !== forLot || lead?.teamId === HUMAN_TEAM) return
            if (nextBidAmount(lead ? lead.newBid : lot.player.basePrice, Boolean(lead)) !== amount || !humanCanBid(amount)) return pass()
            stats.humanBids++
            socket.emit('placeBid', { playerId: room.managerId })
        })
    }
    const pass = () => socket.emit('passLot', { playerId: room.managerId })
    if (plan === 'pass' || !humanCanBid(amount)) return pass()
    if (plan === 'early') { lot.humanPlan = 'pass'; return bid() }
    if (plan === 'war') return setTimeout(bid, 300 + Math.random() * 500)
    if (plan === 'late') {
        lot.humanPlan = 'pass'
        const wait = timerEndsAt * 1000 - Date.now() - 2500
        return setTimeout(bid, Math.max(0, wait))
    }
}

socket.on('timerStarted', ({ timerEndsAt }) => serial(async () => {
    await openLot()
    humanAct(timerEndsAt)
}))

socket.on('bidPlaced', ({ teamId, newBid }) => serial(async () => {
    if (!lot || lot.closed) return check(false, `bid by ${teamId} at ${newBid} outside an open lot`)
    const prev = lot.bids.at(-1)
    check(!prev || prev.teamId !== teamId, `lot ${lot.slNo}: ${teamId} outbid itself`)
    const expected = nextBidAmount(prev ? prev.newBid : lot.player.basePrice, Boolean(prev))
    check(newBid === expected, `lot ${lot.slNo}: bid ${newBid}, expected increment ${expected}`)
    const t = lot.teams.get(teamId)
    check(!bidBlocker({ team: t, lot: { nationality: lot.player.nationality, currentBidderId: prev?.teamId ?? '' }, rules, amount: newBid }), `lot ${lot.slNo}: ${teamId} made an illegal bid`)
    const b = lot.bands.get(teamId)
    if (b) {
        stats.botBids++
        check(newBid <= b.high, `lot ${lot.slNo}: bot ${teamId} bid ${newBid} above its highest possible cap ${b.high} (stale/unsafe decision)`)
    }
    lot.bids.push({ teamId, newBid, at: Date.now() })
}))

const closeLot = (outcome) => serial(async () => {
    check(lot && !lot.closed && Number(outcome.iplPlayerId) === lot.slNo, `close event for ${outcome.iplPlayerId} but open lot is ${lot?.slNo}`)
    lot.closed = true
    const last = lot.bids.at(-1)
    const lead = last?.teamId ?? ''
    const price = last ? last.newBid : null
    if (outcome.sold) {
        stats.sold++
        check(last && outcome.soldTo === last.teamId && outcome.soldFor === last.newBid, `lot ${lot.slNo}: sold to ${outcome.soldTo} at ${outcome.soldFor}, last bid ${lead} at ${price}`)
    } else {
        stats.unsold++
        check(!last, `lot ${lot.slNo}: unsold despite bids`)
    }
    // No bot may be left that was certain to raise.
    const next = last ? nextBidAmount(price, true) : lot.player.basePrice
    for (const [teamId, b] of lot.bands) {
        if (teamId === lead) continue
        const t = lot.teams.get(teamId)
        const blocked = bidBlocker({ team: t, lot: { nationality: lot.player.nationality, currentBidderId: lead }, rules, amount: next })
        check(blocked || b.low < next, `lot ${lot.slNo}: bot ${teamId} (cap ≥ ${b.low}) never raised to ${next} before the lot closed`)
    }
    // Replay the result in the simulator (same mutations as AuctionSim.resolveLot).
    const w = outcome.sold ? idx(outcome.soldTo) : -1
    const p = sim.currentLot()
    if (w >= 0) {
        const tm = sim.teams[w]
        tm.purseLeft -= outcome.soldFor
        tm.purseSpent += outcome.soldFor
        tm.playerCount++
        if (p.nationality === 'Overseas') tm.overseasCount++
        tm.squad.push(p)
    } else sim.unsold.push(p.slNo)
    sim.history.push({ slNo: p.slNo, winner: w >= 0 ? w : null, price: w >= 0 ? outcome.soldFor : null, phase: sim.phase })
    sim.advance()
    if (stats.lots >= LOTS) completed = true
})
socket.on('playerSold', (p) => closeLot({ ...p, sold: true }))
socket.on('playerUnsold', (p) => closeLot({ ...p, sold: false }))

let completed = false
socket.once('auctionCompleted', () => serial(async () => { completed = true }))
const t0 = Date.now()
while (!completed && Date.now() - t0 < 3 * 60 * 60 * 1000) await sleep(500)
await queue
const minutes = ((Date.now() - t0) / 60000).toFixed(1)

// ── End state: Redis vs replay, results persisted ─────────────────────────
if (LOTS === Infinity) {
    check(sim.done, 'replay reached the end of the auction together with the server')
    const state = await readAuctionState(roomId)
    const owners = new Map()
    for (const t of state.teams) {
        const m = sim.teams[idx(t.teamId)]
        const sig = `${t.purseLeft}:${t.playerCount}:${t.overseasCount}:${t.squadSlNos.map(Number).sort((a, b) => a - b).join('.')}`
        const msig = `${m.purseLeft}:${m.playerCount}:${m.overseasCount}:${m.squad.map((p) => p.slNo).sort((a, b) => a - b).join('.')}`
        check(sig === msig, `final ${t.teamId}: redis ${sig} vs replay ${msig}`)
        check(t.purseLeft >= 0 && t.playerCount <= rules.maxPlayers && t.overseasCount <= rules.maxOverseas && t.purseLeft + t.purseSpent === rules.pursePerTeam, `final ${t.teamId}: limits/accounting`)
        for (const sl of t.squadSlNos) {
            check(!owners.has(sl), `player ${sl} in two squads`)
            owners.set(sl, t.teamId)
        }
        if (t.teamId !== HUMAN_TEAM) check(selectBestXI(m.squad).emptySlots === 0, `bot ${t.teamId} ends with a legal XI`)
    }
    check(state.room.status === 'completed', `room status ${state.room.status}`)
    await sleep(3000)
    const res = await fetch(`${SERVER}/room/${roomId}/results`)
    const results = (await res.json()).data
    check(res.ok && results?.teams?.length === teamIds.length, 'results persisted for every team')
    for (const t of results?.teams ?? []) {
        const m = sim.teams[idx(t.teamId)]
        check(Math.abs(t.teamRating - selectBestXI(m.squad).strength) < 0.05, `persisted XI strength ${t.teamId}: ${t.teamRating} vs ${selectBestXI(m.squad).strength}`)
    }
    const keys = []
    let cursor = '0'
    do {
        const [next, batch] = await redis.scan(cursor, 'MATCH', `room:${roomId}*`, 'COUNT', 500)
        cursor = next
        keys.push(...batch)
    } while (cursor !== '0')
    const ttls = await Promise.all(keys.map((k) => redis.ttl(k)))
    console.log(`  redis: ${keys.length} keys for the room; without expiry: ${keys.filter((_, i) => ttls[i] < 0).join(', ') || 'none'}`)
    console.log('  final:')
    for (const id of teamIds) {
        const m = sim.teams[idx(id)]
        const seat = seats.find((s) => s.teamId === id)
        console.log(`    ${id.padEnd(4)} ${(seat ? seat.persona : 'human').padEnd(16)} XI ${selectBestXI(m.squad).strength}  ${m.playerCount} players, ${m.overseasCount} OS, ₹${m.purseLeft}L left`)
    }
}
check(serverErrors.length === 0, `server errors: ${serverErrors.join('; ')}`)
console.log(`  ${JSON.stringify(stats)} in ${minutes} min`)
console.log(failures.length ? `\n✖ ${failures.length} check(s) failed` : '\n✔ all checks passed')
socket.close()
await redis.quit()
await mongoose.disconnect()
process.exitCode = failures.length ? 1 : 0
setTimeout(() => process.exit(), 2000).unref()
