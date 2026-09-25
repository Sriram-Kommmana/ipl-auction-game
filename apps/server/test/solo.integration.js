// End-to-end checks against a RUNNING server (pnpm --filter server start).
//
//   node test/solo.integration.js                → solo game, first 8 lots
//   node test/solo.integration.js --lots 140     → play a whole quick auction
//   node test/solo.integration.js --race         → deadline race check (multiplayer)
//
// SERVER_URL defaults to http://localhost:3001. Every run creates real rooms
// in whatever Redis/MongoDB the server is configured with.
//
// Solo checks:
//   - starting seats exactly 9 AI franchises (4 rule + 5 RL seats)
//   - bots bid on their own; a team never outbids itself
//   - every sold lot goes to the LAST accepted bid, at that price
//   - with the human passing, lots close ~1s after the last bid, well
//     inside the 10s countdown
//   - nothing is bid while the auction is paused
// Race check:
//   - a bid landing right at the countdown deadline is either rejected, or
//     accepted AND honoured by the sale — never accepted and then ignored.

import { io } from 'socket.io-client'

const SERVER = process.env.SERVER_URL || 'http://localhost:3001'
const args = process.argv.slice(2)
const LOTS = Number(args[args.indexOf('--lots') + 1]) || 8
const RACE = args.includes('--race')

const failures = []
const check = (ok, message) => {
    if (!ok) {
        failures.push(message)
        console.log(`  ✖ ${message}`)
    }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

const post = async (path, body) => {
    const res = await fetch(`${SERVER}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })
    const json = await res.json()
    if (!res.ok) throw new Error(`${path} → ${res.status}: ${json.message}`)
    return json.data
}

const connect = (playerId) => new Promise((resolve, reject) => {
    const socket = io(SERVER, { transports: ['websocket'] })
    const timer = setTimeout(() => reject(new Error('stateSync timeout')), 10000)
    socket.once('reconnectError', (e) => reject(new Error(e.message)))
    socket.once('stateSync', (snapshot) => {
        clearTimeout(timer)
        resolve({ socket, snapshot })
    })
    socket.on('connect', () => socket.emit('reconnect', { playerId }))
})

const once = (socket, event, ms = 10000) => new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`timed out waiting for ${event}`)), ms)
    socket.once(event, (payload) => {
        clearTimeout(timer)
        resolve(payload)
    })
})

const soloScenario = async () => {
    console.log(`\n● Solo game — watching ${LOTS} lots`)
    const room = await post('/room/create', {
        managerNickname: 'IntegrationTest', managerPin: '1234', mode: 'solo', pool: 'quick'
    })
    console.log(`  room ${room.roomId} (${room.mode}, ${room.pool})`)
    check(room.mode === 'solo', 'createRoom echoes solo mode')

    const { socket, snapshot } = await connect(room.managerId)
    check(snapshot.room.mode === 'solo', 'stateSync reports solo mode')
    check(snapshot.room.timerDuration === 10, `solo timer is 10s (got ${snapshot.room.timerDuration})`)

    socket.emit('startAuction', { playerId: room.managerId })
    const early = await once(socket, 'startAuctionError')
    check(/franchise/i.test(early.message), `start refused until the human picks a team ("${early.message}")`)

    const errors = []
    for (const e of ['startAuctionError', 'bidError', 'passError', 'pauseError', 'resumeError', 'auctionError']) {
        socket.on(e, (p) => errors.push(`${e}: ${p.message}`))
    }

    socket.emit('selectTeam', { playerId: room.managerId, teamId: 'MI' })
    await once(socket, 'teamSelected')

    const seated = once(socket, 'stateSync')
    const started = once(socket, 'auctionStarted')
    socket.emit('startAuction', { playerId: room.managerId })
    const roster = await seated
    await started

    const bots = roster.players.filter((p) => p.isBot)
    check(bots.length === 9, `9 bots seated (got ${bots.length})`)
    check(bots.filter((b) => b.botKind === 'rule').length === 4, '4 rule-based bots')
    check(bots.filter((b) => b.botKind === 'rl').length === 5, '5 RL seats')
    check(roster.teams.length === 10, `10 franchises in play (got ${roster.teams.length})`)
    check(bots.every((b) => b.status === 'online' && b.teamId), 'bots are online and own a team')
    console.log(`  seated: ${bots.map((b) => `${b.teamId}=${b.nickname}`).join(', ')}`)

    // ── Watch lots ───────────────────────────────────────────────────────
    let lot = { bids: [], openedAt: Date.now() }
    let lotsDone = 0
    let paused = false
    const pausedBids = []
    const summaries = []
    const durations = []
    const gameStarted = Date.now()

    socket.on('bidPlaced', ({ teamId, newBid }) => {
        if (paused) pausedBids.push(teamId)
        const prev = lot.bids.at(-1)
        check(!prev || prev.teamId !== teamId, `${teamId} outbid itself at ${newBid}`)
        check(teamId !== 'MI', 'nobody bid for the human (the human never bids here)')
        lot.bids.push({ teamId, newBid, at: Date.now() })
    })

    const endLot = (outcome) => {
        const last = lot.bids.at(-1)
        const closeGap = last ? Date.now() - last.at : Date.now() - lot.openedAt
        if (outcome.status === 'sold') {
            check(last && outcome.soldTo === last.teamId, `lot sold to the last bidder (${outcome.soldTo} vs ${last?.teamId})`)
            check(last && outcome.soldFor === last.newBid, `lot sold at the last bid (${outcome.soldFor} vs ${last?.newBid})`)
        } else {
            check(lot.bids.length === 0, 'an unsold lot had no bids')
        }
        check(closeGap < 9000, `lot closed early via auto-close (${closeGap}ms after last action)`)
        const lotSeconds = (Date.now() - lot.openedAt) / 1000
        durations.push(lotSeconds)
        summaries.push(`${outcome.playerName.padEnd(22)} ${outcome.status === 'sold' ? `→ ${outcome.soldTo.padEnd(4)} ₹${String(outcome.soldFor).padStart(4)}L` : '  unsold        '}  ${String(lot.bids.length).padStart(2)} bids in ${lotSeconds.toFixed(1).padStart(4)}s, closed ${(closeGap / 1000).toFixed(1)}s after last bid`)
        lotsDone++
        lot = { bids: [], openedAt: Date.now() }
    }
    socket.on('playerSold', (p) => endLot({ ...p, status: 'sold' }))
    socket.on('playerUnsold', (p) => endLot({ ...p, status: 'unsold' }))

    // The human passes on every lot as soon as its timer starts.
    socket.on('timerStarted', () => socket.emit('passLot', { playerId: room.managerId }))

    const deadline = Date.now() + LOTS * 90000
    let pauseTested = false
    while (lotsDone < LOTS && Date.now() < deadline) {
        await sleep(250)
        if (!pauseTested && lotsDone === 2 && lot.bids.length > 0) {
            pauseTested = true
            // Count from the server's confirmation, not from our request: a bid
            // that landed before the pause did is legitimate. Socket.IO keeps
            // event order, so any bidPlaced after timerPaused is a real bug.
            socket.once('timerPaused', () => { paused = true })
            socket.emit('pauseAuction', { playerId: room.managerId })
            await once(socket, 'timerPaused')
            await sleep(4000)
            check(pausedBids.length === 0, `no bids while paused (got ${pausedBids.length})`)
            paused = false
            socket.emit('resumeAuction', { playerId: room.managerId })
            await once(socket, 'timerResumed')
            console.log('  pause/resume: bots stayed silent for 4s while paused')
        }
    }
    check(lotsDone >= LOTS, `watched ${LOTS} lots (got ${lotsDone})`)
    check(errors.length === 0, `no server errors (${errors.join('; ')})`)
    summaries.forEach((s) => console.log(`  ${s}`))
    const avg = durations.reduce((a, b) => a + b, 0) / Math.max(1, durations.length)
    console.log(`  ${lotsDone} lots in ${((Date.now() - gameStarted) / 60000).toFixed(1)} min — ${avg.toFixed(1)}s per lot on average`)

    if (LOTS >= 140) {
        const done = await Promise.race([once(socket, 'auctionCompleted', 60000), sleep(60000).then(() => null)])
        check(done, 'auction completed after the last lot')
        await sleep(3000)
        const res = await fetch(`${SERVER}/room/${room.roomId}/results`)
        const results = (await res.json()).data
        check(res.ok && results?.teams?.length === 10, 'results persisted for 10 teams')
        if (results?.teams) {
            const board = [...results.teams].sort((a, b) => b.teamRating - a.teamRating)
            check(board.every((t) => t.teamRating <= 100), 'XI strength on a 0-100 scale')
            console.log('  final leaderboard (XI strength):')
            board.forEach((t, i) => console.log(`    ${String(i + 1).padStart(2)}. ${t.teamId.padEnd(4)} ${String(t.teamRating).padStart(5)}  ${t.isBot ? `${t.botKind}:${t.botPersona}` : 'human'}  ${t.playerCount} players, ₹${t.purseLeft}L left`))
        }
    }
    socket.close()
}

const raceScenario = async () => {
    console.log('\n● Deadline race (multiplayer, 30s lots) — this takes a few minutes')
    const room = await post('/room/create', {
        managerNickname: 'RaceManager', managerPin: '1111', roomPin: '2222', mode: 'multiplayer'
    })
    const guest = await post('/room/join', { roomId: room.roomId, roomPin: '2222', nickname: 'RaceGuest', playerPin: '3333' })
    const a = await connect(room.managerId)
    const b = await connect(guest.playerId)
    a.socket.emit('selectTeam', { playerId: room.managerId, teamId: 'CSK' })
    await once(a.socket, 'teamSelected')
    b.socket.emit('selectTeam', { playerId: guest.playerId, teamId: 'RCB' })
    await once(b.socket, 'teamSelected')

    const accepted = []
    a.socket.on('bidPlaced', (p) => accepted.push(p))
    let timerStartedAt = 0
    a.socket.on('timerStarted', () => { timerStartedAt = Date.now() })

    a.socket.emit('startAuction', { playerId: room.managerId })
    await once(a.socket, 'auctionStarted')

    // Offsets (ms) relative to the expected deadline at which the late bid fires.
    // Redis may be remote (tens to hundreds of ms away), so sweep well before
    // the deadline too — the interesting case is a bid ACCEPTED just before
    // the timeout fires. Override with --offsets -500,-200,-100
    const offsetsArg = args.includes('--offsets') ? args[args.indexOf('--offsets') + 1] : null
    const offsets = offsetsArg ? offsetsArg.split(',').map(Number) : [-600, -350, -250, -180, -120, -80, -40]
    for (const [i, offset] of offsets.entries()) {
        // Open the lot with a bid from CSK so there's a leader to beat.
        await sleep(300)
        a.socket.emit('placeBid', { playerId: room.managerId })
        await sleep(300)
        const before = accepted.length
        const fireAt = timerStartedAt + 30000 + offset
        await sleep(Math.max(0, fireAt - Date.now()))
        b.socket.emit('placeBid', { playerId: guest.playerId })

        const result = await Promise.race([
            once(a.socket, 'playerSold', 75000).then((p) => ({ ...p, status: 'sold' })),
            once(a.socket, 'playerUnsold', 75000).then((p) => ({ ...p, status: 'unsold' }))
        ])
        const last = accepted.at(-1)
        const lateBidAccepted = accepted.slice(before).some((p) => p.teamId === 'RCB')
        check(result.status === 'sold' && result.soldTo === last.teamId && result.soldFor === last.newBid,
            `offset ${offset}ms: sale honours the last accepted bid (sold to ${result.soldTo} at ${result.soldFor}, last bid ${last.teamId} at ${last.newBid})`)
        console.log(`  lot ${i + 1}, late bid at ${offset >= 0 ? '+' : ''}${offset}ms: ${lateBidAccepted ? 'accepted → lot extended' : 'rejected (lot already closing)'}; sold to ${result.soldTo} at ₹${result.soldFor}L ✔`)
        await once(a.socket, 'timerStarted', 10000).catch(() => {})
    }
    a.socket.close()
    b.socket.close()
}

try {
    await (RACE ? raceScenario() : soloScenario())
} catch (err) {
    failures.push(err.message)
    console.error(`  ✖ ${err.message}`)
}
console.log(failures.length ? `\n✖ ${failures.length} check(s) failed` : '\n✔ all checks passed')
// Let the closed sockets finish tearing down: calling process.exit() while a
// handle is still closing aborts Node on Windows (libuv async.c assertion).
process.exitCode = failures.length ? 1 : 0
setTimeout(() => process.exit(), 2000).unref()
