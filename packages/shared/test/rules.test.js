import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { bidBlocker, bidIncrement, botSpendLimit, nextBidAmount } from '../src/rules.js'

test('first bid on a lot is the base price', () => {
    assert.equal(nextBidAmount(150, false), 150)
})

test('increment ladder is chosen by the price before the increment', () => {
    const ladder = [190]
    for (let i = 0; i < 9; i++) ladder.push(nextBidAmount(ladder.at(-1), true))
    assert.deepEqual(ladder, [190, 200, 220, 240, 260, 280, 300, 350, 400, 450])
    assert.equal(bidIncrement(199), 10)
    assert.equal(bidIncrement(299), 20)
    assert.equal(bidIncrement(300), 50)
})

// The live engine enforces the ladder inside a Lua script, which can't
// import this module. Pin the two together so neither changes alone.
test('Lua placeBidAtomic uses the same ladder', () => {
    const lua = readFileSync(
        fileURLToPath(new URL('../../../apps/server/src/redis/registerLuaScripts.js', import.meta.url)),
        'utf8'
    )
    assert.match(lua, /if currentBid < 200 then\s+increment = 10/)
    assert.match(lua, /elseif currentBid < 300 then\s+increment = 20/)
    assert.match(lua, /else\s+increment = 50/)
})

test('bidBlocker mirrors the server checks and messages', () => {
    const rules = { pursePerTeam: 12500, maxPlayers: 25, maxOverseas: 8 }
    const team = { teamId: 'MI', purseLeft: 500, playerCount: 10, overseasCount: 8 }
    const indian = { nationality: 'Indian', currentBidderId: '' }
    const overseas = { nationality: 'Overseas', currentBidderId: '' }

    assert.equal(bidBlocker({ team, lot: indian, rules, amount: 500 }), null)
    assert.equal(bidBlocker({ team, lot: { ...indian, currentBidderId: 'MI' }, rules, amount: 100 }), 'You are already the highest bidder.')
    assert.equal(bidBlocker({ team, lot: indian, rules, amount: 510 }), 'Insufficient purse for this bid.')
    assert.equal(bidBlocker({ team: { ...team, playerCount: 25 }, lot: indian, rules, amount: 20 }), 'Your squad is full.')
    assert.equal(bidBlocker({ team, lot: overseas, rules, amount: 20 }), 'Overseas player limit reached.')
})

test('bots always keep enough to finish an XI at the cheapest base price', () => {
    // 3 players owned → buying a 4th leaves 7 slots to fill at ₹20L
    assert.equal(botSpendLimit({ purseLeft: 1000, playerCount: 3 }), 1000 - 7 * 20)
    assert.equal(botSpendLimit({ purseLeft: 1000, playerCount: 15 }), 1000)
})
