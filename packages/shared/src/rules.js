// Auction rules — the single source of truth for everything that decides
// whether a bid is legal and how much it costs.
//
// The live server enforces these inside the `placeBidAtomic` Lua script
// (apps/server/src/redis/registerLuaScripts.js) because the check and the
// write must be one atomic step. Lua can't import JS, so that script keeps
// its own copy — test/rules.test.js pins both to the same ladder.

export const ROLES = ['BATSMAN', 'BOWLER', 'ALL ROUNDER', 'WICKET KEEPER']

export const DEFAULT_RULES = {
    pursePerTeam: 12500, // lakhs
    maxPlayers: 25,
    maxOverseas: 8
}

// Cheapest base price in the player pool — used to reserve enough money
// to still complete a playing XI.
export const MIN_BASE_PRICE = 20

export const XI_SIZE = 11

// Increment ladder, chosen by the price BEFORE the increment:
// 190 → 200 → 220 → … → 300 → 350
export const bidIncrement = (currentBid) => {
    if (currentBid < 200) return 10
    if (currentBid < 300) return 20
    return 50
}

// The amount the next bid will be. The first bid on a lot is the base price
// itself; every later bid adds exactly one increment. There are no jump bids.
export const nextBidAmount = (currentBid, hasBidder) =>
    hasBidder ? currentBid + bidIncrement(currentBid) : currentBid

// Same checks, in the same order, as placeBidAtomic (minus the timer check,
// which only the live engine knows about). Returns null when the bid is
// allowed, otherwise the exact error string the server would send.
export const bidBlocker = ({ team, lot, rules = DEFAULT_RULES, amount }) => {
    if (lot.currentBidderId && lot.currentBidderId === team.teamId) {
        return 'You are already the highest bidder.'
    }
    if (team.purseLeft < amount) return 'Insufficient purse for this bid.'
    if (team.playerCount >= rules.maxPlayers) return 'Your squad is full.'
    if (lot.nationality === 'Overseas' && team.overseasCount >= rules.maxOverseas) {
        return 'Overseas player limit reached.'
    }
    return null
}

// Bot-only spending guard. Humans may blow their whole purse on one player
// (the engine allows it), but bots keep enough back to still fill an XI —
// otherwise a single bidding war can leave a bot with an unplayable squad
// for the rest of the auction. `reservePerSlot` should be what a cheap
// player realistically costs in the pool that's left (see deriveLotFacts);
// it defaults to the cheapest base price in the game.
export const botSpendLimit = (team, reservePerSlot = MIN_BASE_PRICE) => {
    const slotsToFillAfterThis = Math.max(0, XI_SIZE - team.playerCount - 1)
    return team.purseLeft - slotsToFillAfterThis * reservePerSlot
}

// What a cheap signing costs in the players still to come: the 25th
// percentile base price. In the full pool that's ₹20L; in the quick pool
// (mostly top players) it's higher, and reserving only ₹20L per slot would
// be fiction.
export const cheapSlotPrice = (upcoming) => {
    if (!upcoming || upcoming.length === 0) return MIN_BASE_PRICE
    const prices = upcoming.map((p) => p.basePrice).sort((a, b) => a - b)
    return Math.max(MIN_BASE_PRICE, prices[Math.floor(prices.length * 0.25)])
}
