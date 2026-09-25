// Auction order — which players come up, and in what sequence.
//
// Shared so the live server and the training simulator build pools the
// same way. Order: sets ascending (set 1 = marquee), shuffled within each set.

export const POOL_MODES = ['full', 'quick']

// Quick pool: the best players in each role, not simply "sets 1-5".
// Sets 1-5 hold 133 players but only 3 wicket keepers — with 10 teams and a
// Best XI that needs a keeper, 7 teams could never field a complete side.
// These quotas give 140 players with enough of every role to go round.
export const QUICK_POOL_QUOTAS = Object.freeze({
    'WICKET KEEPER': 20,
    BATSMAN: 35,
    'ALL ROUNDER': 35,
    BOWLER: 50
})

export const shuffle = (array, rng = Math.random) => {
    const arr = [...array]
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1))
        ;[arr[i], arr[j]] = [arr[j], arr[i]]
    }
    return arr
}

const selectQuickPlayers = (players) => {
    const picked = []
    for (const [role, quota] of Object.entries(QUICK_POOL_QUOTAS)) {
        picked.push(
            ...players
                .filter((p) => p.role === role)
                .sort((a, b) => (b.rating - a.rating) || (a.slNo - b.slNo))
                .slice(0, quota)
        )
    }
    return picked
}

// Returns an array of slNo in auction order.
export const buildAuctionPool = (players, { mode = 'full', rng = Math.random } = {}) => {
    const source = mode === 'quick' ? selectQuickPlayers(players) : players

    const sets = new Map()
    for (const p of source) {
        if (!sets.has(p.setNo)) sets.set(p.setNo, [])
        sets.get(p.setNo).push(p.slNo)
    }

    const pool = []
    for (const setNo of [...sets.keys()].sort((a, b) => a - b)) {
        pool.push(...shuffle(sets.get(setNo), rng))
    }
    return pool
}

// Full pools re-auction unsold players once; quick pools don't (they exist
// to keep solo sessions short).
export const hasReauction = (mode) => mode !== 'quick'
