// Auction order — which players come up, and in what sequence.
//
// Shared so the live server and the training simulator build pools the
// same way. Every game auctions the whole player list. Order: sets ascending
// (set 1 = marquee), shuffled within each set. Unsold players get one
// re-auction round at the end.

export const shuffle = (array, rng = Math.random) => {
    const arr = [...array]
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1))
        ;[arr[i], arr[j]] = [arr[j], arr[i]]
    }
    return arr
}

// Returns an array of slNo in auction order.
export const buildAuctionPool = (players, { rng = Math.random } = {}) => {
    const sets = new Map()
    for (const p of players) {
        if (!sets.has(p.setNo)) sets.set(p.setNo, [])
        sets.get(p.setNo).push(p.slNo)
    }

    const pool = []
    for (const setNo of [...sets.keys()].sort((a, b) => a - b)) {
        pool.push(...shuffle(sets.get(setNo), rng))
    }
    return pool
}
