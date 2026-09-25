import Player from '../db/models/Player.js'

// Every bot decision needs player details (role, rating, stats) for the lot,
// every squad and the players still to come. They never change during a
// game, so load them from MongoDB once per process and serve from memory
// (323 small documents).
let cache = null
let loading = null

const loadPlayers = async () => {
    if (cache) return cache
    if (!loading) {
        loading = Player.find({})
            .select('slNo playerName country nationality role basePrice rating stats setNo -_id')
            .lean()
            .then((docs) => {
                cache = new Map(docs.map((d) => [d.slNo, d]))
                return cache
            })
            .finally(() => { loading = null })
    }
    return loading
}

const getPlayer = (players, slNo) => players.get(Number(slNo)) || null

export { loadPlayers, getPlayer }
