// Search real episodes (adversarial controllers, training seeds) for act-v3
// final-path decision states; print the first hits with their action lists.
import { readFileSync, writeFileSync } from 'node:fs'
const base = new URL('../../../../packages/shared/', import.meta.url)
const { parsePlayersCsv } = await import(new URL('src/playersCsv.js', base))
const RL = await import(new URL('src/rl/index.js', base))
const players = parsePlayersCsv(readFileSync(new URL('../../apps/server/src/db/players.csv', base), 'utf8'))
const hits = []
let scanned = 0
for (let seed = 1_000_500; seed < 1_000_700 && hits.length < 4; seed++) {
    for (const [name, ctrl] of Object.entries(RL.ADVERSARIAL)) {
        const entry = RL.sampleEpisode(seed)
        const ep = new RL.RlEpisode({ players, entry, tremble: 0.01 })
        const rng = ep.learnerRng
        ep.reset()
        const actions = []
        let fp = 0, step
        do {
            if (ep.pending.mask.shield.finalPath) fp++
            const a = ctrl.act(ep, rng)
            actions.push(a)
            step = ep.step(a)
        } while (!step.done)
        scanned++
        if (fp) { hits.push({ seed, controller: name, finalPathStates: fp, entry, actions, legalXI: step.info.episode.legalXI }); console.log(`hit: seed ${seed} ${name} finalPath states ${fp} legalXI ${step.info.episode.legalXI}`) }
    }
}
console.log(`scanned ${scanned} episodes, ${hits.length} with final-path states`)
writeFileSync(process.argv[2] ?? 'final_path_hits.json', JSON.stringify(hits))
