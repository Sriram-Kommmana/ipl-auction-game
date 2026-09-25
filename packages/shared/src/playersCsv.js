// Reads the seed CSV (apps/server/src/db/players.csv) into the same shape as
// the MongoDB Player documents — so the simulator, tests and training bridge
// can run without a database.
//
// Mirrors apps/server/src/db/seed.js, including how nationality is derived.

export const parsePlayersCsv = (text) => {
    const lines = text.trim().split(/\r?\n/)
    const header = lines[0].split(',').map((h) => h.trim())
    const col = (name) => {
        const i = header.indexOf(name)
        if (i === -1) throw new Error(`players.csv is missing column ${name}`)
        return i
    }
    const idx = {
        slNo: col('SL_NO'), name: col('PLAYER_NAME'), country: col('COUNTRY'), role: col('ROLE'),
        basePrice: col('BASE PRICE'), rating: col('RATING'), bat: col('BAT'), pwr: col('PWR'),
        bwl: col('BWL'), tec: col('TEC'), clt: col('CLT'), setNo: col('SET_NO')
    }

    return lines.slice(1).filter(Boolean).map((line) => {
        const c = line.split(',').map((v) => v.trim())
        const country = c[idx.country]
        return {
            slNo: Number(c[idx.slNo]),
            playerName: c[idx.name],
            country,
            nationality: country === 'INDIA' ? 'Indian' : 'Overseas',
            role: c[idx.role],
            basePrice: Number(c[idx.basePrice]),
            rating: Number(c[idx.rating]),
            isActive: true,
            stats: {
                bat: Number(c[idx.bat]), pwr: Number(c[idx.pwr]), bwl: Number(c[idx.bwl]),
                tec: Number(c[idx.tec]), clt: Number(c[idx.clt])
            },
            setNo: Number(c[idx.setNo])
        }
    })
}
