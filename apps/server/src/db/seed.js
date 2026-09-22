import 'dotenv/config';
import csvtojson from 'csvtojson';
import path from 'path';
import { fileURLToPath } from 'url';
import mongoose from 'mongoose';
import Player from './models/Player.js';
import connectMongoDB from './client.js';

// __dirname isn't defined under ESM ("type": "module" in package.json) —
// this is the standard replacement, derived from the module's own URL.
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const seedPlayers = async () => {
    await connectMongoDB()

    const players = await csvtojson().fromFile(path.join(__dirname, 'players.csv'))

    const formatted = players.map(p => ({
        slNo:        Number(p.SL_NO),
        playerName:  p.PLAYER_NAME.trim(),
        country:     p.COUNTRY.trim(),
        nationality: p.COUNTRY.trim() === 'INDIA' ? 'Indian' : 'Overseas',
        role:        p.ROLE.trim(),
        basePrice:   Number(p['BASE PRICE']),
        rating:      Number(p.RATING),
        isActive:    true,
        stats: {
            bat: Number(p.BAT),
            pwr: Number(p.PWR),
            bwl: Number(p.BWL),
            tec: Number(p.TEC),
            clt: Number(p.CLT),
        },
        setNo: Number(p.SET_NO),
    }))

    await Player.deleteMany({})
    await Player.insertMany(formatted)

    console.log(`${formatted.length} players seeded successfully`)

    await mongoose.connection.close()
    process.exit(0)
}

seedPlayers().catch(err => {
    console.error('Seeding failed:', err)
    process.exit(1)
})