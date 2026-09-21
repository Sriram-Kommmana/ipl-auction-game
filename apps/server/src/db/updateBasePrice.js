// One-off data migration: players seeded with a base price of 250 are
// repriced to 200. Nothing else about the player documents is touched.
//
//   node src/db/updateBasePrice.js --dry    → report only, writes nothing
//   node src/db/updateBasePrice.js          → applies the update
//
// Safe to re-run: it only matches documents that still have basePrice 250,
// so a second run is a no-op.
import 'dotenv/config'
import mongoose from 'mongoose'
import connectMongoDB from './client.js'
import Player from './models/Player.js'

const FROM_PRICE = 250
const TO_PRICE = 200

const run = async () => {
    const isDryRun = process.argv.includes('--dry')

    await connectMongoDB()

    const distribution = await Player.aggregate([
        { $group: { _id: '$basePrice', count: { $sum: 1 } } },
        { $sort: { _id: 1 } }
    ])

    const affected = await Player.find({ basePrice: FROM_PRICE })
        .select('slNo playerName -_id')
        .sort({ slNo: 1 })
        .lean()

    console.log(`[updateBasePrice] Base price distribution before:`)
    for (const { _id, count } of distribution) {
        console.log(`   ₹${_id}L → ${count} players`)
    }
    console.log(`[updateBasePrice] ${affected.length} players at ₹${FROM_PRICE}L:`)
    for (const p of affected) {
        console.log(`   #${p.slNo} ${p.playerName}`)
    }

    if (isDryRun) {
        console.log(`[updateBasePrice] DRY RUN — no changes written.`)
    } else {
        const result = await Player.updateMany(
            { basePrice: FROM_PRICE },
            { $set: { basePrice: TO_PRICE } }
        )
        console.log(
            `[updateBasePrice] matched ${result.matchedCount}, modified ${result.modifiedCount} → ₹${TO_PRICE}L`
        )

        const after = await Player.aggregate([
            { $group: { _id: '$basePrice', count: { $sum: 1 } } },
            { $sort: { _id: 1 } }
        ])
        console.log(`[updateBasePrice] Base price distribution after:`)
        for (const { _id, count } of after) {
            console.log(`   ₹${_id}L → ${count} players`)
        }
    }

    await mongoose.connection.close()
    process.exit(0)
}

run().catch(async (err) => {
    console.error('[updateBasePrice] Failed:', err)
    await mongoose.connection.close().catch(() => {})
    process.exit(1)
})
