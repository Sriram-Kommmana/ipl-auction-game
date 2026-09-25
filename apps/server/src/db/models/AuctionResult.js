import mongoose from 'mongoose'

const statsSchema = new mongoose.Schema({
    bat: Number,
    pwr: Number,
    bwl: Number,
    tec: Number,
    clt: Number
}, { _id: false })

const squadPlayerSchema = new mongoose.Schema({
    slNo:        { type: Number, required: true },
    playerName:  { type: String, required: true },
    role:        { type: String, required: true },
    nationality: { type: String, required: true },
    boughtFor:   { type: Number, required: true },
    rating:      { type: Number, required: true },
    stats:       statsSchema
}, { _id: false })

const teamSchema = new mongoose.Schema({
    teamId:        String,
    teamName:      String,
    ownerId:       String,
    ownerNickname: String,
    isBot:         { type: Boolean, default: false },
    botKind:       { type: String, default: null },   // 'rule' | 'rl'
    botPersona:    { type: String, default: null },
    purseSpent:    Number,
    purseLeft:     Number,
    playerCount:   Number,
    overseasCount: Number,
    teamRating:    Number,   // best-XI strength (v2; v1 was avg squad rating)
    squad:         [squadPlayerSchema]
}, { _id: false })

const historyEntrySchema = new mongoose.Schema({
    auctionOrder: Number,
    slNo:         { type: Number, required: true },
    playerName:   { type: String, required: true },
    soldTo:       { type: String, default: null },
    soldFor:      { type: Number, default: null },
    status:       {
        type:     String,
        enum:     ['sold', 'unsold', 'skipped'],
        required: true
    },
    auctionedAt:  { type: Date, required: true }
}, { _id: false })

const auctionResultSchema = new mongoose.Schema({
    roomId:      { type: String, required: true, unique: true },
    version:     { type: Number, default: 1 },   // schema version for future migrations
    mode:        { type: String, default: 'multiplayer' },
    startedAt:   { type: Date,   required: true },
    completedAt: { type: Date,   required: true },
    teams:       [teamSchema],
    history:     [historyEntrySchema]
}, {
    timestamps: true
})

export default mongoose.model('AuctionResult', auctionResultSchema)