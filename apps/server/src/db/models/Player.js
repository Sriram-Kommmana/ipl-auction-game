import mongoose from 'mongoose';

const playerSchema = new mongoose.Schema(
    {
        slNo: {
            type: Number,
            required: true,
            unique: true,
        },

        playerName: {
            type: String,
            required: true,
            trim: true,
        },

        country: {
            type: String,
            required: true,
            trim: true,
        },

        nationality: {
            type: String,
            required: true,
            enum: ["Indian", "Overseas"],
        },

        role: {
            type: String,
            required: true,
            enum: ["BATSMAN", "BOWLER", "ALL ROUNDER", "WICKET KEEPER",],
        },

        basePrice: {
            type: Number,
            required: true,
            min: 0,
        },

        rating: {
            type: Number,
            required: true,
            min: 0,
            max: 100,
        },

        isActive: {
            type: Boolean,
            default: true,
        },

        stats: {
            bat: { type: Number, min: 0, max: 100 },
            pwr: { type: Number, min: 0, max: 100 },
            bwl: { type: Number, min: 0, max: 100 },
            tec: { type: Number, min: 0, max: 100 },
            clt: { type: Number, min: 0, max: 100 },
        },

        setNo: {
            type: Number,
            required: true,
            index: true,
        },
    },
    {
        timestamps: true,
    }
)

export default mongoose.model("Player", playerSchema);