import mongoose from 'mongoose'

const roomSchema = new mongoose.Schema({
    roomId: {
        type: String,
        required: true,
    },
    createdAt: {
        type: Date,
        default: Date.now,
    },
    expiresAt: {
        type: Date,
        required: true,
    }
})

export default mongoose.model('Room', roomSchema)