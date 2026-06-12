require('dotenv').config()
const Redis = require('ioredis');

const redis = new Redis(process.env.REDIS_URL)

redis.on('connect', () => {
    console.log("[Redis] TCP connection established.")
})

redis.on('error', (err) => {
    console.error('Redis background error:', err)
})

module.exports = redis