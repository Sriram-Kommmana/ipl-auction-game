import 'dotenv/config';
import Redis from 'ioredis';

const redis = new Redis(process.env.REDIS_URL)

redis.on('connect', () => {
    console.log("[Redis] TCP connection established.")
})

redis.on('error', (err) => {
    console.error('Redis background error:', err)
})

export default redis