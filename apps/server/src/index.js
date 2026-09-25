import 'dotenv/config';

import './redis/client.js';
import './redis/registerLuaScripts.js'
import connectMongoDB from './db/client.js';
import express from 'express';
import http from 'node:http';
import cors from 'cors';
import { Server } from 'socket.io';
import roomRoutes from './http/roomRoutes.js';
import { getStats } from './http/getStats.js';
import { errorHandler } from './middlewares/errorHandler.js'
import { registerSocketHandlers } from './socket/index.js'
import { initBots } from './bots/botManager.js'



const app = express();
const server = http.createServer(app);

const io = new Server(server, {
    cors: {
        // Same origin as the Express CORS below — the game client is the
        // only thing that ever opens a socket connection.
        origin: process.env.CLIENT_URL,
        methods: ['GET', 'POST']
    }
});

registerSocketHandlers(io)
initBots(io)

app.use(cors({
    origin: process.env.CLIENT_URL
}));

app.use(express.json());

app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

// Public read-only counters for the static landing page
app.get('/stats', getStats);

app.use('/room', roomRoutes);


//after all routes
app.use(errorHandler)

const PORT = process.env.PORT || 3001;

connectMongoDB().
then(() => {
    server.listen(PORT, () => {
        console.log(`Server running on the port: ${PORT}`)
    });
})
.catch((err) => {
    console.error("Mongo db connection failed ",err);
})
