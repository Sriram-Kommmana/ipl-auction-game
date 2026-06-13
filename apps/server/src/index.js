import 'dotenv/config';

import './redis/client.js';
import connectMongoDB from './db/client.js';
import express from 'express';
import http from 'node:http';
import cors from 'cors';
import { Server } from 'socket.io';
import roomRoutes from './http/roomRoutes.js';


const app = express();
const server = http.createServer(app);

const io = new Server(server, {
    cors: {
        origin: process.env.CLIENT_URL,
        methods: ['GET', 'POST']
    }
});

app.use(cors({
    origin: process.env.CLIENT_URL
}));

app.use(express.json());

app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

app.use('/room', roomRoutes);

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
