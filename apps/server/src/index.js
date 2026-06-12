require('dotenv').config()

require('./redis/client');
const connectMongoDB = require('./db/client')
const express = require('express');
const http = require('node:http');
const cors = require('cors');
const { Server } = require('socket.io');
const roomRoutes = require('./http/roomRoutes')


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
