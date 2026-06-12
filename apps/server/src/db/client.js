require('dotenv').config()
const {DB_NAME} = require('../constants')
const mongoose = require('mongoose')

const dns = require("dns")
dns.setServers([
    '1.1.1.1',
    '8.8.8.8'
]);

const connectMongoDB = async () => {
    try {
        // console.log(`${process.env.MONGODB_URI}${DB_NAME}`)
        const connectionInstance = await mongoose.connect(`${process.env.MONGODB_URI}${DB_NAME}`);
        console.log(`MongoDB connected: ${connectionInstance.connection.host}`);
    } catch (error) {
        console.error("MONGODB connection error ", error);
        process.exit(1);
    }
}

module.exports = connectMongoDB;