import 'dotenv/config';
import { DB_NAME } from '../constants.js';
import mongoose from 'mongoose';

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

export default connectMongoDB;