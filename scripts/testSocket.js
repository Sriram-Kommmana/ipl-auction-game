import { io } from "socket.io-client";

const socket = io("http://localhost:3001");

socket.on("connect", () => {
    console.log("Connected!");
    console.log("Socket ID:", socket.id);
});

socket.on("disconnect", () => {
    console.log("Disconnected");
});