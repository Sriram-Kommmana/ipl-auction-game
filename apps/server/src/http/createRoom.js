import { ApiError } from '../utils/ApiError';
import { ApiResponse } from '../utils/ApiResponse';

const createRoom = async (req, res) => {
    try{

        // get details from frontend
        const {managerNickname, managerPin, roomPin, pursePerTeam} = req.body;

        // validation
        if(!managerNickname || !managerPin || !roomPin || !pursePerTeam){
            throw new ApiError(400,"All fields are required")
        }
        if (!/^\d{4}$/.test(managerPin)) {
            throw new Error("managerPin must be exactly 4 digits");
        }

        if (!/^\d{4}$/.test(roomPin)) {
            throw new Error("roomPin must be exactly 4 digits");
        }


    } catch(error){

    }
}