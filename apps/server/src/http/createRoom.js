const createRoom = async (req, res) => {
    try{
        const {managerNickname, managerPin, roomPin, pursePerTeam} = req.body;
        if(!managerNickname || !managerPin || !roomPin || !pursePerTeam){
            return res.status(400).json({
                success: false,
                message: "All fields are required"
            })
            
        }
    } catch(error){

    }
}