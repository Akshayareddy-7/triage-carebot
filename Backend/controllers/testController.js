const testUserController =  (req,res) =>{
    try {
        res.status(200).send("<h1>Server running and fetching api successfully</h1>");
    } catch (error) {
        console.log("error in test api",error);
    }
};
module.exports={testUserController};