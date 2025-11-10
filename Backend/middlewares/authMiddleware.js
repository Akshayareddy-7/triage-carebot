const JWT = require('jsonwebtoken');

module.exports = async(req,res,next)=>{
    try {
        const token = req.headers["authorization"].split(" ")[1];
        JWT.verify(token,process.env.JWT_SECRET,(err,decode)=>{
            if(err)
            {
                return res.status(401).send({
                    success:false,
                    message:'Un-Authorized user',
                });
            }
            else{
                req.user = decode;
                next();
            }
        });
    } catch (error) {
        console.log(error);
        res.status(500).send({
            success: false,
            message:'please provide Auth token',
            error,
        });
    }
    
};