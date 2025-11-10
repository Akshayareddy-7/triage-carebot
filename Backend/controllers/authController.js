const userModel = require("../models/userModel");
const bcrypt = require('bcryptjs');
const JWT = require('jsonwebtoken');
// REGISTER
const registerController = async (req, res) => {
  try {
    const { userName, email, password, phone, address , answer} = req.body;
    //validation
    if (!userName || !email || !password || !address || !phone || !answer) {
      return res.status(400).send({
        success: false,
        message: "Please provide all fields",
      });
    }

    // Check if user already exists
    const existing = await userModel.findOne({ email });
    if (existing) {
      return res.status(400).send({
        success: false,
        message: "Email already registered, please login",
      });
    }
    //hashing 
    var salt = bcrypt.genSaltSync(10);
    const hashedPassword = await bcrypt.hash(password, salt);


    // Create new user
    const user = await userModel.create({
      userName,
      email,
      password: hashedPassword,
      address,
      phone,
      answer,
    });

    res.status(201).send({
      success: true,
      message: "Successfully Registered",
      user,
    });
  } catch (error) {
    console.log(error);
    res.status(500).send({
      success: false,
      message: "Error in register API",
      error,
    });
  }
};

//LOGIN
const loginController =async(req,res) =>
{
    try {
        const {email,password} = req.body;
        if(!email || !password)
        {
            return res.status(500).send({
                success:false,
                message:'please provide email or password',
            });
        }
        const user = await userModel.findOne({email});
        if(!user)
        {
            return res.status(404).send({
                success:false,
                message:'User not found',
            });
        }
        // check user password | compare password
        const isMatch = await bcrypt.compare(password, user.password);
        if(!isMatch)
        {
            return res.status(500).send({
                success:false,
                message:'Invalid Credintials',
            });
        }
        //token
        const token = JWT.sign({id:user._id}, process.env.JWT_SECRET,{
            expiresIn: '7d',
        });
        // user.password = undefined;
        res.status(200).send({
            success:true,
            message:'Login successfully',
            token,
            user,
        });
        return res.redirect("http://127.0.0.1:8000/");
    } catch (error) {
        console.log(error);
        res.status(500).send({
            success:false,
            message:'Error in login api',
            error,
        })
    }
};
module.exports = { registerController,loginController};
