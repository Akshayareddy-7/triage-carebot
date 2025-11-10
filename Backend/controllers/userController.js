const userModel = require("../models/userModel");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");

// ================== Register User ==================
const registerUserController = async (req, res) => {
  try {
    const { userName, email, password, address, phone } = req.body;

    // Validate required fields
    if (!userName || !email || !password || !phone) {
      return res.status(400).send({
        success: false,
        message: "Please provide all required fields",
      });
    }

    // Check if user already exists
    const existingUser = await userModel.findOne({ email });
    if (existingUser) {
      return res.status(400).send({
        success: false,
        message: "Email already registered",
      });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Create user
    const user = await userModel.create({
      userName,
      email,
      password: hashedPassword,
      address,
      phone,
    });

    res.status(201).send({
      success: true,
      message: "User registered successfully",
      user,
    });
  } catch (error) {
    console.error(error);
    res.status(500).send({
      success: false,
      message: "Error in register API",
      error: error.message,
    });
  }
};

// ================== Login User ==================
const loginUserController = async (req, res) => {
  try {
    const { email, password } = req.body;

    // Validate inputs
    if (!email || !password) {
      return res.status(400).send({
        success: false,
        message: "Email and password are required",
      });
    }

    // Find user
    const user = await userModel.findOne({ email });
    if (!user) {
      return res.status(404).send({
        success: false,
        message: "User not found",
      });
    }

    // Compare passwords
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).send({
        success: false,
        message: "Invalid credentials",
      });
    }

    // Create JWT token
    const token = jwt.sign(
      { id: user._id },
      process.env.JWT_SECRET || "defaultsecret", // use env variable in production
      { expiresIn: "1d" }
    );

    res.status(200).send({
      success: true,
      message: "Login successful",
      token,
      user: {
        id: user._id,
        userName: user.userName,
        email: user.email,
        phone: user.phone,
        address: user.address,
      },
    });
  } catch (error) {
    console.error(error);
    res.status(500).send({
      success: false,
      message: "Error in login API",
      error: error.message,
    });
  }
};

// ================== Get User by ID ==================
const getUserController = async (req, res) => {
  try {
    const id = req.body.id;

    if (!id) {
      return res.status(400).send({
        success: false,
        message: "User ID is required",
      });
    }

    const user = await userModel.findById(id).select("-password");

    if (!user) {
      return res.status(404).send({
        success: false,
        message: "User not found",
      });
    }

    res.status(200).send({
      success: true,
      message: "User retrieved successfully",
      user,
    });
  } catch (error) {
    console.error(error);
    res.status(500).send({
      success: false,
      message: "Error in get user API",
      error: error.message,
    });
  }
};

// ================== Update User ==================
const updateUserController = async (req, res) => {
  try {
    const id = req.body.id;

    if (!id) {
      return res.status(400).send({
        success: false,
        message: "User ID is required",
      });
    }

    const user = await userModel.findById(id);
    if (!user) {
      return res.status(404).send({
        success: false,
        message: "User not found",
      });
    }

    const { userName, address, phone } = req.body;
    if (userName) user.userName = userName;
    if (address) user.address = address;
    if (phone) user.phone = phone;

    await user.save();

    res.status(200).send({
      success: true,
      message: "User updated successfully",
      user,
    });
  } catch (error) {
    console.error(error);
    res.status(500).send({
      success: false,
      message: "Error in update user API",
      error: error.message,
    });
  }
};

module.exports = {
  registerUserController,
  loginUserController,
  getUserController,
  updateUserController,
};
