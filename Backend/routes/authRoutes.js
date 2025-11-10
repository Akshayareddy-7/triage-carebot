const express = require('express');
const { registerController} = require('../controllers/authController');
const { loginController } = require('../controllers/authController');
const router = express.Router();

router.post('/register',registerController);

router.post('/login',loginController);
module.exports = router;
