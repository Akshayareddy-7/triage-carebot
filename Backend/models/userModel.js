const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
    userName: {
        type: String,
        required: [true, 'user name is required']
    },
    email: {
        type: String,
        required: [true, 'email is required'],
        unique: true
    },
    password: {
        type: String,
        required: [true, 'password is required']
    },
    address: {
        type: Array,
    },
    phone: {
        type: String,
        required: [true, 'phone number is required']
    },
    answer: {
        type: String,
        required: [true, 'answer is required']   // 👈 correct closing now
    }
}, { timestamps: true });

module.exports = mongoose.model('User', userSchema);
