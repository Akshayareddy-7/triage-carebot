// const express = require('express');
// const colors = require('colors');
// const cors = require('cors');
// const morgan = require('morgan');
// const dotenv = require('dotenv');
// const connectDb = require('./config/db');

// //dot en config
// dotenv.config();

// //DB conenction
// connectDb();

// //rest object
// const app = express();
 
// //middlewares
// app.use(cors());
// app.use(express.json());
// app.use(morgan("dev")); 
// app.use(express.static('static'));
// // //route => URL=> http://localhost:8080
// app.use('/api/v1/test',require('./routes/testRoutes'));
// app.use('/api/v1/auth',require('./routes/authRoutes'));
// app.use('/api/v1/user',require('./routes/userRoutes'));
// app.get('/', (req,res)=>{
//     return res.status(200).send("<h1>Welcome to Carebot SERVER project</h1>");
// });

// //PORT
// const PORT = process.env.PORT;

// console.log("Script is running...");
// console.log("PORT from .env:", process.env.PORT);

// app.listen(PORT, () => {
//     console.log(`Server Running on http://localhost:${PORT}`);
// });
const express = require('express');
const colors = require('colors');
const cors = require('cors');
const morgan = require('morgan');
const dotenv = require('dotenv');
const path = require('path');
const connectDb = require('./config/db');

// dotenv config
dotenv.config();

// DB connection
connectDb();

// rest object
const app = express();

// middlewares
app.use(cors());
app.use(express.json());
app.use(morgan("dev"));

// serve static files from public folder
app.use(express.static(path.join(__dirname, 'public')));

// API routes
app.use('/api/v1/test', require('./routes/testRoutes'));
app.use('/api/v1/auth', require('./routes/authRoutes'));
app.use('/api/v1/user', require('./routes/userRoutes'));

// Serve frontend pages
app.get('/register', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'register.html'));
});

app.get('/login', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'login.html'));
});

app.get('/chat', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'chat.html'));
});

// default route
app.get('/', (req, res) => {
    return res.status(200).send("<h1>Welcome to Carebot SERVER project</h1><br><a href='/register'>Register</a> | <a href='/login'>Login</a>");
});

// PORT
const PORT = process.env.PORT || 8080;

console.log("Script is running...");
console.log("PORT from .env:", process.env.PORT);

app.listen(PORT, () => {
    console.log(`Server Running on http://localhost:${PORT}`);
});
