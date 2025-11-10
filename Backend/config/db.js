// Backend/config/db.js
const mongoose = require("mongoose");
const dotenv = require("dotenv");

dotenv.config({ override: true });

function getMongoUri() {
  // support either name, in case your code used MONGO_URI before
  const raw = (process.env.MONGO_URL || process.env.MONGO_URI || "").trim();
  // strip accidental surrounding quotes
  const uri = raw.replace(/^['"]|['"]$/g, "");
  if (!uri) throw new Error("MONGO_URL (or MONGO_URI) is missing in .env");
  if (!uri.startsWith("mongodb://") && !uri.startsWith("mongodb+srv://")) {
    throw new Error(`MONGO_URL must start with mongodb:// or mongodb+srv://. Got: ${uri}`);
  }
  return uri;
}

async function connectDb() {
  const uri = getMongoUri();
  console.log("Connecting to MongoDB...");
  await mongoose.connect(uri);
  console.log("✅ MongoDB connected");
}

module.exports = connectDb;
