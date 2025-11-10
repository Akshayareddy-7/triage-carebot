# # from fastapi import FastAPI, Depends, HTTPException, status
# # from fastapi.staticfiles import StaticFiles
# # from fastapi.responses import FileResponse, JSONResponse
# # from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# # from pydantic import BaseModel
# # from passlib.context import CryptContext
# # from jose import JWTError, jwt
# # import motor.motor_asyncio
# # import os, time
# # from datetime import datetime, timedelta

# # from doctor_agent import doctor_reply
# # from symptom_extractor import extract_structured
# # from triage_engine import evaluate_triage
# # from guideline_verifier import verify
# # from utils import log_session, APP_PORT

# # # ---------------------------
# # # Setup
# # # ---------------------------
# # app = FastAPI()
# # SESSIONS = {}
# # app.mount("/static", StaticFiles(directory="static"), name="static")

# # # Load .env file
# # from dotenv import load_dotenv
# # load_dotenv()

# # # MongoDB Atlas connection
# # MONGO_URL = os.getenv("MONGO_URL")
# # client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
# # db = client["carecompanion"]
# # users = db["users"]


# # # JWT + Password hashing
# # SECRET_KEY = os.getenv("SECRET_KEY")
# # ALGORITHM = os.getenv("ALGORITHM")
# # ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

# # pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# # oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# # def hash_password(password: str):
# #     return pwd_context.hash(password)

# # def verify_password(password: str, hashed: str):
# #     return pwd_context.verify(password, hashed)

# # def create_access_token(data: dict, expires_delta: timedelta = None):
# #     to_encode = data.copy()
# #     expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
# #     to_encode.update({"exp": expire})
# #     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# # async def get_current_user(token: str = Depends(oauth2_scheme)):
# #     try:
# #         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
# #         email: str = payload.get("sub")
# #         if email is None:
# #             raise HTTPException(status_code=401, detail="Invalid token")
# #         user = await users.find_one({"email": email})
# #         if not user:
# #             raise HTTPException(status_code=401, detail="User not found")
# #         return user
# #     except JWTError:
# #         raise HTTPException(status_code=401, detail="Invalid token")

# # # ---------------------------
# # # Models
# # # ---------------------------
# # class User(BaseModel):
# #     email: str
# #     password: str

# # class ChatRequest(BaseModel):
# #     session_id: str
# #     message: str

# # # ---------------------------
# # # Routes
# # # ---------------------------
# # @app.post("/register")
# # async def register(user: User):
# #     if await users.find_one({"email": user.email}):
# #         raise HTTPException(status_code=400, detail="Email already registered")
# #     hashed_pw = hash_password(user.password)
# #     await users.insert_one({"email": user.email, "password": hashed_pw})
# #     return {"message": "User registered successfully"}

# # @app.post("/login")
# # async def login(form_data: OAuth2PasswordRequestForm = Depends()):
# #     user = await users.find_one({"email": form_data.username})
# #     if not user or not verify_password(form_data.password, user["password"]):
# #         raise HTTPException(status_code=401, detail="Invalid credentials")
# #     token = create_access_token({"sub": user["email"]})
# #     return {"access_token": token, "token_type": "bearer"}

# # @app.get("/")
# # def serve_home():
# #     return FileResponse(os.path.join("static", "login.html"))  # redirect to login first

# # @app.get("/chat")
# # def serve_chat():
# #     return FileResponse(os.path.join("static", "index.html"))

# # @app.post("/api/chat")
# # async def chat(req: ChatRequest, current_user: dict = Depends(get_current_user)):
# #     # same chatbot logic you already have
# #     sid = req.session_id
# #     if sid not in SESSIONS:
# #         SESSIONS[sid] = []
# #     SESSIONS[sid].append({"role": "user", "content": req.message})

# #     # Doctor agent reply
# #     reply = doctor_reply(SESSIONS[sid])
# #     SESSIONS[sid].append({"role": "assistant", "content": reply})

# #     # Extract structured data
# #     conv_text = "\n".join([f"{m['role']}: {m['content']}" for m in SESSIONS[sid]])
# #     structured = extract_structured(conv_text)
# #     raw_triage = evaluate_triage(structured)

# #     top_diagnoses = []  
# #     verified = verify(raw_triage, top_diagnoses)

# #     # Urgent/Emergency handling
# #     if verified["level"] in ("Emergency", "Urgent"):
# #         context = (
# #             f"{verified['level']} — {verified['reason']}. Recommended action: "
# #             f"{'go to the nearest emergency department immediately' if verified['level']=='Emergency' else 'seek urgent medical attention'}."
# #         )
# #         final = doctor_reply(SESSIONS[sid], triage_context=context)
# #         SESSIONS[sid].append({"role": "assistant", "content": final})
# #         out_reply = final
# #     else:
# #         out_reply = reply

# #     # Log session
# #     log_session(sid, {
# #         "session": SESSIONS[sid],
# #         "structured": structured,
# #         "triage": verified,
# #         "ts": time.time()
# #     })

# #     return {"reply": out_reply, "triage": verified, "structured": structured}

# # main.py
# import os
# import time
# from datetime import datetime, timedelta
# from typing import Optional

# from fastapi import FastAPI, Depends, HTTPException
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from fastapi.concurrency import run_in_threadpool
# from pydantic import BaseModel
# from passlib.context import CryptContext
# from jose import JWTError, jwt
# import motor.motor_asyncio
# from dotenv import load_dotenv

# # Blocking functions (doctor agent + summarizer)
# from doctor_agent import doctor_reply              # should return (reply: str, end_convo: bool)
# from symptom_extractor import extract_structured
# from triage_engine import evaluate_triage
# from guideline_verifier import verify
# from utils import log_session
# from typing import Optional


# # Summarizer API expected:
# # generate_summary(session_messages) -> str (summary)
# # save_summary_json(summary_text, filepath) -> None/raise
# # save_summary_pdf(summary_text, filepath) -> None/raise

# # ---------------------------
# # Load environment & setup
# # ---------------------------
# load_dotenv()

# app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")

# SESSIONS = {}  # in-memory: { session_id: [ {role:, message:} ] }

# # MongoDB (async)
# MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
# client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
# db = client["carecompanion"]
# users = db["users"]

# # JWT + password hashing
# SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret")
# ALGORITHM = os.getenv("ALGORITHM", "HS256")
# ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# # ---------------------------
# # Utility auth helpers
# # ---------------------------
# def hash_password(password: str) -> str:
#     return pwd_context.hash(password)

# def verify_password(password: str, hashed: str) -> bool:
#     return pwd_context.verify(password, hashed)

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
#     to_encode = data.copy()
#     expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# async def get_current_user(token: str = Depends(oauth2_scheme)):
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         email: str = payload.get("sub")
#         if email is None:
#             raise HTTPException(status_code=401, detail="Invalid token")
#         user = await users.find_one({"email": email})
#         if not user:
#             raise HTTPException(status_code=401, detail="User not found")
#         return user
#     except JWTError:
#         raise HTTPException(status_code=401, detail="Invalid token")


# # ---------------------------
# # Pydantic models
# # ---------------------------
# class User(BaseModel):
#     email: str
#     password: str

# class ChatRequest(BaseModel):
#     session_id: str
#     message: str

# class EndConversationRequest(BaseModel):
#     session_id: str
#     format: str = "json"  # "json" or "pdf"


# # ---------------------------
# # Routes: Auth + static
# # ---------------------------
# @app.post("/register")
# async def register(user: User):
#     if await users.find_one({"email": user.email}):
#         raise HTTPException(status_code=400, detail="Email already registered")
#     hashed_pw = hash_password(user.password)
#     await users.insert_one({"email": user.email, "password": hashed_pw})
#     return {"message": "User registered successfully"}

# @app.post("/login")
# async def login(form_data: OAuth2PasswordRequestForm = Depends()):
#     user = await users.find_one({"email": form_data.username})
#     if not user or not verify_password(form_data.password, user["password"]):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     token = create_access_token({"sub": user["email"]})
#     return {"access_token": token, "token_type": "bearer"}

# @app.get("/")
# def serve_home():
#     return FileResponse(os.path.join("static", "login.html"))

# @app.get("/chat")
# def serve_chat():
#     return FileResponse(os.path.join("static", "index.html"))


# # ---------------------------
# # Chat endpoint
# # ---------------------------
# @app.post("/api/chat")
# async def chat(req: ChatRequest, current_user: dict = Depends(get_current_user)):
#     sid = req.session_id
#     if sid not in SESSIONS:
#         SESSIONS[sid] = []

#     # Append patient message
#     SESSIONS[sid].append({"role": "patient", "message": req.message})

#     # Check for "thank" or "thanks" → end conversation automatically
#     end_convo = False
#     summary_file_path = None
#     if "thank" in req.message.lower():
#         end_convo = True
#         # Generate summary immediately
#         try:
#             summary_text = await run_in_threadpool(generate_summary, SESSIONS[sid])
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             file_basename = f"summary_{sid}_{timestamp}"
#             summaries_dir = os.path.join("static", "summaries")
#             os.makedirs(summaries_dir, exist_ok=True)
#             # Save JSON
#             filename = file_basename + ".json"
#             file_path = os.path.join(summaries_dir, filename)
#             await run_in_threadpool(save_summary_json, summary_text, file_path)
#             summary_file_path = f"/get_summary/{filename}"

#             # Optional: save in MongoDB
#             coll = db["consultations"]
#             doc = {
#                 "session_id": sid,
#                 "summary_file": filename,
#                 "summary_text": summary_text,
#                 "messages": SESSIONS[sid],
#                 "created_at": datetime.utcnow()
#             }
#             await coll.insert_one(doc)
#         except Exception as e:
#             summary_file_path = None

#         # Clear session
#         SESSIONS.pop(sid, None)

#         return {
#             "reply": "Conversation ended. Summary generated automatically.",
#             "ended": True,
#             "summary_file": summary_file_path
#         }

#     # ---------------------------
#     # Normal doctor reply
#     # ---------------------------
#     try:
#         reply = await run_in_threadpool(doctor_reply, SESSIONS[sid])
#     except Exception as e:
#         log_session(sid, {"error": str(e), "ts": time.time()})
#         raise HTTPException(status_code=500, detail=f"Doctor agent error: {e}")

#     # Store doctor's message
#     SESSIONS[sid].append({"role": "doctor", "message": reply})

#     # Extract structured info
#     conv_text = "\n".join([f"{m['role']}: {m['message']}" for m in SESSIONS[sid]])
#     try:
#         structured = await run_in_threadpool(extract_structured, conv_text)
#     except Exception:
#         structured = {}
#     try:
#         raw_triage = await run_in_threadpool(evaluate_triage, structured)
#     except Exception:
#         raw_triage = {"level": "Routine", "reason": "could not evaluate triage"}

#     top_diagnoses = []
#     verified = verify(raw_triage, top_diagnoses)

#     # Emergency handling
#     out_reply = reply
#     if verified.get("level") in ("Emergency", "Urgent"):
#         context = (
#             f"{verified['level']} — {verified.get('reason','')}. Recommended action: "
#             f"{'go to nearest emergency immediately' if verified['level']=='Emergency' else 'seek urgent care'}."
#         )
#         try:
#             final_reply = await run_in_threadpool(doctor_reply, SESSIONS[sid], context)
#             SESSIONS[sid].append({"role": "doctor", "message": final_reply})
#             out_reply = final_reply
#         except Exception:
#             out_reply = reply

#     # Log session
#     try:
#         log_session(sid, {
#             "session": SESSIONS[sid],
#             "structured": structured,
#             "triage": verified,
#             "ts": time.time()
#         })
#     except Exception:
#         pass

#     return {"reply": out_reply, "triage": verified, "structured": structured, "ended": False}



# # ----------------------------------------
# # main.py — Final Stable Version
# # ----------------------------------------

# import os, json, time, random, requests
# from datetime import datetime, timedelta
# from fastapi import FastAPI, Depends, HTTPException, status
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from passlib.context import CryptContext
# from jose import jwt, JWTError
# from pydantic import BaseModel
# import motor.motor_asyncio
# from dotenv import load_dotenv
# from fastapi.concurrency import run_in_threadpool

# # Import AI + utility modules
# from doctor_agent import doctor_reply
# from symptom_extractor import extract_structured
# from triage_engine import evaluate_triage
# from guideline_verifier import verify
# from utils import log_session

# # ----------------------------------------
# # Setup
# # ----------------------------------------
# load_dotenv()
# app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")

# SESSIONS = {}
# message_logs=[]
# # MongoDB setup
# MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
# client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
# db = client["carecompanion"]
# users = db["users"]

# # JWT + Password hashing
# SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
# ALGORITHM = os.getenv("ALGORITHM", "HS256")
# ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# # ----------------------------------------
# # Auth helper functions
# # ----------------------------------------
# def hash_password(password: str): return pwd_context.hash(password)
# def verify_password(password, hashed): return pwd_context.verify(password, hashed)

# def create_access_token(data: dict, expires_delta: timedelta = None):
#     to_encode = data.copy()
#     expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# async def get_current_user(token: str = Depends(oauth2_scheme)):
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         email: str = payload.get("sub")
#         if email is None:
#             raise HTTPException(status_code=401, detail="Invalid token")
#         user = await users.find_one({"email": email})
#         if not user:
#             raise HTTPException(status_code=401, detail="User not found")
#         return user
#     except JWTError:
#         raise HTTPException(status_code=401, detail="Invalid token")


# # ----------------------------------------
# # Models
# # ----------------------------------------
# class User(BaseModel):
#     email: str
#     password: str

# class ChatRequest(BaseModel):
#     session_id: str
#     message: str


# # ----------------------------------------
# # Auth routes
# # ----------------------------------------
# @app.post("/register")
# async def register(user: User):
#     if await users.find_one({"email": user.email}):
#         raise HTTPException(status_code=400, detail="Email already registered")
#     hashed_pw = hash_password(user.password)
#     await users.insert_one({"email": user.email, "password": hashed_pw})
#     return {"message": "User registered successfully"}

# @app.post("/login")
# async def login(form_data: OAuth2PasswordRequestForm = Depends()):
#     user = await users.find_one({"email": form_data.username})
#     if not user or not verify_password(form_data.password, user["password"]):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     token = create_access_token({"sub": user["email"]})
#     return {"access_token": token, "token_type": "bearer"}


# # ----------------------------------------
# # Static pages
# # ----------------------------------------
# @app.get("/")
# def serve_home():
#     return FileResponse(os.path.join("static", "login.html"))

# @app.get("/chat")
# def serve_chat():
#     return FileResponse(os.path.join("static", "index.html"))


# # ----------------------------------------
# # Chat route (✅ fully fixed)
# # ----------------------------------------

# @app.post("/api/chat")
# async def chat(req: ChatRequest, current_user: dict = Depends(get_current_user)):
#     global message_logs
#     sid = req.session_id

#     if sid not in SESSIONS:
#         SESSIONS[sid] = []
#         message_logs.clear()

#     # Store patient message
#     user_message = {"role": "patient", "message": req.message}
#     SESSIONS[sid].append({"role": "user", "content": req.message})
#     message_logs.append(user_message)

#     # -------------------------
#     # Doctor Response Handling
#     # -------------------------
#     try:
#         result = await run_in_threadpool(doctor_reply, SESSIONS[sid])
#         if isinstance(result, tuple):
#             reply, severity_flag = result
#         else:
#             reply, severity_flag = result, False
#     except Exception as e:
#         print(f"❌ Doctor agent error: {e}")
#         raise HTTPException(status_code=500, detail=f"Doctor agent error: {e}")

#     # Append to session
#     SESSIONS[sid].append({"role": "assistant", "content": reply})
#     message_logs.append({"role": "doctor", "message": reply})

#     # -------------------------
#     # Structured data & triage
#     # -------------------------
#     conv_text = "\n".join([f"{m['role']}: {m.get('content', '')}" for m in SESSIONS[sid]])

#     try:
#         structured = await run_in_threadpool(extract_structured, conv_text)
#     except Exception:
#         structured = {}

#     try:
#         raw_triage = await run_in_threadpool(evaluate_triage, structured)
#     except Exception:
#         raw_triage = {"level": "Routine", "reason": "No red flags found; symptoms appear non-urgent."}

#     verified = verify(raw_triage, [])

#     # -------------------------
#     # Map triage level → color & label
#     # -------------------------
#     level = verified.get("level", "Routine")
#     reason = verified.get("reason", "No red flags found; symptoms appear non-urgent.")
#     triage_display = {
#         "Emergency": {"color": "#e63946", "status": "🔴 Emergency — Immediate care required!"},
#         "Urgent": {"color": "#ff8800", "status": "🟠 Urgent — Needs prompt medical attention."},
#         "Routine": {"color": "#2a9d8f", "status": "🟢 Routine — No red flags; symptoms appear non-urgent."},
#         "Normal": {"color": "#2a9d8f", "status": "🟢 Normal — Stable, no emergency detected."}
#     }

#     triage_info = {
#         "level": level,
#         "reason": reason,
#         "color": triage_display.get(level, {}).get("color", "#2a9d8f"),
#         "status": triage_display.get(level, {}).get("status", "🟢 Routine condition"),
#         "severity_flag": severity_flag
#     }

#     # -------------------------
#     # Save conversation log
#     # -------------------------
#     log_session(sid, {
#         "session": SESSIONS[sid],
#         "structured": structured,
#         "triage": triage_info,
#         "ts": time.time()
#     })

#     # Final response
#     return {
#         "reply": reply,
#         "triage": triage_info,
#         "structured": structured
#     }


# # ----------------------------------------
# # Patient Simulator
# # ----------------------------------------
# @app.get("/api/simulate_patient_chat")
# async def simulate_patient_chat():
#     try:
#         file_path = os.path.join(os.getcwd(), "data", "vignettes.json")
#         if not os.path.exists(file_path):
#             raise HTTPException(status_code=500, detail=f"File not found: {file_path}")

#         with open(file_path, "r", encoding="utf-8") as f:
#             vignettes = json.load(f)

#         if not isinstance(vignettes, list) or not vignettes:
#             raise HTTPException(status_code=500, detail="Invalid or empty JSON data")

#         patient = random.choice(vignettes)
#         name = patient.get("name", "Unknown")
#         complaint = patient.get("chief_complaint", "No complaint provided")
#         symptoms = ", ".join(patient.get("symptoms", []))
#         history = patient.get("history", "No history provided")

#         message_history = [
#             {"role": "patient", "message": complaint},
#             {"role": "doctor", "message": f"Can you describe your {symptoms} in a bit more detail?"}
#         ]
#         doctor_response = f"Based on your history ({history}), it seems mild. Please rest and monitor your condition."

#         global message_logs
#         message_logs.clear()
#         message_logs.extend(message_history)
#         message_logs.append({"role": "doctor", "message": doctor_response})

#         print(f"✅ Simulation success for {name}")
#         return {
#             "patient_name": name,
#             "message_history": message_history,
#             "doctor_reply": doctor_response
#         }
#     except Exception as e:
#         print(f"❌ Simulation error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# # ----------------------------------------
# # Generate Summary (✅ works for both chat & simulator)
# # ----------------------------------------
# @app.post("/api/generate_summary")
# async def generate_summary_api(current_user: dict = Depends(get_current_user)):
#     global message_logs
#     if not message_logs or len(message_logs) < 2:
#         print("⚠️ No conversation found yet.")
#         return {"summary": "No conversation found yet."}

#     try:
#         from summary_agent import generate_summary
#         print("🧠 Generating summary using summary_agent.py ...")
#         summary_text = generate_summary(message_logs)

#         summary_doc = {
#             "user_email": current_user["email"],
#             "timestamp": datetime.utcnow(),
#             "summary_text": summary_text,
#             "conversation": message_logs
#         }

#         print("🟢 Saving summary to MongoDB...")
#         result = await db["summaries"].insert_one(summary_doc)
#         print("✅ Summary inserted with ID:", result.inserted_id)

#         return {
#             "summary": summary_text,
#             "message": "✅ Summary saved successfully.",
#             "saved": True
#         }
#     except Exception as e:
#         print(f"❌ Summary generation error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# # ----------------------------------------
# # Generate Differential Diagnosis (Top 5)
# # ----------------------------------------
# @app.post("/api/generate_diagnosis")
# async def generate_differential_diagnosis_api(current_user: dict = Depends(get_current_user)):
#     global message_logs

#     # ✅ Check if we have conversation context
#     if not message_logs or len(message_logs) < 2:
#         print("⚠️ No conversation found yet for diagnosis.")
#         return {"diagnosis": "⚠️ Please have a conversation first before generating diagnosis."}

#     try:
#         from differential_diagnosis import generate_differential_diagnosis
#         print("🧩 Generating differential diagnosis using differential_diagnosis.py ...")
#         diagnosis_text = generate_differential_diagnosis(message_logs)

#         # ✅ Save to MongoDB
#         diagnosis_doc = {
#             "user_email": current_user["email"],
#             "timestamp": datetime.utcnow(),
#             "diagnosis_text": diagnosis_text,
#             "conversation": message_logs
#         }

#         print("🟢 Saving diagnosis to MongoDB...")
#         result = await db["diagnoses"].insert_one(diagnosis_doc)
#         print("✅ Diagnosis inserted with ID:", result.inserted_id)

#         # ✅ Return final structured response
#         return {
#             "diagnosis": diagnosis_text,
#             "message": "✅ Top 5 differential diagnoses generated successfully.",
#             "saved": True
#         }

#     except Exception as e:
#         print(f"❌ Differential diagnosis generation error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


import os, json, time, random, requests
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import jwt, JWTError
from pydantic import BaseModel
import motor.motor_asyncio
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool

# Import AI + utility modules
from doctor_agent import doctor_reply
from symptom_extractor import extract_structured
from triage_engine import evaluate_triage
from guideline_verifier import verify
from utils import log_session

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # you can replace "*" with ["http://localhost:5173"] if using Vite
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------
# Setup
# ----------------------------------------
load_dotenv()
app.mount("/static", StaticFiles(directory="static"), name="static")

SESSIONS = {}
message_logs=[]
# MongoDB setup
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client["carecompanion"]
users = db["users"]

# JWT + Password hashing
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# ----------------------------------------
# Auth helper functions
# ----------------------------------------
# def hash_password(password: str): return pwd_context.hash(password)
def hash_password(password: str):
    # Fix: bcrypt only supports passwords up to 72 bytes. Truncate the password.
    truncated_password = password[:72]
    return pwd_context.hash(truncated_password)
def verify_password(password, hashed): return pwd_context.verify(password, hashed)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = await users.find_one({"email": email})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ----------------------------------------
# Models
# ----------------------------------------
class User(BaseModel):
    email: str
    password: str

class ChatRequest(BaseModel):
    session_id: str
    message: str


# ----------------------------------------
# Auth routes
# ----------------------------------------
# @app.post("/register")
# async def register(user: User):
#     if await users.find_one({"email": user.email}):
#         raise HTTPException(status_code=400, detail="Email already registered")
#     hashed_pw = hash_password(user.password)
#     await users.insert_one({"email": user.email, "password": hashed_pw})
#     return {"message": "User registered successfully"}
from fastapi.responses import JSONResponse

@app.post("/register")
async def register(user: User):
    existing_user = await users.find_one({"email": user.email})
    if existing_user:
        # Explicit JSON response for duplicate email
        return JSONResponse(
            status_code=400,
            content={"detail": "Email already registered"}
        )

    hashed_pw = hash_password(user.password)
    await users.insert_one({"email": user.email, "password": hashed_pw})

    return JSONResponse(
        status_code=200,
        content={"message": "User registered successfully"}
    )

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users.find_one({"email": form_data.username})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user["email"]})
    return {"access_token": token, "token_type": "bearer"}


# ----------------------------------------
# Static pages
# ----------------------------------------
@app.get("/")
def serve_home():
    return FileResponse(os.path.join("static", "login.html"))

@app.get("/chat")
def serve_chat():
    return FileResponse(os.path.join("static", "index.html"))


# ----------------------------------------
# Chat route (✅ fully fixed)
# ----------------------------------------

@app.post("/api/chat")
async def chat(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    global message_logs
    sid = req.session_id

    if sid not in SESSIONS:
        SESSIONS[sid] = []
        message_logs.clear()

    # Store patient message
    user_message = {"role": "patient", "message": req.message}
    SESSIONS[sid].append({"role": "user", "content": req.message})
    message_logs.append(user_message)

    # -------------------------
    # Doctor Response Handling
    # -------------------------
    try:
        result = await run_in_threadpool(doctor_reply, SESSIONS[sid])
        if isinstance(result, tuple):
            reply, severity_flag = result
        else:
            reply, severity_flag = result, False
    except Exception as e:
        print(f"❌ Doctor agent error: {e}")
        raise HTTPException(status_code=500, detail=f"Doctor agent error: {e}")

    # Append to session
    SESSIONS[sid].append({"role": "assistant", "content": reply})
    message_logs.append({"role": "doctor", "message": reply})

    # -------------------------
    # Structured data & triage
    # -------------------------
    conv_text = "\n".join([f"{m['role']}: {m.get('content', '')}" for m in SESSIONS[sid]])

    try:
        structured = await run_in_threadpool(extract_structured, conv_text)
    except Exception:
        structured = {}

    try:
        raw_triage = await run_in_threadpool(evaluate_triage, structured)
    except Exception:
        raw_triage = {"level": "Routine", "reason": "No red flags found; symptoms appear non-urgent."}

    verified = verify(raw_triage, [])

    # -------------------------
    # Map triage level → color & label
    # -------------------------
    level = verified.get("level", "Routine")
    reason = verified.get("reason", "No red flags found; symptoms appear non-urgent.")
    triage_display = {
        "Emergency": {"color": "#e63946", "status": "🔴 Emergency — Immediate care required!"},
        "Urgent": {"color": "#ff8800", "status": "🟠 Urgent — Needs prompt medical attention."},
        "Routine": {"color": "#2a9d8f", "status": "🟢 Routine — No red flags; symptoms appear non-urgent."},
        "Normal": {"color": "#2a9d8f", "status": "🟢 Normal — Stable, no emergency detected."}
    }

    triage_info = {
        "level": level,
        "reason": reason,
        "color": triage_display.get(level, {}).get("color", "#2a9d8f"),
        "status": triage_display.get(level, {}).get("status", "🟢 Routine condition"),
        "severity_flag": severity_flag
    }

    # -------------------------
    # Save conversation log
    # -------------------------
    log_session(sid, {
        "session": SESSIONS[sid],
        "structured": structured,
        "triage": triage_info,
        "ts": time.time()
    })

    # Final response
    return {
        "reply": reply,
        "triage": triage_info,
        "structured": structured
    }


# ----------------------------------------
# Patient Simulator
# ----------------------------------------
@app.get("/api/simulate_patient_chat")
async def simulate_patient_chat():
    try:
        file_path = os.path.join(os.getcwd(), "data", "vignettes.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail=f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            vignettes = json.load(f)

        if not isinstance(vignettes, list) or not vignettes:
            raise HTTPException(status_code=500, detail="Invalid or empty JSON data")

        patient = random.choice(vignettes)
        name = patient.get("name", "Unknown")
        complaint = patient.get("chief_complaint", "No complaint provided")
        symptoms = ", ".join(patient.get("symptoms", []))
        history = patient.get("history", "No history provided")

        message_history = [
            {"role": "patient", "message": complaint},
            {"role": "doctor", "message": f"Can you describe your {symptoms} in a bit more detail?"}
        ]
        doctor_response = f"Based on your history ({history}), it seems mild. Please rest and monitor your condition."

        global message_logs
        message_logs.clear()
        message_logs.extend(message_history)
        message_logs.append({"role": "doctor", "message": doctor_response})

        print(f"✅ Simulation success for {name}")
        return {
            "patient_name": name,
            "message_history": message_history,
            "doctor_reply": doctor_response
        }
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------
# Generate Summary (✅ works for both chat & simulator)
# ----------------------------------------
@app.post("/api/generate_summary")
async def generate_summary_api(current_user: dict = Depends(get_current_user)):
    global message_logs
    if not message_logs or len(message_logs) < 2:
        print("⚠ No conversation found yet.")
        return {"summary": "No conversation found yet."}

    try:
        from summary_agent import generate_summary
        print("🧠 Generating summary using summary_agent.py ...")
        summary_text = generate_summary(message_logs)

        summary_doc = {
            "user_email": current_user["email"],
            "timestamp": datetime.utcnow(),
            "summary_text": summary_text,
            "conversation": message_logs
        }

        print("🟢 Saving summary to MongoDB...")
        result = await db["summaries"].insert_one(summary_doc)
        print("✅ Summary inserted with ID:", result.inserted_id)

        return {
            "summary": summary_text,
            "message": "✅ Summary saved successfully.",
            "saved": True
        }
    except Exception as e:
        print(f"❌ Summary generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------
# Generate Differential Diagnosis (Top 5)
# ----------------------------------------
@app.post("/api/generate_diagnosis")
async def generate_differential_diagnosis_api(current_user: dict = Depends(get_current_user)):
    global message_logs

    # ✅ Check if we have conversation context
    if not message_logs or len(message_logs) < 2:
        print("⚠ No conversation found yet for diagnosis.")
        return {"diagnosis": "⚠ Please have a conversation first before generating diagnosis."}

    try:
        from differential_diagnosis import generate_differential_diagnosis
        print("🧩 Generating differential diagnosis using differential_diagnosis.py ...")
        diagnosis_text = generate_differential_diagnosis(message_logs)

        # ✅ Save to MongoDB
        diagnosis_doc = {
            "user_email": current_user["email"],
            "timestamp": datetime.utcnow(),
            "diagnosis_text": diagnosis_text,
            "conversation": message_logs
        }

        print("🟢 Saving diagnosis to MongoDB...")
        result = await db["diagnoses"].insert_one(diagnosis_doc)
        print("✅ Diagnosis inserted with ID:", result.inserted_id)

        # ✅ Return final structured response
        return {
            "diagnosis": diagnosis_text,
            "message": "✅ Top 5 differential diagnoses generated successfully.",
            "saved": True
        }

    except Exception as e:
        print(f"❌ Differential diagnosis generation error: {e}")
        raise HTTPException(status_code=500,detail=str(e))
