import os
import random
import requests
import time
from datetime import date
from fastapi import FastAPI, Depends, HTTPException, Body
from sqlalchemy.orm import Session, relationship
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from pydantic import BaseModel
import models, schemas
from database import engine, get_db

# --- CONFIGURATION ---
def _load_env_file(env_path: str) -> None:
    """Load simple KEY=VALUE pairs from a .env file if present."""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"").strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        print(f"⚠️ Failed to load .env file: {exc}")

# Load .env from backend folder to support local dev
_load_env_file(os.path.join(os.path.dirname(__file__), ".env"))

# Get API key from environment variable (NEVER hardcode secrets!)
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError(
        "❌ ERROR: GENAI_API_KEY environment variable not set!\n"
        "Set it with: $env:GENAI_API_KEY='your-key' (PowerShell)\n"
        "Or: set GENAI_API_KEY=your-key (CMD)\n"
        "Or: add GENAI_API_KEY to mindlink_backend/.env"
    )

# Rate limiting: store last request time per user
user_request_times = {} 

# --- NEW: DEFINE TASK MODEL DIRECTLY HERE (The "Academic Layer") ---
# We define this here to avoid editing models.py for the prototype phase
class Task(models.Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(String)
    is_done = Column(Boolean, default=False)
    due_date = Column(String) # Format: YYYY-MM-DD
    user_id = Column(Integer, ForeignKey("users.id"))

    user = relationship("models.User", back_populates="tasks")

# Link the User model to Tasks (Monkey-patching for the prototype)
# This allows us to say user.tasks to get their list
models.User.tasks = relationship("Task", back_populates="user")

# --- NEW: PYDANTIC SCHEMAS FOR TASKS ---
class TaskCreate(BaseModel):
    content: str
    due_date: str 

class TaskUpdate(BaseModel):
    is_done: bool

# Create the database tables (Now includes 'tasks')
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# --- Pydantic Model for Chat Requests ---
class ChatRequest(BaseModel):
    user_id: int
    message: str

# 1. GET - Check if it works
@app.get("/")
def read_root():
    return {"message": "MindLink API is Ready"}

# 2. POST - Register a new user
@app.post("/users/", response_model=schemas.UserResponse)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Fake password hashing for demo
    new_user = models.User(email=user.email, password_hash=user.password + "notreallyhashed")
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# 3. GET - List all users
@app.get("/users/", response_model=list[schemas.UserResponse])
def read_users(db: Session = Depends(get_db)):
    return db.query(models.User).all()

# 4. POST - Login
@app.post("/login/")
def login_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if db_user.password_hash != user.password + "notreallyhashed":
        raise HTTPException(status_code=401, detail="Incorrect password")
    
    return {"message": "Login successful", "user_id": db_user.id}

# 5. POST - Create a Thought (Mood Log)
@app.post("/thoughts/", response_model=schemas.ThoughtResponse)
def create_thought(thought: schemas.ThoughtCreate, user_id: int, db: Session = Depends(get_db)):
    new_thought = models.Thought(
        content=thought.content, 
        mood_score=thought.mood_score, 
        owner_id=user_id
    )
    db.add(new_thought)
    db.commit()
    db.refresh(new_thought)
    return new_thought

# 6. GET - Read my Thoughts
@app.get("/thoughts/{user_id}", response_model=list[schemas.ThoughtResponse])
def read_thoughts(user_id: int, db: Session = Depends(get_db)):
    thoughts = db.query(models.Thought).filter(models.Thought.owner_id == user_id).all()
    return thoughts

# 7. DELETE - Remove a thought
@app.delete("/thoughts/{thought_id}")
def delete_thought(thought_id: int, db: Session = Depends(get_db)):
    thought = db.query(models.Thought).filter(models.Thought.id == thought_id).first()
    if not thought:
        raise HTTPException(status_code=404, detail="Thought not found")
    
    db.delete(thought)
    db.commit()
    return {"message": "Thought deleted successfully"}

# --- 8. ACADEMIC LAYER (NEW TASK ENDPOINTS) ---

@app.post("/tasks/")
def create_task(task: TaskCreate, user_id: int, db: Session = Depends(get_db)):
    db_task = Task(content=task.content, due_date=task.due_date, user_id=user_id)
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

@app.get("/tasks/{user_id}")
def get_tasks(user_id: int, db: Session = Depends(get_db)):
    return db.query(Task).filter(Task.user_id == user_id).all()

@app.put("/tasks/{task_id}")
def update_task_status(task_id: int, update: TaskUpdate, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if task:
        task.is_done = update.is_done
        db.commit()
        return {"message": "Status updated"}
    return {"error": "Task not found"}

# --- 9. THE "SIMULATED AI" (JITAI ENGINE) ---
# This satisfies the "Thesis Hack" logic
@app.get("/jitai/{user_id}")
def get_jitai_intervention(user_id: int, db: Session = Depends(get_db)):
    # A. Get recent mood
    recent_thought = db.query(models.Thought).filter(models.Thought.owner_id == user_id)\
        .order_by(models.Thought.id.desc()).first()
    mood = recent_thought.mood_score if recent_thought else 5
    
    # B. Get overdue tasks
    tasks = db.query(Task).filter(Task.user_id == user_id, Task.is_done == False).all()
    today_str = str(date.today())
    overdue_count = sum(1 for t in tasks if t.due_date < today_str)

    # C. "The Thesis Hack" Logic (Expert System)
    if mood <= 3:
        return {
            "type": "CRISIS", 
            "message": "MindLink Notice: Your mood is critically low. Please prioritize rest over work today."
        }
    
    if overdue_count >= 1 and mood < 6:
         return {
             "type": "ACADEMIC", 
             "message": f"You have {overdue_count} overdue tasks causing stress. Let's focus on finishing just ONE."
         }
         
    if mood >= 8:
        return {
            "type": "MOTIVATION", 
            "message": "You're feeling great! Use this energy to clear your task list."
        }

    return {"type": "NONE", "message": ""}

# --- RATE LIMITING FUNCTION ---
def check_rate_limit(user_id: int, limit_seconds: int = 3) -> bool:
    """Prevent rapid consecutive requests from the same user (prevents quota waste)"""
    now = time.time()
    last_request = user_request_times.get(user_id, 0)

    if now - last_request < limit_seconds:
        return False

    user_request_times[user_id] = now
    return True

# --- 10. INTELLIGENCE LAYER (CHATBOT) ---
@app.post("/chat/")
def chat_with_ai(request: ChatRequest, db: Session = Depends(get_db)):
    # Rate limiting: prevent excessive API calls
    if not check_rate_limit(request.user_id):
        return {
            "response": "Please wait a moment before sending another message.",
            "alert": False,
            "intervention": "RATE_LIMITED"
        }
    
    user_message = request.message.lower()
    
    # --- A. CONTEXT RETRIEVAL ---
    last_thought = db.query(models.Thought).filter(models.Thought.owner_id == request.user_id)\
        .order_by(models.Thought.id.desc()).first()
    
    current_mood = last_thought.mood_score if last_thought else 5

    # --- B. SAFETY PROTOCOL ---
    critical_triggers = ["suicide", "kill myself", "die", "end it all", "hurt myself", "cutting", "overdose"]
    external_harm_triggers = ["kill someone", "kill him", "kill her", "kill them", "hurt others", "murder", "shoot"]

    if any(trigger in user_message for trigger in critical_triggers):
        return {
            "response": "🚨 CRISIS DETECTED: I am sensing severe distress. Please listen to me: You are not alone. I am providing you with the Crisis Resource Card. Please contact the Wellbeing Officer immediately.",
            "alert": True,
            "intervention": "CRITICAL_SELF_HARM"
        }

    if any(trigger in user_message for trigger in external_harm_triggers):
        return {
            "response": "⚠️ SAFETY ALERT: I cannot process requests involving harm to others. If you are feeling out of control, please contact University Counseling immediately at 011-234-5678.",
            "alert": True,
            "intervention": "CRITICAL_EXTERNAL_HARM"
        }

    # --- C. GENERATIVE AI (DIRECT API REQUEST) ---
    try:
        prompt = f"""
        You are a compassionate, empathetic academic counselor for university students.
        CONTEXT:
        - The user's most recent mood score was {current_mood}/10.
        - User's message: "{request.message}"
        INSTRUCTIONS:
        - Respond directly to the user.
        - Keep your answer short (maximum 2 sentences).
        - Be supportive but practical.
        """
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite-001:generateContent?key={GENAI_API_KEY}"
        
        payload = { "contents": [{ "parts": [{"text": prompt}] }] }
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            try:
                ai_reply = data['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError):
                ai_reply = "I'm listening, but I'm having a little trouble forming a response right now."

            return {
                "response": ai_reply,
                "alert": False,
                "intervention": "GEMINI_AI"
            }
        elif response.status_code == 429:
            # QUOTA EXHAUSTED - This is the issue!
            print("\n" + "="*80)
            print("❌ API QUOTA ERROR: Your API key has reached its usage limit!")
            print("="*80)
            print("📋 To fix this:")
            print("   1. Go to: https://ai.google.dev/pricing")
            print("   2. Generate a NEW API key")
            print("   3. Update your .env file or environment variable:")
            print("      PowerShell: $env:GENAI_API_KEY='your-new-key'")
            print("      CMD: set GENAI_API_KEY=your-new-key")
            print("   4. Restart the backend server")
            print("="*80 + "\n")
            return {
                "response": "🚨 API is at its usage limit. The developer needs to generate a new API key.",
                "alert": False,
                "intervention": "QUOTA_EXHAUSTED"
            }
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return {
                "response": "I'm having trouble connecting to my brain right now (API Error). But I'm here to listen.",
                "alert": False,
                "intervention": "API_ERROR"
            }
        
    except Exception as e:
        print(f"Connection Error: {e}")
        return {
            "response": "I'm having trouble connecting to the internet, but I'm here. Can you tell me more about that?",
            "alert": False,
            "intervention": "CONNECTION_ERROR"
        }