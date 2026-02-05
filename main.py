import os
import random
import requests
import time
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# --- THESIS ARCHITECTURE IMPORTS ---
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# --- LOCAL IMPORTS ---
import models  # Imports all classes (User, Thought, Task, ChatHistory)
import schemas
from database import engine, get_db

# --- CONFIGURATION ---
def _load_env_file(env_path: str) -> None:
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

_load_env_file(os.path.join(os.path.dirname(__file__), ".env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Rate limiting storage
user_request_times = {} 

# Create Tables
models.Base.metadata.create_all(bind=engine)

# --- PYDANTIC SCHEMAS ---
class TaskCreate(BaseModel):
    content: str
    due_date: str

class TaskUpdate(BaseModel):
    is_done: bool

class ChatRequest(BaseModel):
    user_id: int
    message: str

class Feedback(BaseModel):
    user_id: int
    outcome: int 

# --- 1. ENHANCED CONTEXT VECTOR (4 DIMENSIONS) ---
print("🧠 Initializing Context-Aware JITAI Model...")

scaler = StandardScaler()

# [Mood (1-10), PendingTasks (Count), HourOfDay (0-23), CompletionRate (0.0-1.0)]
# Classes: 0=Crisis, 1=Academic, 2=Motivation, 3=Neutral/None
X_warmup = [
    # CRISIS (Label 0): Strictly Low Mood (1-3)
    [1, 0, 9, 0.0], [2, 5, 23, 0.1], [3, 0, 3, 0.0], [1, 10, 12, 0.0],

    # ACADEMIC (Label 1): Mid Mood (4-7) + High Tasks (>3)
    # 👇 THIS LINE FIXES YOUR DEMO (Mood 5, 5 Tasks, Day time, 0% Done)
    [5, 5, 12, 0.0], 
    [6, 10, 14, 0.8], [4, 4, 11, 0.2], [5, 8, 16, 0.1],

    # MOTIVATION (Label 2): High Mood (8-10)
    [8, 0, 20, 0.9], [9, 1, 18, 1.0], [7, 0, 9, 0.5],

    # NEUTRAL (Label 3): Mid Mood + Low Tasks
    [5, 0, 12, 1.0], [6, 0, 22, 1.0], [5, 1, 14, 0.5]
]
# Make sure the labels match the number of rows in X_warmup!
y_warmup = [
    0, 0, 0, 0,   # 4 Crisis entries
    1, 1, 1, 1,   # 4 Academic entries
    2, 2, 2,      # 3 Motivation entries
    3, 3, 3       # 3 Neutral entries
]

scaler.fit(X_warmup)
X_scaled = scaler.transform(X_warmup)

jitai_model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, random_state=42)
jitai_model.partial_fit(X_scaled, y_warmup, classes=[0, 1, 2, 3])

print("✅ Model Trained & Ready for Online Learning.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- HELPER: GET 4D CONTEXT VECTOR ---
def get_user_context(user_id: int, db: Session):
    recent_thought = db.query(models.Thought).filter(models.Thought.owner_id == user_id).order_by(models.Thought.id.desc()).first()
    mood = recent_thought.mood_score if recent_thought else 5
    
    all_tasks = db.query(models.Task).filter(models.Task.user_id == user_id).all()
    total = len(all_tasks)
    pending = sum(1 for t in all_tasks if not t.is_done)
    completed = total - pending
    
    completion_rate = 0.5 
    if total > 0:
        completion_rate = completed / total
        
    current_hour = datetime.now().hour

    return [mood, pending, current_hour, completion_rate]

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {"message": "MindLink API Ready"}

@app.post("/users/", response_model=schemas.UserResponse)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = models.User(email=user.email, password_hash=user.password + "notreallyhashed")
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/login/")
def login_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or db_user.password_hash != user.password + "notreallyhashed":
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    return {"message": "Login successful", "user_id": db_user.id}

@app.post("/thoughts/", response_model=schemas.ThoughtResponse)
def create_thought(thought: schemas.ThoughtCreate, user_id: int, db: Session = Depends(get_db)):
    new_thought = models.Thought(content=thought.content, mood_score=thought.mood_score, owner_id=user_id)
    db.add(new_thought)
    db.commit()
    db.refresh(new_thought)
    return new_thought

@app.get("/thoughts/{user_id}", response_model=list[schemas.ThoughtResponse])
def read_thoughts(user_id: int, db: Session = Depends(get_db)):
    return db.query(models.Thought).filter(models.Thought.owner_id == user_id).all()

@app.delete("/thoughts/{thought_id}")
def delete_thought(thought_id: int, db: Session = Depends(get_db)):
    thought = db.query(models.Thought).filter(models.Thought.id == thought_id).first()
    if not thought:
        raise HTTPException(status_code=404, detail="Thought not found")
    db.delete(thought)
    db.commit()
    return {"message": "Thought deleted"}

@app.post("/tasks/")
def create_task(task: TaskCreate, user_id: int, db: Session = Depends(get_db)):
    db_task = models.Task(content=task.content, due_date=task.due_date, user_id=user_id)
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

@app.get("/tasks/{user_id}")
def get_tasks(user_id: int, db: Session = Depends(get_db)):
    return db.query(models.Task).filter(models.Task.user_id == user_id).all()

@app.delete("/tasks/delete/{task_id}")
def delete_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    db.delete(task)
    db.commit()
    return {"message": "Task deleted"}

@app.put("/tasks/{task_id}")
def update_task_status(task_id: int, update: TaskUpdate, db: Session = Depends(get_db)):
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not task:
        return {"error": "Task not found"}
    
    was_pending = not task.is_done
    is_becoming_done = update.is_done
    
    task.is_done = update.is_done
    db.commit()
    
    proactive_msg = None
    
    if was_pending and is_becoming_done:
        context = get_user_context(task.user_id, db)
        hour = context[2]
        pending_count = context[1]
        
        msg = f"Nice work finishing '{task.content}'! ✅"
        
        if pending_count > 0:
             msg += " You're on a roll. Ready to tackle the next one?"
        elif hour >= 22:
             msg += " That was the last one! Time to unwind and sleep?"
        else:
             msg += " All tasks cleared! How are you feeling now?"
             
        db_msg = models.ChatHistory(
            user_id=task.user_id,
            role="assistant",
            content=msg,
            timestamp=datetime.now().isoformat()
        )
        db.add(db_msg)
        db.commit()
        
        proactive_msg = msg

    return {"message": "Status updated", "proactive_feedback": proactive_msg}

def check_rate_limit(user_id: int, limit_seconds: int = 2) -> bool:
    now = time.time()
    last_request = user_request_times.get(user_id, 0)
    if now - last_request < limit_seconds:
        return False
    user_request_times[user_id] = now
    return True

@app.post("/chat/")
def chat_with_ai(request: ChatRequest, db: Session = Depends(get_db)):
    if not check_rate_limit(request.user_id):
        return {"response": "Please wait a moment.", "alert": False, "intervention": "RATE_LIMITED"}
    
    user_message = request.message.strip()
    critical_triggers = ["suicide", "kill myself", "die", "hurt myself"]
    if any(t in user_message.lower() for t in critical_triggers):
        return {"response": "🚨 CRISIS DETECTED. Please contact help.", "alert": True, "intervention": "CRITICAL_SELF_HARM"}

    db.add(models.ChatHistory(user_id=request.user_id, role="user", content=user_message, timestamp=datetime.now().isoformat()))
    db.commit()

    recent_history = db.query(models.ChatHistory).filter(models.ChatHistory.user_id == request.user_id).order_by(models.ChatHistory.id.desc()).limit(6).all()
    recent_history = recent_history[::-1] 

    context_vector = get_user_context(request.user_id, db)
    features_scaled = scaler.transform([context_vector])
    state_id = int(jitai_model.predict(features_scaled)[0])
    state_map = {0: "CRISIS", 1: "ACADEMIC", 2: "MOTIVATION", 3: "NEUTRAL"}
    current_state = state_map.get(state_id, "NEUTRAL")

    system_prompt = f"You are MindLink. User state: {current_state}. Keep it concise."
    messages_payload = [{"role": "system", "content": system_prompt}]
    for msg in recent_history:
        messages_payload.append({"role": msg.role, "content": msg.content})

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "llama-3.3-70b-versatile", "messages": messages_payload, "max_tokens": 150}
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            ai_reply = response.json()['choices'][0]['message']['content'].strip()
            db.add(models.ChatHistory(user_id=request.user_id, role="assistant", content=ai_reply, timestamp=datetime.now().isoformat()))
            db.commit()
            
            intervention = "GROQ_LLAMA3"
            if "<OPEN_PLANNER>" in ai_reply:
                intervention = "ACADEMIC"
                ai_reply = ai_reply.replace("<OPEN_PLANNER>", "").strip()
            return {"response": ai_reply, "alert": False, "intervention": intervention}
            
    except Exception as e:
        print(e)
    
    return {"response": "I'm having trouble thinking.", "alert": False, "intervention": "ERROR"}

@app.get("/jitai/{user_id}")
def get_jitai_intervention(user_id: int, db: Session = Depends(get_db)):
    # 1. Get Real-Time Context
    context_vector = get_user_context(user_id, db)
    
    # 2. Scale Features
    features_scaled = scaler.transform([context_vector])
    
    # 3. Predict State (0, 1, 2, or 3)
    state_id = int(jitai_model.predict(features_scaled)[0])
    
    state_map = {0: "CRISIS", 1: "ACADEMIC", 2: "MOTIVATION", 3: "NONE"}
    jitai_type = state_map.get(state_id, "NONE")
    
    message = ""
    if jitai_type == "ACADEMIC":
        message = "You have pending tasks. Want to focus?"
    elif jitai_type == "CRISIS":
        message = "High stress detected. Let's take a breather."
    elif jitai_type == "MOTIVATION":
        message = "Great progress! Keep the momentum."
    
    # Note: If type is NONE, frontend hides the banner.
    return {"type": jitai_type, "message": message}

@app.post("/jitai/feedback")
def record_feedback(data: Feedback, db: Session = Depends(get_db)):
    """
    Implements Online Learning (Contextual Bandit-style).
    Updates the model weights based on user feedback.
    """
    print(f"📉 Feedback Received for User {data.user_id}: Outcome {data.outcome}")

    # 1. Re-acquire context at the moment of feedback
    # (In a production app, you might pass the context_id from the frontend to be exact)
    context_vector = get_user_context(data.user_id, db)
    features_scaled = scaler.transform([context_vector])
    
    # 2. Get what the model would predict NOW
    current_prediction = int(jitai_model.predict(features_scaled)[0])
    
    # 3. UPDATE THE MODEL (Reinforcement Step)
    if data.outcome == 1:
        # Positive Reinforcement: "Yes, this prediction was helpful in this context."
        # We tell the model: For these features, the correct label IS the current prediction.
        print(f"✅ Reinforcing State {current_prediction} for context {context_vector}")
        jitai_model.partial_fit(features_scaled, [current_prediction])
        
    elif data.outcome == 0:
        # Negative Reinforcement: "No, this was wrong."
        # We nudge the model towards the "Neutral/None" class (Label 3).
        # This teaches the model: "In this context, better to do nothing than to annoy the user."
        print(f"❌ Correcting: Shifting towards Neutral (3) for context {context_vector}")
        jitai_model.partial_fit(features_scaled, [3])

    return {"status": "Model Updated", "weights": "adapted"}