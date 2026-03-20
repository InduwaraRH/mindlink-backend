import os
import json
import time
import requests
from datetime import datetime, timedelta, timezone

import bcrypt

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# THESIS ARCHITECTURE IMPORTS
#
# RandomForestClassifier replaces SGDClassifier for two reasons:
#   1. Non-linear decision boundaries — RF correctly prioritises low mood as a
#      CRISIS signal even when competing features (e.g. high completion rate)
#      pull a linear model toward a different class.
#   2. Enables Option A (confidence gating) via predict_proba(), which SGD
#      does not support reliably with small datasets.
#
# RC1 is implemented as batch retraining (fit on all accumulated examples)
# rather than partial_fit(), because RF does not support incremental learning.
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


import models
import schemas
from database import engine, get_db
# Timezone offset in hours from UTC
# Set TZ_OFFSET=5.5 in Render environment variables for Sri Lanka (UTC+5:30)
# Defaults to 0 (UTC) — local dev already uses system time via datetime.now()

def get_local_hour() -> int:
    if TZ_OFFSET_HOURS == 0:
        return datetime.now().hour  # Local dev — uses PC system time
    tz = timezone(timedelta(hours=TZ_OFFSET_HOURS))
    return datetime.now(tz).hour

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
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
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        print(f"⚠️ Failed to load .env file: {exc}")


_load_env_file(os.path.join(os.path.dirname(__file__), ".env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TZ_OFFSET_HOURS = float(os.getenv("TZ_OFFSET", "0"))

user_request_times = {}
models.Base.metadata.create_all(bind=engine)


# ─────────────────────────────────────────────────────────────────────────────
# PASSWORD HASHING — bcrypt
#
# Replaces the previous stub implementation (password + "notreallyhashed").
# Uses bcrypt directly for secure password storage.
# ─────────────────────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class TaskCreate(BaseModel):
    content: str
    due_date: str


class TaskUpdate(BaseModel):
    is_done: bool


class TaskEdit(BaseModel):
    content: str
    due_date: str


class ChatRequest(BaseModel):
    user_id: int
    message: str


class Feedback(BaseModel):
    event_id: int
    outcome: int  # 1 = helpful, 0 = not helpful


# ─────────────────────────────────────────────────────────────────────────────
# OPTION A — CONFIDENCE GATE THRESHOLD
#
# Academic contribution: Standard RandomForestClassifier always outputs a
# prediction, even when the ensemble is uncertain (e.g. 35% CRISIS, 33%
# ACADEMIC, 32% NEUTRAL). Delivering an intervention based on a near-random
# prediction is clinically unsafe.
#
# This threshold adds a confidence gate: if the model's maximum class
# probability does not exceed CONFIDENCE_THRESHOLD, the system returns NEUTRAL
# rather than acting on an unreliable prediction. This prevents low-confidence
# predictions from triggering inappropriate interventions — a modification not
# present in off-the-shelf RF implementations.
#
# CRISIS exception: The gate is deliberately lowered for CRISIS predictions.
# Clinical safety requires erring on the side of caution — a 45% CRISIS
# probability still warrants a supportive response even if the model is not
# fully certain.
# ─────────────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60         # General gate: must exceed 60% to act
CRISIS_CONFIDENCE_THRESHOLD = 0.40  # Lower gate for CRISIS — safety first


def predict_with_confidence(features_scaled: np.ndarray) -> tuple:
    """
    Option A — Confidence-gated prediction.

    Returns (state_id, confidence, explanation_str).
    If confidence is below threshold the system defaults to NEUTRAL (state 3).
    CRISIS uses a lower threshold (CRISIS_CONFIDENCE_THRESHOLD) because
    under-detecting a crisis is more harmful than over-detecting one.
    """
    probabilities = jitai_model.predict_proba(features_scaled)[0]
    predicted_class = int(np.argmax(probabilities))
    max_confidence = float(np.max(probabilities))

    state_map = {0: "CRISIS", 1: "ACADEMIC", 2: "MOTIVATION", 3: "NEUTRAL"}

    # Apply class-specific confidence threshold
    effective_threshold = CRISIS_CONFIDENCE_THRESHOLD if predicted_class == 0 else CONFIDENCE_THRESHOLD

    if max_confidence < effective_threshold:
        explanation = (
            f"Low confidence ({max_confidence:.0%}) below threshold "
            f"({effective_threshold:.0%}) — defaulting to NEUTRAL. "
            f"Raw: CRISIS={probabilities[0]:.2f}, ACADEMIC={probabilities[1]:.2f}, "
            f"MOTIVATION={probabilities[2]:.2f}, NEUTRAL={probabilities[3]:.2f}"
        )
        print(f"⚠️  Confidence gate triggered: {explanation}")
        return 3, max_confidence, explanation
    else:
        explanation = (
            f"Confidence: {max_confidence:.0%} → {state_map[predicted_class]}. "
            f"Probabilities: CRISIS={probabilities[0]:.2f}, ACADEMIC={probabilities[1]:.2f}, "
            f"MOTIVATION={probabilities[2]:.2f}, NEUTRAL={probabilities[3]:.2f}"
        )
        print(f"✅  Prediction: {explanation}")
        return predicted_class, max_confidence, explanation


# ─────────────────────────────────────────────────────────────────────────────
# WARM-START DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("🌲 Initializing Context-Aware JITAI Model (RandomForest + Options A & B)...")

scaler = StandardScaler()

# Feature vector: [Mood (1–10), PendingTasks (count), HourOfDay (0–23), CompletionRate (0.0–1.0)]
# Classes: 0=CRISIS, 1=ACADEMIC, 2=MOTIVATION, 3=NEUTRAL
X_warmup = [
    # ── CRISIS (0) ────────────────────────────────────────────────────────────
    [1, 8, 23, 0.1],
    [2, 6,  1, 0.2],
    [3, 10, 22, 0.0],
    [2,  4,  0, 0.3],
    [1,  2, 20, 0.9],   # KEY: very low mood + HIGH completion -> still CRISIS
    [2,  1, 14, 0.95],  # KEY: very low mood + daytime + near-complete -> CRISIS
    [3,  0, 12, 1.0],   # KEY: low mood + ALL tasks done -> still CRISIS
    [1,  5, 10, 0.8],   # KEY: very low mood + high completion -> CRISIS
    # ── ACADEMIC (1) ──────────────────────────────────────────────────────────
    [5,  7, 12, 0.2],
    [6, 10, 14, 0.4],
    [4,  5, 11, 0.1],
    [7,  8, 16, 0.3],
    [5,  6,  9, 0.25],
    [6,  9, 13, 0.35],
    # ── MOTIVATION (2) ────────────────────────────────────────────────────────
    [8,  2, 10, 0.8],
    [9,  1, 18, 0.9],
    [10, 0, 15, 1.0],
    [8,  3, 20, 0.7],
    [9,  2, 11, 0.85],
    [7,  1, 16, 0.9],
    # ── NEUTRAL (3) ───────────────────────────────────────────────────────────
    [5,  0, 12, 0.5],
    [6,  1, 21, 0.5],
    [7,  0,  9, 0.6],
    [4,  2, 17, 0.4],
    [6,  0, 15, 0.55],
    [5,  1, 19, 0.45],
]

y_warmup = [
    0, 0, 0, 0, 0, 0, 0, 0,   # CRISIS  (8 examples)
    1, 1, 1, 1, 1, 1,          # ACADEMIC (6 examples)
    2, 2, 2, 2, 2, 2,          # MOTIVATION (6 examples)
    3, 3, 3, 3, 3, 3,          # NEUTRAL (6 examples)
]

# ─────────────────────────────────────────────────────────────────────────────
# OPTION B — CLINICALLY-INFORMED SAMPLE WEIGHTING
# ─────────────────────────────────────────────────────────────────────────────
sample_weights_warmup = [
    3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,  # CRISIS: 3x clinical weight
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,              # ACADEMIC: standard
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,              # MOTIVATION: standard
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,              # NEUTRAL: standard
]

scaler.fit(X_warmup)
X_scaled = scaler.transform(X_warmup)

jitai_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)
jitai_model.fit(X_scaled, y_warmup, sample_weight=sample_weights_warmup)

X_accumulated = list(X_scaled)
y_accumulated = list(y_warmup)
w_accumulated = list(sample_weights_warmup)

print("✅ RandomForest Model Trained & Ready (Option A + Option B active).")
print(f"   Feature importances -> "
      f"Mood={jitai_model.feature_importances_[0]:.3f}, "
      f"Pending={jitai_model.feature_importances_[1]:.3f}, "
      f"Hour={jitai_model.feature_importances_[2]:.3f}, "
      f"Completion={jitai_model.feature_importances_[3]:.3f}")
print(f"   Confidence gate: general={CONFIDENCE_THRESHOLD:.0%}, "
      f"CRISIS={CRISIS_CONFIDENCE_THRESHOLD:.0%}")


# ─────────────────────────────────────────────────────────────────────────────
# JITAI STATE-ADAPTIVE SYSTEM PROMPTS (RC2)
# ─────────────────────────────────────────────────────────────────────────────
STATE_SYSTEM_PROMPTS = {
    "CRISIS": """You are Sage, the MindLink AI companion. The system has detected that this \
user is in a HIGH-STRESS or CRISIS state (low mood score, high task load, or late-night check-in).

Your ONLY goals right now are:
1. Acknowledge their feelings warmly and without judgement. Do NOT minimise what they are feeling.
2. Use brief, calming language. Short sentences. No lists or bullet points.
3. Gently ask ONE open question to help them feel heard (e.g. "What's weighing on you most right now?").
4. If they express any self-harm ideation, immediately respond with: \
"I hear you. Please reach out to a crisis line right now — you don't have to face this alone. \
In Sri Lanka: 1926 (Sumithrayo). You matter."
5. Do NOT offer productivity advice, task suggestions, or solutions. This is not the time.
6. Keep your response under 60 words. Warmth over information.""",

    "ACADEMIC": """You are Sage, the MindLink AI companion. The system has detected that this \
user is in an ACADEMIC PRESSURE state (multiple pending tasks, daytime hours, moderate mood).

Your goals:
1. Be calm, structured, and gently motivating — like a supportive study partner.
2. Help the user break down their workload. If they mention a task, suggest ONE concrete \
first step using the Pomodoro principle (e.g. "Try 25 minutes on X first — what would \
make that feel manageable?").
3. Acknowledge stress briefly but pivot quickly to actionable support.
4. You may reference their task planner ("Have you checked your planner?") to encourage \
engagement with the productivity features.
5. Keep responses concise and structured. Maximum 80 words.
6. Tone: focused, warm, practical. Like a calm librarian who believes in them.""",

    "MOTIVATION": """You are Sage, the MindLink AI companion. The system has detected that this \
user is in a HIGH MOTIVATION state (elevated mood, strong task completion rate, positive momentum).

Your goals:
1. Match their energy — be enthusiastic and celebratory without being over the top.
2. Reinforce their self-efficacy. Use specific affirmations tied to what they've accomplished \
(e.g. "Finishing tasks consistently is a real skill — that's not nothing.").
3. Help them think ambitiously. Ask what they want to tackle next or what goal they are \
working toward.
4. This is a great moment for light goal-setting or reflection (e.g. "What would make \
today a 10/10?").
5. Keep responses upbeat and forward-looking. Maximum 80 words.
6. Tone: energising, genuine, like a coach who just saw them win.""",

    "NEUTRAL": """You are Sage, the MindLink AI companion. The user is in a BASELINE / NEUTRAL \
state — no acute stress, no crisis, no particular academic pressure.

Your goals:
1. Be warm, curious, and open-ended. Let the user set the direction of the conversation.
2. Do not push any agenda. Do not suggest tasks or productivity tools unless the user asks.
3. If the user seems to want to talk, reflect their feelings back gently using active listening \
(e.g. "It sounds like you're feeling... Is that right?").
4. If the user asks for help with something specific, assist naturally and helpfully.
5. Keep responses conversational. Maximum 80 words.
6. Tone: like a calm, trustworthy friend who has time to listen."""
}


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_user_context(user_id: int, db: Session):
    print(f"🔍 get_user_context called for user_id={user_id}")

    recent_thought = (
        db.query(models.Thought)
        .filter(models.Thought.owner_id == user_id)
        .order_by(models.Thought.id.desc())
        .first()
    )
    print(f"   → recent_thought: {recent_thought}, mood={recent_thought.mood_score if recent_thought else 'NONE'}")

    mood = recent_thought.mood_score if recent_thought else 5

    all_tasks = db.query(models.Task).filter(models.Task.user_id == user_id).all()
    print(f"   → tasks found: {len(all_tasks)}")

    total = len(all_tasks)
    pending = sum(1 for t in all_tasks if not t.is_done)
    completed = total - pending
    completion_rate = (completed / total) if total > 0 else 0.5

    current_hour = get_local_hour()
    print(f"   → context vector: [{mood}, {pending}, {current_hour}, {completion_rate:.3f}]")

    return [mood, pending, current_hour, completion_rate]


def check_rate_limit(user_id: int, limit_seconds: int = 2) -> bool:
    now = time.time()
    last_request = user_request_times.get(user_id, 0)
    if now - last_request < limit_seconds:
        return False
    user_request_times[user_id] = now
    return True


def retrain_model(X_train: list, y_train: list, w_train: list) -> None:
    """
    RC1 — Batch retraining with Option B sample weights preserved.
    """
    global jitai_model
    if len(set(y_train)) < 2:
        print("⚠️ Skipping retrain — fewer than 2 classes in training data.")
        return
    jitai_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )
    jitai_model.fit(X_train, y_train, sample_weight=w_train)
    print(f"🔄 Model retrained on {len(X_train)} examples (Option B weights applied).")
    print(f"   Feature importances -> "
          f"Mood={jitai_model.feature_importances_[0]:.3f}, "
          f"Pending={jitai_model.feature_importances_[1]:.3f}, "
          f"Hour={jitai_model.feature_importances_[2]:.3f}, "
          f"Completion={jitai_model.feature_importances_[3]:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "MindLink API Ready"}


@app.post("/users/", response_model=schemas.UserResponse)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = models.User(
        email=user.email,
        password_hash=hash_password(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/login/")
def login_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
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


@app.put("/tasks/edit/{task_id}")
def edit_task(task_id: int, update: TaskEdit, db: Session = Depends(get_db)):
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.content = update.content
    task.due_date = update.due_date
    db.commit()
    return {"message": "Task updated"}


@app.put("/tasks/{task_id}")
def update_task_status(task_id: int, update: TaskUpdate, db: Session = Depends(get_db)):
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

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


@app.post("/chat/")
def chat_with_ai(request: ChatRequest, db: Session = Depends(get_db)):
    # 1. Rate limiting
    if not check_rate_limit(request.user_id):
        return {"response": "Please wait a moment before sending another message.", "alert": False, "intervention": "RATE_LIMITED"}

    user_message = request.message.strip()

    # 2. ── RC3: DETERMINISTIC SAFETY BYPASS ──────────────────────────────────
    critical_triggers = [
        "suicide", "kill myself", "want to die", "end my life",
        "hurt myself", "hurting myself", "harm myself", "harming myself",
        "self harm", "self-harm", "cut myself", "cutting myself",
        "can't go on", "cannot go on"
    ]
    if any(t in user_message.lower() for t in critical_triggers):
        return {
            "response": (
                "🚨 I hear you, and I'm glad you reached out. "
                "What you're feeling right now is real, and you don't have to face it alone. "
                "Please contact a crisis line immediately — in Sri Lanka: Sumithrayo 1926 (24/7). "
                "I'm here with you, but a trained counsellor can help you right now."
            ),
            "alert": True,
            "intervention": "CRITICAL_SELF_HARM"
        }

    # 3. Save user message to chat history
    db.add(models.ChatHistory(
        user_id=request.user_id,
        role="user",
        content=user_message,
        timestamp=datetime.now().isoformat()
    ))
    db.commit()

    # 4. Fetch last 6 messages for LLM memory window
    recent_history = (
        db.query(models.ChatHistory)
        .filter(models.ChatHistory.user_id == request.user_id)
        .order_by(models.ChatHistory.id.desc())
        .limit(6)
        .all()
    )
    recent_history = recent_history[::-1]

    # 5. ── RC2: COMPUTE 4D CONTEXT VECTOR ────────────────────────────────────
    context_vector = get_user_context(request.user_id, db)
    features_scaled = scaler.transform([context_vector])

    state_id, confidence, confidence_explanation = predict_with_confidence(features_scaled)

    state_map = {0: "CRISIS", 1: "ACADEMIC", 2: "MOTIVATION", 3: "NEUTRAL"}
    current_state = state_map.get(state_id, "NEUTRAL")

    # 6. Select state-adaptive system prompt
    system_prompt = STATE_SYSTEM_PROMPTS.get(current_state, STATE_SYSTEM_PROMPTS["NEUTRAL"])

    mood, pending, hour, completion_rate = context_vector
    system_prompt += (
        f"\n\n[LIVE CONTEXT — do not recite these numbers unless helpful]"
        f"\nMood score: {mood}/10 | Pending tasks: {pending} | "
        f"Hour: {hour}:00 | Completion rate: {completion_rate:.0%}"
        f"\nDetected state: {current_state} (confidence: {confidence:.0%})"
    )

    messages_payload = [{"role": "system", "content": system_prompt}]
    for msg in recent_history:
        messages_payload.append({"role": msg.role, "content": msg.content})

    # 7. Call Groq LLM
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages_payload,
            "max_tokens": 150,
            "temperature": 0.7
        }

        response = requests.post(url, json=payload, headers=headers, timeout=10)

        if response.status_code == 200:
            ai_reply = response.json()["choices"][0]["message"]["content"].strip()

            db.add(models.ChatHistory(
                user_id=request.user_id,
                role="assistant",
                content=ai_reply,
                timestamp=datetime.now().isoformat()
            ))
            db.commit()

            intervention = current_state
            if "<OPEN_PLANNER>" in ai_reply:
                intervention = "ACADEMIC"
                ai_reply = ai_reply.replace("<OPEN_PLANNER>", "").strip()

            return {
                "response": ai_reply,
                "alert": False,
                "intervention": intervention,
                "state": current_state,
                "confidence": round(confidence, 3),
                "context_vector": context_vector
            }

    except Exception as e:
        print(f"Groq API error: {e}")

    return {"response": "I'm having trouble thinking right now. Please try again in a moment.", "alert": False, "intervention": "ERROR"}


@app.get("/jitai/{user_id}")
def get_jitai_intervention(user_id: int, db: Session = Depends(get_db)):
    """
    RC2 — Context-aware JITAI delivery with Option A confidence gating.
    """
    cooldown_minutes = 15  # Set to 1 for testing, 15 for production
    last_event = (
        db.query(models.JitaiEvent)
        .filter(models.JitaiEvent.user_id == user_id)
        .order_by(models.JitaiEvent.id.desc())
        .first()
    )

    if last_event and last_event.created_at:
        now = datetime.now(last_event.created_at.tzinfo) if last_event.created_at.tzinfo else datetime.now()
        if (now - last_event.created_at) < timedelta(minutes=cooldown_minutes):
            context_vector = get_user_context(user_id, db)
            features_scaled = scaler.transform([context_vector])
            state_id, confidence, _ = predict_with_confidence(features_scaled)
            state_map = {0: "CRISIS", 1: "ACADEMIC", 2: "MOTIVATION", 3: "NONE"}
            current_type = state_map.get(state_id, "NONE")
            return {
                "type": "NONE",
                "message": "",
                "context_vector": context_vector,
                "state": current_type,
                "confidence": round(confidence, 3),
            }

    context_vector = get_user_context(user_id, db)
    features_scaled = scaler.transform([context_vector])

    state_id, confidence, confidence_explanation = predict_with_confidence(features_scaled)

    state_map = {0: "CRISIS", 1: "ACADEMIC", 2: "MOTIVATION", 3: "NONE"}
    jitai_type = state_map.get(state_id, "NONE")

    if state_id == 3:
        jitai_type = "NONE"

    message = ""
    if jitai_type == "ACADEMIC":
        message = "You have pending tasks. Want to focus?"
    elif jitai_type == "CRISIS":
        message = "High stress detected. Let's take a breather."
    elif jitai_type == "MOTIVATION":
        message = "Great progress! Keep the momentum going."

    event = models.JitaiEvent(
        user_id=user_id,
        jitai_type=jitai_type,
        state_id=state_id,
        context_json=json.dumps(context_vector),
        outcome=None
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    print(f"📢 JITAI event created: type={jitai_type}, confidence={confidence:.0%}, "
          f"event_id={event.id}")

    return {
        "type": jitai_type,
        "message": message,
        "event_id": event.id,
        "context_vector": context_vector,
        "confidence": round(confidence, 3),
        "confidence_explanation": confidence_explanation,
    }


@app.post("/jitai/feedback")
def record_feedback(data: Feedback, db: Session = Depends(get_db)):
    """
    RC1 — Adaptive Feedback Loop with Option B weight preservation.
    """
    global X_accumulated, y_accumulated, w_accumulated

    event = db.query(models.JitaiEvent).filter(models.JitaiEvent.id == data.event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    event.outcome = data.outcome
    db.commit()

    context_vector = json.loads(event.context_json)
    features_scaled = scaler.transform([context_vector])
    predicted_state = int(event.state_id)

    if data.outcome == 1:
        weight = 3.0 if predicted_state == 0 else 1.0
        X_accumulated.append(features_scaled[0].tolist())
        y_accumulated.append(predicted_state)
        w_accumulated.append(weight)
        trained_toward = predicted_state
        print(f"✅ Feedback: reinforcing state {predicted_state} "
              f"(weight={weight}) for context {context_vector}")
    else:
        weight = 2.0 if predicted_state == 0 else 1.0
        X_accumulated.append(features_scaled[0].tolist())
        y_accumulated.append(3)  # NEUTRAL
        w_accumulated.append(weight)
        trained_toward = 3
        print(f"⚠️ Feedback: nudging toward NEUTRAL "
              f"(weight={weight}) for context {context_vector}")

    retrain_model(X_accumulated, y_accumulated, w_accumulated)

    return {
        "status": "Model Retrained",
        "event_id": event.id,
        "outcome": event.outcome,
        "trained_toward": trained_toward,
        "total_training_examples": len(X_accumulated),
        "crisis_weight_applied": weight if predicted_state == 0 else None,
    }