import os
import json
import time
import requests
from datetime import datetime, timedelta, timezone

import bcrypt
import numpy as np

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import models
import schemas
from database import engine, get_db


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

user_request_times = {}
models.Base.metadata.create_all(bind=engine)


# ─────────────────────────────────────────────────────────────────────────────
# TIMEZONE — Sri Lanka Standard Time (UTC+5:30)
# ─────────────────────────────────────────────────────────────────────────────
SRI_LANKA_TZ = timezone(timedelta(hours=5, minutes=30))


def get_local_hour() -> int:
    return datetime.now(SRI_LANKA_TZ).hour


# ─────────────────────────────────────────────────────────────────────────────
# PASSWORD HASHING — bcrypt
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
# ─────────────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60
CRISIS_CONFIDENCE_THRESHOLD = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# TUNED MODEL CONFIGURATION
# Final selected model from evaluation:
# - Balanced synthetic dataset (1000 samples)
# - Feature engineering (9 features total)
# - Hyperparameter-tuned Random Forest
# - Crisis-sensitive weighting retained for safety
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["CRISIS", "ACADEMIC", "MOTIVATION", "NEUTRAL"]
STATE_MAP = {0: "CRISIS", 1: "ACADEMIC", 2: "MOTIVATION", 3: "NEUTRAL"}

BEST_CRISIS_WEIGHT = 2.5
BEST_RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "min_samples_split": 6,
    "max_features": "sqrt",
    "random_state": 42,
    "class_weight": "balanced",
}

N_PER_CLASS = 250
NOISE_RATE = 0.04

FEATURE_NAMES = [
    "Mood",
    "PendingTasks",
    "HourOfDay",
    "CompletionRate",
    "TaskPressure",
    "LateNight",
    "LowMood",
    "LowMoodHighCompletion",
    "LowMoodHighPending",
]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def add_engineered_features(X_input: np.ndarray) -> np.ndarray:
    """
    Input columns:
    0 = Mood
    1 = PendingTasks
    2 = HourOfDay
    3 = CompletionRate

    Output:
    Original 4 features + 5 engineered features
    """
    mood = X_input[:, 0]
    pending = X_input[:, 1]
    hour = X_input[:, 2]
    completion = X_input[:, 3]

    task_pressure = pending * (1 - completion)
    late_night = ((hour >= 22) | (hour <= 4)).astype(int)
    low_mood = (mood <= 3).astype(int)
    low_mood_high_completion = ((mood <= 3) & (completion >= 0.8)).astype(int)
    low_mood_high_pending = ((mood <= 3) & (pending >= 6)).astype(int)

    return np.column_stack([
        X_input,
        task_pressure,
        late_night,
        low_mood,
        low_mood_high_completion,
        low_mood_high_pending,
    ])


def transform_context_vector(context_vector: list[float]) -> np.ndarray:
    """
    Converts raw 4D context vector into engineered 9D scaled feature vector.
    """
    raw = np.array([context_vector], dtype=float)
    engineered = add_engineered_features(raw)
    return scaler.transform(engineered)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC TRAINING DATA GENERATOR
# Mirrors the balanced evaluation dataset used for model selection.
# ─────────────────────────────────────────────────────────────────────────────
def clip_row(mood, pending, hour, completion):
    mood = int(np.clip(round(mood), 1, 10))
    pending = int(np.clip(round(pending), 0, 10))
    hour = int(np.clip(round(hour), 0, 23))
    completion = float(np.clip(completion, 0.0, 1.0))
    return [mood, pending, hour, completion]


def sample_crisis():
    mode = np.random.choice(
        ["classic", "high_completion", "borderline", "ambiguous"],
        p=[0.45, 0.22, 0.23, 0.10]
    )

    if mode == "classic":
        mood = np.random.normal(2.3, 1.1)
        pending = np.random.normal(6.6, 2.0)
        hour = np.random.normal(15, 5)
        completion = np.random.normal(0.24, 0.16)
    elif mode == "high_completion":
        mood = np.random.normal(2.5, 1.0)
        pending = np.random.normal(1.6, 1.2)
        hour = np.random.normal(13, 4)
        completion = np.random.normal(0.84, 0.09)
    elif mode == "borderline":
        mood = np.random.normal(3.6, 1.0)
        pending = np.random.normal(4.8, 1.8)
        hour = np.random.normal(14, 4)
        completion = np.random.normal(0.46, 0.14)
    else:
        mood = np.random.normal(4.0, 0.9)
        pending = np.random.normal(3.8, 1.8)
        hour = np.random.normal(14, 4)
        completion = np.random.normal(0.54, 0.12)

    return clip_row(mood, pending, hour, completion)


def sample_academic():
    mode = np.random.choice(
        ["classic", "borderline_low", "borderline_neutral", "ambiguous"],
        p=[0.50, 0.20, 0.20, 0.10]
    )

    if mode == "classic":
        mood = np.random.normal(5.1, 1.1)
        pending = np.random.normal(6.8, 1.8)
        hour = np.random.normal(13, 3.5)
        completion = np.random.normal(0.30, 0.12)
    elif mode == "borderline_low":
        mood = np.random.normal(4.3, 1.0)
        pending = np.random.normal(5.7, 1.6)
        hour = np.random.normal(14, 3.5)
        completion = np.random.normal(0.37, 0.12)
    elif mode == "borderline_neutral":
        mood = np.random.normal(5.4, 1.0)
        pending = np.random.normal(4.4, 1.6)
        hour = np.random.normal(14, 4)
        completion = np.random.normal(0.44, 0.12)
    else:
        mood = np.random.normal(5.8, 1.0)
        pending = np.random.normal(3.6, 1.6)
        hour = np.random.normal(14, 4)
        completion = np.random.normal(0.49, 0.12)

    return clip_row(mood, pending, hour, completion)


def sample_motivation():
    mode = np.random.choice(
        ["classic", "borderline", "near_neutral", "ambiguous"],
        p=[0.50, 0.20, 0.20, 0.10]
    )

    if mode == "classic":
        mood = np.random.normal(8.0, 1.0)
        pending = np.random.normal(1.7, 1.1)
        hour = np.random.normal(15, 4)
        completion = np.random.normal(0.82, 0.09)
    elif mode == "borderline":
        mood = np.random.normal(7.0, 1.0)
        pending = np.random.normal(2.4, 1.2)
        hour = np.random.normal(14, 4)
        completion = np.random.normal(0.69, 0.10)
    elif mode == "near_neutral":
        mood = np.random.normal(6.6, 1.0)
        pending = np.random.normal(2.8, 1.3)
        hour = np.random.normal(14, 4)
        completion = np.random.normal(0.61, 0.10)
    else:
        mood = np.random.normal(6.2, 1.0)
        pending = np.random.normal(3.1, 1.4)
        hour = np.random.normal(14, 4)
        completion = np.random.normal(0.56, 0.11)

    return clip_row(mood, pending, hour, completion)


def sample_neutral():
    mode = np.random.choice(
        ["classic", "low_side", "high_side", "ambiguous"],
        p=[0.45, 0.20, 0.20, 0.15]
    )

    if mode == "classic":
        mood = np.random.normal(5.8, 1.0)
        pending = np.random.normal(2.2, 1.3)
        hour = np.random.normal(14, 4)
        completion = np.random.normal(0.52, 0.10)
    elif mode == "low_side":
        mood = np.random.normal(4.9, 1.0)
        pending = np.random.normal(3.4, 1.5)
        hour = np.random.normal(14, 4)
        completion = np.random.normal(0.45, 0.11)
    elif mode == "high_side":
        mood = np.random.normal(6.8, 0.9)
        pending = np.random.normal(2.1, 1.3)
        hour = np.random.normal(15, 4)
        completion = np.random.normal(0.60, 0.10)
    else:
        mood = np.random.normal(5.7, 1.0)
        pending = np.random.normal(3.0, 1.5)
        hour = np.random.normal(14, 4)
        completion = np.random.normal(0.54, 0.11)

    return clip_row(mood, pending, hour, completion)


def generate_balanced_synthetic_dataset(seed: int = 42):
    np.random.seed(seed)

    X_list = []
    y_list = []

    for _ in range(N_PER_CLASS):
        X_list.append(sample_crisis())
        y_list.append(0)

    for _ in range(N_PER_CLASS):
        X_list.append(sample_academic())
        y_list.append(1)

    for _ in range(N_PER_CLASS):
        X_list.append(sample_motivation())
        y_list.append(2)

    for _ in range(N_PER_CLASS):
        X_list.append(sample_neutral())
        y_list.append(3)

    X_raw = np.array(X_list, dtype=float)
    y_data = np.array(y_list, dtype=int)

    n_noisy = int(len(y_data) * NOISE_RATE)
    noise_idx = np.random.choice(len(y_data), size=n_noisy, replace=False)

    nearby_class_map = {
        0: [1, 3],
        1: [0, 3],
        2: [3, 1],
        3: [1, 2],
    }

    for idx in noise_idx:
        original = y_data[idx]
        y_data[idx] = np.random.choice(nearby_class_map[original])

    perm = np.random.permutation(len(X_raw))
    X_raw = X_raw[perm]
    y_data = y_data[perm]

    sample_weights = np.where(y_data == 0, BEST_CRISIS_WEIGHT, 1.0)

    return X_raw, y_data, sample_weights


# ─────────────────────────────────────────────────────────────────────────────
# MODEL INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
print("🌲 Initializing MindLink tuned RF model (engineered features + tuned hyperparameters)...")

scaler = StandardScaler()
jitai_model = None
X_accumulated_raw = []
y_accumulated = []
w_accumulated = []


def fit_model_from_raw(X_raw_train, y_train, w_train) -> None:
    """
    Fits scaler + tuned RF on raw 4D context vectors.
    Feature engineering is applied internally before scaling.
    """
    global scaler, jitai_model

    X_raw_np = np.array(X_raw_train, dtype=float)
    y_np = np.array(y_train, dtype=int)
    w_np = np.array(w_train, dtype=float)

    X_engineered = add_engineered_features(X_raw_np)

    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_engineered)

    jitai_model = RandomForestClassifier(**BEST_RF_PARAMS)
    jitai_model.fit(X_scaled_train, y_np, sample_weight=w_np)


def initialize_tuned_model() -> None:
    global X_accumulated_raw, y_accumulated, w_accumulated

    X_raw, y_data, sample_weights = generate_balanced_synthetic_dataset(seed=42)

    fit_model_from_raw(X_raw, y_data, sample_weights)

    X_accumulated_raw = X_raw.tolist()
    y_accumulated = y_data.tolist()
    w_accumulated = sample_weights.tolist()

    print("✅ Tuned RandomForest Model Trained & Ready.")
    print(f"   Training examples: {len(X_accumulated_raw)}")
    print(f"   Engineered features: {len(FEATURE_NAMES)}")
    print("   Feature importances:")
    for name, importance in zip(FEATURE_NAMES, jitai_model.feature_importances_):
        print(f"      {name:<22} = {importance:.3f}")
    print(
        f"   Confidence gate: general={CONFIDENCE_THRESHOLD:.0%}, "
        f"CRISIS={CRISIS_CONFIDENCE_THRESHOLD:.0%}"
    )
    print(
        f"   Tuned params: n_estimators={BEST_RF_PARAMS['n_estimators']}, "
        f"max_depth={BEST_RF_PARAMS['max_depth']}, "
        f"min_samples_leaf={BEST_RF_PARAMS['min_samples_leaf']}, "
        f"min_samples_split={BEST_RF_PARAMS['min_samples_split']}, "
        f"max_features={BEST_RF_PARAMS['max_features']}, "
        f"crisis_weight={BEST_CRISIS_WEIGHT}"
    )


initialize_tuned_model()


# ─────────────────────────────────────────────────────────────────────────────
# OPTION A — CONFIDENCE-GATED PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_with_confidence(context_vector: list[float]) -> tuple:
    """
    Returns:
      (state_id, confidence, explanation_str)

    Uses the tuned RF model over engineered features.
    If confidence is below threshold, defaults to NEUTRAL.
    CRISIS uses a lower threshold to prioritize safety.
    """
    features_scaled = transform_context_vector(context_vector)
    probabilities = jitai_model.predict_proba(features_scaled)[0]

    predicted_class = int(np.argmax(probabilities))
    max_confidence = float(np.max(probabilities))

    effective_threshold = (
        CRISIS_CONFIDENCE_THRESHOLD
        if predicted_class == 0
        else CONFIDENCE_THRESHOLD
    )

    if max_confidence < effective_threshold:
        explanation = (
            f"Low confidence ({max_confidence:.0%}) below threshold "
            f"({effective_threshold:.0%}) — defaulting to NEUTRAL. "
            f"Raw: CRISIS={probabilities[0]:.2f}, ACADEMIC={probabilities[1]:.2f}, "
            f"MOTIVATION={probabilities[2]:.2f}, NEUTRAL={probabilities[3]:.2f}"
        )
        print(f"⚠️  Confidence gate triggered: {explanation}")
        return 3, max_confidence, explanation

    explanation = (
        f"Confidence: {max_confidence:.0%} → {STATE_MAP[predicted_class]}. "
        f"Probabilities: CRISIS={probabilities[0]:.2f}, ACADEMIC={probabilities[1]:.2f}, "
        f"MOTIVATION={probabilities[2]:.2f}, NEUTRAL={probabilities[3]:.2f}"
    )
    print(f"✅ Prediction: {explanation}")
    return predicted_class, max_confidence, explanation


# ─────────────────────────────────────────────────────────────────────────────
# JITAI STATE-ADAPTIVE SYSTEM PROMPTS (RC2)
# ─────────────────────────────────────────────────────────────────────────────
STATE_SYSTEM_PROMPTS = {
    "CRISIS": """You are Sage, the MindLink AI companion. The system has detected that this user is in a HIGH-STRESS or CRISIS state (low mood score, high task load, or late-night check-in).

Your ONLY goals right now are:
1. Acknowledge their feelings warmly and without judgement. Do NOT minimise what they are feeling.
2. Use brief, calming language. Short sentences. No lists or bullet points.
3. Gently ask ONE open question to help them feel heard (e.g. "What's weighing on you most right now?").
4. If they express any self-harm ideation, immediately respond with: "I hear you. Please reach out to a crisis line right now — you don't have to face this alone. In Sri Lanka: 1926 (Sumithrayo). You matter."
5. Do NOT offer productivity advice, task suggestions, or solutions. This is not the time.
6. Keep your response under 60 words. Warmth over information.""",

    "ACADEMIC": """You are Sage, the MindLink AI companion. The system has detected that this user is in an ACADEMIC PRESSURE state (multiple pending tasks, daytime hours, moderate mood).

Your goals:
1. Be calm, structured, and gently motivating — like a supportive study partner.
2. Help the user break down their workload. If they mention a task, suggest ONE concrete first step using the Pomodoro principle (e.g. "Try 25 minutes on X first — what would make that feel manageable?").
3. Acknowledge stress briefly but pivot quickly to actionable support.
4. You may reference their task planner ("Have you checked your planner?") to encourage engagement with the productivity features.
5. Keep responses concise and structured. Maximum 80 words.
6. Tone: focused, warm, practical. Like a calm librarian who believes in them.""",

    "MOTIVATION": """You are Sage, the MindLink AI companion. The system has detected that this user is in a HIGH MOTIVATION state (elevated mood, strong task completion rate, positive momentum).

Your goals:
1. Match their energy — be enthusiastic and celebratory without being over the top.
2. Reinforce their self-efficacy. Use specific affirmations tied to what they've accomplished (e.g. "Finishing tasks consistently is a real skill — that's not nothing.").
3. Help them think ambitiously. Ask what they want to tackle next or what goal they are working toward.
4. This is a great moment for light goal-setting or reflection (e.g. "What would make today a 10/10?").
5. Keep responses upbeat and forward-looking. Maximum 80 words.
6. Tone: energising, genuine, like a coach who just saw them win.""",

    "NEUTRAL": """You are Sage, the MindLink AI companion. The user is in a BASELINE / NEUTRAL state — no acute stress, no crisis, no particular academic pressure.

Your goals:
1. Be warm, curious, and open-ended. Let the user set the direction of the conversation.
2. Do not push any agenda. Do not suggest tasks or productivity tools unless the user asks.
3. If the user seems to want to talk, reflect their feelings back gently using active listening (e.g. "It sounds like you're feeling... Is that right?").
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
    context_vector = [mood, pending, current_hour, completion_rate]

    print(f"   → raw context vector: {context_vector}")

    return context_vector


def check_rate_limit(user_id: int, limit_seconds: int = 2) -> bool:
    now = time.time()
    last_request = user_request_times.get(user_id, 0)
    if now - last_request < limit_seconds:
        return False
    user_request_times[user_id] = now
    return True


def retrain_model(X_raw_train: list, y_train: list, w_train: list) -> None:
    """
    RC1 — Batch retraining using the tuned RF architecture.
    Retraining is done from raw 4D context vectors so that:
    - engineered features are regenerated consistently
    - scaler is refit correctly
    - model remains aligned with the tuned backend pipeline
    """
    if len(set(y_train)) < 2:
        print("⚠️ Skipping retrain — fewer than 2 classes in training data.")
        return

    fit_model_from_raw(X_raw_train, y_train, w_train)

    print(f"🔄 Tuned RF retrained on {len(X_raw_train)} examples.")
    print("   Updated feature importances:")
    for name, importance in zip(FEATURE_NAMES, jitai_model.feature_importances_):
        print(f"      {name:<22} = {importance:.3f}")


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
        return {
            "response": "Please wait a moment before sending another message.",
            "alert": False,
            "intervention": "RATE_LIMITED"
        }

    user_message = request.message.strip()

    # 2. RC3 — DETERMINISTIC SAFETY BYPASS
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

    # 3. Save user message
    db.add(models.ChatHistory(
        user_id=request.user_id,
        role="user",
        content=user_message,
        timestamp=datetime.now().isoformat()
    ))
    db.commit()

    # 4. Fetch recent chat history
    recent_history = (
        db.query(models.ChatHistory)
        .filter(models.ChatHistory.user_id == request.user_id)
        .order_by(models.ChatHistory.id.desc())
        .limit(6)
        .all()
    )
    recent_history = recent_history[::-1]

    # 5. Compute live context and tuned model prediction
    context_vector = get_user_context(request.user_id, db)
    state_id, confidence, confidence_explanation = predict_with_confidence(context_vector)
    current_state = STATE_MAP.get(state_id, "NEUTRAL")

    # 6. Select state-adaptive prompt
    system_prompt = STATE_SYSTEM_PROMPTS.get(current_state, STATE_SYSTEM_PROMPTS["NEUTRAL"])

    mood, pending, hour, completion_rate = context_vector
    system_prompt += (
        f"\n\n[LIVE CONTEXT — do not recite these numbers unless helpful]"
        f"\nMood score: {mood}/10 | Pending tasks: {pending} | "
        f"Hour: {hour}:00 | Completion rate: {completion_rate:.0%}"
        f"\nDetected state: {current_state} (confidence: {confidence:.0%})"
        f"\nDecision note: {confidence_explanation}"
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
                "context_vector": context_vector,
                "confidence_explanation": confidence_explanation,
            }

    except Exception as e:
        print(f"Groq API error: {e}")

    return {
        "response": "I'm having trouble thinking right now. Please try again in a moment.",
        "alert": False,
        "intervention": "ERROR"
    }


@app.get("/jitai/{user_id}")
def get_jitai_intervention(user_id: int, db: Session = Depends(get_db)):
    """
    RC2 — Context-aware JITAI delivery with tuned RF + engineered features.
    """
    cooldown_minutes = 15
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
            state_id, confidence, _ = predict_with_confidence(context_vector)
            current_type = {0: "CRISIS", 1: "ACADEMIC", 2: "MOTIVATION", 3: "NONE"}.get(state_id, "NONE")
            return {
                "type": "NONE",
                "message": "",
                "context_vector": context_vector,
                "state": current_type,
                "confidence": round(confidence, 3),
            }

    context_vector = get_user_context(user_id, db)
    state_id, confidence, confidence_explanation = predict_with_confidence(context_vector)

    jitai_type = {0: "CRISIS", 1: "ACADEMIC", 2: "MOTIVATION", 3: "NONE"}.get(state_id, "NONE")
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

    print(
        f"📢 JITAI event created: type={jitai_type}, confidence={confidence:.0%}, "
        f"event_id={event.id}"
    )

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
    RC1 — Adaptive feedback loop with tuned RF retraining.
    Stores raw 4D context and regenerates engineered features during retrain.
    """
    global X_accumulated_raw, y_accumulated, w_accumulated

    event = db.query(models.JitaiEvent).filter(models.JitaiEvent.id == data.event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    event.outcome = data.outcome
    db.commit()

    context_vector = json.loads(event.context_json)
    predicted_state = int(event.state_id)

    if data.outcome == 1:
        trained_toward = predicted_state
        weight = BEST_CRISIS_WEIGHT if trained_toward == 0 else 1.0

        X_accumulated_raw.append(context_vector)
        y_accumulated.append(trained_toward)
        w_accumulated.append(weight)

        print(
            f"✅ Feedback: reinforcing state {trained_toward} "
            f"(weight={weight}) for raw context {context_vector}"
        )
    else:
        trained_toward = 3  # NEUTRAL
        weight = 1.0

        X_accumulated_raw.append(context_vector)
        y_accumulated.append(trained_toward)
        w_accumulated.append(weight)

        print(
            f"⚠️ Feedback: nudging toward NEUTRAL "
            f"(weight={weight}) for raw context {context_vector}"
        )

    retrain_model(X_accumulated_raw, y_accumulated, w_accumulated)

    return {
        "status": "Tuned Model Retrained",
        "event_id": event.id,
        "outcome": event.outcome,
        "trained_toward": trained_toward,
        "total_training_examples": len(X_accumulated_raw),
        "crisis_weight_applied": weight if trained_toward == 0 else None,
        "model_type": "Tuned RandomForest + engineered features",
    }