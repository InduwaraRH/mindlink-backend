from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)

    thoughts = relationship("Thought", back_populates="owner")
    tasks = relationship("Task", back_populates="user")


class Thought(Base):
    __tablename__ = "thoughts"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String)
    mood_score = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="thoughts")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String)
    is_done = Column(Boolean, default=False)
    due_date = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))

    user = relationship("User", back_populates="tasks")


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    role = Column(String)  # "user" or "assistant"
    content = Column(String)
    timestamp = Column(String)


class JitaiEvent(Base):
    """
    Stores the exact decision-time context + predicted state for a shown intervention.
    This makes /jitai/feedback thesis-accurate (Context -> Action -> Reward -> Update).
    """
    __tablename__ = "jitai_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)

    # "ACADEMIC", "CRISIS", "MOTIVATION", "NONE"
    jitai_type = Column(String, index=True)

    # 0-3
    state_id = Column(Integer)

    # JSON string of [mood, pending, hour, completion_rate]
    context_json = Column(String)

    # outcome is null until user gives feedback (1 = helpful, 0 = not helpful)
    outcome = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())