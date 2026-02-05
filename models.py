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
    tasks = relationship("Task", back_populates="user") # Added relationship

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