from pydantic import BaseModel
from datetime import datetime  # <--- 1. IMPORT THIS

# --- USER SCHEMAS ---
class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    class Config:
        from_attributes = True

# --- THOUGHT SCHEMAS ---
class ThoughtBase(BaseModel):
    content: str
    mood_score: int  

class ThoughtCreate(ThoughtBase):
    pass 

class ThoughtResponse(ThoughtBase):
    id: int
    owner_id: int
    created_at: datetime  # <--- 2. ADD THIS LINE
    
    class Config:
        from_attributes = True