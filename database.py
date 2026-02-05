from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# ⚠️ CHANGE 'password' to your real PostgreSQL password!
# Format: postgresql://username:password@localhost/databasename
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:admin123@localhost/mindlink"

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()