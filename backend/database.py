import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Retrieve database connection string from environment, fallback to SQLite
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./backend/summaries.db")

# SQLite needs check_same_thread=False for thread safety in multi-threaded environments (like FastAPI)
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# Create connection engine
engine = create_engine(DATABASE_URL, connect_args=connect_args)

# Create SessionLocal class for database transactions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative Base for models
Base = declarative_base()

class Summary(Base):
    """Database model for webpage summaries."""
    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(255), nullable=True)
    url = Column(Text, nullable=True)  # Using Text for URL to support long parameters
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class Flashcard(Base):
    """Database model for study flashcards."""
    __tablename__ = "flashcards"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    summary_id = Column(Integer, ForeignKey("summaries.id", ondelete="CASCADE"), nullable=False)
    front = Column(Text, nullable=False)
    back = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class Quiz(Base):
    """Database model for multiple-choice quizzes."""
    __tablename__ = "quizzes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    summary_id = Column(Integer, ForeignKey("summaries.id", ondelete="CASCADE"), nullable=False)
    question = Column(Text, nullable=False)
    options = Column(JSON, nullable=False) # stored as JSON string automatically
    answer = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class Chat(Base):
    """Database model for summary chat histories."""
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    summary_id = Column(Integer, ForeignKey("summaries.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)  # "user" or "assistant"
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)




def init_db():
    """Create all tables in the database if they do not exist."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency provider for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
