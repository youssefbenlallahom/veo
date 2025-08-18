from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone
import sqlite3

# Simple SQLite database for reports
DATABASE_URL = "sqlite:///./candidate_reports.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class CandidateReport(Base):
    __tablename__ = "candidate_reports"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Core ReportModel fields that match the JSON structure exactly
    applied_job_title = Column(String(255), nullable=False, index=True)
    applied_job_description = Column(Text, nullable=False)
    candidate_name = Column(String(255), nullable=False, index=True)
    candidate_job_title = Column(String(255), nullable=True, index=True)  # What candidate currently works as
    candidate_experience = Column(String(100), nullable=False)
    candidate_background = Column(Text, nullable=False)
    requirements_analysis = Column(JSON, nullable=False)  # List[str]
    match_results = Column(JSON, nullable=False)  # Dict[str, str]
    scoring_weights = Column(JSON, nullable=False)  # Dict[str, float]
    score_details = Column(JSON, nullable=False)  # List[ScoreDetail]
    total_weighted_score = Column(Float, nullable=False, default=0.0, index=True)
    strengths = Column(JSON, nullable=False)  # List[str]
    gaps = Column(JSON, nullable=False)  # List[str]
    rationale = Column(Text, nullable=False)
    risk = Column(Text, nullable=True)
    next_steps = Column(JSON, nullable=True)  # Optional[List[str]]
    # Recommendation flag (persisted)
    is_recommended = Column(Boolean, nullable=False, default=False, server_default="0", index=True)
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

# Create tables only if they don't exist (preserve existing data)
Base.metadata.create_all(bind=engine)

# Lightweight migration: ensure is_recommended column exists (SQLite only)
def _ensure_is_recommended_column():
    try:
        with engine.connect() as conn:
            # PRAGMA table_info returns rows with 'name' field for column names
            result = conn.exec_driver_sql("PRAGMA table_info(candidate_reports)")
            cols = [row[1] for row in result.fetchall()]  # row[1] is column name
            if "is_recommended" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE candidate_reports ADD COLUMN is_recommended BOOLEAN NOT NULL DEFAULT 0"
                )
    except Exception:
        # Best-effort; avoid breaking app startup if migration fails
        pass

_ensure_is_recommended_column()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
