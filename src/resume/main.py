import sys
import warnings
import os
# REMOVED: import streamlit as st
import PyPDF2
import tempfile
import re
import asyncio
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from src.resume.crew import Resume
from src.resume.skill_extraction import extract_skills_from_pdf  # <-- Import the skill extraction function
import sys
import os

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from .database import get_db, CandidateReport
import uvicorn

load_dotenv()
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

app = FastAPI()

# Allow CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---


# New request/response models
from pydantic import BaseModel
from typing import List, Dict, Optional

class CandidateInput(BaseModel):
    name: str
    skills: Dict[str, List[str]]

class MultiSkillMatchRequest(BaseModel):
    # Job title is required to save analyzed/recommended candidates per job
    job_title: str
    candidates: List[CandidateInput]
    job_skills: Dict[str, List[str]]
    threshold: Optional[int] = 40  # Add threshold parameter with default 40

class MultiSkillMatchResponse(BaseModel):
    matched_candidates: List[str]
    errors: Optional[Dict[str, str]] = None

@app.post("/analyze-skill-match-multi", response_model=MultiSkillMatchResponse)
def api_analyze_skill_match_multi(data: MultiSkillMatchRequest):
    """
    API endpoint to analyze skill match for multiple candidates against a job using Azure AI LLM.
    """
    print("/analyze-skill-match-multi called with:")
    print("Job Title:", data.job_title)
    print("Candidates:", [c.name for c in data.candidates])
    print("Job Skills:", json.dumps(data.job_skills, indent=2))
    print("Threshold:", data.threshold)
    matched = []
    errors = {}
    for candidate in data.candidates:
        result = analyze_skill_match(candidate.skills, data.job_skills, threshold=data.threshold)
        if "Error" in result:
            errors[candidate.name] = result["Error"]
        elif result.get("is_match", False):
            matched.append(candidate.name)
    
    # Persist analyzed and recommended candidates in SQLite under job_required_skills
    try:
        import sqlite3
        # Resolve DB path (same DB used elsewhere in this project)
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'candidate_reports.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Ensure base table exists (keeps compatibility with other APIs)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_required_skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_title TEXT NOT NULL,
                required_skills_json TEXT NOT NULL,
                barem_json TEXT
            )
        ''')

        # Ensure new columns exist (SQLite lacks IF NOT EXISTS on ADD COLUMN)
        cursor.execute("PRAGMA table_info(job_required_skills)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        if 'analyzed_candidates' not in existing_cols:
            cursor.execute("ALTER TABLE job_required_skills ADD COLUMN analyzed_candidates TEXT")
        if 'recommended_candidates' not in existing_cols:
            cursor.execute("ALTER TABLE job_required_skills ADD COLUMN recommended_candidates TEXT")

        analyzed_names = json.dumps([c.name for c in data.candidates])
        recommended_names = json.dumps(matched)

        # Upsert by job_title
        cursor.execute('SELECT id FROM job_required_skills WHERE job_title = ?', (data.job_title,))
        row = cursor.fetchone()
        if row:
            cursor.execute(
                'UPDATE job_required_skills SET analyzed_candidates = ?, recommended_candidates = ? WHERE job_title = ?',
                (analyzed_names, recommended_names, data.job_title)
            )
        else:
            cursor.execute(
                'INSERT INTO job_required_skills (job_title, required_skills_json, barem_json, analyzed_candidates, recommended_candidates) VALUES (?, ?, ?, ?, ?)',
                (data.job_title, json.dumps(data.job_skills), None, analyzed_names, recommended_names)
            )
        conn.commit()
    except Exception as e:
        # Log but don't fail the endpoint if DB write has an issue
        print(f"[WARN] Failed to persist analyzed/recommended candidates: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Print the final API response
    response = MultiSkillMatchResponse(matched_candidates=matched, errors=errors if errors else None)
    print("=== API RESPONSE ===")
    print("Matched candidates:", matched)
    if errors:
        print("Errors:", errors)
    print("=== END API RESPONSE ===")
    
    return response


def _extract_json_fallback(response_text: str) -> dict:
    """Fallback JSON extraction for malformed LLM responses."""
    import re
    
    # Try to find JSON-like content between braces
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not json_match:
        return None
    
    json_str = json_match.group(0)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix common JSON issues
        try:
            # Remove trailing commas
            fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)
            # Fix unquoted keys
            fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)
            return json.loads(fixed_json)
        except json.JSONDecodeError:
            # Last resort: create a minimal valid response
            return {
                "match_percentage": 0,
                "is_match": False,
                "matched_skills": [],
                "missing_skills": []
            }


def analyze_skill_match(candidate_skills: dict, job_skills: dict, threshold: int = 50, debug: bool = False) -> dict:
    import os
    import json
    from crewai.llm import LLM

    model = os.getenv("model")
    api_key = os.getenv("AZURE_AI_API_KEY")
    base_url = os.getenv("AZURE_AI_ENDPOINT")
    api_version = os.getenv("AZURE_AI_API_VERSION")

    if not model or not api_key or not base_url:
        return {"Error": "Azure AI credentials not found."}

    llm = LLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
        temperature=0.0,
        stream=False,
        format="json"
    )

    # ----------------------
    # System prompt: mention synonyms explicitly
    # ----------------------
    system_prompt = (
        "You are an expert HR assistant. Compare a candidate's skills to a job's required skills. "
        "Count as matches both exact skills and recognized synonyms or closely related technologies. "
        "For example:\n"
        "- 'Power BI' or 'Microsoft Power BI' should match 'Dashboards' and 'Visualizations'.\n"
        "- 'SQL', 'PostgreSQL', 'SQL Server', 'MySQL', or 'SQLite' are considered equivalent.\n"
        "- 'Excel' or 'Microsoft Excel' counts as 'Reporting'.\n"
        "Do NOT count unrelated or inferred skills beyond these synonyms. "
        "Compute match_percentage based on total job skills matched by exact skills or synonyms. "
        f"If match_percentage >= {threshold}, set is_match = true. "
        "Return ONLY valid JSON with these fields: match_percentage, is_match, matched_skills, missing_skills. "
        "Do not include explanations, guesses, or extra text."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps({
                "candidate_skills": candidate_skills,
                "job_skills": job_skills
            })
        }
    ]

    try:
        if debug:
            print("[skill-match-llm] request:")
            print("  candidate_skills:", json.dumps(candidate_skills))
            print("  job_skills:", json.dumps(job_skills))
            print("  threshold:", threshold)
        
        # Retry logic for LLM calls
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = llm.call(messages)
                if debug:
                    print(f"[skill-match-llm] raw response (attempt {attempt + 1}):", response)
                
                # Try to parse JSON
                result = json.loads(response)
                break  # Success, exit retry loop
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(f"[skill-match-llm] JSON decode error on attempt {attempt + 1}, retrying...")
                    continue
                else:
                    # Last attempt failed, try to extract JSON from response
                    print(f"[skill-match-llm] All attempts failed, trying fallback parsing...")
                    result = _extract_json_fallback(response)
                    if result is None:
                        return {"Error": f"Invalid JSON returned by LLM after {max_retries} attempts", "raw_response": response}
        
        # Verify and fix LLM's math - count total job skills and matched skills
        total_job_skills = sum(len(skills) for skills in job_skills.values())
        
        # Count matched skills (handle both list and dict formats)
        matched_skills = result.get("matched_skills", [])
        if isinstance(matched_skills, dict):
            matched_count = sum(len(skills) for skills in matched_skills.values())
        elif isinstance(matched_skills, list):
            matched_count = len(matched_skills)
        else:
            matched_count = 0
        
        # Recalculate correct percentage
        correct_percentage = round((matched_count / total_job_skills) * 100, 2) if total_job_skills > 0 else 0
        
        # Update the result with correct values
        result["match_percentage"] = correct_percentage
        result["is_match"] = correct_percentage >= threshold
        
        # Add verification info
        result["_verification"] = {
            "total_job_skills": total_job_skills,
            "matched_skills_count": matched_count,
            "llm_original_percentage": result.get("match_percentage", 0),
            "corrected_percentage": correct_percentage
        }
        
        # Print the analyze_skill_match function output
        print("=== analyze_skill_match OUTPUT ===")
        print(json.dumps(result, indent=2))
        print("=== END OUTPUT ===")
        
        return result
    except json.JSONDecodeError:
        return {"Error": "Invalid JSON returned by LLM", "raw_response": response}
    except Exception as e:
        return {"Error": str(e)}





def _ensure_category_dict_llm(x) -> Dict[str, List[str]]:
    """Coerce various skill structures into {category: [skills]}.
    Accepts:
    - dict of categories -> list[str]
    - dict with nested keys like 'categorized_skills', 'hard_skills', 'technical_skills', 'skills'
    - flat list -> {"General": list}
    """
    def _coerce(d: dict) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, list):
                out[k] = [s for s in v if isinstance(s, str) and s.strip()]
            elif isinstance(v, str) and v.strip():
                out[k] = [v]
        return out

    if isinstance(x, dict):
        # Prefer common nested containers if present
        for key in ("categorized_skills", "hard_skills", "technical_skills", "skills"):
            if key in x:
                val = x.get(key)
                if isinstance(val, dict):
                    return _coerce(val)
                if isinstance(val, list):
                    return {"General": [s for s in val if isinstance(s, str) and s.strip()]}
        # Otherwise assume it's already {category: list|str}
        return _coerce(x)
    if isinstance(x, list):
        return {"General": [s for s in x if isinstance(s, str) and s.strip()]}
    return {}


@app.post("/extract-skills-from-cv")
async def extract_skills_from_cv(
    file: UploadFile | None = File(None),
    job_description: str | None = Form(None),
    job_title: str | None = Form(None)
):
    """Extract technical skills from either an uploaded PDF resume (CV) or a raw job description string.

    - PDF uploads: save into extracted_skills table (existing behavior).
    - Job descriptions: save into job_required_skills table with job_title and required_skills_json.
    """
    import sqlite3
    temp_pdf_path = None
    try:
        # Validate inputs: exactly one of file or job_description
        if (file is None and not job_description) or (file is not None and job_description):
            raise ValueError("Provide exactly one input: either a PDF file or a job_description string.")

        # Prepare input and metadata
        if file is not None:
            # Save uploaded file to a temporary location
            contents = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(contents)
                temp_pdf_path = tmp_file.name
            # Run extraction from PDF
            result = extract_skills_from_pdf(pdf_path=temp_pdf_path)
            candidate_name = os.path.splitext(file.filename)[0]
            cv_filename = file.filename
        else:
            # Run extraction from job description text
            jd_text = (job_description or "").strip()
            if not jd_text:
                raise ValueError("Provided job_description is empty.")
            jt = (job_title or "").strip()
            if not jt:
                raise ValueError("job_title is required when submitting a job_description.")
            result = extract_skills_from_pdf(job_description=jd_text, job_title=jt)

        # Save to SQLite database (candidate_reports.db in project root)
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'candidate_reports.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if file is not None:
            # Prepare data for DB (PDF flow)
            skills_json = json.dumps(result)
            # Create table if not exists (without created_at)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS extracted_skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_name TEXT,
                    cv_filename TEXT,
                    skills_json TEXT
                )
            ''')
            # Check if candidate already exists (only for PDF uploads where name is stable)
            cursor.execute('SELECT id FROM extracted_skills WHERE candidate_name = ?', (candidate_name,))
            exists = cursor.fetchone()
            if not exists:
                # Insert row (without created_at)
                cursor.execute(
                    'INSERT INTO extracted_skills (candidate_name, cv_filename, skills_json) VALUES (?, ?, ?)',
                    (candidate_name, cv_filename, skills_json)
                )
                conn.commit()
        else:
            # JD flow: save into new job_required_skills table (without created_at)
            required_skills_json = json.dumps(result)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS job_required_skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_title TEXT NOT NULL,
                    required_skills_json TEXT NOT NULL,
                    barem_json TEXT
                )
            ''')
            # Check if job_title already exists
            cursor.execute('SELECT id, barem_json FROM job_required_skills WHERE job_title = ?', (jt,))
            job_exists = cursor.fetchone()
            if not job_exists:
                cursor.execute(
                    'INSERT INTO job_required_skills (job_title, required_skills_json, barem_json) VALUES (?, ?, ?)',
                    (jt, required_skills_json, None)
                )
                conn.commit()
            else:
                # Update existing record with new skills, preserve existing barem
                cursor.execute(
                    'UPDATE job_required_skills SET required_skills_json = ? WHERE job_title = ?',
                    (required_skills_json, jt)
                )
                conn.commit()

        conn.close()

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"Error": [str(e)]}, status_code=500)
    finally:
        # Clean up temp file if created
        try:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
        except Exception:
            pass
    
# --- Display Extracted Skills Endpoint ---
@app.get("/display_skills")
def display_skills():
    """Return all candidates and their skills from extracted_skills table."""
    import sqlite3
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'candidate_reports.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS extracted_skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT,
            cv_filename TEXT,
            skills_json TEXT
        )
    ''')
    cursor.execute('SELECT candidate_name, cv_filename, skills_json FROM extracted_skills')
    rows = cursor.fetchall()
    conn.close()
    # Prepare response
    result = []
    for row in rows:
        candidate_name, cv_filename, skills_json = row
        try:
            skills = json.loads(skills_json)
        except Exception:
            skills = skills_json
        result.append({
            "candidate_name": candidate_name,
            "cv_filename": cv_filename,
            "skills": skills
        })
    return JSONResponse(content={"candidates": result})

@app.get("/job-barem/{job_title}")
def get_job_barem(job_title: str):
    """Get assessment criteria (barem) for a specific job title."""
    import sqlite3
    
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'candidate_reports.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_required_skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_title TEXT NOT NULL,
            required_skills_json TEXT NOT NULL,
            barem_json TEXT
        )
    ''')
    cursor.execute('SELECT required_skills_json, barem_json FROM job_required_skills WHERE job_title = ?', (job_title,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail=f"No data found for job title: {job_title}")
    
    required_skills_json, barem_json = row
    
    try:
        required_skills = json.loads(required_skills_json) if required_skills_json else {}
        barem = json.loads(barem_json) if barem_json else {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing JSON data: {str(e)}")
    
    return {
        "job_title": job_title,
        "barem": barem
    }

@app.get("/job-skills/{job_title}")
def get_job_skills(job_title: str):
    """Get required skills for a specific job title."""
    import sqlite3

    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'candidate_reports.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_required_skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_title TEXT NOT NULL,
            required_skills_json TEXT NOT NULL,
            barem_json TEXT
        )
    ''')
    cursor.execute('SELECT required_skills_json FROM job_required_skills WHERE job_title = ?', (job_title,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"No data found for job title: {job_title}")

    required_skills_json = row[0]

    try:
        required_skills = json.loads(required_skills_json) if required_skills_json else {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing JSON data: {str(e)}")

    return {
        "job_title": job_title,
        "required_skills": required_skills
    }

class ExtractSkillsRequest(BaseModel):
    job_title: str
    job_description: str

class ExtractSkillsResponse(BaseModel):
    hard_skills: List[str]
    categorized_skills: Dict[str, List[str]]

class BaremRequest(BaseModel):
    job_title: str
    skills_weights: Dict[str, int]
    categorized_skills: Optional[Dict[str, List[str]]] = None

class BaremResponse(BaseModel):
    barem: Dict[str, dict]

class ReportAnalysis(BaseModel):
    strengths: str
    gaps: str

class ReportScoring(BaseModel):
    overall_score: float
    detailed_breakdown: str

class ReportMetadata(BaseModel):
    candidate_name: str
    position_title: str
    evaluation_date: str

class JsonReport(BaseModel):
    report_metadata: ReportMetadata
    scoring: ReportScoring
    executive_summary: str
    analysis: ReportAnalysis

class AnalyzeResult(BaseModel):
    id: int
    candidate_name: str
    job_title: str
    total_weighted_score: float
    rationale: str
    full_report_json: Dict  # The complete JSON report
    created_at: datetime

class AnalyzeResponse(BaseModel):
    results: List[AnalyzeResult]

# --- Database Models ---
class SavedCandidateReport(BaseModel):
    id: int
    candidate_name: str
    job_title: str
    total_weighted_score: float
    rationale: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# Recommendation toggle models
class ToggleRecommendRequest(BaseModel):
    value: bool = True

class BulkRecommendRequest(BaseModel):
    candidate_names: List[str]
    value: bool = True
    job_title: Optional[str] = None

# --- Endpoints ---

@app.post("/extract-skills", response_model=ExtractSkillsResponse)
def extract_skills(data: ExtractSkillsRequest):
    # Use Azure AI for skill extraction
    skills_data = get_key_skills_from_azure_ai(data.job_title, data.job_description)
    return ExtractSkillsResponse(
        hard_skills=skills_data["hard_skills"],
        categorized_skills=skills_data["categorized_skills"]
    )

@app.post("/barem", response_model=BaremResponse)
def create_barem(data: BaremRequest):
    import sqlite3
    
    print("=== BAREM API CALLED ===")
    print(f"Job Title: {data.job_title}")
    print(f"Skills Weights: {data.skills_weights}")
    print(f"Categorized Skills: {data.categorized_skills}")
    
    barem = create_custom_barem(data.skills_weights, data.categorized_skills)
    
    print("=== GENERATED BAREM ===")
    print(json.dumps(barem, indent=2))
    
    # Save barem to database
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'candidate_reports.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Ensure table exists with barem_json column
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_required_skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_title TEXT NOT NULL,
                required_skills_json TEXT NOT NULL,
                barem_json TEXT
            )
        ''')
        
        # Check if job already exists
        cursor.execute('SELECT id, required_skills_json, barem_json FROM job_required_skills WHERE job_title = ?', (data.job_title,))
        existing_job = cursor.fetchone()
        
        barem_json = json.dumps(barem)
        
        print("=== DATABASE OPERATION ===")
        if existing_job:
            print(f"UPDATING existing job: ID={existing_job[0]}")
            print(f"Previous barem: {existing_job[2]}")
            # Update existing record with barem
            cursor.execute(
                'UPDATE job_required_skills SET barem_json = ? WHERE job_title = ?',
                (barem_json, data.job_title)
            )
            print(f"✅ UPDATED barem for job: {data.job_title}")
        else:
            print(f"CREATING new job record: {data.job_title}")
            # Create new record with empty skills (barem can be created before skills extraction)
            cursor.execute(
                'INSERT INTO job_required_skills (job_title, required_skills_json, barem_json) VALUES (?, ?, ?)',
                (data.job_title, json.dumps({}), barem_json)
            )
            print(f"✅ CREATED new job with barem: {data.job_title}")
        
        print("=== SAVED TO DATABASE ===")
        print(f"Job Title: {data.job_title}")
        print(f"Barem JSON Length: {len(barem_json)} characters")
        print(f"Barem Keys: {list(barem.keys())}")
        print("=== END BAREM SAVE ===")
        
        conn.commit()
    finally:
        conn.close()
    
    return BaremResponse(barem=barem)

# --- Database Endpoints ---
@app.get("/candidate/{candidate_name}/reports")
def get_candidate_reports(candidate_name: str, db: Session = Depends(get_db)):
    """Get all reports for a specific candidate by name."""
    reports = db.query(CandidateReport).filter(
        CandidateReport.candidate_name == candidate_name
    ).order_by(CandidateReport.created_at.desc()).all()
    
    if not reports:
        raise HTTPException(status_code=404, detail=f"No reports found for candidate: {candidate_name}")
    
    return {
        "candidate_name": candidate_name,
        "total_reports": len(reports),
        "reports": reports
    }

@app.get("/candidate/{candidate_name}/latest")
def get_candidate_latest_report(candidate_name: str, db: Session = Depends(get_db)):
    """Get the latest report for a specific candidate."""
    report = db.query(CandidateReport).filter(
        CandidateReport.candidate_name == candidate_name
    ).order_by(CandidateReport.created_at.desc()).first()
    
    if not report:
        raise HTTPException(status_code=404, detail=f"No reports found for candidate: {candidate_name}")
    
    return report

@app.get("/candidates")
def get_all_candidates(db: Session = Depends(get_db)):
    """Get list of all candidates with report counts."""
    from sqlalchemy import func
    
    candidates = db.query(
        CandidateReport.candidate_name,
        func.count(CandidateReport.id).label('report_count'),
        func.max(CandidateReport.created_at).label('latest_report')
    ).group_by(CandidateReport.candidate_name).all()
    
    return {
        "candidates": [
            {
                "name": candidate.candidate_name,
                "report_count": candidate.report_count,
                "latest_report": candidate.latest_report
            }
            for candidate in candidates
        ]
    }

@app.get("/reports")
def get_all_reports(
    skip: int = 0,
    limit: int = 100,
    candidate_name: str = None,
    job_title: str = None,
    is_recommended: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Get all reports with optional filtering."""
    query = db.query(CandidateReport)
    
    if candidate_name:
        query = query.filter(CandidateReport.candidate_name.contains(candidate_name))
    if job_title:
        query = query.filter(CandidateReport.applied_job_title.contains(job_title))
    if is_recommended is not None:
        query = query.filter(CandidateReport.is_recommended == is_recommended)
    
    total = query.count()
    reports = query.offset(skip).limit(limit).order_by(CandidateReport.created_at.desc()).all()
    
    return {"reports": reports, "total": total}

@app.get("/reports/{report_id}")
def get_report(report_id: int, db: Session = Depends(get_db)):
    """Get a specific report by ID, including the full JSON."""
    report = db.query(CandidateReport).filter(CandidateReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    job_title: str = Form(...),
    job_description: str = Form(...),
    barem: str = Form(...),  # JSON string
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)  # Add database dependency
):
    # Parse barem JSON
    barem_dict = json.loads(barem)
    results = []
    semaphore = asyncio.Semaphore(5)
    async def analyze_one(file: UploadFile):
        # Save uploaded file to temp
        try:
            contents = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(contents)
                resume_path = tmp_file.name
            candidate_name = os.path.splitext(file.filename)[0]
            report_filename = await analyze_resume_async(
                resume_path=resume_path,
                job_title=job_title,
                job_description=job_description,
                candidate_name=candidate_name,
                barem=barem_dict
            )
            
            # Read the JSON report directly (no markdown parsing needed)
            with open(report_filename, 'r', encoding='utf-8') as json_file:
                json_report = json.load(json_file)

            # Use only AI-extracted name from resume content
            final_candidate_name = json_report.get('candidate_name', 'Unknown Candidate')
            if not final_candidate_name or final_candidate_name.strip() == "":
                final_candidate_name = 'Unknown Candidate'  # Default if AI couldn't extract name

            # Check if this candidate already has a report for the same job (case-insensitive)
            existing_report = db.query(CandidateReport).filter(
                CandidateReport.candidate_name.ilike(final_candidate_name),
                CandidateReport.applied_job_title.ilike(json_report['applied_job_title'])
            ).first()
            
            if existing_report:
                # Update existing record for the same candidate and job
                existing_report.applied_job_description = json_report['applied_job_description']
                existing_report.candidate_job_title = json_report.get('candidate_job_title')
                existing_report.candidate_experience = json_report['candidate_experience']
                existing_report.candidate_background = json_report['candidate_background']
                existing_report.requirements_analysis = json_report['requirements_analysis']
                existing_report.match_results = json_report['match_results']
                existing_report.scoring_weights = json_report['scoring_weights']
                existing_report.score_details = json_report['score_details']
                existing_report.total_weighted_score = json_report['total_weighted_score']
                existing_report.strengths = json_report['strengths']
                existing_report.gaps = json_report['gaps']
                existing_report.rationale = json_report['rationale']
                existing_report.risk = json_report.get('risk')
                existing_report.next_steps = json_report.get('next_steps')
                existing_report.created_at = datetime.now(timezone.utc)  # Update timestamp
                
                db.commit()
                db.refresh(existing_report)
                db_report = existing_report
            else:
                # Create new record for new candidate or same candidate applying to different job
                db_report = CandidateReport(
                    applied_job_title=json_report['applied_job_title'],
                    applied_job_description=json_report['applied_job_description'],
                    candidate_name=final_candidate_name,  # Use the verified candidate name
                    candidate_job_title=json_report.get('candidate_job_title'),  # Optional field
                    candidate_experience=json_report['candidate_experience'],
                    candidate_background=json_report['candidate_background'],
                    requirements_analysis=json_report['requirements_analysis'],
                    match_results=json_report['match_results'],
                    scoring_weights=json_report['scoring_weights'],
                    score_details=json_report['score_details'],
                    total_weighted_score=json_report['total_weighted_score'],
                    strengths=json_report['strengths'],
                    gaps=json_report['gaps'],
                    rationale=json_report['rationale'],
                    risk=json_report.get('risk'),  # Optional field
                    next_steps=json_report.get('next_steps')  # Optional field
                )
                
                db.add(db_report)
                db.commit()
                db.refresh(db_report)
            
            # Keep the JSON report file (don't delete it)
            # The report is now stored in the database and preserved as a file
            
            # Clean up temporary files
            try:
                if os.path.exists(resume_path):
                    os.remove(resume_path)
            except Exception:
                pass
            
            return {
                "id": db_report.id,
                "candidate_name": final_candidate_name,  # Use AI-extracted name only
                "job_title": db_report.applied_job_title,
                "total_weighted_score": db_report.total_weighted_score,
                "rationale": db_report.rationale,
                "full_report_json": json_report,  # Return the complete JSON report
                "created_at": db_report.created_at
            }
        except Exception as e:
            # In case of error, you might want to log it and return an error response
            return {
                "id": -1,
                "candidate_name": 'Unknown Candidate',
                "job_title": job_title,
                "total_weighted_score": 0,
                "rationale": str(e),
                "full_report_json": {
                    "applied_job_title": job_title,
                    "applied_job_description": job_description,
                    "candidate_name": 'Unknown Candidate',
                    "candidate_experience": "Unknown",
                    "candidate_background": "Error occurred during analysis",
                    "requirements_analysis": [],
                    "match_results": {},
                    "scoring_weights": {},
                    "score_details": [],
                    "total_weighted_score": 0.0,
                    "strengths": [],
                    "gaps": [],
                    "rationale": str(e),
                    "risk": None,
                    "next_steps": None
                },
                "created_at": datetime.now(timezone.utc)
            }
    tasks = [analyze_one(file) for file in files]
    analyzed = await asyncio.gather(*tasks)
    # Convert to AnalyzeResult models
    results = [AnalyzeResult(**r) for r in analyzed]
    return AnalyzeResponse(results=results)


def sanitize_text(text):
    """Sanitize text to remove problematic characters for display."""
    if not text:
        return ""

    # First try with UTF-8 encoding/decoding with error handling
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception:
        pass

    # Remove null bytes
    text = text.replace('\x00', '')

    # Remove other problematic control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\xFF]', '', text)

    # As a last resort, keep only ASCII characters
    if '\ufffd' in text or any(ord(c) > 127 for c in text):
        text = ''.join(c for c in text if ord(c) < 128)

    return text.strip()


def validate_pdf(file_path):
    """Validate that the PDF is readable."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            if num_pages == 0:
                return False, "PDF has no pages"

            # Try to read first page text
            first_page = pdf_reader.pages[0]
            text = first_page.extract_text()

            return True, f"PDF is valid with {num_pages} pages"
    except Exception as e:
        return False, f"PDF validation failed: {str(e)}"


import re

def clean_extracted_text(text):
    cleaned_lines = []
    prev_blank = False
    for line in text.splitlines():
        # Remove lines that are only pipes and spaces (table delimiters)
        if re.fullmatch(r'\s*\|[\s\|]*\|?\s*', line):
            continue
        # Remove lines that are only pipes or empty after pipes
        if re.fullmatch(r'\s*\|+\s*', line):
            continue
        # Remove lines that are only whitespace
        if not line.strip():
            if prev_blank:
                continue  # Avoid multiple consecutive blanks
            prev_blank = True
            cleaned_lines.append('')
            continue
        prev_blank = False
        # Remove trailing pipes and spaces
        line = re.sub(r'\s*\|+\s*$', '', line)
        # Optionally: collapse multiple spaces
        line = re.sub(r' {2,}', ' ', line)
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines).strip()

async def analyze_resume_async(resume_path, job_title, job_description, candidate_name, barem=None):
    """Run analysis asynchronously and save the report with a unique filename."""
    try:
        # Create Resume instance with candidate name
        resume_crew = Resume(pdf_path=resume_path, candidate_name=candidate_name)
        
        inputs = {
            'pdf': resume_path,
            'job_title': job_title,
            'job_description': job_description,
            'current_year': str(datetime.now().year)
        }
        if barem is not None:
            inputs['barem'] = barem

        # Run the analysis asynchronously
        result = await resume_crew.crew().kickoff_async(inputs=inputs)

        # The report should now be generated with the candidate name in the reports directory
        safe_name = re.sub(r'[^\w\-_\. ]', '_', candidate_name)
        report_filename = f'reports/report_{safe_name}.json'
        
        # If the old report.json exists, rename it to the proper location
        if os.path.exists('report.json') and not os.path.exists(report_filename):
            os.makedirs('reports', exist_ok=True)
            os.rename('report.json', report_filename)

        return report_filename

    except Exception as e:
        raise RuntimeError(f"Error analyzing resume for {candidate_name}: {str(e)}")


async def analyze_single_resume_with_error_handling(resume_pdf, job_title, job_description, barem, semaphore):
    """Analyze a single resume with error handling and concurrency control."""
    async with semaphore:  # Limit concurrent executions
        resume_path = None
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(resume_pdf.getbuffer())
                resume_path = tmp_file.name

            # Validate PDF
            is_valid, validation_message = validate_pdf(resume_path)
            if not is_valid:
                return {
                    "filename": resume_pdf.name,
                    "valid": False,
                    "error": validation_message,
                    "score": 0,
                    "recommendation": "Failed",
                    "report_content": ""
                }

            # Extract candidate name from the filename
            candidate_name = os.path.splitext(resume_pdf.name)[0]

            # Analyze resume asynchronously
            report_filename = await analyze_resume_async(
                resume_path=resume_path,
                job_title=job_title,
                job_description=job_description,
                candidate_name=candidate_name,
                barem=barem
            )

            # Parse the generated report
            candidate_result = parse_report(report_filename, resume_pdf.name)
            return candidate_result

        except Exception as e:
            return {
                "filename": resume_pdf.name,
                "valid": False,
                "error": str(e),
                "score": 0,
                "recommendation": "Error",
                "report_content": ""
            }

        finally:
            # Clean up the temporary file silently
            if resume_path and os.path.exists(resume_path):
                try:
                    os.remove(resume_path)
                except Exception:
                    pass  # Silent cleanup


async def analyze_all_resumes_async(resume_pdfs, job_title, job_description, barem, progress_callback=None):
    """Analyze all resumes concurrently with progress tracking."""
    # Create semaphore to limit concurrent executions (prevent memory issues)
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent analyses
    
    # Create tasks for all resumes
    tasks = [
        analyze_single_resume_with_error_handling(
            resume_pdf, job_title, job_description, barem, semaphore
        ) 
        for resume_pdf in resume_pdfs
    ]
    
    # Execute with progress tracking
    results = []
    completed = 0
    
    for future in asyncio.as_completed(tasks):
        result = await future
        results.append(result)
        completed += 1
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback(completed, len(resume_pdfs))
    
    return results


def parse_report_to_json(report_path):
    """Parse the markdown report into a structured JSON object."""
    with open(report_path, 'r', encoding='utf-8') as file:
        content = file.read()

    def extract_section(title, text):
        pattern = re.compile(f"## {re.escape(title)}\n(.*?)(?=\n## |\Z)", re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)
        return match.group(1).strip() if match else ""

    # Metadata
    candidate_name = (re.search(r"\*\*Candidate:\*\*\s*(.*)", content) or ["", ""])[1].strip()
    position_title = (re.search(r"\*\*Position:\*\*\s*(.*)", content) or ["", ""])[1].strip()
    evaluation_date = (re.search(r"\*\*Evaluation Date:\*\*\s*(.*)", content) or ["", ""])[1].strip()
    
    # Scoring
    overall_score_match = re.search(r"\*\*Overall Score:\*\*\s*([\d.]+)/10", content)
    overall_score = float(overall_score_match.group(1)) if overall_score_match else 0.0
    
    detailed_breakdown = extract_section("Detailed Scoring Breakdown", content)
    
    # Summary and Analysis
    executive_summary = extract_section("Executive Summary", content)
    strengths_analysis = extract_section("✅ Strengths Analysis", content)
    gaps_analysis = extract_section("❌ Critical Gaps Analysis", content)

    report = {
        "report_metadata": {
            "candidate_name": candidate_name,
            "position_title": position_title,
            "evaluation_date": evaluation_date
        },
        "scoring": {
            "overall_score": overall_score,
            "detailed_breakdown": detailed_breakdown
        },
        "executive_summary": executive_summary,
        "analysis": {
            "strengths": strengths_analysis,
            "gaps": gaps_analysis
        }
    }
    return report

def parse_report(report_path, filename):
    """This function is now replaced by parse_report_to_json. Kept for compatibility if needed elsewhere."""
    try:
        json_report = parse_report_to_json(report_path)
        return {
            "filename": filename,
            "valid": True,
            "error": "",
            "score": json_report['scoring']['overall_score'],
            "strengths": [s.strip() for s in json_report['analysis']['strengths'].split('\n') if s.strip()],
            "gaps": [g.strip() for g in json_report['analysis']['gaps'].split('\n') if g.strip()],
            "report_content": json.dumps(json_report, indent=2)
        }
    except Exception as e:
        return {
            "filename": filename,
            "valid": False,
            "error": str(e),
            "score": 0,
            "recommendation": "ERROR",
            "strengths": [],
            "gaps": [],
            "report_content": ""
        }


# Remove the entire display_comparison_table and display_individual_reports functions, as they are Streamlit UI and not used in FastAPI


def get_key_skills_from_azure_ai(job_title, job_description):
    """Extract key skills from job description using Azure AI LLM, grouped by Hard and Soft Skills."""
    from crewai.llm import LLM
    import json, re, os
    
    try:
        # Use Azure AI configuration
        model = os.getenv("model")
        azure_api_key = os.getenv("AZURE_AI_API_KEY")
        base_url = os.getenv("AZURE_AI_ENDPOINT")
        api_version = os.getenv("AZURE_AI_API_VERSION")
        
        if not model or not azure_api_key or not base_url:
            raise ValueError("Azure AI credentials not configured")
        
        llm = LLM(
            model=model,
            api_key=azure_api_key,
            base_url=base_url,
            api_version=api_version,
            temperature=0.0,
            stream=False,
            format="json"
        )
        
        prompt = f"""
### Role
You are an AI skill extraction specialist designed to help HR teams automatically categorize technical skills from job descriptions into standardized groups.

### Task
Analyze the job description below and:
1. Extract ALL mentioned/implied hard/soft skills
2. **Group skills into standardized categories** (e.g., "Power BI" → "Business Intelligence")
3. Map variations to standardized terms (e.g., "Tableau" → "Business Intelligence")
4. Output in structured JSON format

### Standardized Skill Categories
HARD SKILLS CATEGORIES (with examples):
1. **Business Intelligence**: Tableau, Power BI, QlikView, Looker, Data Warehousing
2. **Programming**: Python, Java, JavaScript, SQL, C++
3. **Data & Analytics**: Data Modeling, ETL, Data Mining, Machine Learning
4. **Cloud & DevOps**: AWS, Azure, Docker, Kubernetes, Git
5. **Productivity Tools**: Excel Advanced, Microsoft Office Suite, SharePoint
6. **Design & UX**: Figma, Adobe Creative Suite, UI/UX Design
7. **Industry-Specific**: SAP, Salesforce, AutoCAD, Clinical Research
8. **Languages**: French B2, English C1, Spanish B1
9. **Education**: Bachelor's Degree, PMP, CISSP
10. **Experience Levels**: Senior Level, Entry Level

SOFT SKILLS (single category):
Communication, Leadership, Problem Solving, etc.

### Extraction Rules
- **Group first, then list**: Always categorize skills before listing (e.g., "Business Intelligence: Power BI, Tableau")
- **Implied skills**: Infer from context (e.g., "build dashboards" → Business Intelligence tools)
- **Language handling**: Map "anglais courant" → "English B2 Level"
- **Experience mapping**: "5+ years" → "Senior Level Experience"
- **Avoid duplicates**: List each skill only once per category
- **Prioritize relevance**: Omit irrelevant skills

### Few-Shot Examples
Example 1:
**Job Description**: "Seeking Power BI expert to create dashboards. Requires Tableau knowledge and SQL experience."
{{
  "hard_skills": {{
    "Business Intelligence": ["Power BI", "Tableau"],
    "Programming": ["SQL"]
  }},
  "soft_skills": []
}}
Example 2:
Job Description: "Senior developer needed: 5+ years Python, AWS, Docker, CI/CD pipelines."
{{
  "hard_skills": {{
    "Programming": ["Python"],
    "Cloud & DevOps": ["AWS", "Docker", "CI/CD"],
    "Experience Levels": ["Senior Level Experience"]
  }},
  "soft_skills": []
}}
Example 3:
Job Description: "Marketing analyst with Excel Advanced skills. Needs French B2 and teamwork."
{{
  "hard_skills": {{
    "Productivity Tools": ["Excel Advanced"],
    "Languages": ["French B2 Level"]
  }},
  "soft_skills": ["Teamwork"]
}}
Job to Analyze
Job Title: {job_title}
Job Description: {job_description}

Output Format (Strict JSON)
{{
"hard_skills": {{
"Category1": ["SkillA", "SkillB"],
"Category2": ["SkillC"]
}},
"soft_skills": ["SkillD", "SkillE"]
}}

Critical Instructions
Output ONLY valid JSON (no explanations)

Use EXACT category names from standardized list

Include ONLY skills mentioned/implied in description

Skip empty categories

Sort skills alphabetically within categories
"""

        messages = [
            {"role": "system", "content": "You are an expert skill extraction AI. Extract and categorize skills from job descriptions into structured JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        response = llm.call(messages)
        
        # Extract JSON from response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            skills_json = match.group(0)
            skills_data = json.loads(skills_json)
            
            # Ensure we have the expected structure
            hard_skills_dict = skills_data.get('hard_skills', {})
            
            # Return both categorized and flattened skills
            categorized_skills = {}
            all_skills = []
            
            if isinstance(hard_skills_dict, dict):
                for category, skills_list in hard_skills_dict.items():
                    if isinstance(skills_list, list) and skills_list:
                        # Clean and deduplicate skills in this category
                        clean_skills = []
                        for skill in skills_list:
                            if skill and skill.strip():
                                clean_skill = skill.strip()
                                if clean_skill not in clean_skills:
                                    clean_skills.append(clean_skill)
                        
                        if clean_skills:  # Only add non-empty categories
                            categorized_skills[category] = clean_skills
                            all_skills.extend(clean_skills)
            
            # Remove duplicates from flattened list while preserving order
            seen = set()
            unique_skills = []
            for skill in all_skills:
                if skill not in seen:
                    seen.add(skill)
                    unique_skills.append(skill)
            
            # Return both structured and flattened data
            return {
                'hard_skills': unique_skills,
                'categorized_skills': categorized_skills
            }
        else:
            raise ValueError("Failed to extract skills JSON from Azure AI response")
            
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse skills JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error calling Azure AI API: {str(e)}")


def create_custom_barem(skills_weights, categorized_skills=None):
    """Create a barem structure from user-defined skills, weights, and categorized skills."""
    barem = {}
    
    if categorized_skills is None:
        categorized_skills = {}
    
    for skill_or_category, weight in skills_weights.items():
        if weight > 0:  # Only include items with positive weights
            if skill_or_category in categorized_skills:
                # This is a skill category - keep category intact with full weight
                category_skills = categorized_skills[skill_or_category]
                if category_skills:  # Only add if category has skills
                    barem[skill_or_category] = {
                        "weight": weight,
                        "type": "category",
                        "skills": category_skills,
                        "criteria": [
                            f"Has extensive experience and expertise in {skill_or_category} skills: {', '.join(category_skills)}",
                            f"Has some experience with {skill_or_category} skills: {', '.join(category_skills)}",
                            f"Shows potential or basic knowledge in {skill_or_category} skills: {', '.join(category_skills)}"
                        ]
                    }
            else:
                # This is an individual skill (language or custom)
                barem[skill_or_category] = {
                    "weight": weight,
                    "type": "individual",
                    "criteria": [
                        f"Has extensive experience and expertise in {skill_or_category}",
                        f"Has some experience with {skill_or_category}",
                        f"Shows potential or basic knowledge in {skill_or_category}"
                    ]
                }
    
    return barem


# --- Additional Database Endpoints ---

@app.delete("/reports/{report_id}")
def delete_report(report_id: int, db: Session = Depends(get_db)):
    """Delete a specific report by ID."""
    report = db.query(CandidateReport).filter(CandidateReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    db.delete(report)
    db.commit()
    return {"message": f"Report {report_id} deleted successfully"}

@app.get("/stats")
def get_database_stats(db: Session = Depends(get_db)):
    """Get overall database statistics for dashboard."""
    from sqlalchemy import func
    
    total_reports = db.query(func.count(CandidateReport.id)).scalar()
    unique_candidates = db.query(func.count(func.distinct(CandidateReport.candidate_name))).scalar()
    unique_jobs = db.query(func.count(func.distinct(CandidateReport.applied_job_title))).scalar()
    avg_score = db.query(func.avg(CandidateReport.total_weighted_score)).scalar() or 0
    
    # Get top scoring candidates
    top_candidates = db.query(
        CandidateReport.candidate_name,
        func.max(CandidateReport.total_weighted_score).label('highest_score')
    ).group_by(CandidateReport.candidate_name).order_by(
        func.max(CandidateReport.total_weighted_score).desc()
    ).limit(5).all()
    
    # Get recent activity
    recent_reports = db.query(CandidateReport).order_by(
        CandidateReport.created_at.desc()
    ).limit(5).all()
    
    return {
        "total_reports": total_reports,
        "unique_candidates": unique_candidates,
        "unique_job_positions": unique_jobs,
        "average_score": round(avg_score, 2),
        "top_candidates": [
            {"name": candidate.candidate_name, "score": candidate.highest_score}
            for candidate in top_candidates
        ],
        "recent_activity": [
            {
                "id": report.id,
                "candidate_name": report.candidate_name,
                "job_title": report.applied_job_title,
                "score": report.total_weighted_score,
                "date": report.created_at
            }
            for report in recent_reports
        ]
    }

@app.get("/job-positions")
def get_job_positions(db: Session = Depends(get_db)):
    """Get all unique job positions with candidate counts and average scores."""
    from sqlalchemy import func
    
    job_stats = db.query(
        CandidateReport.applied_job_title,
        func.count(CandidateReport.id).label('candidate_count'),
        func.avg(CandidateReport.total_weighted_score).label('avg_score'),
        func.max(CandidateReport.total_weighted_score).label('highest_score'),
        func.min(CandidateReport.total_weighted_score).label('lowest_score')
    ).group_by(CandidateReport.applied_job_title).order_by(
        func.count(CandidateReport.id).desc()
    ).all()
    
    return {
        "job_positions": [
            {
                "title": job.applied_job_title,
                "candidate_count": job.candidate_count,
                "average_score": round(job.avg_score, 2),
                "highest_score": job.highest_score,
                "lowest_score": job.lowest_score
            }
            for job in job_stats
        ]
    }

@app.get("/candidates/search")
def search_candidates(
    name: str = None,
    min_score: float = None,
    max_score: float = None,
    job_title: str = None,
    recommended: Optional[bool] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Search candidates with various filters."""
    query = db.query(CandidateReport)
    
    if name:
        query = query.filter(CandidateReport.candidate_name.ilike(f"%{name}%"))
    if min_score is not None:
        query = query.filter(CandidateReport.total_weighted_score >= min_score)
    if max_score is not None:
        query = query.filter(CandidateReport.total_weighted_score <= max_score)
    if job_title:
        query = query.filter(CandidateReport.applied_job_title.ilike(f"%{job_title}%"))
    if recommended is not None:
        query = query.filter(CandidateReport.is_recommended == recommended)
    
    candidates = query.order_by(
        CandidateReport.total_weighted_score.desc()
    ).limit(limit).all()
    
    return {
        "candidates": candidates,
        "count": len(candidates)
    }

# --- Recommendation Endpoints ---
@app.patch("/reports/{report_id}/recommend")
def set_report_recommendation(report_id: int, data: ToggleRecommendRequest, db: Session = Depends(get_db)):
    """Toggle recommendation status for a specific report."""
    report = db.query(CandidateReport).filter(CandidateReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    report.is_recommended = data.value
    db.commit()
    db.refresh(report)
    return {"id": report.id, "is_recommended": report.is_recommended}

@app.post("/candidates/recommend")
def bulk_recommend(data: BulkRecommendRequest, db: Session = Depends(get_db)):
    """Bulk set recommendation flag by candidate names, optionally filtered by job_title.
    This is useful after running matching logic elsewhere and wanting to persist the status.
    """
    if not data.candidate_names:
        raise HTTPException(status_code=400, detail="candidate_names must not be empty")
    query = db.query(CandidateReport).filter(CandidateReport.candidate_name.in_(data.candidate_names))
    if data.job_title:
        query = query.filter(CandidateReport.applied_job_title.ilike(f"%{data.job_title}%"))
    reports = query.all()
    for r in reports:
        r.is_recommended = data.value
    db.commit()
    return {"updated": len(reports), "value": data.value}

@app.get("/candidates/recommended")
def list_recommended(limit: int = 50, job_title: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(CandidateReport).filter(CandidateReport.is_recommended == True)
    if job_title:
        query = query.filter(CandidateReport.applied_job_title.ilike(f"%{job_title}%"))
    reports = query.order_by(CandidateReport.created_at.desc()).limit(limit).all()
    return {"reports": reports, "count": len(reports)}

@app.get("/reports/compare")
def compare_candidates(
    candidate_ids: str,  # Comma-separated list of report IDs
    db: Session = Depends(get_db)
):
    """Compare multiple candidates side by side."""
    try:
        ids = [int(id.strip()) for id in candidate_ids.split(',')]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid candidate IDs format")
    
    reports = db.query(CandidateReport).filter(CandidateReport.id.in_(ids)).all()
    
    if not reports:
        raise HTTPException(status_code=404, detail="No reports found for the provided IDs")
    
    return {
        "comparison": [
            {
                "id": report.id,
                "candidate_name": report.candidate_name,
                "job_title": report.applied_job_title,
                "total_score": report.total_weighted_score,
                "strengths": report.strengths,
                "gaps": report.gaps,
                "score_details": report.score_details,
                "created_at": report.created_at
            }
            for report in reports
        ]
    }

@app.get("/candidates/top-performers")
def get_top_performers(
    limit: int = 10,
    job_title: str = None,
    db: Session = Depends(get_db)
):
    """Get top performing candidates overall or for a specific job."""
    query = db.query(CandidateReport)
    
    if job_title:
        query = query.filter(CandidateReport.applied_job_title.ilike(f"%{job_title}%"))
    
    top_performers = query.order_by(
        CandidateReport.total_weighted_score.desc()
    ).limit(limit).all()
    
    return {
        "top_performers": [
            {
                "id": report.id,
                "candidate_name": report.candidate_name,
                "job_title": report.applied_job_title,
                "score": report.total_weighted_score,
                "strengths": report.strengths[:3] if len(report.strengths) > 3 else report.strengths,  # Top 3 strengths
                "created_at": report.created_at
            }
            for report in top_performers
        ]
    }

@app.get("/reports/{report_id}/download")
def download_report_json(report_id: int, db: Session = Depends(get_db)):
    """Download the original JSON report file for a specific report."""
    from fastapi.responses import FileResponse
    
    report = db.query(CandidateReport).filter(CandidateReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Try to find the corresponding JSON file
    safe_name = re.sub(r'[^\w\-_\. ]', '_', report.candidate_name)
    json_filename = f"reports/report_{safe_name}.json"
    
    if os.path.exists(json_filename):
        return FileResponse(
            json_filename,
            media_type='application/json',
            filename=f"report_{report.candidate_name}_{report.id}.json"
        )
    else:
        # If file doesn't exist, return the data from database as JSON
        report_data = {
            "applied_job_title": report.applied_job_title,
            "applied_job_description": report.applied_job_description,
            "candidate_name": report.candidate_name,
            "candidate_job_title": report.candidate_job_title,
            "candidate_experience": report.candidate_experience,
            "candidate_background": report.candidate_background,
            "requirements_analysis": report.requirements_analysis,
            "match_results": report.match_results,
            "scoring_weights": report.scoring_weights,
            "score_details": report.score_details,
            "total_weighted_score": report.total_weighted_score,
            "strengths": report.strengths,
            "gaps": report.gaps,
            "rationale": report.rationale,
            "risk": report.risk,
            "next_steps": report.next_steps,
            "created_at": report.created_at.isoformat()
        }
        return JSONResponse(content=report_data)

@app.get("/reports/files")
def list_saved_report_files():
    """List all saved JSON report files."""
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        return {"files": []}
    
    files = []
    for filename in os.listdir(reports_dir):
        if filename.endswith('.json') and filename.startswith('report_'):
            file_path = os.path.join(reports_dir, filename)
            stat = os.stat(file_path)
            files.append({
                "filename": filename,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    
    return {"files": sorted(files, key=lambda x: x['modified'], reverse=True)}


# --- Server Runner ---
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=["src"],
        reload_excludes=[".venv", "__pycache__"]
    )
