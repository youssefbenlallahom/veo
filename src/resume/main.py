"""__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')"""

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
class SkillMatchRequest(BaseModel):
    candidate_skills: dict
    job_skills: dict

class SkillMatchResponse(BaseModel):
    match_percentage: float
    is_match: bool
    error: Optional[str] = None

@app.post("/analyze-skill-match", response_model=SkillMatchResponse)
def api_analyze_skill_match(data: SkillMatchRequest):
    """
    API endpoint to analyze skill match between candidate and job using Azure AI LLM.
    """
    print("/analyze-skill-match called with:")
    print("Candidate Skills:", json.dumps(data.candidate_skills, indent=2))
    print("Job Skills:", json.dumps(data.job_skills, indent=2))
    result = analyze_skill_match(data.candidate_skills, data.job_skills)
    if "Error" in result:
        return SkillMatchResponse(match_percentage=0.0, is_match=False, error=result["Error"])
    return SkillMatchResponse(
        match_percentage=result.get("match_percentage", 0.0),
        is_match=result.get("is_match", False)
    )

def analyze_skill_match(candidate_skills: dict, job_skills: dict) -> dict:
    """
    Use Azure AI LLM to decide if a candidate's skills are enough for potential success in a job.
    Includes transferable skills and obvious overlaps (SQL ↔ PL/SQL, AWS ↔ AWS, etc.).
    """
    import os
    from crewai.llm import LLM
    import json
    import re

    # Flatten candidate and job skills for direct comparison
    def flatten_skills(skills_dict):
        skills = []
        for v in skills_dict.values():
            if isinstance(v, list):
                skills.extend(v)
        return sorted(set([s.strip() for s in skills if s and isinstance(s, str)]))

    flat_candidate_skills = flatten_skills(candidate_skills)
    flat_job_skills = flatten_skills(job_skills)

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
    )

    prompt = (
        "You are an expert HR assistant for a talent sourcing system.\n"
        "Given two lists of skills — one for a candidate and one for a job — determine if the candidate could realistically perform well in this job.\n"
        "Important:\n"
        "1. The candidate does NOT need to have all job-required skills exactly.\n"
        "2. Consider transferable skills, related technologies, and relevant industry experience.\n"
        "3. Recognize obvious overlaps (e.g., SQL ↔ PL/SQL/SQL Server/MySQL, BI tools ↔ dashboards/reporting).\n"
        "4. Be generous in recognizing potential — if the candidate could succeed with minimal upskilling, count it as a match.\n"
        "5. Count each job skill as matched if the candidate has it exactly, has a synonym, or a directly related skill.\n"
        "6. Calculate match_percentage = (matched job skills / total job skills) * 100.\n"
        "7. If match_percentage >= 40, set is_match to true (since this is for recommendations).\n"
        "8. Return ONLY valid JSON in this format:\n"
        "{\n  \"match_percentage\": 0.0,\n  \"is_match\": true\n}\n"
        f"Candidate Skills List:\n{json.dumps(flat_candidate_skills, indent=2)}\n"
        f"Job Skills List:\n{json.dumps(flat_job_skills, indent=2)}\n"
    )

    response = llm.call([{"role": "user", "content": prompt}])
    response_text = response.strip()

    json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
    if not json_match:
        return {"Error": f"No JSON found in response: {response_text[:200]}..."}

    json_text = json_match.group(0)
    try:
        result = json.loads(json_text)
    except Exception as e:
        return {"Error": f"JSON parsing failed: {str(e)}. Response: {response_text[:200]}..."}

    return result


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
                    required_skills_json TEXT NOT NULL
                )
            ''')
            # Check if job_title already exists
            cursor.execute('SELECT id FROM job_required_skills WHERE job_title = ?', (jt,))
            job_exists = cursor.fetchone()
            if not job_exists:
                cursor.execute(
                    'INSERT INTO job_required_skills (job_title, required_skills_json) VALUES (?, ?)',
                    (jt, required_skills_json)
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
class ExtractSkillsRequest(BaseModel):
    job_title: str
    job_description: str

class ExtractSkillsResponse(BaseModel):
    hard_skills: List[str]
    categorized_skills: Dict[str, List[str]]

class BaremRequest(BaseModel):
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

# --- Endpoints ---

@app.post("/extract-skills", response_model=ExtractSkillsResponse)
def extract_skills(data: ExtractSkillsRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    skills_data = get_key_skills_from_gemini(data.job_title, data.job_description, api_key)
    return ExtractSkillsResponse(
        hard_skills=skills_data["hard_skills"],
        categorized_skills=skills_data["categorized_skills"]
    )

@app.post("/barem", response_model=BaremResponse)
def create_barem(data: BaremRequest):
    barem = create_custom_barem(data.skills_weights, data.categorized_skills)
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
    db: Session = Depends(get_db)
):
    """Get all reports with optional filtering."""
    query = db.query(CandidateReport)
    
    if candidate_name:
        query = query.filter(CandidateReport.candidate_name.contains(candidate_name))
    if job_title:
        query = query.filter(CandidateReport.applied_job_title.contains(job_title))
    
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


def get_key_skills_from_gemini(job_title, job_description, api_key):
    """Extract key skills from job description using Gemini, grouped by Hard and Soft Skills."""
    from google import genai
    import json, re
    
    try:
        client = genai.Client(api_key=api_key)
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


        
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt
        )
        
        # Extract JSON from response
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
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
            raise ValueError("Failed to extract skills JSON from Gemini response")
            
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse skills JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error calling Gemini API: {str(e)}")


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
    
    candidates = query.order_by(
        CandidateReport.total_weighted_score.desc()
    ).limit(limit).all()
    
    return {
        "candidates": candidates,
        "count": len(candidates)
    }

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
