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
from datetime import datetime
from dotenv import load_dotenv
from src.resume.crew import Resume
import sys
import os

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional

load_dotenv()
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

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

class AnalyzeResult(BaseModel):
    filename: str
    valid: bool
    error: str
    score: float
    recommendation: str
    strengths: List[str]
    gaps: List[str]
    report_content: str

class AnalyzeResponse(BaseModel):
    results: List[AnalyzeResult]

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

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    job_title: str = Form(...),
    job_description: str = Form(...),
    barem: str = Form(...),  # JSON string
    files: List[UploadFile] = File(...)
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
            candidate_result = parse_report(report_filename, file.filename)
            return candidate_result
        except Exception as e:
            return {
                "filename": file.filename,
                "valid": False,
                "error": str(e),
                "score": 0,
                "recommendation": "Error",
                "strengths": [],
                "gaps": [],
                "report_content": ""
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

def extract_and_save_pdf_text(pdf_path, output_filename=None):
    """
    Extract text from PDF and save to a text file.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_filename (str, optional): Output filename. If None, auto-generated.
    
    Returns:
        str: Path to the saved text file or None if failed
    """
    try:
        from crew import Resume
        
        # Create Resume instance
        resume_crew = Resume(pdf_path=pdf_path)
        
        if not hasattr(resume_crew, 'pdf_tool') or resume_crew.pdf_tool is None:
            print("[ERROR] PDF tool not available")
            return None
        
        # Extract text
        extracted_data = resume_crew.pdf_tool._run('extract_all')
        cleaned_data = clean_extracted_text(extracted_data)
        
        # Generate filename if not provided
        if output_filename is None:
            pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
            sanitized_name = re.sub(r'[^\w\-_\. ]', '_', pdf_basename)
            output_filename = f'extracted_text_{sanitized_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        # Save to file
        with open(output_filename, 'w', encoding='utf-8') as text_file:
            text_file.write(f"=== EXTRACTED PDF CONTENT ===\n")
            text_file.write(f"Source PDF: {pdf_path}\n")
            text_file.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            text_file.write(f"Extraction Tool: {'PyMuPDF' if 'fitz' in str(type(resume_crew.pdf_tool)) else 'PyPDF2/pdfplumber'}\n")
            text_file.write("=" * 50 + "\n\n")
            text_file.write(cleaned_data)
        
        return output_filename
        
    except Exception as e:
        print(f"[ERROR] Failed to extract PDF text: {str(e)}")
        return None

async def analyze_resume_async(resume_path, job_title, job_description, candidate_name, barem=None):
    """Run analysis asynchronously and save the report with a unique filename."""
    try:
        # Create Resume instance and process
        resume_crew = Resume(pdf_path=resume_path)
        # Extract raw data from the PDF and save to text file
        if hasattr(resume_crew, 'pdf_tool') and resume_crew.pdf_tool is not None:
            try:
                extracted_data = resume_crew.pdf_tool._run('extract_all')
                # Clean up table delimiters and excess blank lines
                cleaned_data = clean_extracted_text(extracted_data)
                
                # Save extracted data to text file
                try:
                    # Create a sanitized filename for the extracted text
                    sanitized_name = re.sub(r'[^\w\-_\. ]', '_', candidate_name)
                    extracted_filename = f'extracted_text_{sanitized_name}_async.txt'
                    
                    with open(extracted_filename, 'w', encoding='utf-8') as text_file:
                        text_file.write(f"=== EXTRACTED PDF CONTENT (ASYNC) ===\n")
                        text_file.write(f"Candidate: {candidate_name}\n")
                        text_file.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        text_file.write(f"PDF Path: {resume_path}\n")
                        text_file.write("=" * 50 + "\n\n")
                        text_file.write(cleaned_data)
                    
                    print(f"[DEBUG] Extracted text saved to: {extracted_filename}")
                except Exception as save_error:
                    print(f"[DEBUG] Could not save extracted text: {save_error}")
                    
            except Exception as e:
                print(f"[DEBUG] Could not extract raw PDF data: {e}")
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

        # Generate a unique report filename
        sanitized_name = re.sub(r'[^\w\-_\. ]', '_', candidate_name)  # Replace invalid characters
        report_filename = f'report_{sanitized_name}.md'

        # Ensure the report is renamed after each analysis
        if os.path.exists('report.md'):
            # Check if the target file exists and remove it
            if os.path.exists(report_filename):
                os.remove(report_filename)
            os.rename('report.md', report_filename)

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
            print(f"DEBUG: Parsed result for {resume_pdf.name}: score={candidate_result['score']}")
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


def parse_report(report_path, filename):
    """Parse the report file to extract scores and recommendations."""
    result = {
        "filename": filename,
        "valid": True,
        "error": "",
        "score": 0,
        "recommendation": "Unknown",
        "strengths": [],
        "gaps": [],
        "report_content": ""
    }

    if not os.path.exists(report_path):
        result["valid"] = False
        result["error"] = "Report file not found"
        return result

    try:
        # Read the report content
        with open(report_path, 'r', encoding='utf-8') as report_file:
            report_content = report_file.read()

        # Store full content for display
        result["report_content"] = report_content

        # Try multiple patterns to extract the score
        # Pattern 1: Look for "Overall Score: X/10" or "TOTAL WEIGHTED SCORE: X/10"
        score_match_10 = re.search(
            r'(?:\*\*Overall Score:\*\*|Overall Score:|TOTAL WEIGHTED SCORE:)\s*\[?([\d\.]+)\s*/\s*10\]?',
            report_content,
            re.IGNORECASE
        )
        # Pattern 2: Look for "Overall Score: X/100" or "TOTAL WEIGHTED SCORE: X/100"
        score_match_100 = re.search(
            r'(?:\*\*Overall Score:\*\*|Overall Score:|TOTAL WEIGHTED SCORE:)\s*\[?([\d\.]+)\s*/\s*100\]?',
            report_content,
            re.IGNORECASE
        )
        # Pattern 3: Look for "Overall Score: X" or "TOTAL WEIGHTED SCORE: X" without denominator
        score_match_plain = re.search(
            r'(?:\*\*Overall Score:\*\*|Overall Score:|TOTAL WEIGHTED SCORE:)\s*\[?([\d\.]+)\]?(?!\s*/)',
            report_content,
            re.IGNORECASE
        )
        # Pattern 4: Look for "final score for ... is **X/10**" or "final score is **X.X/10**"
        final_score_match = re.search(
            r'final score.*?is\s*\*\*([\d\.]+)/10\*\*',
            report_content,
            re.IGNORECASE
        )
        # Pattern 5: Look for "The final score for ... is **X.X/10**"
        final_score_match_2 = re.search(
            r'The final score for.*?is\s*\*\*([\d\.]+)/10\*\*',
            report_content,
            re.IGNORECASE
        )
        # Pattern 6: Look for any "final score" with **X.X/10** format  
        final_score_match_3 = re.search(
            r'final score.*?\*\*([\d\.]+)/10\*\*',
            report_content,
            re.IGNORECASE
        )
        # Pattern 7: Look for "## Final Score" section with score
        final_score_section = re.search(
            r'##\s*Final\s*Score.*?([\d\.]+)/10',
            report_content,
            re.IGNORECASE | re.DOTALL
        )
        # Pattern 8: Look for score after "Final Score" heading
        final_score_heading = re.search(
            r'##\s*Final\s*Score.*?(\d+\.?\d*)',
            report_content,
            re.IGNORECASE | re.DOTALL
        )
        # Pattern 6: Look for "Key Contributors:" followed by weighted scores that sum up
        weighted_scores = re.findall(
            r'\[?([\d\.]+)\]?\s*(?:points|point)\s*\(Source',
            report_content
        )
        
        print(f"DEBUG: Parsing report for {filename}")
        print(f"DEBUG: Report content length: {len(report_content)}")
        print(f"DEBUG: Looking for score patterns...")
        
        if score_match_10:
            result["score"] = float(score_match_10.group(1))
            print(f"Found score /10: {result['score']}")
        elif final_score_match:
            result["score"] = float(final_score_match.group(1))
            print(f"Found final score: {result['score']}")
        elif final_score_match_2:
            result["score"] = float(final_score_match_2.group(1))
            print(f"Found final score (pattern 2): {result['score']}")
        elif final_score_match_3:
            result["score"] = float(final_score_match_3.group(1))
            print(f"Found final score (pattern 3): {result['score']}")
        elif final_score_section:
            result["score"] = float(final_score_section.group(1))
            print(f"Found final score in section: {result['score']}")
        elif final_score_heading:
            result["score"] = float(final_score_heading.group(1))
            print(f"Found final score after heading: {result['score']}")
        elif score_match_100:
            # Convert score from /100 to /10 scale
            result["score"] = float(score_match_100.group(1)) / 10
            print(f"Found score /100: {result['score']}")
        elif score_match_plain:
            # Assume it's out of 10 if no denominator is specified
            result["score"] = float(score_match_plain.group(1))
            print(f"Found plain score: {result['score']}")
        elif weighted_scores:
            # Sum up the weighted scores if individual components are found
            try:
                total = sum(float(score) for score in weighted_scores)
                result["score"] = total
                print(f"Calculated score from components: {result['score']}")
            except ValueError:
                pass
        else:
            print("DEBUG: No score pattern matched!")
            print("DEBUG: First 500 characters of report:")
            print(report_content[:500])

        # Extract recommendation
        decision_match = re.search(r'\*\*Decision:\*\* ([^\n]+)', report_content)
        if decision_match:
            result["recommendation"] = decision_match.group(1).strip()

        # Extract strengths
        strengths_match = re.search(r'### ✅ Strengths\n([\s\S]+?)### ❌ Gaps', report_content)
        if strengths_match:
            strengths = strengths_match.group(1).strip().split('\n')
            result["strengths"] = [s.strip('- ').strip() for s in strengths if s.strip()]

        # Extract gaps
        gaps_match = re.search(r'### ❌ Gaps\n([\s\S]+?)## Weighted Score Analysis', report_content)
        if gaps_match:
            gaps = gaps_match.group(1).strip().split('\n')
            result["gaps"] = [g.strip('- ').strip() for g in gaps if g.strip()]

        return result

    except Exception as e:
        result["valid"] = False
        result["error"] = f"Error parsing report: {str(e)}"
        return result


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
                # This is a skill category
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
