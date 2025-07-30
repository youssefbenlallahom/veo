"""__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')"""

import sys
import warnings
import os
import streamlit as st
import PyPDF2
import tempfile
import re
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv
from crew import Resume
import sys
import os

    
load_dotenv()
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


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
        
        print(f"[SUCCESS] PDF text extracted and saved to: {output_filename}")
        print(f"[INFO] Extracted {len(cleaned_data)} characters from PDF")
        
        return output_filename
        
    except Exception as e:
        print(f"[ERROR] Failed to extract PDF text: {str(e)}")
        return None

def analyze_resume(resume_path, job_title, job_description, candidate_name, barem=None):
    """Run analysis and save the report with a unique filename."""
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
                    extracted_filename = f'extracted_text_{sanitized_name}.txt'
                    
                    with open(extracted_filename, 'w', encoding='utf-8') as text_file:
                        text_file.write(f"=== EXTRACTED PDF CONTENT ===\n")
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

        # Run the analysis
        resume_crew.crew().kickoff(inputs=inputs)

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


def display_comparison_table(results_data, placeholder):
    """Display a comparative table of all candidates."""
    # Sort results by score (highest first)
    sorted_results = sorted(results_data, key=lambda x: x["score"], reverse=True)

    # Find the highest score for highlighting
    max_score = max([r["score"] for r in sorted_results if r["valid"]], default=0)

    # Create DataFrame for display
    import pandas as pd

    if sorted_results:
        df = pd.DataFrame([{
            "Rank": idx + 1,
            "Candidate": r["filename"],
            "Score": f"{r['score']:.1f}/10"  # Format as X.X/10
        } for idx, r in enumerate(sorted_results)])

        # Use placeholder to display the table
        with placeholder.container():
            st.subheader("Candidates Comparison")
            st.dataframe(df, use_container_width=True)
    else:
        placeholder.info("No valid results to display") 


def display_individual_reports(results_data):
    """Display individual reports in expandable sections."""
    st.subheader("Individual Analysis Reports")

    # Sort by score (highest first)
    sorted_results = sorted(results_data, key=lambda x: x["score"], reverse=True)

    for idx, result in enumerate(sorted_results):
        # Create expander with score and recommendation in title
        with st.expander(f"#{idx+1}: {result['filename']} - Score: {result['score']:.1f}/10"):
            if not result["valid"]:
                st.error(f"Error: {result['error']}")
            else:
                # Only show sections if they have content
                if result["strengths"]:
                    st.markdown(f"**Strengths:**")
                    for strength in result["strengths"]:
                        st.markdown(f"- {strength}")

                if result["gaps"]:
                    st.markdown(f"**Gaps:**")
                    for gap in result["gaps"]:
                        st.markdown(f"- {gap}")

                if result["report_content"]:
                    st.markdown("**Full Report:**")
                    st.markdown(result["report_content"])


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


def run():
    """
    Run the crew to analyze multiple resumes against a job description.
    """
    # Add GEMINI_API_KEY at the top of the function
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    
    st.markdown('<style>.stButton>button{background-color:#4CAF50;color:white;}</style>', unsafe_allow_html=True)

    # Navigation bar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["HR", "Home", "About"])

    # Display company name
    st.markdown('<h1 style="color:#4CAF50;">VEO</h1>', unsafe_allow_html=True)
    st.title("Resume Analyzer")

    # Initialize session state for workflow step
    if 'workflow_step' not in st.session_state:
        st.session_state.workflow_step = 'input'
    if 'extracted_skills' not in st.session_state:
        st.session_state.extracted_skills = []
    if 'skills_weights' not in st.session_state:
        st.session_state.skills_weights = {}

    # Step 1: Job Information Input
    if st.session_state.workflow_step == 'input':
        st.subheader("Job Information")
        job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer, Data Scientist, Product Manager")
        job_description = st.text_area("Job Description", placeholder="Paste the job description...", height=200)
        
        if st.button("Extract Key Skills from Job Description", disabled=not (job_title and job_description)):
            try:
                with st.spinner("Extracting key skills from job description..."):
                    skills_data = get_key_skills_from_gemini(job_title, job_description, GEMINI_API_KEY)
                    
                    # Extract both flat and categorized skills
                    all_skills = skills_data.get("hard_skills", [])
                    categorized_skills = skills_data.get("categorized_skills", {})
                    
                    # Check if any skills were extracted
                    if not all_skills:
                        st.error("No technical skills could be extracted from the job description. Please check the job description and try again.")
                        return
                    
                    # Display categorized skills preview
                    st.success(f"Successfully extracted {len(all_skills)} technical skills organized in {len(categorized_skills)} categories!")
                    
                    # Show categorized preview
                    with st.expander("📋 Extracted Skills by Category", expanded=True):
                        for category, skills_list in categorized_skills.items():
                            st.markdown(f"**{category}:** {', '.join(skills_list)}")
                    
                    st.session_state.extracted_skills = all_skills
                    st.session_state.categorized_skills = categorized_skills
                    st.session_state.skills_categories = skills_data  # Store the full structure
                    st.session_state.job_title = job_title
                    st.session_state.job_description = job_description
                    
                    # Initialize weights with equal distribution
                    equal_weight = round(100 / len(all_skills))
                    st.session_state.skills_weights = {skill: equal_weight for skill in all_skills}
                    
                    # Adjust for rounding to ensure total is 100
                    total = sum(st.session_state.skills_weights.values())
                    if total != 100 and all_skills:
                        first_skill = list(st.session_state.skills_weights.keys())[0]
                        st.session_state.skills_weights[first_skill] += (100 - total)
                    
                    st.success(f"Successfully extracted {len(all_skills)} technical skills!")
                    st.session_state.workflow_step = 'weighting'
                    st.rerun()
            except Exception as e:
                st.error(f"Error extracting skills: {str(e)}")
                st.write("Please check your job description and try again. Make sure it contains clear skill requirements.")

    # Step 2: Customize skill weights
    elif st.session_state.workflow_step == 'weighting':
        st.subheader("⚖️ Technical Skills Weighting")
        st.markdown("**Adjust the importance of each technical skill. Total must equal 100%.**")
        
        # Enhanced CSS with drag-and-drop styling
        st.markdown("""
        <style>
        .skill-item {
            background: #f8f9fa;
            padding: 0.8rem;
            border-left: 4px solid #007bff;
            margin: 0.3rem 0;
            border-radius: 0 8px 8px 0;
            cursor: move;
            transition: all 0.3s ease;
        }
        .skill-item:hover {
            background: #e3f2fd;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .skill-category-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.7rem 1rem;
            border-radius: 8px;
            margin: 1rem 0 0.5rem 0;
            font-weight: 600;
            font-size: 1.1rem;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .skill-group {
            background: #fff3e0;
            border: 2px dashed #ff9800;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            min-height: 80px;
        }
        .skill-group-header {
            background: #ff9800;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 8px 8px 0 0;
            margin: -1rem -1rem 0.5rem -1rem;
            font-weight: bold;
        }
        .skills-counter {
            background: #e3f2fd;
            color: #1976d2;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            display: inline-block;
            margin: 0.5rem 0;
            font-weight: 500;
            border: 1px solid #bbdefb;
        }
        .remove-button {
            background: none;
            border: 1px solid #dc3545;
            color: #dc3545;
            border-radius: 4px;
            padding: 0.2rem 0.5rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .remove-button:hover {
            background: #dc3545;
            color: white;
        }
        .custom-skill-input {
            background: #f0f8ff;
            border: 2px solid #4caf50;
            border-radius: 8px;
            padding: 0.5rem;
            margin: 0.5rem 0;
        }
        .drag-drop-zone {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            background: #fafafa;
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        .drag-drop-zone:hover {
            border-color: #007bff;
            background: #f0f8ff;
        }
        .group-item {
            background: #fff8e1;
            border-left: 4px solid #ffc107;
            padding: 0.5rem;
            margin: 0.2rem 0;
            border-radius: 0 6px 6px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize session state for skill groups and custom skills
        if 'removed_skills' not in st.session_state:
            st.session_state.removed_skills = set()
        if 'skill_groups' not in st.session_state:
            st.session_state.skill_groups = {}
        if 'custom_skills' not in st.session_state:
            st.session_state.custom_skills = []
        if 'ungrouped_skills' not in st.session_state:
            # Use the flattened skills list from extraction
            st.session_state.ungrouped_skills = st.session_state.get('extracted_skills', [])
        
        # Remove skill group management: Only show one tab for Individual/Custom Skills
        tab1, tab3 = st.tabs(["📊 Skills", "➕ Custom Skills"])
        
        # Tab 1: Individual and Language Skills
        with tab1:
            # Remove the Individual Summary sidebar/column
            # Only use a single main column for all skill weighting and controls
            st.markdown("""
            <style>
            .skill-card {
                background: #f8fafd;
                border: 1px solid #e3e8ee;
                border-radius: 10px;
                padding: 1.1em 1.3em 0.7em 1.3em;
                margin-bottom: 1.2em;
                box-shadow: 0 2px 8px rgba(30, 136, 229, 0.04);
            }
            .category-header {
                font-size: 1.13rem;
                font-weight: 700;
                color: #1976d2;
                margin-bottom: 0.2em;
                margin-top: 0.1em;
            }
            .skill-list {
                color: #333;
                font-size: 0.98rem;
                margin-left: 1.2em;
                margin-bottom: 0.5em;
            }
            .lang-header {
                font-size: 1.05rem;
                font-weight: 600;
                color: #388e3c;
                margin-bottom: 0.2em;
                margin-top: 0.1em;
            }
            .custom-header {
                font-size: 1.05rem;
                font-weight: 600;
                color: #b26a00;
                margin-bottom: 0.2em;
                margin-top: 0.1em;
            }
            hr.section-divider {
                border: none;
                border-top: 1px solid #e0e0e0;
                margin: 0.7em 0 1em 0;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Main skill weighting area (single column)
            with st.container():
                # Get current ungrouped skills excluding removed ones
                ungrouped_skills = [skill for skill in st.session_state.ungrouped_skills 
                                  if skill not in st.session_state.removed_skills]
                
                # Also exclude skills that are already in groups
                grouped_skills = set()
                for group_skills in st.session_state.skill_groups.values():
                    grouped_skills.update(group_skills)
                ungrouped_skills = [skill for skill in ungrouped_skills if skill not in grouped_skills]
                
                new_individual_weights = {}
                
                # Get categorized skills for better display
                categorized_skills = st.session_state.get('categorized_skills', {})
                
                # Prepare lists for UI
                skills_by_category = {}
                uncategorized_skills = []
                language_skills = []
                    
                for skill in ungrouped_skills:
                    found_category = None
                    for category, category_skills in categorized_skills.items():
                        if skill in category_skills:
                            found_category = category
                            break
                    if found_category == "Languages":
                        language_skills.append(skill)
                    elif found_category:
                        if found_category not in skills_by_category:
                            skills_by_category[found_category] = []
                        skills_by_category[found_category].append(skill)
                    else:
                        uncategorized_skills.append(skill)
                
                # Display skill categories (slider only for category, not for individual skills)
                for category, skills_in_category in skills_by_category.items():
                    with st.container():
                        st.markdown(f"<div class='skill-card'><div class='category-header'>{category}</div>", unsafe_allow_html=True)
                        new_weight = st.slider(
                            label="",  # No label, just the slider
                            min_value=0,
                            max_value=100,
                            value=st.session_state.skills_weights.get(category, 0),
                            step=1,
                            key=f"category_weight_{category}",
                            help=f"Weight for {category} (all skills in this category)"
                        )
                        new_individual_weights[category] = new_weight
                        # Update session state immediately with the new weight
                        st.session_state.skills_weights[category] = new_weight
                        st.markdown(f"<div class='skill-list'>{', '.join(skills_in_category)}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Display language skills (each as its own slider)
                if language_skills:
                    st.markdown(f"<div class='skill-card'><div class='lang-header'>Languages</div>", unsafe_allow_html=True)
                    for skill in language_skills:
                        new_weight = st.slider(
                            label=skill,
                            min_value=0,
                            max_value=100,
                            value=st.session_state.skills_weights.get(skill, 0),
                            step=1,
                            key=f"language_weight_{skill}",
                            help=f"Weight for {skill} (Language)"
                        )
                        new_individual_weights[skill] = new_weight
                        # Update session state immediately with the new weight
                        st.session_state.skills_weights[skill] = new_weight
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Display uncategorized skills (custom skills)
                if uncategorized_skills:
                    st.markdown(f"<div class='skill-card'><div class='custom-header'>Custom/Uncategorized Skills</div>", unsafe_allow_html=True)
                    for skill in uncategorized_skills:
                        new_weight = st.slider(
                            label=skill,
                            min_value=0,
                            max_value=100,
                            value=st.session_state.skills_weights.get(skill, 0),
                            step=1,
                            key=f"custom_weight_{skill}",
                            help=f"Weight for {skill} (Custom/Uncategorized)"
                        )
                        new_individual_weights[skill] = new_weight
                        # Update session state immediately with the new weight
                        st.session_state.skills_weights[skill] = new_weight
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    if not skills_by_category and not language_skills:
                        st.info("All skills are either grouped or removed. Use other tabs to manage skills.")
            
            # Right sidebar for individual skills
            # with col2: # This column is removed
            #     st.markdown("### Individual Summary")
            #     total_skills = len(ungrouped_skills)
            #     st.metric("Ungrouped Skills", total_skills)
            #     individual_total = sum(new_individual_weights.get(skill, 0) for skill in new_individual_weights)
            #     if individual_total == 0:
            #         st.info("0%")
            #     else:
            #         st.metric("Individual Weight", f"{individual_total}%")
        
        # Tab 2: Custom Skills
        with tab3:
            st.markdown("### ➕ Add Custom Skills")
            st.markdown("Add skills that weren't automatically detected from the job description.")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                custom_skill_input = st.text_input(
                    "Enter custom skill",
                    placeholder="e.g., Machine Learning, Docker, Kubernetes",
                    key="custom_skill_input"
                )
            
            with col2:
                if st.button("Add Skill", disabled=not custom_skill_input):
                    if custom_skill_input and custom_skill_input not in st.session_state.custom_skills:
                        st.session_state.custom_skills.append(custom_skill_input)
                        st.session_state.ungrouped_skills.append(custom_skill_input)
                        st.session_state.skills_weights[custom_skill_input] = 0
                        st.success(f"Added '{custom_skill_input}'!")
                        st.rerun()
                    else:
                        st.error("Skill already exists or is empty!")
            
            # Display custom skills
            if st.session_state.custom_skills:
                st.markdown("**Custom Skills Added:**")
                for i, skill in enumerate(st.session_state.custom_skills):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.session_state.skills_weights[skill] = st.slider(
                            label=skill,
                            min_value=0,
                            max_value=100,
                            value=st.session_state.skills_weights.get(skill, 0),
                            step=1,
                            key=f"custom_weight_{skill}_{i}"
                        )
                    
                    with col2:
                        if st.button("🗑️", key=f"remove_custom_{skill}_{i}", help="Remove custom skill"):
                            st.session_state.custom_skills.remove(skill)
                            if skill in st.session_state.ungrouped_skills:
                                st.session_state.ungrouped_skills.remove(skill)
                            if skill in st.session_state.skills_weights:
                                del st.session_state.skills_weights[skill]
                            st.rerun()
            else:
                st.info("No custom skills added yet.")
        
        # Overall summary and controls (remove Groups column)
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1:
            # Calculate totals using session state for real-time updates
            all_current_weights = st.session_state.skills_weights
            individual_total = sum(all_current_weights.get(skill, 0) for skill in new_individual_weights.keys())
            custom_individual_total = sum(
                all_current_weights.get(skill, 0) 
                for skill in st.session_state.custom_skills 
            )
            grand_total = individual_total + custom_individual_total
            
            # Display summary
            col_summary1, col_summary2, col_summary3 = st.columns(3)
            with col_summary1:
                st.metric("Categories/Skills", f"{individual_total}%")
            with col_summary2:
                st.metric("Custom", f"{custom_individual_total}%")
            with col_summary3:
                if grand_total == 100:
                    st.success(f"✅ {grand_total}%")
                elif grand_total < 100:
                    st.warning(f"⚠️ {grand_total}%")
                else:
                    st.error(f"❌ {grand_total}%")
        with col2:
            st.markdown("### Controls")
            if st.button("Auto-Distribute", help="Distribute weights automatically"):
                # Get all active items (categories, languages, and custom skills)
                all_active_items = []
                
                # Add skill categories
                categorized_skills = st.session_state.get('categorized_skills', {})
                for category in categorized_skills.keys():
                    if category != "Languages":  # Languages are handled individually
                        all_active_items.append(category)
                
                # Add individual language skills
                for category, skills_list in categorized_skills.items():
                    if category == "Languages":
                        all_active_items.extend(skills_list)
                
                # Add custom skills
                all_active_items.extend(st.session_state.get('custom_skills', []))
                
                # Distribute weights equally
                if all_active_items:
                    equal_weight = round(100 / len(all_active_items))
                    
                    # Clear existing weights
                    st.session_state.skills_weights = {}
                    
                    # Assign equal weights
                    for item in all_active_items:
                        st.session_state.skills_weights[item] = equal_weight
                    
                    # Adjust for rounding to ensure total is 100
                    total = sum(st.session_state.skills_weights.values())
                    if total != 100 and all_active_items:
                        st.session_state.skills_weights[all_active_items[0]] += (100 - total)
                    
                    st.rerun()
            
            # Add debug button to show current weights
            if st.button("🔍 Show Current Weights", help="Display current weight configuration"):
                st.markdown("**Current Weight Configuration:**")
                
                # Create a temporary barem to show what would be sent
                temp_barem = create_custom_barem(
                    st.session_state.skills_weights,
                    st.session_state.get('categorized_skills', {})
                )
                
                # Show inline preview
                with st.expander("Current Barem Preview", expanded=True):
                    for item_name, config in temp_barem.items():
                        st.write(f"**{item_name}**: {config['weight']}% ({config['type']})")
                        if 'skills' in config:
                            st.write(f"  └─ {', '.join(config['skills'])}")
                
                total_weight = sum(st.session_state.skills_weights.values())
                if total_weight == 100:
                    st.success(f"✅ Total: {total_weight}%")
                else:
                    st.warning(f"⚠️ Total: {total_weight}% (Must be 100%)")
        # Navigation
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("← Back"):
                st.session_state.workflow_step = 'input'
                st.rerun()
        with col2:
            # Use the recalculated grand_total
            total_current_weight = sum(st.session_state.skills_weights.values())
            if total_current_weight == 100:
                st.success("Ready!")
            else:
                st.warning("Adjust to 100%")
        with col3:
            # Calculate total active items and weights
            categorized_skills = st.session_state.get('categorized_skills', {})
            custom_skills = st.session_state.get('custom_skills', [])
            
            # Count categories (excluding Languages which are handled individually)
            skill_categories_count = len([cat for cat in categorized_skills.keys() if cat != "Languages"])
            
            # Count individual language skills
            language_skills_count = len(categorized_skills.get("Languages", []))
            
            # Count custom skills
            custom_skills_count = len(custom_skills)
            
            total_items = skill_categories_count + language_skills_count + custom_skills_count
            total_current_weight = sum(st.session_state.skills_weights.values())
            proceed_disabled = total_current_weight != 100 or total_items == 0
            
            if st.button("Continue →", disabled=proceed_disabled):
                # Update extracted_skills with the final configuration
                final_configuration = []
                
                # Add skill categories
                for category in categorized_skills.keys():
                    if category != "Languages" and st.session_state.skills_weights.get(category, 0) > 0:
                        final_configuration.append(category)
                
                # Add individual language skills
                for skill in categorized_skills.get("Languages", []):
                    if st.session_state.skills_weights.get(skill, 0) > 0:
                        final_configuration.append(skill)
                
                # Add custom skills
                for skill in custom_skills:
                    if st.session_state.skills_weights.get(skill, 0) > 0:
                        final_configuration.append(skill)
                
                st.session_state.extracted_skills = final_configuration
                st.session_state.workflow_step = 'upload'
                st.rerun()

    # Step 3: Resume upload and analysis
    elif st.session_state.workflow_step == 'upload':
        st.subheader("Upload Resumes for Analysis")
        
        # Display current configuration
        with st.expander("📋 Configuration Summary", expanded=False):
            st.write(f"**🎯 Job Title:** {st.session_state.job_title}")
            
            # Get current configuration data
            categorized_skills = st.session_state.get('categorized_skills', {})
            custom_skills = st.session_state.get('custom_skills', [])
            skills_weights = st.session_state.get('skills_weights', {})
            
            # Count skill categories and individual skills
            skill_categories = []
            individual_language_skills = []
            individual_custom_skills = []
            
            # Separate skills into categories, language skills, and custom skills
            for weight_key, weight_value in skills_weights.items():
                if weight_value > 0:  # Only show skills with assigned weights
                    if weight_key in categorized_skills:
                        # This is a skill category
                        skill_categories.append((weight_key, weight_value, categorized_skills[weight_key]))
                    elif weight_key in custom_skills:
                        # This is a custom skill
                        individual_custom_skills.append((weight_key, weight_value))
                    else:
                        # Check if it's a language skill from the original categorization
                        is_language = False
                        for category, skills_list in categorized_skills.items():
                            if category == "Languages" and weight_key in skills_list:
                                individual_language_skills.append((weight_key, weight_value))
                                is_language = True
                                break
                        if not is_language:
                            # It's an individual skill from other categories
                            individual_custom_skills.append((weight_key, weight_value))
            
            st.write(f"**📊 Configuration:**")
            st.write(f"• Skill Categories: {len(skill_categories)}")
            st.write(f"• Individual Language Skills: {len(individual_language_skills)}")
            st.write(f"• Custom/Individual Skills: {len(individual_custom_skills)}")
            
            st.write(f"**⚖️ Weight Distribution:**")
            
            # Display skill categories with their weights and included skills
            if skill_categories:
                st.write("**📁 Skill Categories:**")
                for category_name, weight, skills_list in skill_categories:
                    st.write(f"• **{category_name}**: {weight}%")
                    st.write(f"  └─ Skills: {', '.join(skills_list)}")
            
            # Display individual language skills
            if individual_language_skills:
                st.write("**🌐 Language Skills:**")
                for skill, weight in individual_language_skills:
                    st.write(f"• {skill}: {weight}%")
            
            # Display custom/individual skills
            if individual_custom_skills:
                st.write("**� Custom/Individual Skills:**")
                for skill, weight in individual_custom_skills:
                    st.write(f"• {skill}: {weight}%")
            
            # Total weight validation
            total_weight = sum(weight for weight in skills_weights.values())
            
            if total_weight == 100:
                st.success(f"✅ Total Weight: {total_weight}%")
            elif total_weight == 0:
                st.warning("⚠️ No weights assigned yet")
            else:
                st.error(f"❌ Total Weight: {total_weight}% (Must equal 100%)")
        
        # File upload
        resume_pdfs = st.file_uploader("Upload your resume PDFs", type=["pdf"], accept_multiple_files=True)
        
        # Async analysis function
        async def run_analysis_async():
            """Run the async analysis process with custom barem."""
            # Use the custom barem from session state
            barem = st.session_state.get('custom_barem', {})
            job_title = st.session_state.get('job_title', '')
            job_description = st.session_state.get('job_description', '')
            
            # Container for displaying progress
            progress_container = st.container()
            progress_bar = st.progress(0)

            # Placeholder for the comparison table
            table_placeholder = st.empty()

            # Progress callback function
            def update_progress(completed, total):
                progress_percent = int((completed / total) * 100)
                progress_bar.progress(progress_percent)
                progress_container.text(f"Processing resumes concurrently... ({completed}/{total})")

            # Run async analysis
            results_data = await analyze_all_resumes_async(
                resume_pdfs, job_title, job_description, barem, update_progress
            )

            # Complete progress bar
            progress_bar.progress(100)
            progress_container.text("Analysis complete!")

            # Display comparison table
            display_comparison_table(results_data, table_placeholder)

            # Display individual reports in expandable sections
            display_individual_reports(results_data)
        
        if resume_pdfs:
            st.write(f"Uploaded {len(resume_pdfs)} file(s)")
            
            if st.button("Analyze All Resumes"):
                # Create barem from custom weights and categorized skills
                barem = create_custom_barem(
                    st.session_state.skills_weights, 
                    st.session_state.get('categorized_skills', {})
                )
                st.session_state['custom_barem'] = barem
                
                # Run the async analysis
                try:
                    asyncio.run(run_analysis_async())
                except Exception as e:
                    st.error(f"Error processing resumes: {str(e)}")
        
        # Navigation
        if st.button("← Back to Skill Weighting"):
            st.session_state.workflow_step = 'weighting'
            st.rerun()


if __name__ == "__main__":
    run()
