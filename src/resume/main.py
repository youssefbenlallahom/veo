#!/usr/bin/env python
import sys
import warnings
import os
import streamlit as st
import PyPDF2
import tempfile
import re
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from crew import Resume
import sys
import os
import sys
import pysqlite3

sys.modules["sqlite3"] = pysqlite3
    
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

def analyze_resume(resume_path, job_title, job_description, candidate_name, barem=None):
    """Run analysis and save the report with a unique filename."""
    try:
        # Create Resume instance and process
        resume_crew = Resume(pdf_path=resume_path)
        # Extract raw data from the PDF (no saving to file)
        if hasattr(resume_crew, 'pdf_tool') and resume_crew.pdf_tool is not None:
            try:
                extracted_data = resume_crew.pdf_tool._run('extract_all')
                # Clean up table delimiters and excess blank lines
                cleaned_data = clean_extracted_text(extracted_data)
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
        # Extract raw data from the PDF (no saving to file)
        if hasattr(resume_crew, 'pdf_tool') and resume_crew.pdf_tool is not None:
            try:
                extracted_data = resume_crew.pdf_tool._run('extract_all')
                # Clean up table delimiters and excess blank lines
                cleaned_data = clean_extracted_text(extracted_data)
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
    """Extract key skills from job description using Gemini."""
    from google import genai
    import json, re
    client = genai.Client(api_key=api_key)
    prompt = f"""
You are a deterministic skill extraction system. Follow these rules EXACTLY to ensure identical results every time.

Job Title: {job_title}
Job Description: {job_description}

MANDATORY STANDARDIZED SKILL DICTIONARY:
Use ONLY these exact phrases when extracting skills. Do not deviate from this terminology:

HARD SKILLS:
- Education: "Bachelor's Degree" or "Master's Degree" 
- Languages: "French B2 Level", "English B2 Level", "Spanish B2 Level", etc.
- Software: "Microsoft Office Suite", "SAP", "Python Programming", "SQL", "Tableau"
- Technical: "Data Analysis", "Financial Analysis", "Account Reconciliation", "Project Management"
- Industry-specific: "Supplier Account Management", "Payment Processing", "Banking Operations"

SOFT SKILLS:
- Personal: "Detail-Oriented", "Analytical Thinking", "Problem Solving"
- Management: "Time Management", "Stress Management", "Planning and Organization"
- Interpersonal: "Team Collaboration", "Communication Skills", "Leadership"

EXTRACTION ALGORITHM:
1. Read job description once, completely
2. Extract skills in this EXACT order:
   a) Education requirements (if mentioned)
   b) Language requirements (alphabetical by language)
   c) Software/technical tools (alphabetical)
   d) Domain-specific hard skills (alphabetical)
   e) Soft skills (alphabetical within category)

3. For each potential skill, check if it appears in the standardized dictionary above
4. If exact match exists, use the dictionary term
5. If no exact match, use the closest dictionary equivalent
6. Extract exactly 10 skills total

STRICT MATCHING RULES:
- "Bachelor's/Master's" → "Bachelor's Degree"
- "Microsoft Office" → "Microsoft Office Suite"
- "Analytical mindset/thinking" → "Analytical Thinking"
- "Planning/organizational" → "Planning and Organization"
- "Team/teamwork/collaboration" → "Team Collaboration"
- "Time and stress management" → "Time Management"
- "Detail-oriented/meticulous" → "Detail-Oriented"
- "Account reconciliation/reconciliations" → "Account Reconciliation"
- "Supplier account analysis/management" → "Supplier Account Management"

FORBIDDEN VARIATIONS:
- Never use synonyms or alternative phrasings
- Never combine skills (e.g., "Time and Stress Management")
- Never change word order
- Never add descriptive words

OUTPUT FORMAT (exactly 10 skills, no more, no less):
[
    "Skill 1",
    "Skill 2",
    "Skill 3",
    "Skill 4",
    "Skill 5",
    "Skill 6",
    "Skill 7",
    "Skill 8",
    "Skill 9",
    "Skill 10"
]
"""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    # Extract JSON from response
    match = re.search(r'\[.*\]', response.text, re.DOTALL)
    if match:
        skills_json = match.group(0)
        return json.loads(skills_json)
    else:
        raise ValueError("Failed to extract skills JSON from Gemini response")


def create_custom_barem(skills_weights):
    """Create a barem structure from user-defined skills and weights."""
    barem = {}
    for skill, weight in skills_weights.items():
        barem[skill] = {
            "weight": weight,
            "criteria": [
                f"Has extensive experience and expertise in {skill}",
                f"Has some experience with {skill}",
                f"Shows potential or basic knowledge in {skill}"
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
                    extracted_skills = get_key_skills_from_gemini(job_title, job_description, GEMINI_API_KEY)
                    st.session_state.extracted_skills = extracted_skills
                    st.session_state.job_title = job_title
                    st.session_state.job_description = job_description
                    # Initialize weights with equal distribution
                    equal_weight = round(100 / len(extracted_skills))
                    st.session_state.skills_weights = {skill: equal_weight for skill in extracted_skills}
                    # Adjust for rounding to ensure total is 100
                    total = sum(st.session_state.skills_weights.values())
                    if total != 100:
                        first_skill = list(st.session_state.skills_weights.keys())[0]
                        st.session_state.skills_weights[first_skill] += (100 - total)
                    st.session_state.workflow_step = 'weighting'
                    st.rerun()
            except Exception as e:
                st.error(f"Error extracting skills: {str(e)}")

    # Step 2: Customize skill weights
    elif st.session_state.workflow_step == 'weighting':
        st.subheader("Customize Skill Weights")
        st.write("Adjust the importance of each skill (total must equal 100%):")
        
        # Create columns for better layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create sliders for each skill
            new_weights = {}
            for skill in st.session_state.extracted_skills:
                new_weights[skill] = st.slider(
                    f"{skill}",
                    min_value=0,
                    max_value=100,
                    value=st.session_state.skills_weights.get(skill, 0),
                    step=1,
                    key=f"weight_{skill}"
                )
        
        with col2:
            # Display current total
            current_total = sum(new_weights.values())
            if current_total == 100:
                st.success(f"Total: {current_total}%")
            else:
                st.error(f"Total: {current_total}%")
            
            # Reset to equal weights button
            if st.button("Reset to Equal Weights"):
                equal_weight = round(100 / len(st.session_state.extracted_skills))
                for skill in st.session_state.extracted_skills:
                    st.session_state[f"weight_{skill}"] = equal_weight
                st.rerun()
        
        # Update session state
        st.session_state.skills_weights = new_weights
        
        # Navigation buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("← Back to Job Input"):
                st.session_state.workflow_step = 'input'
                st.rerun()
        
        with col2:
            # Auto-normalize button
            if st.button("Auto-Normalize to 100%") and current_total > 0:
                normalized_weights = {skill: round((weight / current_total) * 100) for skill, weight in new_weights.items()}
                # Adjust for rounding errors
                total_normalized = sum(normalized_weights.values())
                if total_normalized != 100:
                    first_skill = list(normalized_weights.keys())[0]
                    normalized_weights[first_skill] += (100 - total_normalized)
                
                # Update sliders
                for skill, weight in normalized_weights.items():
                    st.session_state[f"weight_{skill}"] = weight
                st.session_state.skills_weights = normalized_weights
                st.rerun()
        
        with col3:
            if st.button("Proceed to Resume Upload →", disabled=current_total != 100):
                st.session_state.workflow_step = 'upload'
                st.rerun()

    # Step 3: Resume upload and analysis
    elif st.session_state.workflow_step == 'upload':
        st.subheader("Upload Resumes for Analysis")
        
        # Display current configuration
        with st.expander("Current Configuration", expanded=False):
            st.write(f"**Job Title:** {st.session_state.job_title}")
            st.write(f"**Skills & Weights:**")
            for skill, weight in st.session_state.skills_weights.items():
                st.write(f"• {skill}: {weight}%")
        
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
                # Create barem from custom weights
                barem = create_custom_barem(st.session_state.skills_weights)
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
