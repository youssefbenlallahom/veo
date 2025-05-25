#!/usr/bin/env python
import sys
import warnings
import os
import streamlit as st
import PyPDF2
import tempfile
import re
from datetime import datetime
from dotenv import load_dotenv

# Configure ChromaDB for Streamlit Cloud
os.environ["CHROMA_SERVER_AUTHN_PROVIDER"] = ""
os.environ["CHROMA_SERVER_AUTHN_CREDENTIALS"] = ""
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from crew import Resume

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
        # Extract and save raw data from the PDF
        if hasattr(resume_crew, 'pdf_tool') and resume_crew.pdf_tool is not None:
            try:
                extracted_data = resume_crew.pdf_tool._run('extract_all')
                # Clean up table delimiters and excess blank lines
                cleaned_data = clean_extracted_text(extracted_data)
                extracted_filename = f"extracted_{re.sub(r'[^\w\-_\. ]', '_', candidate_name)}.txt"
                with open(extracted_filename, 'w', encoding='utf-8') as exf:
                    exf.write(cleaned_data)
            except Exception as e:
                print(f"[DEBUG] Could not extract or save raw PDF data: {e}")
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

        # Extract overall score (handle multiple formats - both /10 and /100)
        score_match_10 = re.search(
            r'(?:\*\*Overall Score:\*\*|Overall Score:|TOTAL WEIGHTED SCORE:)\s*\[?([\d\.]+)\s*/\s*10\]?',
            report_content,
            re.IGNORECASE
        )
        score_match_100 = re.search(
            r'(?:\*\*Overall Score:\*\*|Overall Score:|TOTAL WEIGHTED SCORE:)\s*\[?([\d\.]+)\s*/\s*100\]?',
            report_content,
            re.IGNORECASE
        )
        
        if score_match_10:
            result["score"] = float(score_match_10.group(1))
        elif score_match_100:
            # Convert score from /100 to /10 scale
            result["score"] = float(score_match_100.group(1)) / 10

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
        with st.expander(f"#{idx+1}: {result['filename']} - Score: {result['score']} - {result['recommendation']}"):
            if not result["valid"]:
                st.error(f"Error: {result['error']}")
            else:
                st.markdown(f"**Strengths:**")
                for strength in result["strengths"]:
                    st.markdown(f"- {strength}")

                st.markdown(f"**Gaps:**")
                for gap in result["gaps"]:
                    st.markdown(f"- {gap}")

                st.markdown("**Full Report:**")
                st.markdown(result["report_content"])


def run():
    """
    Run the crew to analyze multiple resumes against a job description.
    """
    st.markdown('<style>.stButton>button{background-color:#4CAF50;color:white;}</style>', unsafe_allow_html=True)

    # Navigation bar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["HR", "Home", "About"])

    # Display company name
    st.markdown('<h1 style="color:#4CAF50;">VEO</h1>', unsafe_allow_html=True)
    st.title("Resume Analyzer")

    # Multiple file uploader
    resume_pdfs = st.file_uploader("Upload your resume PDFs", type=["pdf"], accept_multiple_files=True)

    # Updated input fields for job information
    st.subheader("Job Information")
    job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer, Data Scientist, Product Manager")
    job_description = st.text_area("Job Description", placeholder="Paste the job description...", height=200)

    # Gemini LLM barem extraction utility
    def get_barem_from_gemini(job_title, job_description, api_key):
        from google import genai
        import json, re
        client = genai.Client(api_key=api_key)
        prompt = f"""
I will give you a job description. Based on that, create a detailed evaluation barème (scoring rubric) out of 100 points for assessing resumes of applicants to this role.

Divide the rubric into clear sections based on the responsibilities and qualifications mentioned. For each section:
- Assign a weight in points (e.g., 10, 15, 20, etc.).
- Describe what earns full points, partial points, and zero points.
- Include a short explanation of why each category is important for this role.

Also include a "Bonus" section (up to 5 points) for exceptional or relevant extra experience (like certifications, industry familiarity, or advanced technologies).

Output your answer as a JSON object in the following format (use double curly braces for JSON structure):
{{
  "rubric": [
    {{
      "section": "Section Name",
      "weight": 20,
      "scoring": {{
        "full_points": "What earns full points.",
        "partial_points": "What earns partial points.",
        "zero_points": "What earns zero points."
      }},
      "explanation": "Why this section is important."
    }},
    ...
    {{
      "section": "Bonus",
      "weight": 5,
      "scoring": {{
        "full_points": "Exceptional or extra qualifications.",
        "partial_points": "Some relevant extras.",
        "zero_points": "No extras."
      }},
      "explanation": "Bonus for outstanding or additional relevant experience."
    }}
  ]
}}

Use the job description below:
Job Title: {job_title}
Job Description:
{job_description}
"""
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        # Extract JSON from response
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            barem_json = match.group(0)
            return json.loads(barem_json)
        else:
            raise ValueError("Failed to extract barem JSON from Gemini response")

    if st.button("Analyze All Resumes"):
        if resume_pdfs and len(resume_pdfs) > 0 and job_title and job_description:
            # --- Barem extraction and caching ---
            import json
            barem_filename = "barem_gemini.json"
            barem = None
            # Try to load the barem from file if it exists
            if os.path.exists(barem_filename):
                with open(barem_filename, "r", encoding="utf-8") as f:
                    barem = json.load(f)
            else:
                GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
                barem = get_barem_from_gemini(job_title, job_description, GEMINI_API_KEY)
                with open(barem_filename, "w", encoding="utf-8") as f:
                    json.dump(barem, f, ensure_ascii=False, indent=2)
            # Save to session_state for the rest of the app
            st.session_state["barem_gemini"] = barem
            # Container for displaying progress
            progress_container = st.container()
            progress_bar = st.progress(0)

            # Placeholder for the comparison table
            table_placeholder = st.empty()

            # Data structure to store results
            results_data = []

            # Process each resume
            total_files = len(resume_pdfs)

            for idx, resume_pdf in enumerate(resume_pdfs):
                # Update progress
                progress_percent = int((idx / total_files) * 100)
                progress_bar.progress(progress_percent)
                progress_container.text(f"Processing {resume_pdf.name} ({idx+1}/{total_files})")

                # Create temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(resume_pdf.getbuffer())
                    resume_path = tmp_file.name

                # Validate PDF
                is_valid, validation_message = validate_pdf(resume_path)
                if not is_valid:
                    results_data.append({
                        "filename": resume_pdf.name,
                        "valid": False,
                        "error": validation_message,
                        "score": 0,
                        "recommendation": "Failed",
                        "report_content": ""
                    })
                    os.unlink(resume_path)
                    continue

                try:
                    # Extract candidate name from the filename
                    candidate_name = os.path.splitext(resume_pdf.name)[0]

                    # Analyze resume and generate a unique report
                    report_filename = analyze_resume(
                        resume_path=resume_path,
                        job_title=job_title,
                        job_description=job_description,
                        candidate_name=candidate_name,
                        barem=barem
                    )

                    # Parse the generated report
                    candidate_result = parse_report(report_filename, resume_pdf.name)
                    results_data.append(candidate_result)

                except Exception as e:
                    results_data.append({
                        "filename": resume_pdf.name,
                        "valid": False,
                        "error": str(e),
                        "score": 0,
                        "recommendation": "Error",
                        "report_content": ""
                    })

                finally:
                    # Clean up the temporary file silently
                    if os.path.exists(resume_path):
                        try:
                            os.remove(resume_path)
                        except Exception:
                            pass  # Silent cleanup

            # Complete progress bar
            progress_bar.progress(100)
            progress_container.text("Analysis complete!")

            # Display comparison table
            display_comparison_table(results_data, table_placeholder)

            # Display individual reports in expandable sections
            display_individual_reports(results_data)
        else:
            st.error("Please upload at least one resume, enter a job title, and provide a job description.")


if __name__ == "__main__":
    run()
