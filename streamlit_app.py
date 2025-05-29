#!/usr/bin/env python
"""
VEO Resume Analyzer - Streamlit Cloud Compatible Version
This version bypasses ChromaDB/CrewAI compatibility issues by using direct Gemini API calls.
"""
import sys
import warnings
import os
import streamlit as st
import PyPDF2
import tempfile
import re
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

def sanitize_text(text):
    """Sanitize text to remove problematic characters for display."""
    if not text:
        return ""

    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception:
        pass

    text = text.replace('\x00', '')
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\xFF]', '', text)

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

            first_page = pdf_reader.pages[0]
            text = first_page.extract_text()

            return True, f"PDF is valid with {num_pages} pages"
    except Exception as e:
        return False, f"PDF validation failed: {str(e)}"

def clean_extracted_text(text):
    """Clean extracted PDF text."""
    cleaned_lines = []
    prev_blank = False
    for line in text.splitlines():
        if re.fullmatch(r'\s*\|[\s\|]*\|?\s*', line):
            continue
        if re.fullmatch(r'\s*\|+\s*', line):
            continue
        if not line.strip():
            if prev_blank:
                continue
            prev_blank = True
            cleaned_lines.append('')
            continue
        prev_blank = False
        line = re.sub(r'\s*\|+\s*$', '', line)
        line = re.sub(r' {2,}', ' ', line)
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines).strip()

def extract_pdf_text(file_path):
    """Extract text from PDF file."""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return clean_extracted_text(text)
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def get_barem_from_gemini(job_title, job_description, api_key):
    """Generate evaluation criteria using Gemini."""
    try:
        from google import genai
        
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
I will give you a job description. Based on that, create a detailed evaluation barème (scoring rubric) out of 100 points for assessing resumes of applicants to this role.

Divide the rubric into clear sections based on the responsibilities and qualifications mentioned. For each section:
- Assign a weight in points (e.g., 10, 15, 20, etc.).
- Describe what earns full points, partial points, and zero points.
- Include a short explanation of why each category is important for this role.

Also include a "Bonus" section (up to 5 points) for exceptional or relevant extra experience.

Output your answer as a JSON object in the following format:
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
    }}
  ]
}}

Job Title: {job_title}
Job Description:
{job_description}
"""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt
        )
        
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            barem_json = match.group(0)
            return json.loads(barem_json)
        else:
            raise ValueError("Failed to extract barem JSON from Gemini response")
            
    except Exception as e:
        st.error(f"Error generating rubric: {e}")
        return None

def analyze_resume_with_gemini(resume_text, job_title, job_description, barem, candidate_name):
    """Analyze resume using Gemini API."""
    try:
        from google import genai
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"error": "GEMINI_API_KEY not found"}
            
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
Analyze this resume against the job requirements and provide a comprehensive evaluation.

Job Title: {job_title}
Job Description: {job_description}

Candidate Name: {candidate_name}

Resume Content:
{resume_text}

Evaluation Rubric: {json.dumps(barem, indent=2)}

Please provide your analysis in this EXACT format:

# Resume Analysis for {candidate_name}

## Overall Score
**Overall Score:** [X.X]/10

## Decision
**Decision:** [RECOMMENDED/NOT RECOMMENDED]

## Contact Information
- **Name:** {candidate_name}
- **Experience:** [X years]
- **Background:** [Current role/background]

## Job Requirements Analysis
[Analyze how well the candidate meets the specific job requirements]

## Alignment Analysis

### ✅ Strengths
- [Strength 1 - specific to job requirements]
- [Strength 2 - specific to job requirements]
- [Strength 3 - specific to job requirements]

### ❌ Gaps
- [Gap 1 - missing requirement]
- [Gap 2 - missing requirement]
- [Gap 3 - missing requirement]

## Weighted Score Analysis
[For each rubric section, provide score and reasoning]
- **[Section 1]:** [Score]/[Weight] points - [Reasoning]
- **[Section 2]:** [Score]/[Weight] points - [Reasoning]
- **[Section 3]:** [Score]/[Weight] points - [Reasoning]

**Total Weighted Score:** [X]/10

## Recommendation Summary
**Rationale:** [Brief explanation of recommendation]
**Risk Assessment:** [Main concerns if any]
**Next Steps:** [If recommended, what development areas to focus on]
"""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt
        )
        
        return {"analysis": response.text}
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def parse_analysis(analysis_text, filename):
    """Parse the analysis response."""
    result = {
        "filename": filename,
        "valid": True,
        "error": "",
        "score": 0,
        "recommendation": "Unknown",
        "strengths": [],
        "gaps": [],
        "report_content": analysis_text
    }
    
    # Extract score
    score_match = re.search(r'Overall Score:\*\*\s*([\d\.]+)/10', analysis_text)
    if score_match:
        result["score"] = float(score_match.group(1))
    
    # Extract recommendation
    rec_match = re.search(r'Decision:\*\*\s*(RECOMMENDED|NOT RECOMMENDED)', analysis_text)
    if rec_match:
        result["recommendation"] = rec_match.group(1)
    
    # Extract strengths
    strengths_match = re.search(r'### ✅ Strengths\n(.*?)### ❌ Gaps', analysis_text, re.DOTALL)
    if strengths_match:
        strengths = [s.strip('- ').strip() for s in strengths_match.group(1).strip().split('\n') if s.strip() and s.strip().startswith('-')]
        result["strengths"] = strengths
    
    # Extract gaps
    gaps_match = re.search(r'### ❌ Gaps\n(.*?)## Weighted Score Analysis', analysis_text, re.DOTALL)
    if gaps_match:
        gaps = [g.strip('- ').strip() for g in gaps_match.group(1).strip().split('\n') if g.strip() and g.strip().startswith('-')]
        result["gaps"] = gaps
    
    return result

def display_comparison_table(results_data, placeholder):
    """Display a comparative table of all candidates."""
    sorted_results = sorted(results_data, key=lambda x: x["score"], reverse=True)
    max_score = max([r["score"] for r in sorted_results if r["valid"]], default=0)

    import pandas as pd

    if sorted_results:
        df = pd.DataFrame([{
            "Rank": idx + 1,
            "Candidate": r["filename"],
            "Score": f"{r['score']:.1f}/10",
            "Status": "✅ Highest" if r["score"] == max_score else "✅ Valid" if r["valid"] else "❌ Failed"
        } for idx, r in enumerate(sorted_results)])

        with placeholder.container():
            st.subheader("Candidates Comparison")
            st.dataframe(df, use_container_width=True)
    else:
        placeholder.info("No valid results to display")

def display_individual_reports(results_data):
    """Display individual reports in expandable sections."""
    st.subheader("Individual Analysis Reports")
    sorted_results = sorted(results_data, key=lambda x: x["score"], reverse=True)

    for idx, result in enumerate(sorted_results):
        with st.expander(f"#{idx+1}: {result['filename']} - Score: {result['score']:.1f} - {result['recommendation']}"):
            if not result["valid"]:
                st.error(f"Error: {result['error']}")
            else:
                st.markdown("**Strengths:**")
                for strength in result["strengths"]:
                    st.markdown(f"- {strength}")

                st.markdown("**Gaps:**")
                for gap in result["gaps"]:
                    st.markdown(f"- {gap}")

                st.markdown("**Full Report:**")
                st.markdown(result["report_content"])

def main():
    """Main application function."""
    st.set_page_config(
        page_title="VEO Resume Analyzer",
        page_icon="📄",
        layout="wide"
    )
    
    st.markdown('<style>.stButton>button{background-color:#4CAF50;color:white;}</style>', unsafe_allow_html=True)
    
    # Display company name
    st.markdown('<h1 style="color:#4CAF50;">VEO</h1>', unsafe_allow_html=True)
    st.title("Resume Analyzer")
    st.caption("🚀 Streamlit Cloud Compatible Version - Powered by Google Gemini")
    
    # Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("❌ GEMINI_API_KEY environment variable is required!")
        st.info("Please set your Google Gemini API key in the Streamlit Cloud secrets or environment variables.")
        st.stop()
    
    # Navigation bar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["HR Analysis", "About", "Help"])
    
    if page == "About":
        st.markdown("""
        ## About VEO Resume Analyzer
        
        This application helps HR professionals efficiently screen and analyze multiple resumes against job descriptions using AI.
        
        **Features:**
        - Batch resume processing
        - AI-powered scoring and analysis
        - Comparative candidate ranking
        - Detailed individual reports
        
        **Technology:**
        - Google Gemini AI for analysis
        - Streamlit for the interface
        - PyPDF2 for PDF processing
        """)
        return
    elif page == "Help":
        st.markdown("""
        ## How to Use
        
        1. **Upload PDFs**: Select multiple resume PDF files
        2. **Job Details**: Enter the job title and complete job description
        3. **Analyze**: Click "Analyze All Resumes"
        4. **Review**: Check the comparative table and individual reports
        
        ## Tips
        - Ensure PDFs contain extractable text (not scanned images)
        - Provide detailed job descriptions for better analysis
        - The system works best with well-structured resumes
        """)
        return
    
    # Multiple file uploader
    resume_pdfs = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)
    
    if resume_pdfs:
        st.success(f"📁 {len(resume_pdfs)} PDF(s) uploaded successfully")
    
    # Job information
    st.subheader("Job Information")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
    
    with col2:
        job_description = st.text_area("Job Description", placeholder="Paste the complete job description here...", height=150)
    
    if st.button("🔍 Analyze All Resumes", type="primary"):
        if resume_pdfs and len(resume_pdfs) > 0 and job_title and job_description:
            
            # Generate barème
            progress_container = st.container()
            with progress_container:
                with st.spinner("🧠 Generating evaluation criteria..."):
                    api_key = os.environ["GEMINI_API_KEY"]
                    barem = get_barem_from_gemini(job_title, job_description, api_key)
                    
                    if not barem:
                        st.error("❌ Failed to generate evaluation criteria")
                        return
                    
                    st.success("✅ Evaluation criteria generated successfully")
            
            # Process each resume
            results_data = []
            progress_bar = st.progress(0)
            table_placeholder = st.empty()
            
            total_files = len(resume_pdfs)
            
            for idx, resume_pdf in enumerate(resume_pdfs):
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
                        "strengths": [],
                        "gaps": [],
                        "report_content": ""
                    })
                    os.unlink(resume_path)
                    continue
                
                # Extract text and analyze
                candidate_name = os.path.splitext(resume_pdf.name)[0]
                resume_text = extract_pdf_text(resume_path)
                
                if "Error extracting text" in resume_text:
                    results_data.append({
                        "filename": resume_pdf.name,
                        "valid": False,
                        "error": resume_text,
                        "score": 0,
                        "recommendation": "Failed",
                        "strengths": [],
                        "gaps": [],
                        "report_content": ""
                    })
                else:
                    analysis_result = analyze_resume_with_gemini(
                        resume_text, job_title, job_description, barem, candidate_name
                    )
                    
                    if "error" in analysis_result:
                        results_data.append({
                            "filename": resume_pdf.name,
                            "valid": False,
                            "error": analysis_result["error"],
                            "score": 0,
                            "recommendation": "Error",
                            "strengths": [],
                            "gaps": [],
                            "report_content": ""
                        })
                    else:
                        parsed = parse_analysis(analysis_result["analysis"], resume_pdf.name)
                        results_data.append(parsed)
                
                # Clean up
                os.unlink(resume_path)
                
                # Update table in real-time
                if results_data:
                    display_comparison_table(results_data, table_placeholder)
            
            # Complete progress
            progress_bar.progress(100)
            progress_container.text("✅ Analysis complete!")
            
            # Final results display
            display_comparison_table(results_data, table_placeholder)
            display_individual_reports(results_data)
            
        else:
            st.error("❌ Please upload at least one resume, enter a job title, and provide a job description.")

if __name__ == "__main__":
    main()
