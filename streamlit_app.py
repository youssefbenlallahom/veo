#!/usr/bin/env python
"""
Streamlit Cloud compatible version that avoids ChromaDB issues
by using a simplified CrewAI configuration.
"""
import sys
import warnings
import os
import streamlit as st
import PyPDF2
import tempfile
import re
from datetime import datetime
from dotenv import load_dotenv

# Set environment variables to disable ChromaDB features
os.environ["CHROMA_SERVER_AUTHN_PROVIDER"] = ""
os.environ["CHROMA_SERVER_AUTHN_CREDENTIALS"] = ""
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["DISABLE_CHROMA"] = "True"

load_dotenv()
warnings.filterwarnings("ignore")

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

def extract_pdf_text(file_path):
    """Extract text from PDF file."""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def get_barem_from_gemini(job_title, job_description, api_key):
    """Generate evaluation criteria using Gemini."""
    try:
        from google import generativeai as genai
        import json
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
Create a detailed evaluation rubric for assessing resumes for this job position.

Job Title: {job_title}
Job Description: {job_description}

Create a JSON rubric with sections, weights (totaling 100), and scoring criteria.
Format:
{{
  "rubric": [
    {{
      "section": "Technical Skills",
      "weight": 30,
      "scoring": {{
        "full_points": "Has all required technical skills",
        "partial_points": "Has most required technical skills",
        "zero_points": "Missing critical technical skills"
      }},
      "explanation": "Technical skills are crucial for this role"
    }}
  ]
}}
"""
        
        response = model.generate_content(prompt)
        # Extract JSON from response
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            barem_json = match.group(0)
            return json.loads(barem_json)
        else:
            raise ValueError("Failed to extract rubric JSON from Gemini response")
            
    except Exception as e:
        st.error(f"Error generating rubric: {e}")
        return None

def analyze_resume_simple(resume_text, job_title, job_description, barem):
    """Simple resume analysis without CrewAI."""
    try:
        from google import generativeai as genai
        
        # Configure Gemini
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"error": "GEMINI_API_KEY not found"}
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
Analyze this resume against the job requirements and provide a score out of 10.

Job Title: {job_title}
Job Description: {job_description}

Resume Content:
{resume_text}

Evaluation Rubric: {barem}

Provide your analysis in this format:
SCORE: X.X/10
RECOMMENDATION: RECOMMENDED/NOT RECOMMENDED
STRENGTHS:
- Strength 1
- Strength 2

GAPS:
- Gap 1
- Gap 2

ANALYSIS:
Detailed analysis...
"""
        
        response = model.generate_content(prompt)
        return {"analysis": response.text}
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def parse_analysis(analysis_text):
    """Parse the analysis response."""
    result = {
        "score": 0,
        "recommendation": "Unknown",
        "strengths": [],
        "gaps": [],
        "analysis": analysis_text
    }
    
    # Extract score
    score_match = re.search(r'SCORE:\s*([\d\.]+)/10', analysis_text)
    if score_match:
        result["score"] = float(score_match.group(1))
    
    # Extract recommendation
    rec_match = re.search(r'RECOMMENDATION:\s*(RECOMMENDED|NOT RECOMMENDED)', analysis_text)
    if rec_match:
        result["recommendation"] = rec_match.group(1)
    
    # Extract strengths
    strengths_match = re.search(r'STRENGTHS:(.*?)GAPS:', analysis_text, re.DOTALL)
    if strengths_match:
        strengths = [s.strip('- ').strip() for s in strengths_match.group(1).strip().split('\n') if s.strip()]
        result["strengths"] = strengths
    
    # Extract gaps
    gaps_match = re.search(r'GAPS:(.*?)ANALYSIS:', analysis_text, re.DOTALL)
    if gaps_match:
        gaps = [g.strip('- ').strip() for g in gaps_match.group(1).strip().split('\n') if g.strip()]
        result["gaps"] = gaps
    
    return result

def main():
    """Main application."""
    st.markdown('<style>.stButton>button{background-color:#4CAF50;color:white;}</style>', unsafe_allow_html=True)
    
    # Display company name
    st.markdown('<h1 style="color:#4CAF50;">VEO</h1>', unsafe_allow_html=True)
    st.title("Resume Analyzer (Streamlit Cloud Compatible)")
    
    st.info("🚀 This version uses direct Gemini API calls to avoid ChromaDB compatibility issues on Streamlit Cloud.")
    
    # Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("❌ GEMINI_API_KEY environment variable is required!")
        st.stop()
    
    # Navigation bar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["HR", "Home", "About"])
    
    # Multiple file uploader
    resume_pdfs = st.file_uploader("Upload your resume PDFs", type=["pdf"], accept_multiple_files=True)
    
    # Job information
    st.subheader("Job Information")
    job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
    job_description = st.text_area("Job Description", placeholder="Paste the job description...", height=200)
    
    if st.button("Analyze All Resumes"):
        if resume_pdfs and len(resume_pdfs) > 0 and job_title and job_description:
            # Generate barème
            with st.spinner("Generating evaluation criteria..."):
                api_key = os.environ["GEMINI_API_KEY"]
                barem = get_barem_from_gemini(job_title, job_description, api_key)
                
                if not barem:
                    st.error("Failed to generate evaluation criteria")
                    return
            
            # Process each resume
            results_data = []
            progress_bar = st.progress(0)
            
            for idx, resume_pdf in enumerate(resume_pdfs):
                progress_bar.progress((idx + 1) / len(resume_pdfs))
                
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
                        "recommendation": "Failed"
                    })
                    os.unlink(resume_path)
                    continue
                
                # Extract text and analyze
                resume_text = extract_pdf_text(resume_path)
                analysis_result = analyze_resume_simple(resume_text, job_title, job_description, barem)
                
                if "error" in analysis_result:
                    results_data.append({
                        "filename": resume_pdf.name,
                        "valid": False,
                        "error": analysis_result["error"],
                        "score": 0,
                        "recommendation": "Error"
                    })
                else:
                    parsed = parse_analysis(analysis_result["analysis"])
                    results_data.append({
                        "filename": resume_pdf.name,
                        "valid": True,
                        "error": "",
                        "score": parsed["score"],
                        "recommendation": parsed["recommendation"],
                        "strengths": parsed["strengths"],
                        "gaps": parsed["gaps"],
                        "analysis": parsed["analysis"]
                    })
                
                # Clean up
                os.unlink(resume_path)
            
            # Display results
            progress_bar.progress(1.0)
            
            # Sort by score
            results_data.sort(key=lambda x: x["score"], reverse=True)
            
            # Display table
            import pandas as pd
            df = pd.DataFrame([{
                "Rank": idx + 1,
                "Candidate": r["filename"],
                "Score": f"{r['score']:.1f}/10",
                "Status": "✅ Valid" if r["valid"] else "❌ Failed"
            } for idx, r in enumerate(results_data)])
            
            st.subheader("Results")
            st.dataframe(df, use_container_width=True)
            
            # Individual reports
            st.subheader("Individual Reports")
            for idx, result in enumerate(results_data):
                with st.expander(f"#{idx+1}: {result['filename']} - Score: {result['score']:.1f}"):
                    if result["valid"]:
                        st.markdown(f"**Recommendation:** {result['recommendation']}")
                        st.markdown("**Strengths:**")
                        for strength in result.get("strengths", []):
                            st.markdown(f"- {strength}")
                        st.markdown("**Gaps:**")
                        for gap in result.get("gaps", []):
                            st.markdown(f"- {gap}")
                        st.markdown("**Full Analysis:**")
                        st.text(result.get("analysis", "No analysis available"))
                    else:
                        st.error(f"Error: {result['error']}")
        else:
            st.error("Please upload at least one resume, enter a job title, and provide a job description.")

if __name__ == "__main__":
    main()
