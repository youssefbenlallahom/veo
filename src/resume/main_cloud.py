#!/usr/bin/env python
"""
Streamlit Cloud compatible version of the resume analyzer.
This version handles ChromaDB initialization issues gracefully.
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

# Essential environment setup for Streamlit Cloud
def setup_environment():
    """Setup environment variables for Streamlit Cloud compatibility."""
    # ChromaDB configuration
    os.environ.setdefault("CHROMA_SERVER_AUTHN_PROVIDER", "")
    os.environ.setdefault("CHROMA_SERVER_AUTHN_CREDENTIALS", "")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_SERVER_HOST", "localhost")
    os.environ.setdefault("CHROMA_SERVER_HTTP_PORT", "8000")
    os.environ.setdefault("IS_PERSISTENT", "TRUE")
    
    # Disable telemetry and warnings
    os.environ.setdefault("STREAMLIT_TELEMETRY", "False")
    warnings.filterwarnings("ignore")

# Setup environment before any imports
setup_environment()

# Try to import crew with error handling
try:
    from crew import Resume
    CREW_AVAILABLE = True
except Exception as e:
    CREW_AVAILABLE = False
    st.error(f"CrewAI initialization failed: {e}")
    st.info("This might be due to ChromaDB compatibility issues on Streamlit Cloud.")

load_dotenv()

def main():
    """Main application function."""
    st.markdown('<style>.stButton>button{background-color:#4CAF50;color:white;}</style>', unsafe_allow_html=True)
    
    # Display company name
    st.markdown('<h1 style="color:#4CAF50;">VEO</h1>', unsafe_allow_html=True)
    st.title("Resume Analyzer")
    
    if not CREW_AVAILABLE:
        st.error("❌ CrewAI is not available due to environment issues.")
        st.info("""
        This is likely due to ChromaDB compatibility issues on Streamlit Cloud.
        
        **To fix this:**
        1. Ensure all required environment variables are set
        2. Check that all dependencies are properly installed
        3. Consider running this locally or on a different platform
        """)
        return
    
    # Navigation bar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["HR", "Home", "About"])
    
    # Multiple file uploader
    resume_pdfs = st.file_uploader("Upload your resume PDFs", type=["pdf"], accept_multiple_files=True)
    
    # Updated input fields for job information
    st.subheader("Job Information")
    job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer, Data Scientist, Product Manager")
    job_description = st.text_area("Job Description", placeholder="Paste the job description...", height=200)
    
    if st.button("Analyze All Resumes"):
        if resume_pdfs and len(resume_pdfs) > 0 and job_title and job_description:
            try:
                # Your existing analysis code here
                st.success("Analysis would run here when CrewAI is properly initialized.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")
        else:
            st.error("Please upload at least one resume, enter a job title, and provide a job description.")

if __name__ == "__main__":
    main()
