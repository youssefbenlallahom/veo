#!/usr/bin/env python
"""
Streamlit web interface for the Resume Analysis CrewAI project.
This file contains only UI logic, with business logic handled by main.py
"""

import sys
import os
import re
from pathlib import Path

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import streamlit as st
import tempfile
import pandas as pd
from typing import List, Dict, Any

# Import the main analysis engine
from main import ResumeAnalysisEngine, run_batch_analysis
from utils.file_handler import FileHandler

# Configure Streamlit page
st.set_page_config(
    page_title="VEO Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .candidate-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .score-high { color: #4CAF50; font-weight: bold; }
    .score-medium { color: #ff9800; font-weight: bold; }
    .score-low { color: #f44336; font-weight: bold; }
    .recommendation-strong { color: #2e7d32; font-weight: bold; }
    .recommendation-recommended { color: #4CAF50; font-weight: bold; }
    .recommendation-conditional { color: #ff9800; font-weight: bold; }
    .recommendation-not { color: #f44336; font-weight: bold; }
    .confidence-high { color: #4CAF50; font-weight: bold; }
    .confidence-medium { color: #ff9800; font-weight: bold; }
    .confidence-low { color: #f44336; font-weight: bold; }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .executive-summary {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .critical-concern {
        background: #ffebee;
        padding: 0.5rem;
        border-radius: 4px;
        border-left: 4px solid #f44336;
        margin: 0.25rem 0;
    }
    .strength-item {
        background: #e8f5e8;
        padding: 0.5rem;
        border-radius: 4px;
        border-left: 4px solid #4CAF50;
        margin: 0.25rem 0;
    }
    .gap-item {
        background: #fff3e0;
        padding: 0.5rem;
        border-radius: 4px;
        border-left: 4px solid #ff9800;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Place these helper functions at the top of the file (after imports) so they are always defined before use
import re

def extract_score_from_markdown(md):
    # Match numbers like 8.7 in lines like '**Overall Score:** **8.7/10**' or '- **Overall Score:** **8.7/10**' or 'Overall Score: 8.7/10'
    match = re.search(r'Overall Score[\s\S]*?(\d+(?:\.\d+)?)\s*/\s*10', md)
    if match:
        return float(match.group(1))
    return 0.0

def extract_recommendation_from_markdown(md):
    match = re.search(r'Decision: ?([A-Z ]+)', md)
    if match:
        return match.group(1).strip()
    return "Unknown"


class StreamlitApp:
    """Main Streamlit application for resume analysis."""
    
    def __init__(self):
        self.file_handler = FileHandler(Path("output"))
        
    def run(self):
        """Run the Streamlit application."""
        # Header
        st.markdown('<h1 style="color:#4CAF50;">VEO</h1>', unsafe_allow_html=True)
        st.title("🤖 AI Resume Analyzer")
        st.markdown("*Powered by CrewAI - Intelligent Resume Analysis*")
        
        # Sidebar navigation
        self._render_sidebar()
        
        # Main content area
        page = st.session_state.get('page', 'Home')
        
        if page == 'Home':
            self._render_home_page()
        elif page == 'HR':
            self._render_hr_page()
        elif page == 'About':
            self._render_about_page()
    
    def _render_sidebar(self):
        """Render the sidebar navigation."""
        st.sidebar.title("📋 Navigation")
        
        pages = {
            'Home': '🏠 Home',
            'HR': '👥 HR Dashboard',
            'About': 'ℹ️ About'
        }
        
        for key, display_name in pages.items():
            if st.sidebar.button(display_name, key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()
        
        # System status
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 System Status")
        
        # Check environment variables
        required_vars = ['GEMINI_API_KEY']
        all_good = True
        
        for var in required_vars:
            if os.getenv(var):
                st.sidebar.success(f"✅ {var}")
            else:
                st.sidebar.error(f"❌ {var}")
                all_good = False
        
        if all_good:
            st.sidebar.success("🚀 System Ready")
        else:
            st.sidebar.error("⚠️ Configuration Issues")
    
    def _render_home_page(self):
        """Render the home page with resume upload and analysis."""
        st.header("📄 Resume Analysis")
        
        # File upload section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📁 Upload Resumes")
            resume_files = st.file_uploader(
                "Select resume PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload one or more PDF resumes for analysis"
            )
            
            if resume_files:
                st.success(f"📊 {len(resume_files)} resume(s) uploaded")
                
                # Display uploaded files
                for i, file in enumerate(resume_files, 1):
                    st.write(f"{i}. {file.name}")
        
        with col2:
            st.subheader("📊 Upload Summary")
            if resume_files:
                st.write("✅ Ready for analysis")
                st.write(f"**Files:** {len(resume_files)} resume(s)")
            else:
                st.write("📁 No files uploaded yet")
                st.write("Upload PDF resumes to begin")
        
        # Job information section
        st.subheader("💼 Job Requirements")
        
        job_title = st.text_input(
            "Job Title",
            placeholder="e.g., Senior Software Engineer",
            help="Enter the position title"
        )
        
        job_description = st.text_area(
            "Job Description",
            placeholder="Paste the complete job description here...",
            height=200,
            help="Provide detailed job requirements, responsibilities, and qualifications"
        )
        
        # Analysis button
        if st.button("🚀 Start Analysis", type="primary"):
            self._run_analysis(resume_files, job_title, job_description)
    
    def _render_hr_page(self):
        """Render the HR dashboard page."""
        st.header("👥 HR Dashboard")
        
        # Metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Analyses", st.session_state.get('total_analyses', 0))
        
        with col2:
            st.metric("📈 Success Rate", f"{st.session_state.get('success_rate', 0):.1f}%")
        
        with col3:
            st.metric("⭐ Avg Score", f"{st.session_state.get('avg_score', 0):.1f}/10")
        
        with col4:
            st.metric("🕒 Last Analysis", st.session_state.get('last_analysis', 'None'))
        
        # Recent analyses section
        st.subheader("📋 Recent Analyses")
        
        if 'recent_analyses' in st.session_state and st.session_state.recent_analyses:
            # Display recent analyses in a table
            df = pd.DataFrame(st.session_state.recent_analyses)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent analyses available. Start analyzing resumes to see results here.")
        
        # Batch analysis section
        st.subheader("📦 Batch Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("Upload multiple resumes for batch processing:")
            batch_files = st.file_uploader(
                "Select multiple resume PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                key="batch_upload"
            )
        
        with col2:
            if batch_files:
                st.success(f"📊 {len(batch_files)} files ready for batch analysis")
                
                batch_job_title = st.text_input("Batch Job Title", key="batch_job_title")
                batch_job_description = st.text_area(
                    "Batch Job Description", 
                    height=100, 
                    key="batch_job_description"
                )
                
                if st.button("🚀 Start Batch Analysis", type="primary"):
                    if batch_job_title and batch_job_description:
                        self._run_batch_analysis(batch_files, batch_job_title, batch_job_description)
                    else:
                        st.error("Please provide job title and description for batch analysis.")
    
    def _render_about_page(self):
        """Render the about page with system information."""
        st.header("ℹ️ About VEO Resume Analyzer")
        
        # System overview
        st.markdown("""
        ### 🤖 AI-Powered Resume Analysis
        
        **VEO Resume Analyzer** leverages CrewAI and advanced LLMs to automate and standardize 
        resume screening for HR teams and recruiters with comprehensive JSON-based reporting.
        
        #### Key Features:
        - **📊 Comprehensive Scoring**: Multi-dimensional evaluation with detailed scoring breakdown
        - **📋 Executive Summary**: Clear recommendations with confidence levels and decision factors
        - **🎯 Business Impact Assessment**: Immediate and long-term value analysis
        - **📈 Competitive Analysis**: Market positioning and salary alignment insights
        - **📄 Multi-format Support**: Analyze PDF resumes with advanced text extraction
        - **🔍 Detailed Reports**: Structured JSON reports with strengths, gaps, and recommendations
        - ** Batch Processing**: Handle multiple resumes simultaneously
        - **🔒 Privacy-First**: No data storage, all processing is temporary
        
        #### New JSON Structure Benefits:
        - **📊 Structured Data**: Machine-readable format for easy integration
        - **🎯 Executive Focus**: Clear executive summary for quick decision-making
        - **📈 Business Metrics**: Impact assessment and competitive analysis
        - **🔍 Quality Assurance**: Built-in validation and confidence scoring
        - **📋 Comprehensive Coverage**: Multi-dimensional candidate evaluation
        """)
        
        # Technical specifications
        with st.expander("🔧 Technical Specifications"):
            st.markdown("""
            - **AI Models**: Google Gemini, Azure OpenAI
            - **Framework**: CrewAI for multi-agent orchestration
            - **PDF Processing**: pypdf, PDFplumber for text extraction
            - **Interface**: Streamlit for web UI
            - **Output Format**: Structured JSON with comprehensive analysis
            - **Scoring System**: Multi-dimensional weighted scoring
            - **Security**: API-key based authentication, no data persistence
            """)
        
        # Usage instructions
        with st.expander("📚 How to Use"):
            st.markdown("""
            1. **Setup**: Ensure GEMINI_API_KEY is configured in environment
            2. **Upload**: Select one or more PDF resumes
            3. **Job Details**: Provide job title and description
            4. **Analysis**: Click 'Start Analysis' to begin processing
            5. **Results**: View comprehensive JSON-structured reports including:
               - Executive Summary with recommendations
               - Detailed scoring breakdown
               - Business impact assessment
               - Competitive analysis
               - Job requirements matching
            6. **Export**: Download structured reports for record-keeping
            """)
        
        # New features highlight
        with st.expander("🆕 New Features in JSON Format"):
            st.markdown("""
            - **🎯 Executive Summary**: Quick decision-making insights
            - **📊 Detailed Scoring**: Multi-dimensional evaluation metrics
            - **💼 Business Impact**: Immediate and long-term value assessment
            - **🏆 Competitive Analysis**: Market positioning and advantages
            - **📋 Requirements Matching**: Detailed job fit analysis
            - **🔍 Quality Assurance**: Confidence levels and data validation
            - **📈 Stakeholder Impact**: Analysis for different organizational levels
            """)
        
        # Contact and support
        st.markdown("""
        ---
        **Contact**: [Your Company](mailto:contact@yourcompany.com)  
        **Version**: 1.0.0  
        **Last Updated**: July 2025
        """)
    
    def _run_analysis(self, resume_files, job_title, job_description):
        """Run the resume analysis process."""
        resume_file_list = []  # Ensure this is always defined
        if not resume_files:
            st.error("Please upload at least one resume.")
            return
        
        if not job_title or not job_description:
            st.error("Please provide job title and description.")
            return
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize analysis engine
            engine = ResumeAnalysisEngine()
            
            # Prepare file list for batch processing
            for i, uploaded_file in enumerate(resume_files):
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    candidate_name = Path(uploaded_file.name).stem
                    resume_file_list.append((tmp_file.name, candidate_name))
            
            # Update initial status
            status_text.text("Starting analysis...")
            
            # Run batch analysis
            results = engine.analyze_multiple_resumes(
                resume_file_list, 
                job_title, 
                job_description,
                progress_callback=lambda idx, total, candidate: self._update_progress(progress_bar, status_text, idx, total, candidate)
            )
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")
            
            # Display results
            self._display_results(results)
            
            # Update session state
            self._update_session_state(results)
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
        finally:
            # Clean up temporary files
            for file_path, _ in resume_file_list:
                try:
                    os.unlink(file_path)
                except:
                    pass
    
    def _run_batch_analysis(self, batch_files, job_title, job_description):
        """Run batch analysis for multiple resumes."""
        # Convert uploaded files to format expected by run_batch_analysis
        resume_file_list = []
        
        for uploaded_file in batch_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                candidate_name = Path(uploaded_file.name).stem
                resume_file_list.append((tmp_file.name, candidate_name))
        
        # Run batch analysis
        with st.spinner("Running batch analysis..."):
            results = run_batch_analysis(resume_file_list, job_title, job_description)
        
        # Display results
        self._display_results(results)
        
        # Update session state
        self._update_session_state(results)
        
        # Clean up temporary files
        for file_path, _ in resume_file_list:
            try:
                os.unlink(file_path)
            except:
                pass
    
    def _display_results(self, results):
        """Display analysis results in a formatted way using the new JSON structure."""
        if not results:
            st.warning("No results to display.")
            return
        
        st.success(f"✅ Analysis completed for {len(results)} candidates")
        
        # Results summary
        st.subheader("📊 Results Summary")
        
        # Create results dataframe
        df_data = []
        for result in results:
            executive_summary = result.get('executive_summary', {})
            candidate_profile = result.get('candidate_profile', {})
            score_raw = executive_summary.get('overall_score', result.get('score', 0))
            try:
                score = float(score_raw)
            except (ValueError, TypeError):
                score = 0.0
            recommendation = executive_summary.get('overall_recommendation', result.get('recommendation', 'Unknown'))
            if isinstance(recommendation, (list, dict)):
                recommendation = str(recommendation)
            # Fallback: extract from markdown if missing
            if (not score or score == 0) and result.get('report_content'):
                score = extract_score_from_markdown(result['report_content'])
            if (not recommendation or recommendation == 'Unknown') and result.get('report_content'):
                recommendation = extract_recommendation_from_markdown(result['report_content'])
            confidence = executive_summary.get('confidence_level', 'Unknown')
            df_data.append({
                'Candidate': result.get('candidate_name', 'Unknown'),
                'Score': f"{score:.1f}/10",
                'Recommendation': recommendation,
                'Confidence': confidence,
                'Experience': 'Unknown',
                'Status': '✅ Success' if result.get('valid', True) else '❌ Failed'
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # Detailed results
        st.subheader("📋 Detailed Results")
        for i, result in enumerate(results, 1):
            candidate_name = result.get('candidate_name', 'Unknown')
            executive_summary = result.get('executive_summary', {})
            score_raw = executive_summary.get('overall_score', result.get('score', 0))
            try:
                score = float(score_raw)
            except (ValueError, TypeError):
                score = 0.0
            recommendation = executive_summary.get('overall_recommendation', result.get('recommendation', 'Unknown'))
            if isinstance(recommendation, (list, dict)):
                recommendation = str(recommendation)
            # Fallback: extract from markdown if missing
            if (not score or score == 0) and result.get('report_content'):
                score = extract_score_from_markdown(result['report_content'])
            if (not recommendation or recommendation == 'Unknown') and result.get('report_content'):
                recommendation = extract_recommendation_from_markdown(result['report_content'])
            strengths = result.get('strengths', [])
            gaps = result.get('gaps', [])
            report_content = result.get('report_content', '')

            with st.expander(f"#{i}: {candidate_name} - {score:.1f}/10"):
                if result.get('valid', True):
                    st.markdown("### 🎯 Executive Summary")
                    st.write(f"**Candidate Name:** {executive_summary.get('candidate_name', '')}")
                    st.write(f"**Position Applied:** {executive_summary.get('position_applied', '')}")
                    st.write(f"**Overall Recommendation:** {executive_summary.get('overall_recommendation', '')}")
                    st.write(f"**Overall Score:** {executive_summary.get('overall_score', '')}")
                    st.write(f"**Confidence Level:** {executive_summary.get('confidence_level', '')}")
                    st.write("**Key Decision Factors:**")
                    for factor in executive_summary.get('key_decision_factors', []):
                        st.markdown(f"- {factor}")
                    st.write("**Critical Concerns:**")
                    for concern in executive_summary.get('critical_concerns', []):
                        st.markdown(f"- {concern}")
                    st.write(f"**Summary:** {executive_summary.get('recommendation_summary', '')}")

                    candidate_profile = result.get('candidate_profile', {})
                    st.markdown("### 👤 Candidate Profile")
                    st.write(f"**Professional Summary:** {candidate_profile.get('professional_summary', '')}")
                    st.write(f"**Career Trajectory:** {candidate_profile.get('career_trajectory', '')}")
                    st.write(f"**Total Experience:** {candidate_profile.get('total_experience', '')}")
                    st.write(f"**Relevant Experience:** {candidate_profile.get('relevant_experience', '')}")
                    st.write(f"**Education Level:** {candidate_profile.get('education_level', '')}")
                    st.write(f"**Current Status:** {candidate_profile.get('current_status', '')}")
                    st.write(f"**Geographic Considerations:** {candidate_profile.get('geographic_considerations', '')}")
                    st.write(f"**Career Progression Analysis:** {candidate_profile.get('career_progression_analysis', '')}")
                    st.write(f"**Industry Experience:** {', '.join(candidate_profile.get('industry_experience', []))}")
                    st.write(f"**Company Sizes Worked:** {', '.join(candidate_profile.get('company_sizes_worked', []))}")

                    st.markdown("### ✅ Strengths and Differentiators")
                    if strengths:
                        for strength in strengths:
                            st.markdown(f"- {strength}")
                    else:
                        st.markdown("No strengths found.")

                    st.markdown("### ❌ Gaps and Risk Assessment")
                    if gaps:
                        for gap in gaps:
                            st.markdown(f"- {gap}")
                    else:
                        st.markdown("No gaps found.")

                    st.markdown("### 📄 Full Report (JSON)")
                    with st.expander("Show raw JSON output"):
                        st.json(result)
                else:
                    st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
    
    def _update_session_state(self, results):
        """Update session state with analysis results using the new JSON structure."""
        if not results:
            return
        
        # Update metrics
        valid_results = [r for r in results if r.get('valid', True)]
        
        st.session_state.total_analyses = st.session_state.get('total_analyses', 0) + len(results)
        st.session_state.success_rate = (len(valid_results) / len(results)) * 100 if results else 0
        
        # Calculate average score from new structure
        total_score = 0
        score_count = 0
        for result in valid_results:
            executive_summary = result.get('executive_summary', {})
            score_raw = executive_summary.get('overall_score', 0)
            try:
                score = float(score_raw)
            except (ValueError, TypeError):
                score = 0.0
            if score > 0:
                total_score += score
                score_count += 1
        
        st.session_state.avg_score = total_score / score_count if score_count > 0 else 0
        st.session_state.last_analysis = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
        
        # Update recent analyses
        recent_analyses = st.session_state.get('recent_analyses', [])
        
        for result in results:
            executive_summary = result.get('executive_summary', {})
            score_raw = executive_summary.get('overall_score', 0)
            try:
                score = float(score_raw)
            except (ValueError, TypeError):
                score = 0.0
            recent_analyses.append({
                'Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                'Candidate': executive_summary.get('candidate_name', 'Unknown'),
                'Score': f"{score:.1f}/10",
                'Recommendation': executive_summary.get('overall_recommendation', 'Unknown'),
                'Confidence': executive_summary.get('confidence_level', 'Unknown'),
                'Status': '✅ Success' if result.get('valid', True) else '❌ Failed'
            })
        
        # Keep only last 10 analyses
        st.session_state.recent_analyses = recent_analyses[-10:]
    
    def _update_progress(self, progress_bar, status_text, idx, total, candidate):
        """Update progress bar and status text during analysis."""
        if total > 0:
            progress = (idx + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Processing {candidate} ({idx + 1}/{total})...")
        
    def _get_score_color_class(self, score: float) -> str:
        """Get CSS class for score color based on value."""
        if score >= 8.0:
            return "score-high"
        elif score >= 6.0:
            return "score-medium"
        else:
            return "score-low"
    
    def _get_recommendation_color_class(self, recommendation: str) -> str:
        """Get CSS class for recommendation color based on value."""
        recommendation_lower = recommendation.lower()
        if "strongly recommended" in recommendation_lower:
            return "recommendation-strong"
        elif "recommended" in recommendation_lower and "not" not in recommendation_lower:
            return "recommendation-recommended"
        elif "conditional" in recommendation_lower:
            return "recommendation-conditional"
        else:
            return "recommendation-not"
    
    def _get_confidence_color_class(self, confidence: str) -> str:
        """Get CSS class for confidence color based on value."""
        confidence_lower = confidence.lower()
        if confidence_lower == "high":
            return "confidence-high"
        elif confidence_lower == "medium":
            return "confidence-medium"
        else:
            return "confidence-low"
    
    def _format_score_display(self, score: float) -> str:
        """Format score for display with color coding."""
        color_class = self._get_score_color_class(score)
        return f'<span class="{color_class}">{score:.1f}/10</span>'
    
    def _format_recommendation_display(self, recommendation: str) -> str:
        """Format recommendation for display with color coding."""
        color_class = self._get_recommendation_color_class(recommendation)
        return f'<span class="{color_class}">{recommendation}</span>'
    
    def _format_confidence_display(self, confidence: str) -> str:
        """Format confidence for display with color coding."""
        color_class = self._get_confidence_color_class(confidence)
        return f'<span class="{color_class}">{confidence}</span>'


def main():
    """Main entry point for the Streamlit application."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()